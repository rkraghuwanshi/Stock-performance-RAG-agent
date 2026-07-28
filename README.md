# AI Agents with LangGraph — Local RAG Agent

A retrieval-augmented generation (RAG) agent built with [LangGraph](https://langchain-ai.github.io/langgraph/) that answers questions about a PDF document. Everything runs locally: a quantized Llama 2 model via `llama.cpp` for generation, a HuggingFace sentence-transformer for embeddings, and ChromaDB for vector storage. No API keys, no network calls at inference time.

The bundled document is `Stock_Market_Performance_2024.pdf`.

## How it works

The agent is a two-node LangGraph state machine with a loop back to the model, so the LLM can call the retriever as many times as it needs before answering.

```
              ┌─────────────────┐
  entry ────▶ │      llm        │ ──── no tool calls ────▶ END
              └─────────────────┘
                 ▲           │
                 │      tool calls
                 │           ▼
              ┌─────────────────┐
              │ retriever_agent │
              └─────────────────┘
```

| Piece | Where | What it does |
| --- | --- | --- |
| `AgentState` | [RAG_agent.py:121](RAG_agent.py#L121) | Single `messages` key, accumulated with `operator.add` |
| `call_llm` | [RAG_agent.py:144](RAG_agent.py#L144) | Prepends the system prompt and invokes the tool-bound LLM |
| `take_action` | [RAG_agent.py:153](RAG_agent.py#L153) | Executes each requested tool call, returns `ToolMessage`s |
| `should_conitnue` | [RAG_agent.py:126](RAG_agent.py#L126) | Conditional edge — routes to the retriever if the last message has `tool_calls` |
| `retriever_tool` | [RAG_agent.py:91](RAG_agent.py#L91) | Similarity search over ChromaDB, top 5 chunks |

Ingestion pipeline: `PyPDFLoader` → `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap) → `all-MiniLM-L6-v2` embeddings → Chroma collection `stock_market`.

## Requirements

- Python 3.12 (the checked-in bytecode cache is `cpython-312`)
- ~4 GB free disk for the GGUF model, plus RAM to load it
- A C compiler toolchain for `llama-cpp-python` (on macOS: `xcode-select --install`)

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model file is expected at `model/llama-2-7b-langchain-chat-q4_0.gguf` (3.6 GB, not tracked in git). Download any Llama 2 7B chat GGUF quantization and place it there, or point `model_path` at [RAG_agent.py:19](RAG_agent.py#L19) to a model you already have.

## Running

```bash
python RAG_agent.py
```

The script ingests the PDF, builds the vector store, then drops into an interactive prompt:

```
=== RAG AGENT===

What is your question: How did the S&P 500 perform in 2024?
```

Type `exit` or `quit` to leave.

## Configuration

Tunables, all near the top of [RAG_agent.py](RAG_agent.py):

| Setting | Line | Default |
| --- | --- | --- |
| `model_path` | [19](RAG_agent.py#L19) | `model/llama-2-7b-langchain-chat-q4_0.gguf` |
| `n_ctx` (context window) | [22](RAG_agent.py#L22) | `5120` |
| `max_tokens` | [21](RAG_agent.py#L21) | `512` |
| `pdf_path` | [28](RAG_agent.py#L28) | `Stock_Market_Performance_2024.pdf` |
| Chunk size / overlap | [48-49](RAG_agent.py#L48-L49) | `1000` / `200` |
| `persist_directory` | [55](RAG_agent.py#L55) | absolute path — see below |
| Retrieved chunks (`k`) | [80](RAG_agent.py#L80) | `5` |
| System prompt | [131](RAG_agent.py#L131) | Stock-market-specific |

To point the agent at a different PDF, change `pdf_path`, pick a fresh `collection_name`, and update the system prompt so it describes the new document.

## Notes and known rough edges

- **`persist_directory` is hardcoded** to `/Users/rohit/Documents/rk/AI Agents with LangGraph/` at [RAG_agent.py:55](RAG_agent.py#L55). It won't work on another machine — make it relative (e.g. `"./chroma_db"`) before sharing.
- **The PDF is re-ingested on every run.** `Chroma.from_documents` is called unconditionally, so each run appends another copy of the chunks to the persisted collection and duplicates pile up in `chroma.sqlite3`. To fix, load the existing store when the persist directory already has data and only call `from_documents` on first run.
- **The vector store lives in the repo root**: `chroma.sqlite3` plus the UUID-named index directory (`59bdc6e2-…`). Both are generated artifacts and belong in `.gitignore`, along with `model/*.gguf` and `__pycache__/`.
- **`requirements.txt` lists unused packages** — `dotenv`, `langchain_openai`, and `langchain` are not imported by the current code, which uses `ChatLlamaCpp` locally instead of a hosted model.
- **Tool-calling with a local Llama 2 model is unreliable.** `bind_tools` works, but a 4-bit 7B chat model frequently answers without invoking the retriever. If retrieval isn't firing, that's the usual reason — a larger or more instruction-tuned model handles it better.
