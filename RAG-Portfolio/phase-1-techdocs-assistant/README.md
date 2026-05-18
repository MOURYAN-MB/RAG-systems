# Phase 1 — TechDocs Assistant

> **Goal:** Build the simplest possible production-quality RAG chatbot from scratch, understanding every component.

A Streamlit document Q&A assistant that answers questions grounded in uploaded PDFs and text files. Swap between Ollama (local), Anthropic Claude, and OpenAI at runtime from the sidebar.

## Architecture

```
PDF / Text File
      │  PyPDF + text splitter
      ▼
 ChromaDB  ←─── HuggingFace Embeddings (all-MiniLM-L6-v2)
      │  top-k vector similarity
      ▼
 Prompt (question + retrieved chunks)
      │
      ▼
 LLM  (Ollama / Claude / GPT-4)
      │
      ▼
 Answer + Source chunks with similarity scores
```

## Key Features

- Upload any PDF or text file → immediately queryable
- Multi-provider LLM: switch between Ollama, Claude, and GPT from sidebar
- Source transparency: every answer shows the exact chunks used
- Persistent ChromaDB vectorstore — no re-embedding on restart
- Chat history with Streamlit session state

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Vector Store | ChromaDB |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) |
| Chunking | Recursive text splitter — 500 tokens, 50 overlap |
| LLM | Ollama (llama3.2) / Anthropic Claude / OpenAI GPT |

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

cp .env.example .env
# Edit .env: set OLLAMA_MODEL or add ANTHROPIC_API_KEY / OPENAI_API_KEY

# Local LLM (optional)
ollama pull llama3.2

streamlit run app.py
# → http://localhost:8501
```

## What This Phase Covers

- Core RAG loop: embed → store → retrieve → generate
- ChromaDB collection management and persistence
- Prompt engineering to ground answers in retrieved context
- LangChain LCEL chain composition
- Multi-provider LLM abstraction