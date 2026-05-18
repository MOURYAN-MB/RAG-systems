# Phase 2 — LangChain RAG with MMR Retrieval

> **Goal:** Replace basic similarity search with MMR retrieval and build a proper LangChain LCEL pipeline over real academic PDFs.

An AI research paper assistant that loads landmark papers (Transformers, GPT-3, RAG, etc.), indexes them with page-level metadata, and answers questions with citations including exact page numbers.

## Architecture

```
PDF Papers (Transformers, GPT-3, RAG...)
      │  PyPDFLoader → page metadata
      ▼
 ChromaDB  ←─── HuggingFace Embeddings
      │  MMR retrieval (relevance + diversity)
      ▼
 LangChain LCEL Chain
 PromptTemplate → LLM → StrOutputParser
      │
      ▼
 Answer + expandable source docs with page numbers
```

## Key Features

- **MMR Retrieval** — Max Marginal Relevance balances relevance *and* diversity, preventing the LLM from seeing 5 nearly identical chunks
- Page-level metadata preserved through the whole pipeline
- Source documents shown with page numbers in expandable sections
- Build/rebuild index on demand from sidebar
- Auto-loads existing vectorstore on startup (no re-embedding)
- Proper LangChain LCEL chain (`prompt | llm | parser`)

## What Changed vs Phase 1

| Feature | Phase 1 | Phase 2 |
|---------|---------|---------|
| Retrieval | Cosine similarity (top-k) | MMR (relevance + diversity) |
| Documents | Any uploaded file | Real academic PDFs with metadata |
| Page tracking | No | Yes (shown in sources) |
| Chain | Basic | LCEL (`prompt \| llm \| parser`) |
| Chunking | Simple splitter | RecursiveCharacterTextSplitter with separators |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Vector Store | ChromaDB + LangChain |
| Retrieval | MMR (Max Marginal Relevance) |
| Embeddings | HuggingFace via LangChain |
| Chain | LangChain LCEL |
| LLM | Ollama / Anthropic / OpenAI |

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Set OLLAMA_MODEL or API keys

# Add PDFs to data/ folder, then build index from the sidebar
streamlit run app.py
```

## What This Phase Covers

- MMR vs cosine similarity trade-offs
- LangChain LCEL composition pattern
- Document metadata preservation through chunking and retrieval
- RecursiveCharacterTextSplitter separator hierarchy
- Building reusable retriever abstractions