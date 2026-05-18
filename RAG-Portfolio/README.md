# GenAI & RAG Engineering Portfolio

A 5-phase progression from basic RAG to production-grade adaptive retrieval systems — built with local LLMs (Ollama), LangChain, LangGraph, FastAPI, PostgreSQL, Elasticsearch, and Airflow.

Each phase adds a meaningful layer of complexity: better retrieval, agentic decision-making, multi-agent coordination, and finally automated quality evaluation with self-healing.

---

## Phases at a Glance

| Phase | Project | What It Does | Key Innovation |
|-------|---------|-------------|----------------|
| 1 | [TechDocs Assistant](./phase-1-techdocs-assistant) | Document Q&A chatbot | Core RAG loop: embed → store → retrieve → generate |
| 2 | [LangChain RAG](./phase-2-langchain-rag) | AI paper research assistant | MMR retrieval + LCEL chains + page metadata |
| 3 | [LangGraph Agent](./phase-3-langgraph-agent) | Agentic research assistant | Agent decides: papers vs live web vs no tool |
| 4 | [Multi-Agent Supervisor](./phase-4-multi-agent-supervisor) | Research → report pipeline | Task decomposition: research agent + writer agent |
| 5 | [Adaptive RAG Engine](./phase-5-adaptive-rag-engine) | Production RAG platform | Hybrid retrieval + RAGAS eval + self-healing |

---

## Architecture Progression

```
Phase 1: Question → ChromaDB → LLM → Answer
         (cosine similarity, any document)

Phase 2: Question → ChromaDB (MMR) → LCEL chain → Answer + page citations
         (diverse retrieval, academic PDFs)

Phase 3: Question → LangGraph Agent → [search papers | search web] → Answer
         (agent decides which tool to use, multi-step reasoning)

Phase 4: Task → Supervisor → Research Agent → Writer Agent → Report
         (specialized agents, task decomposition, downloadable output)

Phase 5: Question → Hybrid Retrieval (BM25 + vector + RRF + rerank) → LLM → Answer
                 ↓ async
         RAGAS evaluation → quality scores → Prometheus → Grafana
                 ↓ if quality drops
         Airflow DAG → automatic re-indexing → alert resolved
```

---

## Technology Comparison

| Feature | Ph 1 | Ph 2 | Ph 3 | Ph 4 | Ph 5 |
|---------|------|------|------|------|------|
| Vector DB | ChromaDB | ChromaDB | ChromaDB | — | pgvector |
| Full-text search | — | — | — | — | Elasticsearch |
| Retrieval | cosine | MMR | agent-routed | agent-routed | BM25 + vector + RRF |
| Reranking | — | — | — | — | Cross-encoder |
| Pipeline | chain | LCEL | LangGraph | LangGraph | FastAPI + background |
| Agents | — | — | 1 | 3 | — |
| Evaluation | — | — | — | — | RAGAS |
| Monitoring | — | — | — | — | Prometheus + Grafana |
| Orchestration | — | — | — | — | Airflow DAG |
| UI | Streamlit | Streamlit | Streamlit (streaming) | Streamlit (2-col) | FastAPI + Streamlit |
| Infrastructure | local | local | local | local | Docker Compose (6 services) |

---

## LLM Providers

All phases support switching between providers at runtime:

| Provider | Model | Notes |
|----------|-------|-------|
| **Ollama** (default) | `llama3.2`, `llama3.1` | Fully local, no API key needed |
| **Anthropic** | Claude Haiku / Sonnet / Opus | Fastest inference, best quality |
| **OpenAI** | GPT-4o mini / GPT-4o | Widely supported |

---

## Prerequisites

```bash
# Required for all phases
python >= 3.11
pip

# For local LLM (Phases 1-4, and Phase 5 default)
ollama                   # https://ollama.ai
ollama pull llama3.2

# For Phase 3 & 4 web search
# TAVILY_API_KEY in .env  (free at https://app.tavily.com)

# For Phase 5 infrastructure
docker + docker-compose
```

---

## Running Each Phase

Each phase is self-contained. Navigate to the phase folder and follow its README:

```bash
# Example: Phase 1
cd phase-1-techdocs-assistant
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py

# Example: Phase 5
cd phase-5-adaptive-rag-engine
docker-compose up -d postgres elasticsearch redis
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --port 8080 --reload
```

---

## Key Concepts Demonstrated

**Retrieval Quality**
- Why cosine similarity alone fails (duplicate chunks, no diversity)
- MMR: balancing relevance and diversity in retrieved context
- Hybrid BM25 + vector: keyword precision + semantic understanding
- RRF fusion: merging ranked lists without score calibration
- Cross-encoder reranking: expensive but accurate final re-scoring

**Agentic Patterns**
- Tool routing: when to search documents vs the live web
- Agent state machines with LangGraph `StateGraph`
- Supervisor pattern vs fully agentic routing (trade-offs)
- Task decomposition: research ≠ synthesis

**Production RAG**
- Async evaluation: RAGAS scores without adding latency
- Quality thresholds and automated alerting
- Self-healing: quality drops trigger automatic re-indexing
- Prometheus metrics for RAG-specific quality signals
- Dockerized infrastructure for repeatable deployments

---

## Project Structure

```
RAG-Portfolio/
├── README.md                              ← This file
├── phase-1-techdocs-assistant/
│   ├── README.md
│   ├── .env.example
│   ├── app.py
│   ├── requirements.txt
│   └── src/
├── phase-2-langchain-rag/
│   ├── README.md
│   ├── .env.example
│   ├── app.py
│   ├── requirements.txt
│   └── src/
├── phase-3-langgraph-agent/
│   ├── README.md
│   ├── .env.example
│   ├── app.py
│   ├── requirements.txt
│   └── src/
├── phase-4-multi-agent-supervisor/
│   ├── README.md
│   ├── .env.example
│   ├── app.py
│   ├── requirements.txt
│   └── src/
└── phase-5-adaptive-rag-engine/
    ├── README.md
    ├── .env.example
    ├── docker-compose.yml
    ├── requirements.txt
    ├── api/
    ├── dashboard/
    ├── dags/
    ├── src/
    └── prometheus/
```

---

## About

A self-directed GenAI engineering portfolio demonstrating the full spectrum of RAG system design — from fundamentals to production.
