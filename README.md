# RAG Systems — GenAI Engineering Portfolio

> A 5-phase hands-on journey: from a 50-line document chatbot to a production RAG system with hybrid retrieval, automated quality evaluation, and self-healing.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?logo=chainlink&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.13-005571?logo=elasticsearch&logoColor=white)
![Apache Airflow](https://img.shields.io/badge/Airflow-2.9-017CEE?logo=apacheairflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)

**Runs fully offline. No mandatory API keys.**

---

## What is RAG and Why Does It Matter?

Large Language Models know everything up to their training cutoff — and nothing after. Ask GPT-4 about your internal documents, last week's report, or a proprietary dataset: it hallucinates or refuses.

**Retrieval-Augmented Generation (RAG)** fixes this by injecting your own documents into the LLM's context at query time:

```mermaid
flowchart LR
    subgraph Without RAG
        Q1[Question] --> LLM1[LLM\nfrozen knowledge]
        LLM1 --> A1[Answer\nmay hallucinate]
    end

    subgraph With RAG
        Q2[Question] --> R[Search your docs]
        R --> C[Relevant chunks]
        C --> LLM2[LLM + context]
        LLM2 --> A2[Grounded answer\nwith citations]
    end
```

The result: accurate, up-to-date, source-cited answers over **your own data** — even content the LLM has never seen. RAG is now the dominant architecture for enterprise AI: legal document analysis, medical knowledge bases, customer support, and internal knowledge management.

---

## Why 5 Phases?

Building a production RAG system means solving many distinct problems. Each phase introduces **exactly one new concept**, keeping the learning incremental and the reasoning traceable.

```mermaid
graph LR
    P1["🗂 Phase 1\nBasic RAG\nCore loop"] -->|smarter retrieval| P2
    P2["📚 Phase 2\nLangChain RAG\nMMR + LCEL"] -->|add agency| P3
    P3["🤖 Phase 3\nLangGraph Agent\nTool routing"] -->|add specialisation| P4
    P4["👥 Phase 4\nMulti-Agent\nSupervisor pattern"] -->|add production| P5
    P5["⚙️ Phase 5\nAdaptive RAG\nHybrid + RAGAS + Airflow"]

    style P1 fill:#e8f4f8,stroke:#2196F3
    style P2 fill:#e8f5e9,stroke:#4CAF50
    style P3 fill:#fff3e0,stroke:#FF9800
    style P4 fill:#fce4ec,stroke:#E91E63
    style P5 fill:#ede7f6,stroke:#673AB7
```

| Phase | Question it answers | Key addition |
|-------|-------------------|--------------|
| 1 | Can I build a working RAG at all? | ChromaDB · cosine search · LangChain |
| 2 | Can I retrieve smarter, not just more? | MMR retrieval · LCEL · page metadata |
| 3 | Can the system decide what to do? | LangGraph StateGraph · tool routing · web search |
| 4 | Can agents specialise and collaborate? | Supervisor pattern · Research Agent · Writer Agent |
| 5 | Can this run in production with quality guarantees? | Hybrid search · RAGAS · Prometheus · Airflow |

---

## Projects

### Phase 1 — TechDocs Assistant
**[→ View Project](./RAG-Portfolio/phase-1-techdocs-assistant)**

Upload any PDF. Ask questions. Get grounded answers with source chunks.

```mermaid
flowchart TD
    PDF[PDF / Text File] -->|PyPDF + text splitter\n500 tokens · 50 overlap| CHUNK[Chunks]
    CHUNK -->|all-MiniLM-L6-v2\n384-dim embeddings| CHROMA[(ChromaDB)]
    Q[User Question] -->|embed| CHROMA
    CHROMA -->|top-k cosine similarity| CTX[Retrieved Chunks]
    CTX --> PROMPT[Prompt Builder\nquestion + context]
    PROMPT --> LLM[LLM\nOllama / Claude / GPT]
    LLM --> ANS[Answer + Source Chunks]
```

---

### Phase 2 — LangChain RAG with MMR
**[→ View Project](./RAG-Portfolio/phase-2-langchain-rag)**

Real academic PDFs, page-level citations, and MMR retrieval that avoids redundant context chunks.

```mermaid
flowchart TD
    PDF[Academic PDFs\nTransformers · GPT-3 · RAG] -->|PyPDFLoader\npreserves page numbers| CHUNK[Chunks + Metadata]
    CHUNK --> CHROMA[(ChromaDB)]
    Q[User Question] --> MMR{MMR Retrieval}
    CHROMA --> MMR
    MMR -->|"score = λ·relevance − (1−λ)·max_sim\nRelevant AND diverse"| CTX[Top-k Diverse Chunks]
    CTX --> LCEL["LCEL Chain\nprompt_template | llm | parser"]
    LCEL --> ANS["Answer\n+ source docs with page numbers"]
```

> **Why MMR?** Cosine similarity returns the 5 *most similar* chunks — which are often nearly identical paragraphs. MMR returns chunks that are relevant *and* different from each other, making better use of the LLM's context window.

---

### Phase 3 — LangGraph Agentic RAG
**[→ View Project](./RAG-Portfolio/phase-3-langgraph-agent)**

The LLM stops following a fixed pipeline and starts making decisions — which tool to call, when to loop, when to stop.

```mermaid
flowchart TD
    Q[User Question] --> AGENT[Agent Node\nLLM decides next action]
    AGENT -->|tool_call: search_papers| PAPERS[search_ai_papers\nChromaDB semantic search]
    AGENT -->|tool_call: web_search| WEB[web_search\nTavily live results]
    AGENT -->|no tool needed| END([Final Answer])
    PAPERS -->|results added to state| AGENT
    WEB -->|results added to state| AGENT

    style END fill:#c8e6c9,stroke:#388E3C
```

Every tool call, result, and reasoning step is shown live in the Streamlit UI as the agent thinks.

---

### Phase 4 — Multi-Agent Supervisor System
**[→ View Project](./RAG-Portfolio/phase-4-multi-agent-supervisor)**

Complex tasks need specialisation. A Supervisor routes work between a Research Agent and a Writer Agent, each optimised for its role.

```mermaid
flowchart TD
    TASK[User Task\ne.g. Write a report on RAG evaluation] --> SUP[Supervisor Node\ndeterministic routing]

    SUP -->|Step 1| RA[Research Agent]
    RA -->|search_papers| DB[(ChromaDB)]
    RA -->|web_search| WEB[Tavily]
    RA -->|synthesised notes| SUP

    SUP -->|Step 2| WA[Writer Agent]
    WA -->|polished markdown| SUP

    SUP -->|Step 3| DONE([FINISH])
    DONE --> RPT[Downloadable Report]

    style DONE fill:#c8e6c9,stroke:#388E3C
    style RPT fill:#e3f2fd,stroke:#1976D2
```

---

### Phase 5 — Adaptive RAG Engine
**[→ View Project](./RAG-Portfolio/phase-5-adaptive-rag-engine)**

Every component gets a production-grade upgrade: hybrid retrieval, automated quality scoring, Prometheus metrics, and an Airflow DAG that re-indexes automatically when quality drops.

#### Retrieval Pipeline

```mermaid
flowchart TD
    Q[User Question] --> BM25[BM25 Full-text Search\nElasticsearch 8\ntop-20 candidates]
    Q --> VEC[Vector Search\npgvector HNSW\ntop-20 candidates]
    BM25 --> RRF[RRF Fusion\nscore = Σ 1 divided by k+rank\nno score calibration needed]
    VEC --> RRF
    RRF --> RERANK[Cross-Encoder Reranking\nms-marco-MiniLM\ntop-20 → top-5]
    RERANK --> LLM[LLM Synthesis\nOllama / Claude / GPT]
    LLM --> RESP[HTTP Response\nanswer + sources + latency]
    LLM -.->|BackgroundTask| RAGAS[RAGAS Evaluation\nfaithfulness · answer_relevancy]
    RAGAS --> DB[(PostgreSQL\nEvaluation records)]

    style RESP fill:#c8e6c9,stroke:#388E3C
    style RAGAS fill:#fff3e0,stroke:#FF9800
```

#### Quality Monitoring & Self-Healing

```mermaid
flowchart LR
    DB[(PostgreSQL\nEvaluation scores)] --> PROM[Prometheus\nscrapes /metrics]
    PROM --> GRAF[Grafana\ndashboards]

    DAG[Airflow DAG\nevery 30 min] --> CHECK{faithfulness\n< 0.70?}
    CHECK -->|yes| INGEST[Re-index documents]
    CHECK -->|no| DONE[✓ Quality OK]
    INGEST --> RESOLVE[Resolve alerts]

    style DONE fill:#c8e6c9,stroke:#388E3C
    style INGEST fill:#fce4ec,stroke:#E91E63
```

#### Infrastructure

```mermaid
graph TD
    subgraph Docker Compose
        PG[(PostgreSQL 16\n+ pgvector\nport 5432)]
        ES[(Elasticsearch 8\nBM25 index\nport 9200)]
        RD[(Redis 7\ncache\nport 6379)]
        PR[Prometheus\nport 9090]
        GR[Grafana\nport 3000]
        AF[Airflow\nport 8088]
    end
    API[FastAPI\nport 8080] --> PG
    API --> ES
    API --> RD
    API --> PR
    DASH[Streamlit\nport 8501] --> API
    AF --> API
```

---

## Tech Stack

| Layer | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|-------|:-------:|:-------:|:-------:|:-------:|:-------:|
| UI | Streamlit | Streamlit | Streamlit streaming | Streamlit 2-col | FastAPI + Streamlit |
| Vector DB | ChromaDB | ChromaDB | ChromaDB | — | pgvector |
| Full-text | — | — | — | — | Elasticsearch |
| Retrieval | Cosine | MMR | Agent-routed | Agent-routed | BM25 + Vector + RRF |
| Reranking | — | — | — | — | Cross-encoder |
| Orchestration | LangChain | LCEL | LangGraph | LangGraph | FastAPI |
| Agents | — | — | 1 (ReAct) | 3 (Supervisor) | — |
| Evaluation | — | — | — | — | RAGAS |
| Monitoring | — | — | — | — | Prometheus + Grafana |
| Scheduling | — | — | — | — | Airflow DAG |
| Infrastructure | local | local | local | local | Docker Compose |

---

## Key Engineering Decisions

**Why RRF instead of normalising BM25 + cosine scores?**
BM25 and cosine similarity live on different scales. Normalization requires per-query calibration. RRF works on *ranks*, not scores — no calibration needed, robust across any corpus.

**Why async RAGAS evaluation?**
RAGAS calls the LLM again to score answers. With a local model this takes 3–5 minutes. Running it synchronously would make every query 5× slower. Background evaluation keeps user-facing latency at LLM generation time only.

**Why PostgreSQL for vectors instead of a dedicated vector DB?**
pgvector with HNSW indexing lets vector search, metadata filters, and evaluation records live in one transactional store. For production systems, avoiding a second database simplifies ops and consistency.

**Why Airflow for re-indexing instead of a cron job?**
Airflow gives retry logic, task dependency tracking, backfill, and a UI for monitoring DAG runs — all needed when a scheduled task has multiple dependent steps (check quality → ingest → resolve alerts).

---

## Quick Start

```bash
# Phases 1–4 (Streamlit, ~5 min setup)
cd RAG-Portfolio/phase-1-techdocs-assistant  # or phase-2, 3, 4
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2          # local LLM — free, no API key
streamlit run app.py          # http://localhost:8501

# Phase 5 (full production stack)
cd RAG-Portfolio/phase-5-adaptive-rag-engine
docker-compose up -d postgres elasticsearch redis prometheus grafana
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --port 8080 --reload
streamlit run dashboard/app.py
# API docs:  http://localhost:8080/docs
# Dashboard: http://localhost:8501
# Grafana:   http://localhost:3000
```
