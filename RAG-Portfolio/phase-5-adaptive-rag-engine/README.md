# Phase 5 — Adaptive RAG Evaluation Engine

> Production-grade RAG with hybrid retrieval, automated RAGAS evaluation, Prometheus monitoring, and a self-healing Airflow DAG that re-indexes when quality degrades.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+pgvector-336791?logo=postgresql&logoColor=white)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.13-005571?logo=elasticsearch&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-2.9-017CEE?logo=apacheairflow&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Grafana-E6522C?logo=prometheus&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

---

## What Makes This Different

A standard RAG pipeline is static — index documents, serve queries, hope nothing breaks. This system treats retrieval quality as a **live signal**. Every answer is automatically scored. Scores are tracked over time. When the sliding window average drops below threshold, an Airflow DAG fires and re-indexes without human intervention.

Three layers no standard RAG tutorial includes:

| Layer | What it does | Why it matters |
|---|---|---|
| **Hybrid retrieval** | BM25 + vector search fused with RRF + cross-encoder reranking | Neither approach alone is sufficient in production |
| **RAGAS evaluation** | Every query scored on faithfulness + answer relevancy + context recall | Hallucination detection as a continuous signal, not a one-off test |
| **Self-healing loop** | Airflow DAG checks scores every 30 min, re-indexes on quality alert | System degrades silently without this — the "adaptive" in the name |

---

## Architecture

### Retrieval Pipeline

```
User Question
    │
    ├──► pgvector (HNSW)      ─── top-20 by cosine similarity
    │    [dense vector search]
    │
    └──► Elasticsearch (BM25) ─── top-20 by TF-IDF keyword score
         [sparse keyword search]
              │
              ▼
         RRF Fusion
         score = Σ 1/(k + rank)   ← rank-based, scale-agnostic
              │
              ▼
    Cross-Encoder Reranking       ← joint (query, doc) scoring
    ms-marco-MiniLM-L-6-v2        ← top-20 → top-5
              │
              ▼
    LLM Synthesis                 ← Ollama / Claude / OpenAI / Gemini
    [answer + citations]          ← async RAGAS eval in background
```

### Evaluation & Self-Healing Loop

```
Every query ──► Background Task ──► Embedding-based metrics (fast)
                                        faithfulness   (0–1)
                                        context_recall (0–1)
                                    + RAGAS LLM metric
                                        answer_relevancy (0–1)
                                    ──► PostgreSQL evaluation log

Every 30 min ──► Airflow DAG
                    check_quality  ──► GET /quality
                    maybe_reindex  ──► POST /ingest/rebuild (if alert)
                                   ──► Alerts resolved after re-index

Prometheus ──► Grafana ──► Real-time dashboards
```

### Infrastructure (Docker Compose)

| Service | Image | Port | Role |
|---|---|---|---|
| PostgreSQL + pgvector | pgvector/pgvector:pg16 | 5432 | Chunks + evaluations + alerts |
| Elasticsearch | elasticsearch:8.13.0 | 9200 | BM25 keyword index |
| Redis | redis:7-alpine | 6379 | Cache + rate limiting |
| Airflow | apache/airflow:2.9.1 | 8081 | Re-indexing DAG scheduler |
| Prometheus | prom/prometheus | 9090 | Metrics collection |
| Grafana | grafana/grafana | 3000 | Dashboards |

---

## Tech Stack

```
FastAPI          ─── REST API, async background tasks, /metrics endpoint
pgvector         ─── HNSW vector index inside PostgreSQL (no separate vector DB)
Elasticsearch    ─── BM25 full-text retrieval
SQLAlchemy       ─── ORM for PostgreSQL (chunks, evaluations, alerts)
HuggingFace      ─── all-MiniLM-L6-v2 (embeddings) + ms-marco-MiniLM (reranker)
RAGAS            ─── answer_relevancy metric (LLM-based)
NumPy            ─── faithfulness + context_recall (embedding cosine similarity)
Prometheus       ─── custom Gauges for RAGAS scores + FastAPI HTTP metrics
Grafana          ─── 5-panel dashboard: scores, latency, volume, alert status
Airflow          ─── quality_monitor_reindex DAG (every 30 min)
Streamlit        ─── analytics dashboard with Altair score trend chart
Docker Compose   ─── reproducible local stack (6 services, all pinned versions)
```

---

## Setup

### Prerequisites
- Docker Desktop running
- Ollama installed with `ollama pull llama3.2`
- Python 3.11+

### Start infrastructure

```bash
git clone <repo>
cd phase-5-adaptive-rag-engine
docker-compose up -d
```

Wait ~30 seconds for Elasticsearch to become healthy.

### Start application

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
cp .env.example .env            # fill in API keys if using cloud LLMs

uvicorn api.main:app --port 8080 --reload
```

### Start dashboard

```bash
streamlit run dashboard/app.py
```

### Add documents and run first query

1. Drop PDF files into `data/raw/`
2. Click **Rebuild Index** in the Streamlit sidebar (or `POST /ingest/rebuild`)
3. Ask a question — evaluation runs in the background within ~10 seconds
4. Check the **Analytics** tab for scores

### Access points

| Interface | URL | Default credentials |
|---|---|---|
| Streamlit dashboard | http://localhost:8501 | — |
| FastAPI docs | http://localhost:8080/docs | — |
| Airflow UI | http://localhost:8081 | admin / admin |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Submit question, get answer + sources + RAGAS scheduled |
| `/ingest/rebuild` | POST | Re-chunk, re-embed, re-index all documents in `data/raw/` |
| `/ingest/status` | GET | Chunk count per document |
| `/quality` | GET | Sliding window RAGAS scores + alert status |
| `/health` | GET | PostgreSQL / Elasticsearch / Redis connectivity |
| `/metrics` | GET | Prometheus metrics endpoint |

### Query request/response

```json
// POST /query
{ "question": "How does HNSW indexing work?", "provider": "ollama", "model": "llama3.2:latest" }

// Response
{
  "answer": "HNSW builds a multi-layer graph where...",
  "sources": [{ "source": "vector_databases.pdf", "page": 1, "rerank_score": 4.76, "text_preview": "..." }],
  "latency_ms": 4200,
  "eval_scheduled": true
}
```

---

## Evaluation Metrics

| Metric | Method | Target | What it measures |
|---|---|---|---|
| **Faithfulness** | Embedding cosine similarity | ≥ 0.70 | Fraction of answer sentences grounded in retrieved context |
| **Answer Relevancy** | RAGAS LLM (llama3.2:1b) | — | Do hypothetical questions from the answer match the original query? |
| **Context Recall** | Embedding cosine similarity | ≥ 0.60 | Mean grounding strength of answer sentences to context |

**Quality alert logic:** If `avg_faithfulness < 0.70` OR `avg_context_recall < 0.60` across the last 50 queries, the system writes an alert record and the Airflow DAG triggers re-indexing.

---

## Key Engineering Decisions

**Why RRF instead of normalising BM25 + cosine scores?**
BM25 and cosine live on different scales. Normalisation needs per-query calibration. RRF uses ranks, not scores — no calibration needed, robust across any corpus. Published by Cormack et al. (2009), still one of the most reliable fusion methods.

**Why pgvector instead of Pinecone/Qdrant?**
At portfolio scale (< 10M vectors): single database for vectors, evaluation logs, and alerts — ACID transactions, SQL joins, no extra infra. The trade-off is a scaling ceiling at ~10M vectors where a dedicated store would be needed. Deliberate choice, articulable in interviews.

**Why async RAGAS evaluation?**
Faithfulness scoring makes sequential LLM calls (one per answer statement). Running it synchronously adds 30–120s to query latency. Background tasks keep user-facing latency at LLM generation time only.

**Why the cross-encoder comes after RRF, not instead of it?**
Cross-encoders process (query, document) pairs jointly — far more accurate than bi-encoder similarity — but too slow to run over the full index. The cascade: bi-encoder → top-20 candidates → RRF fusion → cross-encoder rerank → top-5 to LLM.

**Why Airflow instead of a cron job?**
Retry logic, task dependency graph, backfill, and a UI for inspecting DAG run history. When a scheduled job has multiple dependent steps (check quality → ingest → resolve alerts), those properties matter.

---

## Project Structure

```
adaptive-rag-engine/
├── api/
│   ├── main.py              # FastAPI app, Prometheus metrics, /health, /quality
│   └── routers/
│       ├── ingest.py        # POST /ingest/rebuild, GET /ingest/status
│       └── query.py         # POST /query  (hybrid retrieval + async eval)
├── src/
│   ├── config.py            # All env vars + thresholds
│   ├── database.py          # SQLAlchemy models (Chunk, Evaluation, Alert)
│   ├── ingestion.py         # PDF extraction + chunking + dual-indexing
│   ├── retrieval.py         # Vector search + BM25 + RRF + cross-encoder
│   ├── llm.py               # LLM client (Ollama / Anthropic / OpenAI / Google)
│   ├── evaluation.py        # Embedding faithfulness + RAGAS answer_relevancy
│   └── monitor.py           # Sliding window quality status + alert creation
├── dags/
│   └── reindex_dag.py       # Airflow DAG: check_quality → maybe_reindex
├── dashboard/
│   └── app.py               # Streamlit: query interface + analytics + score trends
├── grafana/
│   └── provisioning/        # Auto-provisioned Prometheus datasource + dashboard
├── prometheus/
│   └── prometheus.yml       # Scrape config (FastAPI /metrics)
├── data/raw/                # Drop PDFs here
├── docker-compose.yml       # Full 6-service stack
├── create_test_pdfs.py      # Generates 3 sample PDFs (transformers, vectors, LLM eval)
└── .env                     # DB / ES / Redis / LLM config
```
