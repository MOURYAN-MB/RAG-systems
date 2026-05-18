# Phase 5 — Adaptive RAG Evaluation Engine

> **Goal:** Build a production-grade RAG system with hybrid retrieval, automated quality evaluation, monitoring, and self-healing re-indexing.

A fully Dockerized RAG platform with a FastAPI backend, hybrid BM25 + vector retrieval, RAGAS quality evaluation, Prometheus metrics, Grafana dashboards, and an Airflow DAG that automatically re-indexes when quality drops.

## Architecture

```
                     ┌─────────────────────────────┐
                     │       FastAPI Backend        │
                     │    http://localhost:8080      │
                     └──┬─────────────┬─────────────┘
                        │             │
               ┌────────▼───┐   ┌─────▼──────────┐
               │  /ingest   │   │    /query       │
               │  Upload PDF│   │  Question →     │
               │  Chunk     │   │  Answer         │
               │  Embed     │   └─────┬────────────┘
               └────┬───────┘         │
                    │           Hybrid Retrieval
              ┌─────▼──────┐   ┌──────┴──────────────────┐
              │ pgvector   │   │ BM25 (Elasticsearch)     │
              │ PostgreSQL │   │ + Vector (pgvector)      │
              │ (vectors)  │   │ → RRF Fusion             │
              └────────────┘   │ → Cross-encoder Rerank   │
                               └──────────────────────────┘
                                         │
                               ┌─────────▼──────────┐
                               │  LLM Synthesis      │
                               │  Ollama / Claude /  │
                               │  OpenAI             │
                               └─────────┬──────────┘
                                         │
                          ┌──────────────▼───────────────┐
                          │  Background: RAGAS Evaluation │
                          │  faithfulness + answer_rel    │
                          │  → PostgreSQL (Evaluation)    │
                          └──────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    Monitoring Stack                      │
│  Prometheus → Grafana (http://localhost:3000)           │
│  /metrics endpoint auto-instrumented                    │
│  Quality gauges: faithfulness, answer_relevancy         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Airflow DAG                           │
│  Runs every 30 min → checks RAGAS scores                │
│  If faithfulness < 0.70 → triggers re-indexing         │
│  Self-healing: quality drop → automatic recovery       │
└─────────────────────────────────────────────────────────┘
```

## Key Features

- **Hybrid retrieval pipeline**: BM25 (Elasticsearch) + vector search (pgvector) fused with RRF, then re-ranked by cross-encoder
- **Async RAGAS evaluation**: every query is scored for faithfulness and answer relevancy in a background task — zero added latency
- **Quality monitoring**: sliding window averages, threshold alerts (faithfulness < 0.70 triggers alert)
- **Self-healing**: Airflow DAG polls quality metrics and re-indexes automatically when scores drop
- **Prometheus + Grafana**: RAG quality metrics exposed as Prometheus gauges, visualized in Grafana
- **Streamlit analytics dashboard**: Query Interface tab + Analytics tab with score trends and recent queries table
- **Health endpoint** (`/health`): checks PostgreSQL, Elasticsearch, and Redis connectivity
- **Dockerized**: all 6 services managed with `docker-compose up`

## Services

| Service | Port | Purpose |
|---------|------|---------|
| FastAPI | 8080 | RAG API (ingest, query, health, quality) |
| Streamlit | 8501 | Analytics dashboard |
| PostgreSQL + pgvector | 5432 | Document chunks + evaluation records |
| Elasticsearch | 9200 | BM25 full-text index |
| Redis | 6379 | Query result caching |
| Prometheus | 9090 | Metrics scraping |
| Grafana | 3000 | Dashboards |

## Retrieval Pipeline

```
Question
   │
   ├─ BM25 search (Elasticsearch) → top-20 candidates
   ├─ Vector search (pgvector HNSW) → top-20 candidates
   │
   ▼
 RRF Fusion (Reciprocal Rank Fusion, k=60)
   │
   ▼
 Cross-encoder reranking → top-5 final chunks
   │
   ▼
 LLM synthesis with grounded context
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + uvicorn |
| Vector DB | PostgreSQL 16 + pgvector |
| Full-text search | Elasticsearch 8 |
| Cache | Redis 7 |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Evaluation | RAGAS (faithfulness, answer_relevancy) |
| Orchestration | Apache Airflow 2.9 |
| Monitoring | Prometheus + Grafana |
| Dashboard | Streamlit |
| LLM | Ollama (llama3.2) / Anthropic / OpenAI |

## Quick Start

```bash
# 1. Start all infrastructure services
docker-compose up -d postgres elasticsearch redis prometheus grafana

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — DB credentials already match docker-compose defaults

# 5. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# 6. Start the dashboard (new terminal)
streamlit run dashboard/app.py

# 7. Ingest documents
# Place PDFs in data/raw/ then POST http://localhost:8080/ingest/rebuild

# 8. Query
# POST http://localhost:8080/query
# Body: {"question": "...", "provider": "ollama", "model": "llama3.2:latest"}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check PostgreSQL, Elasticsearch, Redis |
| `/quality` | GET | Current RAGAS scores + alert status |
| `/ingest/rebuild` | POST | Re-ingest all documents in data/raw/ |
| `/ingest/status` | GET | Count of indexed documents and chunks |
| `/query` | POST | Ask a question, get answer + sources |
| `/metrics` | GET | Prometheus metrics endpoint |

## What This Phase Covers

- Production RAG architecture decisions: why hybrid retrieval beats pure vector search
- RRF fusion algorithm and cross-encoder reranking
- pgvector HNSW indexing for approximate nearest neighbor search
- RAGAS evaluation: faithfulness and answer relevancy as automated quality signals
- Prometheus custom gauges for domain-specific metrics
- Airflow DAG design for quality-triggered automation
- FastAPI BackgroundTasks for non-blocking evaluation
- Docker Compose multi-service orchestration