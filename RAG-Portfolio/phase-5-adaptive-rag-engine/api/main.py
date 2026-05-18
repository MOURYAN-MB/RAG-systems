from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge

from api.routers import ingest, query
from src.monitor import compute_quality_status

app = FastAPI(
    title="Adaptive RAG Evaluation Engine",
    description="Hybrid retrieval + RAGAS evaluation + self-healing indexing",
    version="1.0.0",
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(ingest.router)
app.include_router(query.router)

# ── Prometheus metrics ─────────────────────────────────────────────────────
# Auto-instruments all HTTP endpoints with latency histograms + request counts
Instrumentator().instrument(app).expose(app)

# Custom gauges for RAGAS quality scores
faithfulness_gauge    = Gauge("rag_faithfulness",    "Sliding window avg faithfulness score")
answer_rel_gauge      = Gauge("rag_answer_relevancy", "Sliding window avg answer relevancy")
context_recall_gauge  = Gauge("rag_context_recall",  "Sliding window avg context recall")
quality_alert_gauge   = Gauge("rag_quality_alert",   "1 if quality alert active, 0 if healthy")


@app.on_event("startup")
def startup():
    """Update Prometheus gauges with current quality status on startup."""
    _refresh_quality_gauges()


def _refresh_quality_gauges():
    status = compute_quality_status()
    if status["status"] != "no_data":
        faithfulness_gauge.set(status.get("avg_faithfulness", 0))
        answer_rel_gauge.set(status.get("avg_answer_relevancy", 0))
        context_recall_gauge.set(status.get("avg_context_recall", 0))
        quality_alert_gauge.set(1 if status["status"] == "alert" else 0)


@app.get("/health")
def health():
    """System health check — confirms all backends are reachable."""
    checks = {}

    # PostgreSQL
    try:
        from sqlalchemy import text
        from src.database import SessionLocal
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = str(e)

    # Elasticsearch
    try:
        from elasticsearch import Elasticsearch
        from src.config import ES_HOST, ES_PORT
        es = Elasticsearch(f"http://{ES_HOST}:{ES_PORT}")
        checks["elasticsearch"] = "ok" if es.ping() else "unreachable"
    except Exception as e:
        checks["elasticsearch"] = str(e)

    # Redis
    try:
        import redis
        from src.config import REDIS_HOST, REDIS_PORT
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = str(e)

    overall = "healthy" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "services": checks}


@app.get("/quality")
def quality_status():
    """Current sliding window quality scores + alert status."""
    status = compute_quality_status()
    _refresh_quality_gauges()
    return status