import time
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from src.retrieval import hybrid_search, RetrievedChunk
from src.llm import build_llm, synthesize
from src.evaluation import run_ragas
from src.monitor import check_and_create_alert
from src.database import SessionLocal, Evaluation
from src.config import OLLAMA_MODEL

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    provider: str = "ollama"
    model: str = OLLAMA_MODEL


class ChunkInfo(BaseModel):
    chunk_id:     int
    source:       str
    page:         int
    rerank_score: float | None
    text_preview: str


class QueryResponse(BaseModel):
    answer:         str
    sources:        list[ChunkInfo]
    latency_ms:     int
    eval_scheduled: bool


def _log_evaluation(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
    latency_ms: int,
    model: str,
    provider: str,
):
    """Run RAGAS evaluation and persist results — called as a background task."""
    scores = run_ragas(query, answer, chunks, provider=provider, model=model)
    check_and_create_alert(scores)

    db = SessionLocal()
    try:
        db.add(Evaluation(
            query=query,
            answer=answer,
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("answer_relevancy"),
            context_recall=scores.get("context_recall"),
            latency_ms=latency_ms,
            model=model,
            retrieval_method="hybrid",
        ))
        db.commit()
    finally:
        db.close()


@router.post("", response_model=QueryResponse)
async def query(req: QueryRequest, background_tasks: BackgroundTasks):
    t0 = time.time()

    chunks, _pipeline_meta = hybrid_search(req.question)
    llm = build_llm(req.provider, req.model)
    answer = synthesize(req.question, chunks, llm)

    latency_ms = int((time.time() - t0) * 1000)

    # Schedule RAGAS evaluation asynchronously — user gets response immediately
    background_tasks.add_task(
        _log_evaluation,
        req.question, answer, chunks, latency_ms, req.model, req.provider,
    )

    return QueryResponse(
        answer=answer,
        sources=[
            ChunkInfo(
                chunk_id=c.chunk_id,
                source=c.source,
                page=c.page,
                rerank_score=c.rerank_score,
                text_preview=c.text[:300] + "..." if len(c.text) > 300 else c.text,
            )
            for c in chunks
        ],
        latency_ms=latency_ms,
        eval_scheduled=True,
    )