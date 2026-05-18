from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from elasticsearch import Elasticsearch
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import (
    EMBEDDING_MODEL, ES_HOST, ES_PORT, ES_INDEX,
    TOP_K_CANDIDATES, TOP_K_RERANK, TOP_K_FINAL, RRF_K,
)
from src.database import SessionLocal

# ── Lazy-loaded singletons (expensive to initialise) ──────────────────────
_embedder: HuggingFaceEmbeddings | None = None
_reranker: CrossEncoder | None = None


def _get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


@dataclass
class RetrievedChunk:
    chunk_id:   int
    text:       str
    source:     str
    page:       int
    vector_rank:  int | None = None
    bm25_rank:    int | None = None
    rrf_score:    float = 0.0
    rerank_score: float | None = None


def _vector_search(query_vec: list[float], k: int) -> list[RetrievedChunk]:
    """Cosine similarity search via pgvector."""
    db: Session = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT id, chunk_text, source_file, page_number,
                   1 - (embedding <=> CAST(:vec AS vector)) AS score
            FROM chunks
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
        """), {"vec": str(query_vec), "k": k}).fetchall()

        return [
            RetrievedChunk(
                chunk_id=r.id,
                text=r.chunk_text,
                source=r.source_file or "",
                page=r.page_number or 0,
                vector_rank=i + 1,
            )
            for i, r in enumerate(rows)
        ]
    finally:
        db.close()


def _bm25_search(query: str, k: int) -> list[RetrievedChunk]:
    """BM25 keyword search via Elasticsearch."""
    es = Elasticsearch(f"http://{ES_HOST}:{ES_PORT}")
    resp = es.search(
        index=ES_INDEX,
        body={"query": {"match": {"chunk_text": query}}, "size": k},
    )
    results = []
    for rank, hit in enumerate(resp["hits"]["hits"], 1):
        src = hit["_source"]
        results.append(RetrievedChunk(
            chunk_id=src["chunk_id"],
            text=src["chunk_text"],
            source=src.get("source", ""),
            page=src.get("page", 0),
            bm25_rank=rank,
        ))
    return results


def _reciprocal_rank_fusion(
    vector_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = sum of 1/(k + rank) across result sets.
    Rank-based (not score-based) so incompatible score scales don't matter.
    """
    fused: dict[int, RetrievedChunk] = {}

    for chunk in vector_results:
        fused[chunk.chunk_id] = chunk
        fused[chunk.chunk_id].rrf_score += 1 / (k + (chunk.vector_rank or 99))

    for chunk in bm25_results:
        if chunk.chunk_id in fused:
            fused[chunk.chunk_id].bm25_rank = chunk.bm25_rank
            fused[chunk.chunk_id].rrf_score += 1 / (k + (chunk.bm25_rank or 99))
        else:
            chunk.rrf_score = 1 / (k + (chunk.bm25_rank or 99))
            fused[chunk.chunk_id] = chunk

    return sorted(fused.values(), key=lambda c: c.rrf_score, reverse=True)


def _cross_encoder_rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = TOP_K_FINAL,
) -> list[RetrievedChunk]:
    """
    Re-score the fused candidates using a cross-encoder.
    Unlike the bi-encoder used for embedding, the cross-encoder reads
    (query, document) pairs jointly — much more accurate but too slow
    to run over the full index, which is why we candidate-filter first.
    """
    reranker = _get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)

    return sorted(chunks, key=lambda c: c.rerank_score or 0, reverse=True)[:top_k]


def hybrid_search(query: str) -> tuple[list[RetrievedChunk], dict]:
    """
    Full hybrid retrieval pipeline:
      1. Vector search  (pgvector)    → TOP_K_CANDIDATES results
      2. BM25 search    (Elasticsearch) → TOP_K_CANDIDATES results
      3. RRF fusion                   → merged ranked list
      4. Cross-encoder reranking      → TOP_K_FINAL results

    Returns (final_chunks, pipeline_metadata) where metadata contains
    intermediate results for the observability dashboard.
    """
    embedder = _get_embedder()
    query_vec = embedder.embed_query(query)

    vector_results = _vector_search(query_vec, TOP_K_CANDIDATES)
    bm25_results   = _bm25_search(query, TOP_K_CANDIDATES)
    fused          = _reciprocal_rank_fusion(vector_results, bm25_results)
    final          = _cross_encoder_rerank(query, fused[:TOP_K_RERANK])

    metadata = {
        "vector_results": vector_results[:5],
        "bm25_results":   bm25_results[:5],
        "fused_results":  fused[:5],
        "final_results":  final,
    }
    return final, metadata