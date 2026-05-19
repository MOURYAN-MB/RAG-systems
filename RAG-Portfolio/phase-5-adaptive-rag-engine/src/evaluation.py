"""
Evaluation pipeline — runs asynchronously after every query.

answer_relevancy : RAGAS LLM metric (fast, uses embeddings + LLM).
faithfulness     : Embedding cosine similarity — fraction of answer sentences
                   grounded in the retrieved context (threshold 0.45).
context_recall   : Mean max-similarity of answer sentences to context sentences
                   (soft grounding score 0-1, no ground-truth needed).

Both embedding metrics are fast, reliable, and require no cloud API.
"""
import logging
import math
import re

import numpy as np
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy

from src.config import (
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAGAS_MODEL,
    RAGAS_PROVIDER,
)
from src.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


# ── RAGAS LLM factory (used only for answer_relevancy) ───────────────────────

def _get_ragas_llm(provider: str = "ollama", model: str = None):
    model = model or OLLAMA_MODEL
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0, format="json")
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from src.config import ANTHROPIC_API_KEY
        llm = ChatAnthropic(model=model, api_key=ANTHROPIC_API_KEY, temperature=0)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        from src.config import OPENAI_API_KEY
        llm = ChatOpenAI(model=model, api_key=OPENAI_API_KEY, temperature=0)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=GOOGLE_API_KEY, temperature=0)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return LangchainLLMWrapper(llm)


# ── Embedding-based faithfulness + context_recall ────────────────────────────

def _sentences(text: str) -> list[str]:
    """Split text into sentences longer than 20 characters."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 20]


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_n @ b_n.T


def _embedding_metrics(
    answer: str, contexts: list[str], embedder
) -> tuple[float | None, float | None]:
    """
    Returns (faithfulness, context_recall).

    faithfulness   : fraction of answer sentences whose max cosine-sim to any
                     context sentence is >= 0.45  (binary grounding check).
    context_recall : mean of those max-sim values across all answer sentences
                     (soft score — how well the context covers the answer).
    """
    ans_sents = _sentences(answer)
    ctx_sents = _sentences(" ".join(contexts))
    if not ans_sents or not ctx_sents:
        return None, None

    # Single batched embed call — fast
    all_embs = np.array(embedder.embed_documents(ans_sents + ctx_sents))
    ans_embs = all_embs[: len(ans_sents)]
    ctx_embs = all_embs[len(ans_sents) :]

    sim = _cosine_matrix(ans_embs, ctx_embs)   # [n_ans × n_ctx]
    max_sims = sim.max(axis=1)                  # best context match per sentence

    faithfulness    = round(float(np.mean(max_sims >= 0.45)), 4)
    context_recall  = round(float(np.mean(max_sims)), 4)
    return faithfulness, context_recall


# ── Public evaluation entry-point ─────────────────────────────────────────────

def run_ragas(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
    provider: str = None,
    model: str = None,
) -> dict:
    """
    Score a single query-answer pair. Never raises — returns None values on failure.

    faithfulness + context_recall : embedding cosine similarity (always fast).
    answer_relevancy              : RAGAS LLM metric (best-effort, falls back to None).
    """
    contexts = [c.text for c in chunks]
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def _safe(v):
        if v is None:
            return None
        f = float(v)
        return round(f, 4) if not math.isnan(f) else None

    # ── faithfulness + context_recall (embedding-based, no LLM, always works) ──
    faith, ctx_recall = None, None
    try:
        faith, ctx_recall = _embedding_metrics(answer, contexts, embedder)
        logger.info("[EVAL] faithfulness=%.3f  context_recall=%.3f", faith or 0, ctx_recall or 0)
    except Exception as e:
        logger.warning("[EVAL] embedding metrics failed: %s", e)

    # ── answer_relevancy (RAGAS LLM — best-effort) ────────────────────────────
    rel = None
    try:
        ragas_llm = _get_ragas_llm(RAGAS_PROVIDER, RAGAS_MODEL)
        ragas_emb = LangchainEmbeddingsWrapper(embedder)

        dataset = Dataset.from_dict({
            "question": [query],
            "answer":   [answer],
            "contexts": [contexts],
        })
        run_cfg = RunConfig(timeout=120, max_retries=1, max_wait=30)
        result = evaluate(
            dataset,
            metrics=[answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_emb,
            raise_exceptions=False,
            run_config=run_cfg,
        )
        rel_raw = result["answer_relevancy"]
        if hasattr(rel_raw, "__iter__"):
            rel_raw = list(rel_raw)[0]
        rel = _safe(rel_raw)
        logger.info("[EVAL] answer_relevancy=%.3f", rel or 0)
    except Exception as exc:
        logger.warning("[EVAL] answer_relevancy failed: %s", exc)

    return {
        "faithfulness":     faith,
        "answer_relevancy": rel,
        "context_recall":   ctx_recall,
    }
