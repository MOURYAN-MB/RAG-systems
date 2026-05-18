"""
RAGAS-based evaluation — runs asynchronously after every query so it
never adds latency to the user-facing response.

Evaluates faithfulness and answer_relevancy using the local Ollama LLM.
context_recall is omitted because it requires real ground-truth answers.
"""
import logging
import math
from ragas import evaluate
from ragas import RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, EMBEDDING_MODEL
from src.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def _get_ragas_llm(provider: str = "ollama", model: str = None):
    model = model or OLLAMA_MODEL
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from src.config import ANTHROPIC_API_KEY
        llm = ChatAnthropic(model=model, api_key=ANTHROPIC_API_KEY, temperature=0)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        from src.config import OPENAI_API_KEY
        llm = ChatOpenAI(model=model, api_key=OPENAI_API_KEY, temperature=0)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return LangchainLLMWrapper(llm)


def run_ragas(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
    provider: str = "ollama",
    model: str = None,
) -> dict:
    """
    Score a single query-answer pair using RAGAS faithfulness + answer_relevancy.
    context_recall requires real ground truth so is excluded here.
    Falls back to None values on failure (LLM timeout, API error, etc.).
    """
    try:
        ragas_llm = _get_ragas_llm(provider, model)
        ragas_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )

        dataset = Dataset.from_dict({
            "question": [query],
            "answer":   [answer],
            "contexts": [[c.text for c in chunks]],
        })

        # 10-minute timeout per metric — Ollama local inference is slow
        run_cfg = RunConfig(timeout=600, max_retries=2, max_wait=600)

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_emb,
            raise_exceptions=False,
            run_config=run_cfg,
        )

        # result is a ragas Result object; index by column name gives a float
        faith = result["faithfulness"]
        rel   = result["answer_relevancy"]

        # Handle both scalar and sequence returns across ragas versions
        if hasattr(faith, "__iter__"):
            faith = list(faith)[0]
        if hasattr(rel, "__iter__"):
            rel = list(rel)[0]

        def _safe(v):
            """Convert to float, return None for nan/None."""
            if v is None:
                return None
            f = float(v)
            return round(f, 4) if not math.isnan(f) else None

        return {
            "faithfulness":     _safe(faith),
            "answer_relevancy": _safe(rel),
            "context_recall":   None,  # requires ground truth — not available
        }

    except Exception as exc:
        logger.error("[RAGAS] Evaluation failed: %s", exc, exc_info=True)
        return {"faithfulness": None, "answer_relevancy": None, "context_recall": None}