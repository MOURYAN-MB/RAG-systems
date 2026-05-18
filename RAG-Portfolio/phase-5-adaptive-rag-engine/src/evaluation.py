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

        # answer_relevancy uses embeddings (fast, works locally).
        # faithfulness requires N sequential LLM calls (one per statement) — with
        # Ollama on CPU this takes 5-10 min and reliably times out. Run it separately
        # so a faithfulness timeout cannot block answer_relevancy from being saved.
        run_cfg = RunConfig(timeout=600, max_retries=1, max_wait=60)

        def _safe(v):
            if v is None:
                return None
            f = float(v)
            return round(f, 4) if not math.isnan(f) else None

        # --- answer_relevancy (fast) ---
        rel_result = evaluate(
            dataset,
            metrics=[answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_emb,
            raise_exceptions=False,
            run_config=run_cfg,
        )
        rel_raw = rel_result["answer_relevancy"]
        if hasattr(rel_raw, "__iter__"):
            rel_raw = list(rel_raw)[0]
        rel = _safe(rel_raw)

        # --- faithfulness (slow — best-effort, may stay None on small machines) ---
        faith = None
        try:
            faith_result = evaluate(
                dataset,
                metrics=[faithfulness],
                llm=ragas_llm,
                embeddings=ragas_emb,
                raise_exceptions=False,
                run_config=RunConfig(timeout=600, max_retries=1, max_wait=60),
            )
            faith_raw = faith_result["faithfulness"]
            if hasattr(faith_raw, "__iter__"):
                faith_raw = list(faith_raw)[0]
            faith = _safe(faith_raw)
        except Exception as fe:
            logger.warning("[RAGAS] faithfulness timed out: %s", fe)

        return {
            "faithfulness":     faith,
            "answer_relevancy": rel,
            "context_recall":   None,  # requires ground truth — not available
        }

    except Exception as exc:
        logger.error("[RAGAS] Evaluation failed: %s", exc, exc_info=True)
        return {"faithfulness": None, "answer_relevancy": None, "context_recall": None}
