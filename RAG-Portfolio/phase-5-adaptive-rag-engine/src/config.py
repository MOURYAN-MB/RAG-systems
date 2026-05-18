from pathlib import Path
from dotenv import load_dotenv
import os

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# ── Database ───────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpass")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
ASYNC_DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Elasticsearch ──────────────────────────────────────────────────────────
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_INDEX = os.getenv("ES_INDEX", "rag_chunks")

# ── Redis ──────────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# ── LLM providers ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

ANTHROPIC_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ── Retrieval pipeline ─────────────────────────────────────────────────────
TOP_K_CANDIDATES = 20   # fetched from each retriever (BM25 + vector)
TOP_K_RERANK = 10       # sent to cross-encoder after RRF fusion
TOP_K_FINAL = 5         # passed to LLM for synthesis
RRF_K = 60              # RRF constant (standard value)

# ── Quality monitoring ─────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.70"))
CONTEXT_RECALL_THRESHOLD = float(os.getenv("CONTEXT_RECALL_THRESHOLD", "0.60"))
QUALITY_WINDOW = int(os.getenv("QUALITY_WINDOW", "50"))

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data" / "raw"


def get_ollama_models() -> list[str]:
    try:
        import urllib.request, json
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
            return models if models else [OLLAMA_MODEL]
    except Exception:
        return [OLLAMA_MODEL]