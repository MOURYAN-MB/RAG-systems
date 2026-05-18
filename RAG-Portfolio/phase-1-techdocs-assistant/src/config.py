from pathlib import Path
from dotenv import load_dotenv
import os

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

DATA_DIR = _ROOT / "data" / "raw"
VECTORSTORE_DIR = _ROOT / "vectorstore"
COLLECTION_NAME = "techdocs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
MAX_TOKENS = 1024

ANTHROPIC_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]


def get_ollama_models() -> list[str]:
    try:
        import urllib.request, json
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
            return models if models else [OLLAMA_MODEL]
    except Exception:
        return [OLLAMA_MODEL]