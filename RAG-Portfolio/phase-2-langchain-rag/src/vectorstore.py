from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import VECTORSTORE_DIR, EMBEDDING_MODEL, COLLECTION_NAME


def _embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(chunks) -> Chroma:
    print(f"Embedding {len(chunks)} chunks and persisting to {VECTORSTORE_DIR}...")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
        collection_name=COLLECTION_NAME,
    )
    print("Vectorstore built.")
    return vs


def load_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def vectorstore_exists() -> bool:
    return (VECTORSTORE_DIR / "chroma.sqlite3").exists()