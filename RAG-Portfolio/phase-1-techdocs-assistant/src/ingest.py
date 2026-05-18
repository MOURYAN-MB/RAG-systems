"""
Ingestion pipeline: reads documents from data/raw/, chunks them,
generates embeddings, and stores them in ChromaDB.
"""
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

from src.config import (
    DATA_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)

COLLECTION_NAME = "techdocs"


def load_documents(data_dir: Path) -> list[dict]:
    """Read all .txt and .md files from data/raw/."""
    docs = []
    for ext in ("*.txt", "*.md"):
        for file in data_dir.glob(ext):
            text = file.read_text(encoding="utf-8")
            docs.append({"filename": file.name, "text": text})
    print(f"Loaded {len(docs)} document(s) from {data_dir}")
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest(force: bool = False):
    """
    Full ingestion pipeline.
    Set force=True to re-embed even if the collection already exists.
    """
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Skip re-ingestion unless forced
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing and not force:
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping ingestion.")
        print("Run with force=True to re-ingest.")
        return

    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    docs = load_documents(DATA_DIR)
    if not docs:
        print(f"No .txt or .md files found in {DATA_DIR}. Add documents and retry.")
        return

    all_chunks, all_ids, all_metadata = [], [], []
    chunk_index = 0

    for doc in docs:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_index}")
            all_metadata.append({"source": doc["filename"]})
            chunk_index += 1

    print(f"Embedding {len(all_chunks)} chunks using '{EMBEDDING_MODEL}'...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadata,
    )

    print(f"Done. {len(all_chunks)} chunks stored in ChromaDB collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest()