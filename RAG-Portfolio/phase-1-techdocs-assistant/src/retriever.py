from sentence_transformers import SentenceTransformer
import chromadb

from src.config import VECTORSTORE_DIR, EMBEDDING_MODEL, TOP_K

COLLECTION_NAME = "techdocs"

_model = None
_collection = None


def _load():
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    if _collection is None:
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _collection = client.get_collection(COLLECTION_NAME)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and return the top_k most relevant chunks.
    Each result has keys: 'text', 'source', 'distance'.
    """
    _load()
    query_embedding = _model.encode(query).tolist()
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": meta.get("source", "unknown"),
            "distance": round(dist, 4),
        })
    return chunks


if __name__ == "__main__":
    query = "What is RAG?"
    print(f"Query: {query}\n")
    for i, chunk in enumerate(retrieve(query), 1):
        print(f"[{i}] (distance={chunk['distance']}) source={chunk['source']}")
        print(chunk["text"])
        print()