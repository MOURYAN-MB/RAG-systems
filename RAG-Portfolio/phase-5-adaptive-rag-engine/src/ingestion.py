import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch
from sqlalchemy.orm import Session

from src.config import (
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, ES_HOST, ES_PORT, ES_INDEX,
)
from src.database import Document, Chunk, SessionLocal


def _get_embedder():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _get_es() -> Elasticsearch:
    return Elasticsearch(f"http://{ES_HOST}:{ES_PORT}", request_timeout=60)


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _ensure_es_index(es: Elasticsearch):
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(index=ES_INDEX, body={
            "mappings": {
                "properties": {
                    "chunk_id":   {"type": "integer"},
                    "chunk_text": {"type": "text", "analyzer": "english"},
                    "source":     {"type": "keyword"},
                    "page":       {"type": "integer"},
                }
            }
        })


def ingest_documents(data_dir: Path = DATA_DIR) -> dict:
    """
    Load PDFs → chunk → embed → dual-index into pgvector + Elasticsearch.
    Skips files whose hash matches an already-indexed document.
    Returns a summary dict.
    """
    embedder = _get_embedder()
    es = _get_es()
    _ensure_es_index(es)

    pdf_files = list(data_dir.glob("**/*.pdf"))
    if not pdf_files:
        return {"status": "no_files", "indexed": 0, "skipped": 0}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    db: Session = SessionLocal()
    indexed_files = 0
    skipped_files = 0
    total_chunks = 0

    try:
        for pdf_path in pdf_files:
            file_hash = _file_hash(pdf_path)

            # Skip if already indexed with same content
            existing = db.query(Document).filter_by(file_hash=file_hash).first()
            if existing and existing.status == "indexed":
                skipped_files += 1
                continue

            # Load pages
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Chunk
            chunks = splitter.split_documents(pages)

            # Record document
            if not existing:
                doc_record = Document(
                    filename=pdf_path.name,
                    file_hash=file_hash,
                    page_count=len(pages),
                    status="pending",
                )
                db.add(doc_record)
                db.flush()
                doc_id = doc_record.id
            else:
                doc_id = existing.id
                existing.status = "pending"
                db.flush()

            # Strip NUL bytes — PostgreSQL rejects strings containing \x00
            for c in chunks:
                c.page_content = c.page_content.replace('\x00', '')

            # Embed all chunks in one batch
            texts = [c.page_content for c in chunks]
            vectors = embedder.embed_documents(texts)

            es_bulk = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                page_num = chunk.metadata.get("page", 0)

                # Insert into pgvector
                chunk_record = Chunk(
                    document_id=doc_id,
                    chunk_text=chunk.page_content,
                    embedding=vector,
                    chunk_index=i,
                    page_number=page_num,
                    source_file=pdf_path.name,
                    chunk_size=CHUNK_SIZE,
                    embedding_model=EMBEDDING_MODEL,
                )
                db.add(chunk_record)
                db.flush()

                # Prepare Elasticsearch bulk action
                es_bulk.append({
                    "index": {"_index": ES_INDEX, "_id": str(chunk_record.id)}
                })
                es_bulk.append({
                    "chunk_id":   chunk_record.id,
                    "chunk_text": chunk.page_content,
                    "source":     pdf_path.name,
                    "page":       page_num,
                })

            # Bulk-index into Elasticsearch in batches of 100 actions (50 docs)
            batch_size = 100
            for i in range(0, len(es_bulk), batch_size):
                es.bulk(body=es_bulk[i:i + batch_size], refresh=True)

            # Mark document as indexed
            doc_obj = db.query(Document).get(doc_id)
            doc_obj.status = "indexed"
            db.commit()

            indexed_files += 1
            total_chunks += len(chunks)

    except Exception as exc:
        db.rollback()
        raise exc
    finally:
        db.close()

    return {
        "status": "ok",
        "indexed_files": indexed_files,
        "skipped_files": skipped_files,
        "total_chunks": total_chunks,
    }