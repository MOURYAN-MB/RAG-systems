import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from src.config import DATA_DIR
from src.ingestion import ingest_documents

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a PDF and trigger background ingestion."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest = DATA_DIR / file.filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(ingest_documents)
    return {"status": "accepted", "filename": file.filename, "message": "Ingestion started in background."}


@router.post("/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    """Re-ingest all documents (use after adding new files manually)."""
    background_tasks.add_task(ingest_documents)
    return {"status": "accepted", "message": "Re-ingestion started in background."}


@router.get("/status")
def ingestion_status():
    """Return count of indexed documents and chunks."""
    from src.database import SessionLocal, Document, Chunk
    db = SessionLocal()
    try:
        n_docs   = db.query(Document).filter_by(status="indexed").count()
        n_chunks = db.query(Chunk).count()
        return {"indexed_documents": n_docs, "total_chunks": n_chunks}
    finally:
        db.close()