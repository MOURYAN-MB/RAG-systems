from sqlalchemy import (
    create_engine, Column, Integer, Text, Float, ARRAY,
    DateTime, String, func
)
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

from src.config import DATABASE_URL, EMBEDDING_DIM

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id         = Column(Integer, primary_key=True)
    filename   = Column(Text, nullable=False)
    file_hash  = Column(Text, nullable=False, unique=True)
    page_count = Column(Integer)
    status     = Column(String, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"
    id              = Column(Integer, primary_key=True)
    document_id     = Column(Integer)
    chunk_text      = Column(Text, nullable=False)
    embedding       = Column(Vector(EMBEDDING_DIM))
    chunk_index     = Column(Integer)
    page_number     = Column(Integer)
    source_file     = Column(Text)
    chunk_size      = Column(Integer)
    embedding_model = Column(Text)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


class Evaluation(Base):
    __tablename__ = "evaluations"
    id                  = Column(Integer, primary_key=True)
    query               = Column(Text, nullable=False)
    answer              = Column(Text, nullable=False)
    retrieved_chunk_ids = Column(ARRAY(Integer))
    faithfulness        = Column(Float)
    answer_relevancy    = Column(Float)
    context_recall      = Column(Float)
    latency_ms          = Column(Integer)
    model               = Column(Text)
    retrieval_method    = Column(Text, default="hybrid")
    created_at          = Column(DateTime(timezone=True), server_default=func.now())


class Alert(Base):
    __tablename__ = "alerts"
    id           = Column(Integer, primary_key=True)
    metric       = Column(Text, nullable=False)
    metric_value = Column(Float, nullable=False)
    threshold    = Column(Float, nullable=False)
    status       = Column(String, default="open")
    triggered_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at  = Column(DateTime(timezone=True))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()