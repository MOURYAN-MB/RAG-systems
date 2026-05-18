-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a separate database for Airflow metadata
CREATE DATABASE airflow;

-- ── Documents ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    file_hash   TEXT NOT NULL UNIQUE,   -- skip re-ingestion if unchanged
    page_count  INTEGER,
    status      TEXT DEFAULT 'pending', -- pending | indexed | failed
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Chunks (pgvector) ─────────────────────────────────────────────────────
-- 384 dims = all-MiniLM-L6-v2
CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text      TEXT NOT NULL,
    embedding       vector(384),
    chunk_index     INTEGER,            -- position within document
    page_number     INTEGER,
    source_file     TEXT,
    chunk_size      INTEGER,            -- chunking params used (for lineage)
    embedding_model TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast approximate nearest-neighbour search
-- ef_construction=64, m=16 are good defaults for 384-dim vectors
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (ef_construction = 64, m = 16);

-- ── Evaluation log ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS evaluations (
    id                  SERIAL PRIMARY KEY,
    query               TEXT NOT NULL,
    answer              TEXT NOT NULL,
    retrieved_chunk_ids INTEGER[],
    faithfulness        FLOAT,          -- RAGAS: is answer grounded in context?
    answer_relevancy    FLOAT,          -- RAGAS: does answer address the question?
    context_recall      FLOAT,          -- RAGAS: did retrieval cover what was needed?
    latency_ms          INTEGER,        -- end-to-end response time
    model               TEXT,
    retrieval_method    TEXT DEFAULT 'hybrid',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── Quality alerts ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id              SERIAL PRIMARY KEY,
    metric          TEXT NOT NULL,      -- e.g. 'faithfulness'
    metric_value    FLOAT NOT NULL,
    threshold       FLOAT NOT NULL,
    status          TEXT DEFAULT 'open', -- open | resolved
    triggered_at    TIMESTAMPTZ DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ
);