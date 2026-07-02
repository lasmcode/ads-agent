-- src/ads_agent/infrastructure/vector_store/schema.sql
--
-- Schema for the ADS Agent knowledge base (Phase 3 — RAG pipeline).
--
-- Why raw SQL instead of Alembic:
--   This is a single append-mostly table with no evolving business logic —
--   there is no ORM model that needs migration tooling to stay in sync.
--   A plain, idempotent (CREATE ... IF NOT EXISTS) script run once at
--   startup (see connection.py::setup_schema) is simpler to reason about
--   and matches the project's "no ORM" style (raw psycopg elsewhere too).
--   If this table gains multiple evolving variants or FK relationships to
--   other tables, revisit this decision and introduce Alembic then.

-- pgvector: adds the `vector` type and ANN index access methods (HNSW/IVFFlat).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url   TEXT NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    content      TEXT NOT NULL,
    -- SHA-256 hex digest of `content`. Used for idempotent upserts: re-ingesting
    -- an unchanged chunk is a no-op; a changed chunk updates the existing row
    -- instead of inserting a duplicate.
    content_hash TEXT NOT NULL,
    -- text-embedding-004 was the originally specified model but Google has
    -- decommissioned it; gemini-embedding-001 (requested at 768 dimensions)
    -- is its supported, verified successor — see core/settings.py.
    embedding    vector(768) NOT NULL,
    -- Generated column: PostgreSQL maintains this automatically from
    -- title + content, so ingestion never has to compute it by hand.
    tsv          tsvector GENERATED ALWAYS AS (
                     to_tsvector('english', coalesce(title, '') || ' ' || content)
                 ) STORED,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW over IVFFlat: HNSW gives logarithmic-time approximate search with no
-- training/build step and better recall at our (sub-million row) scale —
-- IVFFlat requires choosing `lists` ahead of time based on row count and
-- degrades until the index is retrained after significant growth.
-- cosine distance matches the similarity metric used by Gemini embeddings.
CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_hnsw_idx
    ON knowledge_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search (BM25-style ranking via ts_rank/ts_rank_cd).
CREATE INDEX IF NOT EXISTS knowledge_chunks_tsv_gin_idx
    ON knowledge_chunks
    USING gin (tsv);

-- Enforces idempotent ingestion at the database level: a document that
-- produces the same (source_url, content_hash) pair as an existing row
-- cannot be duplicated, even under concurrent ingestion.
CREATE UNIQUE INDEX IF NOT EXISTS knowledge_chunks_source_hash_uniq_idx
    ON knowledge_chunks (source_url, content_hash);

-- Speeds up "delete stale chunks for this URL" during re-ingestion.
CREATE INDEX IF NOT EXISTS knowledge_chunks_source_url_idx
    ON knowledge_chunks (source_url);
