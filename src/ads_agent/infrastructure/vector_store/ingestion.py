# src/ads_agent/infrastructure/vector_store/ingestion.py
"""
Ingestion pipeline: fetch -> chunk -> embed (batch) -> upsert.

Idempotency: each chunk's SHA-256 content hash participates in the
(source_url, content_hash) unique constraint from schema.sql.
    - Unchanged chunk (same hash already stored)  -> UPDATE, no new row.
    - Changed chunk    (new hash for this URL)     -> INSERT, new row.
    - Removed chunk    (old hash no longer present) -> DELETEd as stale.
This means re-ingesting the exact same document twice is a true no-op at
the row level, and re-ingesting an edited document converges the table to
match only the latest version — never accumulating duplicates or orphans.
"""

from __future__ import annotations

import hashlib

import litellm
import numpy as np
from psycopg.types.json import Jsonb
import structlog
import trafilatura

from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.mcp.extract import fetch_page
from ads_agent.infrastructure.vector_store.chunker import chunk_document
from ads_agent.infrastructure.vector_store.connection import get_pool

log = structlog.get_logger(__name__)

# ON CONFLICT keeps the row keyed by (source_url, content_hash): re-ingesting
# unchanged content updates nothing meaningful (values are identical), while
# a genuinely new chunk hash inserts a fresh row.
_UPSERT_SQL = """
    INSERT INTO knowledge_chunks (source_url, title, content, content_hash, embedding, metadata)
    VALUES (%(source_url)s, %(title)s, %(content)s, %(content_hash)s, %(embedding)s, %(metadata)s)
    ON CONFLICT (source_url, content_hash)
    DO UPDATE SET
        title = EXCLUDED.title,
        embedding = EXCLUDED.embedding,
        metadata = EXCLUDED.metadata
"""

# Removes chunks left over from a previous version of the same document —
# e.g. a section that was deleted upstream must not linger in the knowledge
# base forever. `<> ALL(array)` is the array-parameter-friendly form of
# `NOT IN (...)`.
_DELETE_STALE_SQL = """
    DELETE FROM knowledge_chunks
    WHERE source_url = %(source_url)s
      AND content_hash <> ALL(%(keep_hashes)s)
"""


def _content_hash(content: str) -> str:
    """Stable fingerprint used for idempotent upserts (see module docstring)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _extract_markdown(html_or_text: str, source_url: str) -> str:
    """
    Extract markdown with headers preserved so the chunker can split on
    document structure. Plain-text/markdown sources pass through unchanged
    since trafilatura is a no-op-ish pass-through on already-clean text.
    """
    markdown = trafilatura.extract(
        html_or_text,
        url=source_url,
        output_format="markdown",
        include_formatting=True,
        include_tables=True,
        include_comments=False,
    )
    return markdown or ""


async def _embed_batch(texts: list[str]) -> list[np.ndarray]:
    """
    Embed a batch of chunk texts in a single LiteLLM call (not one-by-one).

    Returned as numpy arrays — pgvector's registered psycopg adapter (see
    connection.py's pool `configure` hook) sends these as `vector` literals,
    matching the column type exactly instead of relying on implicit casts.
    """
    settings = get_settings()
    response = await litellm.aembedding(
        model=settings.embedding_model,
        input=texts,
        dimensions=settings.embedding_dimensions,
    )
    return [np.array(item["embedding"], dtype=np.float32) for item in response.data]


async def ingest_document(url: str) -> int:
    """
    Fetch, chunk, embed, and upsert a single document into the knowledge base.

    Returns:
        Number of chunks currently stored for this URL after ingestion
        (i.e. inserted + updated), or 0 if no extractable content was found.
    """
    log.info("ingestion_started", url=url)

    page = await fetch_page(url)
    markdown = _extract_markdown(page.content, page.final_url)
    if not markdown.strip():
        log.warning("ingestion_no_content", url=url)
        return 0

    drafts = chunk_document(markdown)
    if not drafts:
        log.warning("ingestion_no_chunks", url=url)
        return 0

    embeddings = await _embed_batch([draft.content for draft in drafts])
    hashes = [_content_hash(draft.content) for draft in drafts]

    pool = await get_pool()
    async with pool.connection() as conn, conn.transaction():
        for draft, content_hash, embedding in zip(drafts, hashes, embeddings, strict=True):
            await conn.execute(
                _UPSERT_SQL,
                {
                    "source_url": page.final_url,
                    "title": draft.title,
                    "content": draft.content,
                    "content_hash": content_hash,
                    "embedding": embedding,
                    "metadata": Jsonb({}),
                },
            )

        await conn.execute(
            _DELETE_STALE_SQL,
            {"source_url": page.final_url, "keep_hashes": hashes},
        )

    log.info("ingestion_completed", url=page.final_url, chunks=len(drafts))
    return len(drafts)
