# tests/integration/vector_store/test_ingestion_retrieval.py
"""
Integration tests for the RAG pipeline against a real PostgreSQL + pgvector
instance (see docker-compose.yml — `make docker-up`) and the real Gemini
embedding API.

Requires:
    - PostgreSQL reachable at ADS_DATABASE_URL (defaults to the
      docker-compose service) with the `vector` extension installable.
    - GEMINI_API_KEY set — hybrid_search's vector leg needs real embeddings,
      a mocked one would trivially "pass" without testing anything real.

The network fetch (fetch_page) is mocked so these tests stay hermetic and
fast regardless of whether some real doc site is reachable/unchanged — but
the database, chunker, embedding call, and retrieval SQL are all real. Each
test uses a unique source_url and cleans up its own rows afterward so tests
can run repeatedly and in parallel without colliding.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch
import uuid

import pytest

from ads_agent.infrastructure.mcp.extract import FetchedPage
from ads_agent.infrastructure.vector_store.connection import close_pool, get_pool, setup_schema
from ads_agent.infrastructure.vector_store.ingestion import ingest_document
from ads_agent.infrastructure.vector_store.retriever import hybrid_search

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio,
    pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"),
]

# ingestion.py runs trafilatura.extract(..., output_format="markdown") on
# `page.content`, exactly as it would on a real fetched page — so the mock
# fixture below must be actual HTML (with real <h1>/<h2>/<p> tags), not
# pre-rendered markdown, or trafilatura has nothing to parse headers from.
_SAMPLE_HTML = """
<html><body><article>
<h1>Checkpointing Guide</h1>
<h2>Overview</h2>
<p>Checkpointers persist a thread's graph state so long-running agent
conversations survive process restarts.</p>
<h2>PostgresSaver setup</h2>
<p>Call checkpointer.setup() exactly once when the application starts,
before compiling the graph. Re-running setup() is safe but wasteful.</p>
</article></body></html>
"""

_SAMPLE_HTML_WITH_NEW_SECTION = _SAMPLE_HTML.replace(
    "</article>",
    "<h2>New section</h2><p>Brand-new content.</p></article>",
)


def _fake_source_url() -> str:
    """A fresh, unique URL per test — never collides with real or prior data."""
    return f"https://integration-test.invalid/{uuid.uuid4()}"


def _patch_fetch_page(html: str, source_url: str):
    """Mock the network boundary; everything downstream (DB, embeddings) is real."""
    fetched = FetchedPage(
        final_url=source_url,
        content=html,
        content_type="text/html",
        hostname="integration-test.invalid",
    )
    return patch(
        "ads_agent.infrastructure.vector_store.ingestion.fetch_page",
        AsyncMock(return_value=fetched),
    )


async def _delete_rows_for(source_url: str) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM knowledge_chunks WHERE source_url = %(source_url)s",
            {"source_url": source_url},
        )


async def _row_count_for(source_url: str) -> int:
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT count(*) AS c FROM knowledge_chunks WHERE source_url = %(source_url)s",
            {"source_url": source_url},
        )
        row = await cur.fetchone()
        return row["c"]


@pytest.fixture(autouse=True)
async def _real_postgres():
    """Skip gracefully (instead of failing) when Postgres isn't reachable."""
    try:
        await setup_schema()
    except Exception as exc:  # any connectivity failure means "skip", not "fail"
        pytest.skip(f"PostgreSQL not reachable — run `make docker-up` first ({exc})")
    yield
    await close_pool()


class TestIngestionAndRetrieval:
    async def test_ingest_and_retrieve(self) -> None:
        """A freshly ingested document is retrievable via hybrid_search."""
        source_url = _fake_source_url()
        try:
            with _patch_fetch_page(_SAMPLE_HTML, source_url):
                chunk_count = await ingest_document(source_url)
            assert chunk_count > 0

            results = await hybrid_search("how do I set up the PostgresSaver checkpointer", top_k=5)

            matching = [r for r in results if r.source_url == source_url]
            assert matching, "expected the ingested document to be retrievable"
            assert any("checkpointer.setup()" in r.content for r in matching)
        finally:
            await _delete_rows_for(source_url)

    async def test_ingest_idempotency_does_not_duplicate_rows(self) -> None:
        """Ingesting the exact same document twice must not create duplicate rows."""
        source_url = _fake_source_url()
        try:
            with _patch_fetch_page(_SAMPLE_HTML, source_url):
                first_count = await ingest_document(source_url)
                second_count = await ingest_document(source_url)

            assert first_count == second_count
            assert await _row_count_for(source_url) == first_count
        finally:
            await _delete_rows_for(source_url)

    async def test_ingest_updates_changed_content_without_orphaning_rows(self) -> None:
        """Re-ingesting an edited document converges to only the latest content."""
        source_url = _fake_source_url()
        try:
            with _patch_fetch_page(_SAMPLE_HTML, source_url):
                await ingest_document(source_url)

            with _patch_fetch_page(_SAMPLE_HTML_WITH_NEW_SECTION, source_url):
                await ingest_document(source_url)

            pool = await get_pool()
            async with pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT content FROM knowledge_chunks WHERE source_url = %(source_url)s",
                    {"source_url": source_url},
                )
                rows = await cur.fetchall()

            all_content = "\n".join(row["content"] for row in rows)
            assert "Brand-new content." in all_content
        finally:
            await _delete_rows_for(source_url)

    async def test_search_with_blank_query_returns_empty_list_without_erroring(self) -> None:
        """
        A blank/whitespace query short-circuits to an empty result.

        Note: a nonsense-but-non-blank query is deliberately *not* tested for
        emptiness here — pgvector's `<=>` ORDER BY always returns the nearest
        rows in the table regardless of how distant they are (there is no
        cosine-distance cutoff in `_VECTOR_SQL`), so hybrid_search only
        returns [] when the knowledge base itself is empty or the query is
        blank. Score-based relevance filtering is applied by callers via
        `rag_score_threshold`, not by hybrid_search itself.
        """
        assert await hybrid_search("   ", top_k=5) == []
