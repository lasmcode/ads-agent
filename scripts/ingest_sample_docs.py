#!/usr/bin/env python
# scripts/ingest_sample_docs.py
"""
Demo/smoke-test script for the RAG pipeline.

Ingests a handful of real LangGraph/LangChain documentation pages into the
pgvector-backed knowledge base, then runs a sample hybrid_search query and
prints the fused results. Useful for manually verifying end-to-end wiring
(fetch -> chunk -> embed -> upsert -> hybrid retrieve) against a live
PostgreSQL instance without going through the full agent pipeline.

Prerequisites:
    - PostgreSQL reachable at ADS_DATABASE_URL (see docker-compose.yml)
    - GEMINI_API_KEY set (real embedding calls, not mocked)

Usage:
    uv run python scripts/ingest_sample_docs.py
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

from ads_agent.infrastructure.asyncio_compat import run as run_async
from ads_agent.infrastructure.vector_store.connection import close_pool, setup_schema
from ads_agent.infrastructure.vector_store.ingestion import ingest_document
from ads_agent.infrastructure.vector_store.retriever import hybrid_search

if TYPE_CHECKING:
    from ads_agent.core.entities.chunk import Chunk

log = structlog.get_logger(__name__)

SAMPLE_URLS = [
    "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    "https://langchain-ai.github.io/langgraph/how-tos/persistence/",
    "https://python.langchain.com/docs/concepts/",
]

SAMPLE_QUERY = "how to implement checkpointing in LangGraph"


def _configure_logging() -> None:
    """
    Human-readable console logs for this standalone script.

    Uses structlog's PrintLogger (not stdlib) as the backend, so level
    filtering comes solely from `make_filtering_bound_logger` — mixing in
    `structlog.stdlib.filter_by_level` here would crash, since that
    processor expects a stdlib `logging.Logger` instance, not a PrintLogger.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


async def _ingest_all(urls: list[str]) -> None:
    for url in urls:
        try:
            chunk_count = await ingest_document(url)
            log.info("sample_ingest_done", url=url, chunks=chunk_count)
        except Exception:
            # One bad page (network hiccup, paywall, layout trafilatura can't
            # parse) shouldn't abort the whole demo run.
            log.exception("sample_ingest_failed", url=url)


def _print_results(query: str, results: list[Chunk]) -> None:
    print()
    print(f"=== hybrid_search results for: {query!r} ===")
    if not results:
        print("(no results — did ingestion succeed above?)")
        return
    for rank, chunk in enumerate(results, start=1):
        preview = chunk.content[:160].replace("\n", " ")
        print(f"\n#{rank}  score={chunk.score:.4f}  source={chunk.source_url}")
        print(f"    title:   {chunk.title or '(untitled)'}")
        print(f"    preview: {preview}...")


async def main() -> None:
    _configure_logging()

    await setup_schema()
    await _ingest_all(SAMPLE_URLS)

    results = await hybrid_search(SAMPLE_QUERY, top_k=5)
    _print_results(SAMPLE_QUERY, results)

    await close_pool()


if __name__ == "__main__":
    run_async(main())
