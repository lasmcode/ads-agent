# src/ads_agent/infrastructure/vector_store/connection.py
"""
Async connection pool management for the pgvector-backed knowledge store.

A single process-wide AsyncConnectionPool is lazily created on first use and
reused by both the ingestion and retrieval adapters. This mirrors the
recommended psycopg3 pattern: construct the pool with `open=False` and open
it explicitly inside the running event loop — opening in `__init__` (the
pre-3.2 default) risks binding the pool to the wrong event loop when the
    CLI's `asyncio_compat.run()` creates a fresh loop per invocation — on
    Windows it uses `loop_factory=SelectorEventLoop` because psycopg async
    is incompatible with the default ProactorEventLoop.

Bootstrap ordering note:
    Every pooled connection registers pgvector's `vector` type adapter via
    `configure=` so callers can pass/receive Python lists transparently.
    That registration requires the `vector` extension to already exist in
    the target database. `setup_schema()` therefore deliberately uses its
    own standalone connection instead of borrowing one from the pool — this
    avoids a chicken-and-egg failure on a brand-new database where the pool
    would try to register a type that doesn't exist yet. Callers MUST invoke
    `setup_schema()` once before the pool is used for ingestion/retrieval,
    exactly like `AsyncPostgresSaver.setup()` for the LangGraph checkpointer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from pgvector.psycopg import register_vector_async
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
import structlog

from ads_agent.core.settings import get_settings

if TYPE_CHECKING:
    from psycopg import AsyncConnection

log = structlog.get_logger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_pool: AsyncConnectionPool | None = None
_pool_lock = asyncio.Lock()


async def _configure_connection(conn: AsyncConnection) -> None:
    """Register pgvector's `vector` type adapter on every pooled connection."""
    await register_vector_async(conn)


async def get_pool() -> AsyncConnectionPool:
    """
    Return the process-wide connection pool, opening it on first call.

    Guarded by an asyncio.Lock so concurrent first-callers (e.g. parallel
    hybrid_search + ingest_document invocations) don't race to open two pools.
    """
    global _pool

    if _pool is not None:
        return _pool

    async with _pool_lock:
        if _pool is None:
            settings = get_settings()
            pool = AsyncConnectionPool(
                conninfo=settings.database_url,
                min_size=1,
                max_size=10,
                open=False,
                kwargs={"autocommit": True, "row_factory": dict_row},
                configure=_configure_connection,
            )
            await pool.open(wait=True, timeout=10.0)
            log.info("vector_store_pool_opened", min_size=1, max_size=10)
            _pool = pool

    return _pool


async def close_pool() -> None:
    """Close the pool and release its connections. Safe to call when unopened."""
    global _pool

    if _pool is not None:
        await _pool.close()
        log.info("vector_store_pool_closed")
        _pool = None


async def setup_schema() -> None:
    """
    Create the `vector` extension, knowledge_chunks table, and indexes if missing.

    Idempotent via `CREATE ... IF NOT EXISTS` — safe to call on every
    application startup. Uses a standalone connection (not the shared pool)
    specifically so it can create the `vector` extension before any pooled
    connection tries to register the type — see module docstring.
    """
    settings = get_settings()
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")

    # No query parameters are used here, so psycopg's simple query protocol
    # accepts the whole semicolon-separated DDL script in a single execute().
    async with await psycopg.AsyncConnection.connect(
        settings.database_url, autocommit=True
    ) as conn:
        await conn.execute(schema_sql)

    log.info("vector_store_schema_ready")
