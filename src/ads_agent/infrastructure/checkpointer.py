# src/ads_agent/infrastructure/checkpointer.py
"""
LangGraph checkpointer factory (Phase 3): MemorySaver -> AsyncPostgresSaver.

Now that PostgreSQL is a real, running dependency (see docker-compose.yml),
pipeline state survives process restarts instead of living only in RAM.
`AsyncPostgresSaver.setup()` creates/migrates the checkpoint tables and is
explicitly documented by LangGraph as a one-time operation — calling it on
every request would re-run migration-version lookups needlessly on the hot
path. `get_postgres_checkpointer()` caches the saver at module scope so a
process calls `.setup()` exactly once, on its first use, regardless of how
many pipeline runs follow.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
import structlog

from ads_agent.core.entities.decision_report import DecisionReport, RecommendationStrength, TradeOff
from ads_agent.core.entities.decision_request import DecisionComplexity, DecisionRequest
from ads_agent.core.entities.execution_receipt import AgentMetrics, AgentStatus, ExecutionReceipt
from ads_agent.core.settings import get_settings

log = structlog.get_logger(__name__)

# LangGraph's msgpack checkpoint serializer warns (and, per CVE-2026-28277,
# will eventually block) deserialization of any type it doesn't recognize —
# this is a deliberate anti-RCE guard against tampered checkpoint rows, not
# a bug. langchain_core message types (HumanMessage, AIMessage, ...) are
# already in LangGraph's built-in SAFE_MSGPACK_TYPES; everything else that
# flows through AgentState is one of our own domain entities, so we register
# them explicitly instead of leaving the (soon-to-be-removed) permissive
# "warn but allow anything" default in place.
#
# NOTE: adding a new custom Pydantic type/enum to AgentState later requires
# adding it here too, or its checkpoints will be BLOCKED — an explicit
# allowlist (unlike the default) rejects anything not listed.
_ALLOWED_MSGPACK_TYPES = [
    DecisionRequest,
    DecisionComplexity,
    DecisionReport,
    RecommendationStrength,
    TradeOff,
    ExecutionReceipt,
    AgentMetrics,
    AgentStatus,
]

CheckpointerPool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]]

_pool: CheckpointerPool | None = None
_checkpointer: AsyncPostgresSaver | None = None
_setup_lock = asyncio.Lock()


async def get_postgres_checkpointer() -> AsyncPostgresSaver:
    """
    Return the process-wide AsyncPostgresSaver, opening its pool and running
    `.setup()` on first call only.

    Uses a dedicated connection pool rather than sharing the knowledge
    store's pool (infrastructure/vector_store/connection.py) — the two have
    different connection requirements (pgvector type registration vs. plain
    dict-row checkpoint payloads) and different lifecycles.
    """
    global _pool, _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    async with _setup_lock:
        if _checkpointer is None:
            settings = get_settings()
            pool: CheckpointerPool = AsyncConnectionPool(
                conninfo=settings.database_url,
                min_size=1,
                max_size=10,
                open=False,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            )
            await pool.open(wait=True, timeout=10.0)

            serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_TYPES)
            checkpointer = AsyncPostgresSaver(conn=pool, serde=serde)
            await checkpointer.setup()

            _pool = pool
            _checkpointer = checkpointer
            log.info("postgres_checkpointer_ready")

    return _checkpointer


async def close_checkpointer_pool() -> None:
    """Close the checkpointer's connection pool. Safe to call when unopened."""
    global _pool, _checkpointer

    if _pool is not None:
        await _pool.close()
        log.info("postgres_checkpointer_pool_closed")
        _pool = None
        _checkpointer = None
