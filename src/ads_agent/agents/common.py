# src/ads_agent/agents/common.py
"""Shared utilities for LangGraph agent nodes."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import wraps
import inspect
from typing import TYPE_CHECKING, cast

import structlog

from ads_agent.core.entities.execution_receipt import AgentMetrics, AgentStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ads_agent.agents.state import AgentState

log = structlog.get_logger(__name__)


def _handle_node_failure(agent_name: str, state: AgentState, exc: Exception) -> dict:
    error_msg = f"{agent_name}: {exc}"
    log.exception("node_failed", agent_name=agent_name, error=str(exc))

    receipt = state.get("receipt")
    if receipt:
        failed_at = datetime.now(UTC)
        receipt.add_agent_metrics(
            AgentMetrics(
                agent_name=agent_name,
                status=AgentStatus.FAILED,
                started_at=failed_at,
                completed_at=failed_at,
                error_message=str(exc),
            )
        )

    return {"error": error_msg, "receipt": receipt}


def safe_node(
    agent_name: str,
) -> Callable[
    [Callable[..., dict | Awaitable[dict]]],
    Callable[..., dict | Awaitable[dict]],
]:
    """
    Wrap a graph node to capture failures without crashing the pipeline.

    Supports both sync and async node functions.
    On exception, writes state['error'] and records FAILED AgentMetrics on the receipt.
    The supervisor routes to FINISH when error is set.
    """

    def decorator(
        fn: Callable[..., dict | Awaitable[dict]],
    ) -> Callable[..., dict | Awaitable[dict]]:
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(state: AgentState) -> dict:
                try:
                    return await fn(state)
                except Exception as exc:
                    return _handle_node_failure(agent_name, state, exc)

            return async_wrapper

        @wraps(fn)
        def sync_wrapper(state: AgentState) -> dict:
            try:
                return cast("dict", fn(state))
            except Exception as exc:
                return _handle_node_failure(agent_name, state, exc)

        return sync_wrapper

    return decorator
