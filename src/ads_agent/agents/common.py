# src/ads_agent/agents/common.py
"""Shared utilities for LangGraph agent nodes."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING

import structlog

from ads_agent.core.entities.execution_receipt import AgentMetrics, AgentStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from ads_agent.agents.state import AgentState

log = structlog.get_logger(__name__)


def safe_node(
    agent_name: str,
) -> Callable[[Callable[[AgentState], dict]], Callable[[AgentState], dict]]:
    """
    Wrap a graph node to capture failures without crashing the pipeline.

    On exception, writes state['error'] and records FAILED AgentMetrics on the receipt.
    The supervisor routes to FINISH when error is set.
    """

    def decorator(fn: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
        @wraps(fn)
        def wrapper(state: AgentState) -> dict:
            try:
                return fn(state)
            except Exception as exc:
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

        return wrapper

    return decorator
