# src/ads_agent/agents/analysis/nodes.py
"""
Analysis Agent node — Phase 1 stub.

In Phase 1: returns a hardcoded placeholder.
In Phase 4: replaced with structured LLM reasoning using Pydantic output parsing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
import structlog

from ads_agent.agents.common import safe_node
from ads_agent.core.entities.execution_receipt import AgentMetrics, AgentStatus

if TYPE_CHECKING:
    from ads_agent.agents.state import AgentState

log = structlog.get_logger(__name__)


@safe_node("analysis")
def analysis_node(state: AgentState) -> dict:
    """
    Analysis Agent: evaluates trade-offs from research output.

    Phase 1: returns a stub output.
    """
    log.info("analysis_node_started", request_id=state["request"].id)

    started_at = datetime.now(UTC)

    analysis_output = (
        f"[STUB] Trade-off analysis for: '{state['request'].query}'\n\n"
        "This is a placeholder. In Phase 4, this node will:\n"
        "  1. Parse research findings\n"
        "  2. Identify key decision dimensions\n"
        "  3. Score each option per dimension\n"
        "  4. Return a structured TradeOff list via Pydantic parsing"
    )

    completed_at = datetime.now(UTC)

    metrics = AgentMetrics(
        agent_name="analysis",
        status=AgentStatus.COMPLETED,
        started_at=started_at,
        completed_at=completed_at,
    )

    receipt = state.get("receipt")
    if receipt:
        receipt.add_agent_metrics(metrics)

    log.info("analysis_node_completed", duration_s=metrics.duration_seconds)

    return {
        "analysis_output": analysis_output,
        "messages": [AIMessage(content=analysis_output, name="analysis")],
        "receipt": receipt,
    }
