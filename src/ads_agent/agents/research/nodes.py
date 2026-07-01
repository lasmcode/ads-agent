# src/ads_agent/agents/research/nodes.py
"""
Research Agent node — Phase 1 stub.

In Phase 1: returns a hardcoded placeholder to validate graph routing.
In Phase 2: replaced with real MCP tool calls (web search, doc retrieval).
In Phase 3: adds RAG queries against pgvector knowledge base.

The function signature and return contract remain unchanged across phases.
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


@safe_node("research")
def research_node(state: AgentState) -> dict:
    """
    Research Agent: gathers evidence for the technical decision.

    Phase 1: returns a stub output.
    Subsequent phases replace this body while keeping the signature.
    """
    log.info("research_node_started", request_id=state["request"].id)

    started_at = datetime.now(UTC)

    # --- Phase 1 stub output ---
    # Simulates what the real research agent will produce in Phase 2
    research_output = (
        f"[STUB] Research findings for: '{state['request'].query}'\n\n"
        "This is a placeholder. In Phase 2, this node will:\n"
        "  1. Search the web via MCP web_search tool\n"
        "  2. Fetch relevant documentation pages\n"
        "  3. Query the pgvector knowledge base for prior decisions\n"
        "  4. Return structured, cited findings"
    )

    completed_at = datetime.now(UTC)

    # Build agent metrics for the ExecutionReceipt
    metrics = AgentMetrics(
        agent_name="research",
        status=AgentStatus.COMPLETED,
        started_at=started_at,
        completed_at=completed_at,
        input_tokens=0,  # Real token counts come from LiteLLM in Phase 4
        output_tokens=0,
    )

    receipt = state.get("receipt")
    if receipt:
        receipt.add_agent_metrics(metrics)

    log.info("research_node_completed", duration_s=metrics.duration_seconds)

    return {
        "research_output": research_output,
        "messages": [AIMessage(content=research_output, name="research")],
        "receipt": receipt,
    }
