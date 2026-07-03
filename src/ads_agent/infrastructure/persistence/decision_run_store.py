# src/ads_agent/infrastructure/persistence/decision_run_store.py
"""Read pipeline results from LangGraph checkpoints keyed by request_id."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ads_agent.agents.supervisor.graph import build_graph

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.types import RunnableConfig

    from ads_agent.agents.state import AgentState


async def get_state(
    request_id: str,
    checkpointer: BaseCheckpointSaver,
) -> AgentState | None:
    """
    Load the latest checkpoint for a decision run.

    Checkpoints are stored with thread_id equal to DecisionRequest.id
    (see run_pipeline in graph.py).
    """
    graph = build_graph(checkpointer=checkpointer)
    config = cast("RunnableConfig", {"configurable": {"thread_id": request_id}})
    snapshot = await graph.aget_state(config)

    if not snapshot.values:
        return None

    values = cast("dict[str, Any]", snapshot.values)
    request = values.get("request")
    if request is None or request.id != request_id:
        return None

    return cast("AgentState", values)
