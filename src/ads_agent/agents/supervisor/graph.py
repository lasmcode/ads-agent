# src/ads_agent/agents/supervisor/graph.py
"""
LangGraph StateGraph — the compiled multi-agent pipeline.

Graph topology:
    START
      │
  [supervisor]  ←──────────────────────────┐
      │                                      │
  should_continue() ──────────────────────► (conditional edge)
      │                                      │
  ┌───┴──────────────────────────┐           │
  │  "research"  "analysis"  "writer"  "FINISH"
  │      │           │          │
  │  [research]  [analysis]  [writer]
  │      └───────────┴──────────┘
  │                  │
  └──────────────────┘ (all workers loop back to supervisor)

Key design decisions:
  - All workers unconditionally return to supervisor after completion
  - The supervisor holds all routing logic (conditional edges)
  - MemorySaver checkpointer in Phase 1-2 → AsyncPostgresSaver in Phase 3
    (see infrastructure/checkpointer.py::get_postgres_checkpointer). The
    checkpointer parameter accepts any BaseCheckpointSaver so tests keep
    injecting a fast, DB-free MemorySaver.
  - graph.compile() validates edge connectivity at startup, not at runtime
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
import structlog

from ads_agent.agents.analysis.nodes import analysis_node
from ads_agent.agents.research.nodes import research_node
from ads_agent.agents.state import AgentState
from ads_agent.agents.supervisor.nodes import should_continue, supervisor_node
from ads_agent.agents.writer.nodes import writer_node
from ads_agent.core.entities.execution_receipt import ExecutionReceipt
from ads_agent.infrastructure.observability.tracer import (
    capture_trace_id,
    compute_pipeline_scores,
    flush_traces,
    pipeline_trace,
    submit_pipeline_scores,
)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from ads_agent.core.entities.decision_request import DecisionRequest

log = structlog.get_logger(__name__)


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """
    Build and compile the ADS Agent StateGraph.

    Accepts an optional checkpointer for dependency injection:
    - Tests inject MemorySaver directly (fast, no DB required)
    - Production injects AsyncPostgresSaver (Phase 3) via
      infrastructure/checkpointer.py::get_postgres_checkpointer
    - None defaults to a fresh MemorySaver (no persistence across processes)

    Returns a compiled graph ready for invocation.
    """
    builder = StateGraph(AgentState)

    # --- Register nodes ---
    # LangGraph node stubs are stricter than our TypedDict partial-update nodes.
    builder.add_node("supervisor", cast("Any", supervisor_node))
    builder.add_node("research", cast("Any", research_node))
    builder.add_node("analysis", cast("Any", analysis_node))
    builder.add_node("writer", cast("Any", writer_node))

    # --- Entry point ---
    builder.add_edge(START, "supervisor")

    # --- Conditional edge: supervisor decides next node ---
    # should_continue() returns a string key that maps to a node name
    builder.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "research": "research",
            "analysis": "analysis",
            "writer": "writer",
            "FINISH": END,
        },
    )

    # --- All workers loop back to supervisor unconditionally ---
    builder.add_edge("research", "supervisor")
    builder.add_edge("analysis", "supervisor")
    builder.add_edge("writer", "supervisor")

    # Compile validates the graph topology.
    # If any node is unreachable or any edge references a missing node,
    # compile() raises immediately — fail fast, not at runtime.
    graph = builder.compile(checkpointer=checkpointer or MemorySaver())

    log.info("graph_compiled", nodes=list(builder.nodes))
    return graph


async def run_pipeline(
    request: DecisionRequest,
    thread_id: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> tuple[AgentState, ExecutionReceipt]:
    """
    Execute the full agent pipeline for a given DecisionRequest.

    Args:
        request: The validated user query.
        thread_id: Optional ID for checkpointing (resumes prior conversation).
        checkpointer: Injected checkpointer — defaults to MemorySaver. Pass
            `await get_postgres_checkpointer()` for durable persistence.

    Returns:
        Tuple of (final_state, receipt) for the caller to use.
    """
    graph = build_graph(checkpointer=checkpointer)

    # Initialize the ExecutionReceipt — will be updated by each node
    receipt = ExecutionReceipt(request_id=request.id)

    # Build the initial state — only required fields, defaults handle the rest
    initial_state: AgentState = {
        "request": request,
        "messages": [HumanMessage(content=request.query)],
        "next_agent": "",
        "research_output": None,
        "analysis_output": None,
        "final_report": None,
        "receipt": receipt,
        "iterations": 0,
        "error": None,
    }

    # Config carries the thread_id for checkpointing
    config = {"configurable": {"thread_id": thread_id or request.id}}

    log.info(
        "pipeline_started",
        request_id=request.id,
        query_preview=request.query[:80],
    )

    final_state: AgentState | None = None
    session_id = thread_id or request.id

    with pipeline_trace(request.id, session_id=session_id):
        receipt.trace_id = capture_trace_id()

        try:
            final_state = cast(
                "AgentState",
                await graph.ainvoke(
                    cast("Any", initial_state),
                    config=cast("Any", config),
                ),
            )
        except Exception as exc:
            log.error("pipeline_failed", request_id=request.id, error=str(exc))
            receipt.mark_completed()
            raise
        else:
            has_sources, trade_offs_count = compute_pipeline_scores(final_state)
            submit_pipeline_scores(
                receipt.trace_id,
                has_sources=has_sources,
                trade_offs_count=trade_offs_count,
            )

    flush_traces()
    receipt.mark_completed()

    log.info(
        "pipeline_completed",
        request_id=request.id,
        duration_s=receipt.total_duration_seconds,
        agents_run=len(receipt.agents),
        trace_id=receipt.trace_id,
    )

    return final_state, receipt
