# src/ads_agent/agents/supervisor/nodes.py
"""
Supervisor node — the central router of the multi-agent pipeline.

Responsibilities:
  1. Read current pipeline state
  2. Decide which agent to call next (deterministic rules OR LLM)
  3. Enforce the circuit breaker
  4. Update the receipt with iteration count

"""

from __future__ import annotations

import structlog

from ads_agent.agents.state import MAX_ITERATIONS, AgentState
from ads_agent.core.entities.execution_receipt import AgentMetrics, AgentStatus

log = structlog.get_logger(__name__)


def supervisor_node(state: AgentState) -> dict:
    """
    Deterministic supervisor router.

    Reads the current state fields and applies routing rules in order.
    Returns only the fields it modifies — LangGraph merges the rest.

    In Phase 4 this will be replaced with LLM-assisted routing while
    keeping the same function signature and return contract.
    """
    current_iterations = state.get("iterations", 0)
    receipt = state.get("receipt")

    log.info(
        "supervisor_routing",
        iteration=current_iterations,
        has_research=bool(state.get("research_output")),
        has_analysis=bool(state.get("analysis_output")),
        has_report=bool(state.get("final_report")),
    )

    # --- Circuit breaker ---
    # Must be checked before any routing decision
    if current_iterations >= MAX_ITERATIONS:
        log.warning(
            "circuit_breaker_triggered",
            iterations=current_iterations,
            max=MAX_ITERATIONS,
        )
        if receipt:
            receipt.circuit_breaker_triggered = True
        return {
            "next_agent": "FINISH",
            "iterations": current_iterations + 1,
            "receipt": receipt,
        }

    # --- Deterministic routing rules ---
    # Order matters: each rule is checked sequentially.

    # Rule 1: Error in previous node → fail gracefully
    if state.get("error"):
        log.error("supervisor_routing_on_error", error=state["error"])
        next_agent = "FINISH"

    # Rule 2: No research yet → start with research
    elif not state.get("research_output"):
        next_agent = "research"

    # Rule 3: Research done, no analysis yet → run analysis
    elif not state.get("analysis_output"):
        next_agent = "analysis"

    # Rule 4: Analysis done, no report yet → run writer
    elif not state.get("final_report"):
        next_agent = "writer"

    # Rule 5: Report exists → pipeline complete
    else:
        next_agent = "FINISH"

    log.info("supervisor_decision", next_agent=next_agent)

    # Track the supervisor's own metrics in the receipt
    if receipt:
        supervisor_metrics = AgentMetrics(
            agent_name="supervisor",
            status=AgentStatus.COMPLETED,
        )
        receipt.add_agent_metrics(supervisor_metrics)
        receipt.iterations = current_iterations + 1

    return {
        "next_agent": next_agent,
        "iterations": current_iterations + 1,
        "receipt": receipt,
    }


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function — tells LangGraph where to route after supervisor.

    LangGraph calls this function and uses the returned string to look up
    the next node in the graph's edge map.

    Returns: the name of the next node, or END sentinel.
    """
    return state.get("next_agent", "FINISH")
