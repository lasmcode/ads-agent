# src/ads_agent/agents/supervisor/nodes.py
"""
Supervisor node — the central router of the multi-agent pipeline.

Responsibilities:
  1. Read current pipeline state
  2. Decide which agent to call next (deterministic rules + LLM for ambiguous cases)
  3. Enforce the circuit breaker (Python is the final authority)
  4. Update the receipt with iteration count and LLM usage
"""

from __future__ import annotations

from datetime import UTC, datetime
import json

import structlog

from ads_agent.agents.state import MAX_ITERATIONS, AgentState
from ads_agent.agents.supervisor.prompts import (
    SUPERVISOR_ROUTING_TEMPLATE,
    SUPERVISOR_SYSTEM_PROMPT,
)
from ads_agent.core.entities.execution_receipt import AgentMetrics, AgentStatus
from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.llm.client import LLMCompletionResult, complete
from ads_agent.infrastructure.llm.schemas import AnalysisOutput, SupervisorDecision
from ads_agent.infrastructure.observability.tracer import (
    agent_span,
    llm_generation,
    update_generation,
)

log = structlog.get_logger(__name__)

_VALID_AGENTS = frozenset({"research", "analysis", "writer", "FINISH"})
_MIN_OUTPUT_CHARS = 150
_INSUFFICIENCY_MARKERS = (
    "insufficient",
    "unable to",
    "could not find",
    "no reliable",
    "[stub]",
    "[mock]",
)
_PREVIEW_MAX_CHARS = 500


def _truncate(text: str | None, max_chars: int = _PREVIEW_MAX_CHARS) -> str:
    if not text:
        return "(none)"
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _has_insufficiency_markers(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INSUFFICIENCY_MARKERS)


def _is_insufficient_text(text: str | None) -> bool:
    if not text:
        return True
    if len(text.strip()) < _MIN_OUTPUT_CHARS:
        return True
    return _has_insufficiency_markers(text)


def _parse_analysis_trade_off_count(analysis_output: str | None) -> int | None:
    """Return trade-off count if analysis_output is valid JSON, else None."""
    if not analysis_output:
        return None
    try:
        parsed = AnalysisOutput.model_validate_json(analysis_output)
    except json.JSONDecodeError, ValueError:
        return None
    return len(parsed.trade_offs)


def is_ambiguous_state(state: AgentState) -> bool:
    """
    True when existing outputs appear insufficient and LLM routing may help.

    Clear pipeline states (missing fields, complete outputs) are NOT ambiguous.
    """
    research = state.get("research_output")
    analysis = state.get("analysis_output")

    if research and _is_insufficient_text(research):
        return True

    if analysis:
        trade_off_count = _parse_analysis_trade_off_count(analysis)
        if trade_off_count is None or trade_off_count < 3:
            return True
        if _is_insufficient_text(analysis):
            return True

    return False


def deterministic_route(state: AgentState) -> str:
    """
    Pure deterministic routing rules — the circuit breaker fallback.

    Order matters: each rule is checked sequentially.
    """
    if state.get("error"):
        return "FINISH"
    if not state.get("research_output"):
        return "research"
    if not state.get("analysis_output"):
        return "analysis"
    if not state.get("final_report"):
        return "writer"
    return "FINISH"


def validate_llm_route(state: AgentState, llm_choice: str) -> str:
    """
    Validate an LLM routing suggestion against hard pipeline rules.

    Falls back to deterministic_route when the LLM proposes something invalid.
    """
    if llm_choice not in _VALID_AGENTS:
        log.warning("supervisor_llm_invalid_agent", choice=llm_choice)
        return deterministic_route(state)

    if state.get("error") and llm_choice != "FINISH":
        return "FINISH"

    if llm_choice == "research" and state.get("final_report"):
        return deterministic_route(state)

    if llm_choice == "analysis" and not state.get("research_output"):
        return deterministic_route(state)

    if llm_choice == "writer":
        if not state.get("research_output") or not state.get("analysis_output"):
            return deterministic_route(state)
        trade_off_count = _parse_analysis_trade_off_count(state.get("analysis_output"))
        if trade_off_count is None or trade_off_count < 3:
            return deterministic_route(state)

    if (
        llm_choice == "FINISH"
        and not state.get("error")
        and not state.get("final_report")
        and not is_ambiguous_state(state)
    ):
        det = deterministic_route(state)
        if det != "FINISH":
            return det

    return llm_choice


async def llm_route(state: AgentState) -> tuple[str | None, LLMCompletionResult | None]:
    """
    Consult the supervisor LLM for an ambiguous routing decision.

    Returns (choice, result). choice is None on failure.
    """
    settings = get_settings()
    user_content = SUPERVISOR_ROUTING_TEMPLATE.format(
        has_research=bool(state.get("research_output")),
        has_analysis=bool(state.get("analysis_output")),
        has_report=bool(state.get("final_report")),
        last_error=state.get("error") or "none",
        iterations=state.get("iterations", 0),
        research_preview=_truncate(state.get("research_output")),
        analysis_preview=_truncate(state.get("analysis_output")),
    )

    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    model = settings.llm_supervisor_model
    try:
        with llm_generation("supervisor-routing", model, messages) as generation:
            result = await complete(
                messages,
                model,
                response_model=SupervisorDecision,
                receipt=state.get("receipt"),
                agent_name="supervisor",
            )
            output = (
                result.parsed.model_dump()
                if isinstance(result.parsed, SupervisorDecision)
                else result.raw_content
            )
            update_generation(
                generation,
                output=output,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                model=model,
            )
    except Exception as exc:
        log.warning("supervisor_llm_failed", error=str(exc))
        return None, None

    if result.parsed is None or not isinstance(result.parsed, SupervisorDecision):
        log.warning("supervisor_llm_unparseable")
        return None, result

    return result.parsed.next_agent, result


async def supervisor_node(state: AgentState) -> dict:
    """
    Hybrid supervisor router.

    Deterministic rules handle clear states; LLM assists on ambiguous ones.
    Circuit breaker and error rules are always enforced in Python first.
    """
    current_iterations = state.get("iterations", 0)

    with agent_span("supervisor", iteration=current_iterations):
        started_at = datetime.now(UTC)
        receipt = state.get("receipt")
        input_tokens = 0
        output_tokens = 0
        used_llm = False

        log.info(
            "supervisor_routing",
            iteration=current_iterations,
            has_research=bool(state.get("research_output")),
            has_analysis=bool(state.get("analysis_output")),
            has_report=bool(state.get("final_report")),
            ambiguous=is_ambiguous_state(state),
        )

        # --- Circuit breaker (Python authority — no LLM) ---
        if current_iterations >= MAX_ITERATIONS:
            log.warning(
                "circuit_breaker_triggered",
                iterations=current_iterations,
                max=MAX_ITERATIONS,
            )
            if receipt:
                receipt.circuit_breaker_triggered = True
            return _supervisor_return(
                next_agent="FINISH",
                iterations=current_iterations + 1,
                receipt=receipt,
                started_at=started_at,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # --- Routing decision ---
        if state.get("error"):
            log.error("supervisor_routing_on_error", error=state["error"])
            next_agent = "FINISH"
        elif not is_ambiguous_state(state):
            next_agent = deterministic_route(state)
        else:
            used_llm = True
            llm_choice, llm_result = await llm_route(state)
            if llm_result is not None:
                input_tokens = llm_result.input_tokens
                output_tokens = llm_result.output_tokens
            if llm_choice is None:
                next_agent = deterministic_route(state)
            else:
                next_agent = validate_llm_route(state, llm_choice)

        log.info("supervisor_decision", next_agent=next_agent, used_llm=used_llm)

        return _supervisor_return(
            next_agent=next_agent,
            iterations=current_iterations + 1,
            receipt=receipt,
            started_at=started_at,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def _supervisor_return(
    *,
    next_agent: str,
    iterations: int,
    receipt,
    started_at: datetime,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    completed_at = datetime.now(UTC)

    if receipt:
        supervisor_metrics = AgentMetrics(
            agent_name="supervisor",
            status=AgentStatus.COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        receipt.add_agent_metrics(supervisor_metrics)
        receipt.iterations = iterations
        if next_agent == "FINISH":
            receipt.mark_completed()

    return {
        "next_agent": next_agent,
        "iterations": iterations,
        "receipt": receipt,
    }


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function — tells LangGraph where to route after supervisor.

    Returns: the name of the next node, or END sentinel.
    """
    return state.get("next_agent", "FINISH")
