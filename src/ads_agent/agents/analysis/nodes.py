# src/ads_agent/agents/analysis/nodes.py
"""
Analysis Agent node — structured LLM reasoning over research output.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
import structlog

from ads_agent.agents.analysis.prompts import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_USER_TEMPLATE
from ads_agent.agents.common import safe_node
from ads_agent.core.entities.execution_receipt import AgentMetrics, AgentStatus
from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.llm.client import LLMCompletionResult, complete
from ads_agent.infrastructure.llm.schemas import AnalysisOutput
from ads_agent.infrastructure.observability.tracer import (
    agent_span,
    llm_generation,
    update_generation,
)

if TYPE_CHECKING:
    from ads_agent.agents.state import AgentState

log = structlog.get_logger(__name__)


def _format_trade_offs_summary(output: AnalysisOutput) -> str:
    lines = [f"Trade-off analysis ({len(output.trade_offs)} dimensions):"]
    for trade_off in output.trade_offs:
        winner = f" → {trade_off.winner}" if trade_off.winner else ""
        lines.append(
            f"- {trade_off.dimension}: {trade_off.option_a} vs {trade_off.option_b}{winner}"
        )
    return "\n".join(lines)


async def run_analysis_agent(
    query: str,
    research_output: str,
    *,
    receipt=None,
) -> tuple[AnalysisOutput, LLMCompletionResult]:
    """Run the analysis LLM and return structured trade-offs."""
    settings = get_settings()
    user_content = ANALYSIS_USER_TEMPLATE.format(
        query=query,
        research_output=research_output,
    )
    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    model = settings.llm_worker_model
    with llm_generation("analysis-llm", model, messages) as generation:
        result = await complete(
            messages,
            model,
            response_model=AnalysisOutput,
            receipt=receipt,
            agent_name="analysis",
        )
        output = (
            result.parsed.model_dump()
            if isinstance(result.parsed, AnalysisOutput)
            else result.raw_content
        )
        update_generation(
            generation,
            output=output,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            model=model,
        )

    if result.parsed is None or not isinstance(result.parsed, AnalysisOutput):
        msg = "Analysis LLM did not return valid structured output"
        raise ValueError(msg)

    return result.parsed, result


@safe_node("analysis")
async def analysis_node(state: AgentState) -> dict:
    """Analysis Agent: evaluates trade-offs from research output."""
    with agent_span("analysis"):
        log.info("analysis_node_started", request_id=state["request"].id)

        started_at = datetime.now(UTC)
        research_output = state.get("research_output")
        if not research_output:
            msg = "analysis_node requires research_output"
            raise ValueError(msg)

        receipt = state.get("receipt")
        analysis, llm_result = await run_analysis_agent(
            state["request"].query,
            research_output,
            receipt=receipt,
        )

        analysis_output = analysis.model_dump_json()
        summary_message = _format_trade_offs_summary(analysis)
        completed_at = datetime.now(UTC)

        metrics = AgentMetrics(
            agent_name="analysis",
            status=AgentStatus.COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens,
        )

        if receipt:
            receipt.add_agent_metrics(metrics)

        log.info(
            "analysis_node_completed",
            duration_s=metrics.duration_seconds,
            trade_offs=len(analysis.trade_offs),
            tokens=metrics.total_tokens,
        )

        return {
            "analysis_output": analysis_output,
            "messages": [AIMessage(content=summary_message, name="analysis")],
            "receipt": receipt,
        }
