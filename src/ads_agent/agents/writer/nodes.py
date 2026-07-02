# src/ads_agent/agents/writer/nodes.py
"""
Writer Agent node — produces the final structured DecisionReport.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
import structlog

from ads_agent.agents.common import safe_node
from ads_agent.agents.writer.prompts import WRITER_SYSTEM_PROMPT, WRITER_USER_TEMPLATE
from ads_agent.core.entities.decision_report import DecisionReport
from ads_agent.core.entities.execution_receipt import AgentMetrics, AgentStatus
from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.llm.client import LLMCompletionResult, complete
from ads_agent.infrastructure.llm.schemas import AnalysisOutput, WriterDraft
from ads_agent.infrastructure.observability.tracer import (
    agent_span,
    llm_generation,
    update_generation,
)

if TYPE_CHECKING:
    from ads_agent.agents.state import AgentState

log = structlog.get_logger(__name__)


async def run_writer_agent(
    query: str,
    research_output: str,
    analysis: AnalysisOutput,
    *,
    receipt=None,
) -> tuple[WriterDraft, LLMCompletionResult]:
    """Run the writer LLM and return narrative draft fields."""
    settings = get_settings()
    user_content = WRITER_USER_TEMPLATE.format(
        query=query,
        research_output=research_output,
        trade_offs_json=analysis.model_dump_json(indent=2),
    )
    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    model = settings.llm_worker_model
    with llm_generation("writer-llm", model, messages) as generation:
        result = await complete(
            messages,
            model,
            response_model=WriterDraft,
            receipt=receipt,
            agent_name="writer",
        )
        output = (
            result.parsed.model_dump()
            if isinstance(result.parsed, WriterDraft)
            else result.raw_content
        )
        update_generation(
            generation,
            output=output,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            model=model,
        )

    if result.parsed is None or not isinstance(result.parsed, WriterDraft):
        msg = "Writer LLM did not return valid structured output"
        raise ValueError(msg)

    return result.parsed, result


@safe_node("writer")
async def writer_node(state: AgentState) -> dict:
    """Writer Agent: produces the final structured Decision Report."""
    with agent_span("writer"):
        log.info("writer_node_started", request_id=state["request"].id)

        started_at = datetime.now(UTC)
        research_output = state.get("research_output")
        analysis_raw = state.get("analysis_output")

        if not research_output:
            msg = "writer_node requires research_output"
            raise ValueError(msg)
        if not analysis_raw:
            msg = "writer_node requires analysis_output"
            raise ValueError(msg)

        analysis = AnalysisOutput.model_validate_json(analysis_raw)
        receipt = state.get("receipt")

        draft, llm_result = await run_writer_agent(
            state["request"].query,
            research_output,
            analysis,
            receipt=receipt,
        )

        sources = list(receipt.source_urls) if receipt else []

        report = DecisionReport(
            request_id=state["request"].id,
            query=state["request"].query,
            recommendation=draft.recommendation,
            recommendation_strength=draft.recommendation_strength,
            summary=draft.summary,
            trade_offs=analysis.trade_offs,
            key_considerations=draft.key_considerations,
            when_to_choose_alternative=draft.when_to_choose_alternative,
            sources=sources,
        )

        completed_at = datetime.now(UTC)

        metrics = AgentMetrics(
            agent_name="writer",
            status=AgentStatus.COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens,
        )

        if receipt:
            receipt.add_agent_metrics(metrics)

        log.info(
            "writer_node_completed",
            duration_s=metrics.duration_seconds,
            sources=len(sources),
            tokens=metrics.total_tokens,
        )

        return {
            "final_report": report,
            "messages": [AIMessage(content=report.recommendation, name="writer")],
            "receipt": receipt,
        }
