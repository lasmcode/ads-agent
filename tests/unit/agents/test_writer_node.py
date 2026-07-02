# tests/unit/agents/test_writer_node.py
"""Unit tests for the writer agent node."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage
import pytest

from ads_agent.agents.state import AgentState
from ads_agent.agents.writer.nodes import writer_node
from ads_agent.core.entities.decision_report import RecommendationStrength, TradeOff
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.core.entities.execution_receipt import ExecutionReceipt
from ads_agent.infrastructure.llm.client import LLMCompletionResult
from ads_agent.infrastructure.llm.schemas import AnalysisOutput, WriterDraft


@pytest.fixture
def writer_state() -> AgentState:
    request = DecisionRequest(query="pgvector vs Qdrant?")
    analysis = AnalysisOutput(
        trade_offs=[
            TradeOff(dimension="Performance", option_a="pgvector ok", option_b="Qdrant fast"),
            TradeOff(dimension="Ops", option_a="pgvector simple", option_b="Qdrant extra service"),
            TradeOff(dimension="Cost", option_a="pgvector low", option_b="Qdrant moderate"),
        ],
    )
    receipt = ExecutionReceipt(request_id=request.id)
    receipt.add_consulted_sources(
        [
            "https://example.com/pgvector-overview",
            "https://example.com/qdrant-comparison",
        ],
    )
    return AgentState(
        request=request,
        messages=[HumanMessage(content=request.query)],
        next_agent="",
        research_output="Research " * 50,
        analysis_output=analysis.model_dump_json(),
        final_report=None,
        receipt=receipt,
        iterations=0,
        error=None,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_writer_node_uses_receipt_sources(writer_state: AgentState) -> None:
    draft = WriterDraft(
        recommendation="Use pgvector.",
        recommendation_strength=RecommendationStrength.MODERATE,
        summary="Summary text.",
        key_considerations=["Consider ops"],
    )
    llm_result = LLMCompletionResult(
        parsed=draft,
        raw_content=draft.model_dump_json(),
        input_tokens=400,
        output_tokens=200,
        estimated_cost_usd=0.002,
        model="mock",
    )

    with patch(
        "ads_agent.agents.writer.nodes.run_writer_agent",
        new_callable=AsyncMock,
        return_value=(draft, llm_result),
    ):
        result = await writer_node(writer_state)

    report = result["final_report"]
    assert report is not None
    assert report.sources == writer_state["receipt"].source_urls
    assert len(report.trade_offs) == 3
    assert report.recommendation == "Use pgvector."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_writer_node_records_tokens(writer_state: AgentState) -> None:
    draft = WriterDraft(
        recommendation="Use pgvector.",
        recommendation_strength=RecommendationStrength.MODERATE,
        summary="Summary text.",
    )
    llm_result = LLMCompletionResult(
        parsed=draft,
        raw_content=draft.model_dump_json(),
        input_tokens=400,
        output_tokens=200,
        estimated_cost_usd=0.002,
        model="mock",
    )

    with patch(
        "ads_agent.agents.writer.nodes.run_writer_agent",
        new_callable=AsyncMock,
        return_value=(draft, llm_result),
    ):
        result = await writer_node(writer_state)

    metrics = result["receipt"].agents[-1]
    assert metrics.agent_name == "writer"
    assert metrics.input_tokens == 400
