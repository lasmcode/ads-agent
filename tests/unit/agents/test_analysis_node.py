# tests/unit/agents/test_analysis_node.py
"""Unit tests for the analysis agent node."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage
import pytest

from ads_agent.agents.analysis.nodes import analysis_node, run_analysis_agent
from ads_agent.agents.state import AgentState
from ads_agent.core.entities.decision_report import TradeOff
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.core.entities.execution_receipt import ExecutionReceipt
from ads_agent.infrastructure.llm.client import LLMCompletionResult
from ads_agent.infrastructure.llm.schemas import AnalysisOutput


@pytest.fixture
def analysis_state() -> AgentState:
    request = DecisionRequest(query="pgvector vs Qdrant?")
    return AgentState(
        request=request,
        messages=[HumanMessage(content=request.query)],
        next_agent="",
        research_output="Research " * 50,
        analysis_output=None,
        final_report=None,
        receipt=ExecutionReceipt(request_id=request.id),
        iterations=0,
        error=None,
    )


def _sample_analysis() -> AnalysisOutput:
    return AnalysisOutput(
        trade_offs=[
            TradeOff(dimension="Performance", option_a="pgvector ok", option_b="Qdrant fast"),
            TradeOff(dimension="Ops", option_a="pgvector simple", option_b="Qdrant extra service"),
            TradeOff(dimension="Cost", option_a="pgvector low", option_b="Qdrant moderate"),
        ],
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_analysis_agent_returns_structured_output() -> None:
    parsed = _sample_analysis()
    llm_result = LLMCompletionResult(
        parsed=parsed,
        raw_content=parsed.model_dump_json(),
        input_tokens=200,
        output_tokens=100,
        estimated_cost_usd=0.001,
        model="mock",
    )

    with patch(
        "ads_agent.agents.analysis.nodes.complete",
        new_callable=AsyncMock,
        return_value=llm_result,
    ):
        output, result = await run_analysis_agent("query", "research text")

    assert len(output.trade_offs) == 3
    assert result.input_tokens == 200


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analysis_node_records_tokens(analysis_state: AgentState) -> None:
    parsed = _sample_analysis()
    llm_result = LLMCompletionResult(
        parsed=parsed,
        raw_content=parsed.model_dump_json(),
        input_tokens=300,
        output_tokens=150,
        estimated_cost_usd=0.002,
        model="mock",
    )

    with patch(
        "ads_agent.agents.analysis.nodes.run_analysis_agent",
        new_callable=AsyncMock,
        return_value=(parsed, llm_result),
    ):
        result = await analysis_node(analysis_state)

    assert result["analysis_output"] is not None
    metrics = result["receipt"].agents[-1]
    assert metrics.agent_name == "analysis"
    assert metrics.input_tokens == 300
    assert metrics.output_tokens == 150
