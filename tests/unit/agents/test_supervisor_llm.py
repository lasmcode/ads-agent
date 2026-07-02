# tests/unit/agents/test_supervisor_llm.py
"""Unit tests for hybrid supervisor LLM routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage
import pytest

from ads_agent.agents.state import MAX_ITERATIONS, AgentState
from ads_agent.agents.supervisor.nodes import (
    deterministic_route,
    is_ambiguous_state,
    llm_route,
    supervisor_node,
    validate_llm_route,
)
from ads_agent.core.entities.decision_report import TradeOff
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.core.entities.execution_receipt import ExecutionReceipt
from ads_agent.infrastructure.llm.client import LLMCompletionResult
from ads_agent.infrastructure.llm.schemas import AnalysisOutput, SupervisorDecision


@pytest.fixture
def sample_request() -> DecisionRequest:
    return DecisionRequest(query="pgvector vs Qdrant?")


@pytest.fixture
def base_state(sample_request: DecisionRequest) -> AgentState:
    return AgentState(
        request=sample_request,
        messages=[HumanMessage(content=sample_request.query)],
        next_agent="",
        research_output=None,
        analysis_output=None,
        final_report=None,
        receipt=ExecutionReceipt(request_id=sample_request.id),
        iterations=0,
        error=None,
    )


@pytest.mark.unit
class TestIsAmbiguousState:
    def test_clear_missing_research_is_not_ambiguous(self, base_state: AgentState) -> None:
        assert is_ambiguous_state(base_state) is False

    def test_short_research_is_ambiguous(self, base_state: AgentState) -> None:
        base_state["research_output"] = "Too short."
        assert is_ambiguous_state(base_state) is True

    def test_mock_marker_in_research_is_ambiguous(self, base_state: AgentState) -> None:
        base_state["research_output"] = "[MOCK] " + ("x" * 200)
        assert is_ambiguous_state(base_state) is True

    def test_valid_analysis_json_is_not_ambiguous(self, base_state: AgentState) -> None:
        base_state["research_output"] = "x" * 200
        analysis = AnalysisOutput(
            trade_offs=[
                TradeOff(dimension="A", option_a="a", option_b="b"),
                TradeOff(dimension="B", option_a="a", option_b="b"),
                TradeOff(dimension="C", option_a="a", option_b="b"),
            ],
        )
        base_state["analysis_output"] = analysis.model_dump_json()
        assert is_ambiguous_state(base_state) is False

    def test_invalid_analysis_json_is_ambiguous(self, base_state: AgentState) -> None:
        base_state["research_output"] = "x" * 200
        base_state["analysis_output"] = "not-json"
        assert is_ambiguous_state(base_state) is True


@pytest.mark.unit
class TestValidateLlmRoute:
    def test_invalid_agent_falls_back(self, base_state: AgentState) -> None:
        base_state["research_output"] = "x" * 200
        result = validate_llm_route(base_state, "invalid_agent")
        assert result == deterministic_route(base_state)

    def test_writer_without_analysis_falls_back(self, base_state: AgentState) -> None:
        base_state["research_output"] = "x" * 200
        result = validate_llm_route(base_state, "writer")
        assert result == "analysis"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_supervisor_falls_back_when_llm_fails(base_state: AgentState) -> None:
    base_state["research_output"] = "[MOCK] insufficient findings " + ("x" * 200)

    with patch(
        "ads_agent.agents.supervisor.nodes.llm_route",
        new_callable=AsyncMock,
        return_value=(None, None),
    ):
        result = await supervisor_node(base_state)

    assert result["next_agent"] == deterministic_route(base_state)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_circuit_breaker_does_not_call_llm(base_state: AgentState) -> None:
    base_state["iterations"] = MAX_ITERATIONS
    base_state["research_output"] = "[MOCK] " + ("x" * 200)

    with patch(
        "ads_agent.agents.supervisor.nodes.llm_route",
        new_callable=AsyncMock,
    ) as mock_llm:
        result = await supervisor_node(base_state)

    mock_llm.assert_not_called()
    assert result["next_agent"] == "FINISH"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_route_returns_choice_on_success(base_state: AgentState) -> None:
    llm_result = LLMCompletionResult(
        parsed=SupervisorDecision(next_agent="research"),
        raw_content='{"next_agent":"research"}',
        input_tokens=50,
        output_tokens=10,
        estimated_cost_usd=0.001,
        model="mock",
    )

    with patch(
        "ads_agent.agents.supervisor.nodes.complete",
        new_callable=AsyncMock,
        return_value=llm_result,
    ):
        choice, result = await llm_route(base_state)

    assert choice == "research"
    assert result is llm_result
