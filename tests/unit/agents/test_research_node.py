# tests/unit/agents/test_research_node.py
"""Unit tests for the research agent node."""

from __future__ import annotations

from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from ads_agent.agents.research.nodes import (
    ResearchAgentResult,
    _extract_token_usage,
    _extract_urls_from_messages,
    research_node,
    run_research_agent,
)
from ads_agent.agents.state import AgentState
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.core.entities.execution_receipt import ExecutionReceipt


@pytest.fixture
def research_state() -> AgentState:
    request = DecisionRequest(query="pgvector vs Qdrant for RAG?")
    return AgentState(
        request=request,
        messages=[HumanMessage(content=request.query)],
        next_agent="",
        research_output=None,
        analysis_output=None,
        final_report=None,
        receipt=ExecutionReceipt(request_id=request.id),
        iterations=0,
        error=None,
    )


@pytest.mark.unit
class TestResearchHelpers:
    def test_extract_urls_from_messages(self) -> None:
        messages = [
            AIMessage(content="See https://example.com/a and https://example.com/b for details."),
        ]
        urls = _extract_urls_from_messages(messages)
        assert "https://example.com/a" in urls
        assert "https://example.com/b" in urls

    def test_extract_token_usage_from_usage_metadata(self) -> None:
        messages = [
            AIMessage(
                content="done",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            ),
        ]
        inp, out = _extract_token_usage(messages)
        assert inp == 10
        assert out == 5


@pytest.mark.unit
@pytest.mark.asyncio
class TestResearchNode:
    async def test_research_node_contract_and_receipt(
        self,
        research_state: AgentState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test research node directly — bypasses autouse mock via explicit patch."""

        async def _mock_agent(query: str, tools=None) -> ResearchAgentResult:
            return ResearchAgentResult(
                output=f"Findings for {query}",
                input_tokens=42,
                output_tokens=21,
                source_urls=["https://docs.example.com/rag"],
            )

        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.run_research_agent",
            AsyncMock(side_effect=_mock_agent),
        )

        result = await research_node(research_state)

        assert "research_output" in result
        assert result["research_output"] == f"Findings for {research_state['request'].query}"
        assert len(result["messages"]) == 1
        assert result["messages"][0].name == "research"

        receipt = result["receipt"]
        assert receipt is not None
        assert receipt.sources_consulted == 1
        assert "https://docs.example.com/rag" in receipt.source_urls

        research_metrics = [m for m in receipt.agents if m.agent_name == "research"]
        assert len(research_metrics) == 1
        assert research_metrics[0].input_tokens == 42
        assert research_metrics[0].output_tokens == 21

    async def test_run_research_agent_invokes_react_agent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                AIMessage(
                    content="Research complete.",
                    usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                ),
            ],
        }

        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.create_react_agent",
            lambda model, tools, prompt: mock_agent,
        )
        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.get_mcp_tools",
            AsyncMock(return_value=[]),
        )

        result = await run_research_agent("test query", tools=[])

        assert result.output == "Research complete."
        assert result.input_tokens == 5
        assert result.output_tokens == 3
        mock_agent.ainvoke.assert_awaited_once()
