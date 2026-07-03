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
        retrieved_contexts=[],
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
                retrieved_contexts=["chunk content about pgvector"],
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
            "ads_agent.agents.research.nodes.create_agent",
            lambda model, tools, system_prompt=None, **kwargs: mock_agent,
        )
        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.get_mcp_tools",
            AsyncMock(return_value=[]),
        )
        # Isolate from the knowledge base (Phase 3) — its own behavior is
        # covered by test_retrieve_rag_context_* below and the RRF/chunker
        # unit tests; this test only cares about the ReAct agent contract.
        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.hybrid_search",
            AsyncMock(return_value=[]),
        )

        result = await run_research_agent("test query", tools=[])

        assert result.output == "Research complete."
        assert result.input_tokens == 5
        assert result.output_tokens == 3
        mock_agent.ainvoke.assert_awaited_once()

    async def test_run_research_agent_prepends_high_confidence_rag_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """High-confidence knowledge-base chunks are prepended to the agent's query."""
        from ads_agent.core.entities.chunk import Chunk
        from ads_agent.core.settings import get_settings

        threshold = get_settings().rag_score_threshold
        chunk = Chunk(
            id="chunk-1",
            source_url="https://docs.example.com/langgraph/persistence",
            title="Checkpointer vs. store",
            content="Checkpointers persist a thread's graph state.",
            score=threshold + 0.01,
        )
        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.hybrid_search",
            AsyncMock(return_value=[chunk]),
        )

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [AIMessage(content="Research complete.")],
        }
        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.create_agent",
            lambda model, tools, system_prompt=None, **kwargs: mock_agent,
        )

        result = await run_research_agent("test query", tools=[])

        sent_message = mock_agent.ainvoke.call_args[0][0]["messages"][0]
        assert "UNTRUSTED INTERNAL KNOWLEDGE BASE CONTEXT" in sent_message.content
        assert "Checkpointers persist a thread's graph state." in sent_message.content
        assert "test query" in sent_message.content
        assert "https://docs.example.com/langgraph/persistence" in result.source_urls

    async def test_run_research_agent_skips_low_confidence_rag_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Chunks below the confidence threshold are not injected into the query."""
        from ads_agent.core.entities.chunk import Chunk
        from ads_agent.core.settings import get_settings

        threshold = get_settings().rag_score_threshold
        chunk = Chunk(
            id="chunk-1",
            source_url="https://docs.example.com/low-confidence",
            title="Tangentially related",
            content="Not very relevant content.",
            score=threshold / 2,
        )
        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.hybrid_search",
            AsyncMock(return_value=[chunk]),
        )

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [AIMessage(content="Research complete.")],
        }
        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.create_agent",
            lambda model, tools, system_prompt=None, **kwargs: mock_agent,
        )

        result = await run_research_agent("test query", tools=[])

        sent_message = mock_agent.ainvoke.call_args[0][0]["messages"][0]
        assert sent_message.content == "test query"
        assert "https://docs.example.com/low-confidence" not in result.source_urls

    async def test_run_research_agent_tolerates_rag_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A knowledge-base failure (e.g. Postgres unreachable) must not fail research."""
        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.hybrid_search",
            AsyncMock(side_effect=ConnectionError("could not connect to server")),
        )

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [AIMessage(content="Research complete.")],
        }
        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.create_agent",
            lambda model, tools, system_prompt=None, **kwargs: mock_agent,
        )

        result = await run_research_agent("test query", tools=[])

        assert result.output == "Research complete."
        sent_message = mock_agent.ainvoke.call_args[0][0]["messages"][0]
        assert sent_message.content == "test query"
