# tests/unit/conftest.py
"""
Shared fixtures for unit tests.

Phase 2 research uses LLM + MCP; Phase 3 adds a Postgres-backed checkpointer
and knowledge store. Graph routing and CLI unit tests mock all of it to stay
fast, keyless, and DB-free — only tests/integration exercises real services.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from langgraph.checkpoint.memory import MemorySaver
import pytest

from ads_agent.agents.research.nodes import ResearchAgentResult


@pytest.fixture(autouse=True)
def mock_research_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace run_research_agent with a stub for pipeline/CLI unit tests."""

    async def _stub_run_research_agent(query: str, tools=None) -> ResearchAgentResult:
        output = (
            f"[MOCK] Research findings for: '{query}'\n\n"
            "Key findings:\n"
            "- pgvector integrates natively with PostgreSQL\n"
            "- Qdrant offers dedicated vector search performance\n\n"
            "Sources:\n"
            "- https://example.com/pgvector-overview\n"
            "- https://example.com/qdrant-comparison"
        )
        return ResearchAgentResult(
            output=output,
            input_tokens=100,
            output_tokens=50,
            source_urls=[
                "https://example.com/pgvector-overview",
                "https://example.com/qdrant-comparison",
            ],
        )

    monkeypatch.setattr(
        "ads_agent.agents.research.nodes.run_research_agent",
        AsyncMock(side_effect=_stub_run_research_agent),
    )


@pytest.fixture(autouse=True)
def mock_postgres_checkpointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace the CLI's Postgres checkpointer with an in-memory one.

    Unit tests invoke `ads_agent.cli.main()` directly; without this, every
    CLI test would require a live PostgreSQL connection just to obtain a
    checkpointer, even though checkpoint persistence itself isn't under
    test here (see tests/integration for that).
    """
    monkeypatch.setattr(
        "ads_agent.cli.get_postgres_checkpointer",
        AsyncMock(return_value=MemorySaver()),
    )
    monkeypatch.setattr("ads_agent.cli.close_checkpointer_pool", AsyncMock())
    monkeypatch.setattr("ads_agent.cli.close_vector_store_pool", AsyncMock())
