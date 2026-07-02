# tests/unit/conftest.py
"""
Shared fixtures for unit tests.

Phase 2 research uses LLM + MCP; graph routing and CLI unit tests mock it
to stay fast and keyless.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

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
