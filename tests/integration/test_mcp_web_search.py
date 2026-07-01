# tests/integration/test_mcp_web_search.py
"""Integration tests for MCP web search (requires TAVILY_API_KEY)."""

from __future__ import annotations

import os

import pytest

from ads_agent.infrastructure.mcp.search import web_search_impl


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="TAVILY_API_KEY not set")
async def test_web_search_live() -> None:
    """Make a real Tavily search call when API key is configured."""
    result = await web_search_impl("LangGraph supervisor pattern", max_results=2)

    assert not result.startswith("Error:"), result
    assert "LangGraph" in result or "langgraph" in result.lower() or "http" in result
