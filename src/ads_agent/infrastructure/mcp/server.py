# src/ads_agent/infrastructure/mcp/server.py
"""
FastMCP server exposing ADS Agent research tools.

Transport: stdio (default) for local development.
Switch to streamable-http for Phase 8 deployment via ADS_MCP_TRANSPORT.
"""

from __future__ import annotations

import os
from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field
import structlog

from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.mcp.docs import search_tech_docs_impl
from ads_agent.infrastructure.mcp.extract import fetch_url_impl
from ads_agent.infrastructure.mcp.logging_config import configure_stdio_logging
from ads_agent.infrastructure.mcp.search import web_search_impl

log = structlog.get_logger(__name__)

mcp = FastMCP("ads-agent-tools")


@mcp.tool()
async def web_search(
    query: Annotated[
        str,
        Field(
            description="Natural-language search query for the open web",
            examples=["LangGraph checkpointing with PostgreSQL 2026"],
            min_length=1,
            max_length=500,
        ),
    ],
    max_results: Annotated[
        int,
        Field(
            description="Maximum number of search results to return (1-10)",
            examples=[5],
            ge=1,
            le=10,
        ),
    ] = 5,
) -> str:
    """
    Search the open web for current information relevant to a technical decision.

    Uses Tavily Search API. Requires TAVILY_API_KEY environment variable.

    Response format:
        Markdown with numbered result blocks. Each block contains Title, URL, and Snippet.
        On failure returns a single line starting with ``Error:`` describing the problem.
    """
    try:
        return await web_search_impl(query, max_results=max_results)
    except Exception as exc:
        log.exception("web_search_tool_error", error=str(exc))
        return f"Error: Web search failed unexpectedly: {exc}"


@mcp.tool()
async def fetch_url(
    url: Annotated[
        str,
        Field(
            description="Public HTTP or HTTPS URL to fetch and extract text from",
            examples=["https://fastapi.tiangolo.com/features/"],
            min_length=8,
            max_length=2048,
        ),
    ],
) -> str:
    """
    Fetch a public web page and extract its main readable text content.

    URLs pointing to localhost, private networks, or non-http(s) schemes are rejected.

    Response format:
        ``# <page title>`` followed by ``Source: <url>`` and the extracted body text.
        Long content is truncated with a ``[truncated]`` marker.
        On failure returns a single line starting with ``Error:``.
    """
    try:
        return await fetch_url_impl(url)
    except Exception as exc:
        log.exception("fetch_url_tool_error", error=str(exc))
        return f"Error: URL fetch failed unexpectedly: {exc}"


@mcp.tool()
async def search_tech_docs(
    query: Annotated[
        str,
        Field(
            description="Search query scoped to official technical documentation",
            examples=["async checkpoint saver postgres"],
            min_length=1,
            max_length=500,
        ),
    ],
    source: Annotated[
        Literal["langgraph", "mcp", "postgres", "fastapi"],
        Field(
            description=("Documentation source to search: langgraph, mcp, postgres, or fastapi"),
            examples=["langgraph"],
        ),
    ],
) -> str:
    """
    Search official technical documentation for a specific technology stack.

    Results are restricted to official doc domains only (not general web search).

    Response format:
        Header with source name and allowed domains, then numbered markdown result blocks
        (Title, URL, Snippet) matching web_search format.
        On failure returns a single line starting with ``Error:``.
    """
    try:
        return await search_tech_docs_impl(query, source=source)
    except Exception as exc:
        log.exception("search_tech_docs_tool_error", source=source, error=str(exc))
        return f"Error: Technical documentation search failed unexpectedly: {exc}"


def main() -> None:
    """Run the MCP server with transport from settings."""
    settings = get_settings()

    if settings.mcp_transport == "streamable-http":
        log.info(
            "mcp_server_starting",
            transport=settings.mcp_transport,
            host=settings.mcp_http_host,
            port=settings.mcp_http_port,
        )
        mcp.run(
            transport="streamable-http",
            host=settings.mcp_http_host,
            port=settings.mcp_http_port,
        )
    else:
        # stdio: stdout is the MCP wire protocol — silence banner and logging.
        os.environ.setdefault("FASTMCP_SHOW_SERVER_BANNER", "false")
        os.environ.setdefault("FASTMCP_LOG_ENABLED", "false")
        os.environ.setdefault("FASTMCP_CHECK_FOR_UPDATES", "off")
        configure_stdio_logging()
        mcp.run(show_banner=False, log_level="ERROR")


if __name__ == "__main__":
    main()
