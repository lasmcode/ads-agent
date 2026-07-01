# src/ads_agent/infrastructure/mcp/docs.py
"""Official technical documentation search (domain-restricted)."""

from __future__ import annotations

from typing import Literal

import structlog

from ads_agent.infrastructure.mcp.search import web_search_impl

log = structlog.get_logger(__name__)

TechDocSource = Literal["langgraph", "mcp", "postgres", "fastapi", "oracle"]

_SOURCE_DOMAINS: dict[TechDocSource, list[str]] = {
    "langgraph": ["langchain.com", "docs.langchain.com"],
    "mcp": ["modelcontextprotocol.io"],
    "postgres": ["postgresql.org"],
    "fastapi": ["fastapi.tiangolo.com"],
    "oracle": ["oracle.com", "docs.oracle.com"],
}


async def search_tech_docs_impl(query: str, source: TechDocSource) -> str:
    """
    Search official technical documentation for a given source.

    Returns formatted markdown or an error string (never raises).
    """
    domains = _SOURCE_DOMAINS[source]
    header = f"# Technical documentation search — {source}\n**Domains:** {', '.join(domains)}"

    log.info(
        "search_tech_docs_started",
        source=source,
        query_preview=query[:80],
        domains=domains,
    )

    result = await web_search_impl(
        query,
        max_results=5,
        include_domains=domains,
        header=header,
    )

    if result.startswith("Error:"):
        log.warning("search_tech_docs_failed", source=source, error=result)
    else:
        log.info("search_tech_docs_completed", source=source)

    return result
