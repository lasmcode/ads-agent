# src/ads_agent/infrastructure/mcp/docs.py
"""Official technical documentation search (domain-restricted)."""

from __future__ import annotations

import structlog

from ads_agent.core.settings import get_settings
from ads_agent.core.tech_docs import TechDocSource, tech_doc_domains
from ads_agent.infrastructure.mcp.search import web_search_impl

log = structlog.get_logger(__name__)


async def search_tech_docs_impl(query: str, source: TechDocSource) -> str:
    """
    Search official technical documentation for a given source.

    Returns formatted markdown or an error string (never raises).
    """
    settings = get_settings()
    domains = tech_doc_domains(source)
    header = f"# Technical documentation search — {source}\n**Domains:** {', '.join(domains)}"

    log.info(
        "search_tech_docs_started",
        source=source,
        query_preview=query[:80],
        domains=domains,
    )

    result = await web_search_impl(
        query,
        max_results=settings.tech_docs_max_results,
        include_domains=domains,
        header=header,
    )

    if result.startswith("Error:"):
        log.warning("search_tech_docs_failed", source=source, error=result)
    else:
        log.info("search_tech_docs_completed", source=source)

    return result
