# src/ads_agent/infrastructure/mcp/search.py
"""Tavily web search backend (httpx, mock-friendly)."""

from __future__ import annotations

import os
from typing import Any

import httpx
import structlog

from ads_agent.infrastructure.mcp.http import create_async_client

log = structlog.get_logger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _format_search_results(results: list[dict[str, Any]], *, header: str | None = None) -> str:
    """Format Tavily results as numbered markdown blocks."""
    lines: list[str] = []
    if header:
        lines.append(header)
        lines.append("")

    if not results:
        lines.append("No results found.")
        return "\n".join(lines)

    for idx, item in enumerate(results, start=1):
        title = item.get("title") or "Untitled"
        url = item.get("url") or ""
        snippet = item.get("content") or item.get("snippet") or ""
        lines.append(f"## {idx}. {title}")
        lines.append(f"**URL:** {url}")
        lines.append(f"**Snippet:** {snippet}")
        lines.append("")

    return "\n".join(lines).rstrip()


async def web_search_impl(
    query: str,
    *,
    max_results: int = 5,
    include_domains: list[str] | None = None,
    header: str | None = None,
) -> str:
    """
    Execute a Tavily search and return formatted markdown.

    Returns a readable error string on failure (never raises to callers).
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        log.warning("web_search_no_api_key")
        return "Error: TAVILY_API_KEY is not configured. Set it to enable web search."

    clamped = max(1, min(max_results, 10))
    payload: dict[str, Any] = {
        "query": query,
        "max_results": clamped,
        "search_depth": "basic",
    }
    if include_domains:
        payload["include_domains"] = include_domains

    log.info("web_search_started", query_preview=query[:80], max_results=clamped)

    try:
        async with create_async_client() as client:
            response = await client.post(
                TAVILY_SEARCH_URL,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        log.warning("web_search_failed", reason="timeout")
        return "Error: Web search timed out. Try again with a shorter query."
    except httpx.HTTPStatusError as exc:
        log.warning("web_search_failed", reason="http_error", status=exc.response.status_code)
        return f"Error: Web search failed with HTTP {exc.response.status_code}."
    except httpx.HTTPError as exc:
        log.warning("web_search_failed", reason="network", error=str(exc))
        return f"Error: Web search network error: {exc}"
    except Exception as exc:
        log.warning("web_search_failed", reason="unexpected", error=str(exc))
        return f"Error: Web search failed: {exc}"

    results = data.get("results") or []
    log.info("web_search_completed", result_count=len(results))
    return _format_search_results(results, header=header)
