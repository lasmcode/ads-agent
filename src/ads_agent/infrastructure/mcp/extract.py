# src/ads_agent/infrastructure/mcp/extract.py
"""Fetch URL content and extract readable text."""

from __future__ import annotations

import httpx
import structlog
import trafilatura

from ads_agent.infrastructure.mcp.http import create_async_client
from ads_agent.infrastructure.mcp.url_validator import URLValidationError, validate_url

log = structlog.get_logger(__name__)

MAX_CONTENT_CHARS = 4000


def _validate_redirect_url(url: str) -> None:
    """Re-validate redirect targets to prevent SSRF via open redirects."""
    validate_url(url)


async def fetch_url_impl(url: str) -> str:
    """
    Fetch a URL and return extracted plain text as markdown.

    Response format on success:
        # <title>
        Source: <url>

        <body text>

    Returns a string starting with ``Error:`` on failure (never raises).
    """
    log.info("fetch_url_started", url_preview=url[:120])

    try:
        validated = validate_url(url)
    except URLValidationError as exc:
        log.warning("fetch_url_rejected", reason=str(exc))
        return f"Error: URL rejected for security reasons: {exc}"

    try:
        async with create_async_client(follow_redirects=True) as client:
            # httpx event hooks for redirect validation
            response = await client.get(
                validated.url,
                headers={"User-Agent": "ads-agent/0.1 (MCP fetch_url tool)"},
            )
            # Re-check final URL after redirects
            _validate_redirect_url(str(response.url))
            response.raise_for_status()
            html = response.text
    except URLValidationError as exc:
        log.warning("fetch_url_redirect_rejected", reason=str(exc))
        return f"Error: Redirect target rejected for security reasons: {exc}"
    except httpx.TimeoutException:
        log.warning("fetch_url_failed", reason="timeout", url=validated.url)
        return "Error: Request timed out while fetching URL."
    except httpx.HTTPStatusError as exc:
        log.warning(
            "fetch_url_failed",
            reason="http_error",
            status=exc.response.status_code,
            url=validated.url,
        )
        return f"Error: HTTP {exc.response.status_code} when fetching URL."
    except httpx.HTTPError as exc:
        log.warning("fetch_url_failed", reason="network", error=str(exc), url=validated.url)
        return f"Error: Network error while fetching URL: {exc}"
    except Exception as exc:
        log.warning("fetch_url_failed", reason="unexpected", error=str(exc))
        return f"Error: Failed to fetch URL: {exc}"

    title = validated.hostname
    metadata = trafilatura.extract_metadata(html)
    if metadata and metadata.title:
        title = metadata.title

    text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
    if not text.strip():
        text = "(No readable text content could be extracted from this page.)"

    truncated = False
    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS]
        truncated = True

    lines = [f"# {title}", f"Source: {validated.url}", "", text]
    if truncated:
        lines.append("")
        lines.append("[truncated]")

    log.info("fetch_url_completed", url=validated.url, chars=len(text), truncated=truncated)
    return "\n".join(lines)
