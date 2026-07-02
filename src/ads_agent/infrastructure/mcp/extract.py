# src/ads_agent/infrastructure/mcp/extract.py
"""Fetch URL content and extract readable text."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re
from urllib.parse import urlparse

import httpx
import structlog
import trafilatura

from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.mcp.http import create_async_client
from ads_agent.infrastructure.mcp.url_validator import URLValidationError, validate_url

log = structlog.get_logger(__name__)

_RAW_TEXT_CONTENT_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "application/json",
    }
)
_RAW_TEXT_SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".json")
_HTML_FALLBACK_PATTERNS = (
    r'<article\b(?=[^>]*\bclass\s*=\s*["\']markdown-body["\'])[^>]*>(.*?)</article>',
    r'<div\b(?=[^>]*\bclass\s*=\s*["\']markdown-body["\'])[^>]*>(.*?)</div>',
    r'<div\b(?=[^>]*\bid\s*=\s*["\']readme["\'])[^>]*>(.*?)</div>',
    r"<article\b[^>]*>(.*?)</article>",
)


class ResponseTooLargeError(Exception):
    """Raised when a fetched response exceeds the configured byte limit."""


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping script/style content."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_tags = frozenset({"script", "style", "noscript"})
        self._stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._stack.append(tag.lower())

    def handle_endtag(self, tag: str) -> None:
        if self._stack and self._stack[-1] == tag.lower():
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if any(tag in self._skip_tags for tag in self._stack):
            return
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _validate_redirect_url(url: str) -> None:
    """Re-validate redirect targets to prevent SSRF via open redirects."""
    validate_url(url)


def _normalize_fetch_url(url: str) -> str:
    """Rewrite GitHub blob URLs to raw.githubusercontent.com when possible."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if hostname != "github.com":
        return url

    parts = parsed.path.strip("/").split("/")
    if len(parts) < 5 or parts[2] != "blob":
        return url

    owner, repo, ref = parts[0], parts[1], parts[3]
    path = "/".join(parts[4:])
    if not path:
        return url

    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"


def _prefer_raw_text(url: str, content_type: str) -> bool:
    """Return True when response body should be used as plain text."""
    lower_url = url.lower()
    if "raw.githubusercontent.com" in lower_url:
        return True
    if lower_url.endswith(_RAW_TEXT_SUFFIXES):
        return True
    base_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    return base_type in _RAW_TEXT_CONTENT_TYPES


def _decode_body(body: bytes) -> str:
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("latin-1", errors="replace")


def _extract_html_fallback(html: str) -> str:
    """Fallback extraction for pages where trafilatura returns nothing."""
    for pattern in _HTML_FALLBACK_PATTERNS:
        match = re.search(pattern, html, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            continue
        extractor = _TextExtractor()
        extractor.feed(match.group(1))
        text = extractor.get_text().strip()
        if len(text) > 15:
            return text

    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.get_text().strip()


def _extract_text(*, url: str, content: str, content_type: str) -> str:
    if _prefer_raw_text(url, content_type):
        return content.strip()

    text = trafilatura.extract(content, include_comments=False, include_tables=True) or ""
    if text.strip():
        return text.strip()

    fallback = _extract_html_fallback(content)
    if fallback:
        return fallback

    return "(No readable text content could be extracted from this page.)"


def _resolve_title(*, url: str, content: str, content_type: str, hostname: str) -> str:
    if _prefer_raw_text(url, content_type):
        path = urlparse(url).path.rstrip("/")
        if path:
            return path.rsplit("/", maxsplit=1)[-1]
        return hostname

    metadata = trafilatura.extract_metadata(content)
    if metadata and metadata.title:
        return metadata.title
    return hostname


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep beginning and end sections when content exceeds the char limit."""
    if len(text) <= max_chars:
        return text, False

    marker_budget = 64
    usable = max_chars - marker_budget
    if usable < 200:
        return text[:max_chars], True

    head_size = usable * 2 // 3
    tail_size = usable - head_size

    head = text[:head_size]
    last_break = head.rfind("\n\n")
    if last_break > head_size // 2:
        head = head[:last_break]

    tail = text[-tail_size:]
    first_break = tail.find("\n\n")
    if first_break != -1 and first_break < tail_size // 3:
        tail = tail[first_break + 2 :]

    omitted = len(text) - len(head) - len(tail)
    marker = f"\n\n[... omitted {omitted} characters ...]\n\n"
    return head + marker + tail, True


async def _fetch_body(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int,
) -> tuple[bytes, str, str]:
    """Download response body up to max_bytes. Returns body, final URL, content-type."""
    async with client.stream(
        "GET",
        url,
        headers={"User-Agent": "ads-agent/0.1 (MCP fetch_url tool)"},
    ) as response:
        _validate_redirect_url(str(response.url))
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                msg = f"Response exceeds maximum size of {max_bytes} bytes"
                raise ResponseTooLargeError(msg)
            chunks.append(chunk)

        return b"".join(chunks), str(response.url), content_type


@dataclass(frozen=True)
class FetchedPage:
    """Result of a safely-fetched URL: decoded body plus response metadata."""

    final_url: str
    content: str
    content_type: str
    hostname: str


async def fetch_page(url: str, *, max_bytes: int | None = None) -> FetchedPage:
    """
    Safely fetch a URL and return its decoded body plus metadata.

    Centralizes SSRF/redirect validation and size limiting so every caller
    (the `fetch_url` MCP tool, the RAG ingestion pipeline, ...) shares the
    same security boundary instead of re-implementing it.

    Raises:
        URLValidationError: the URL or a redirect target is blocked.
        ResponseTooLargeError: the response exceeds max_bytes.
        httpx.HTTPError: network failure, timeout, or non-2xx status.
    """
    settings = get_settings()
    resolved_max_bytes = (
        max_bytes if max_bytes is not None else settings.fetch_url_max_response_bytes
    )

    normalized_url = _normalize_fetch_url(url.strip())
    validated = validate_url(normalized_url)

    async with create_async_client(follow_redirects=True) as client:
        body, final_url, content_type = await _fetch_body(
            client,
            validated.url,
            max_bytes=resolved_max_bytes,
        )

    return FetchedPage(
        final_url=final_url,
        content=_decode_body(body),
        content_type=content_type,
        hostname=validated.hostname,
    )


async def fetch_url_impl(url: str) -> str:
    """
    Fetch a URL and return extracted plain text as markdown.

    Response format on success:
        # <title>
        Source: <url>

        <body text>

    Returns a string starting with ``Error:`` on failure (never raises).
    """
    settings = get_settings()
    max_chars = settings.fetch_url_max_chars

    log.info("fetch_url_started", url_preview=url[:120])

    try:
        page = await fetch_page(url)
    except URLValidationError as exc:
        log.warning("fetch_url_rejected", reason=str(exc))
        return f"Error: URL rejected for security reasons: {exc}"
    except ResponseTooLargeError as exc:
        log.warning("fetch_url_failed", reason="response_too_large", url=url)
        return f"Error: {exc}"
    except httpx.TimeoutException:
        log.warning("fetch_url_failed", reason="timeout", url=url)
        return "Error: Request timed out while fetching URL."
    except httpx.HTTPStatusError as exc:
        log.warning(
            "fetch_url_failed",
            reason="http_error",
            status=exc.response.status_code,
            url=url,
        )
        return f"Error: HTTP {exc.response.status_code} when fetching URL."
    except httpx.HTTPError as exc:
        log.warning("fetch_url_failed", reason="network", error=str(exc), url=url)
        return f"Error: Network error while fetching URL: {exc}"
    except Exception as exc:
        log.warning("fetch_url_failed", reason="unexpected", error=str(exc))
        return f"Error: Failed to fetch URL: {exc}"

    final_url, content, content_type = page.final_url, page.content, page.content_type
    text = _extract_text(url=final_url, content=content, content_type=content_type)
    title = _resolve_title(
        url=final_url,
        content=content,
        content_type=content_type,
        hostname=page.hostname,
    )
    text, truncated = _truncate_text(text, max_chars)

    lines = [f"# {title}", f"Source: {final_url}", "", text]
    log.info(
        "fetch_url_completed",
        url=final_url,
        chars=len(text),
        truncated=truncated,
        content_type=content_type.split(";", maxsplit=1)[0],
    )
    return "\n".join(lines)
