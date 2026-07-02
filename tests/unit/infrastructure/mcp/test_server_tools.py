# tests/unit/infrastructure/mcp/test_server_tools.py
"""Unit tests for MCP tool backends (httpx mocks, no real network)."""

from __future__ import annotations

import httpx
import pytest

from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.mcp.extract import fetch_url_impl
from ads_agent.infrastructure.mcp.search import TAVILY_SEARCH_URL, web_search_impl


@pytest.fixture
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.unit
@pytest.mark.asyncio
class TestWebSearchImpl:
    async def test_happy_path_formats_results(
        self,
        httpx_mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        httpx_mock.add_response(
            url=TAVILY_SEARCH_URL,
            method="POST",
            json={
                "results": [
                    {
                        "title": "LangGraph Docs",
                        "url": "https://langchain.com/langgraph",
                        "content": "Graph-based agent framework.",
                    }
                ]
            },
        )

        result = await web_search_impl("LangGraph checkpointing", max_results=3)

        assert "LangGraph Docs" in result
        assert "https://langchain.com/langgraph" in result
        assert "Graph-based agent framework" in result
        assert not result.startswith("Error:")

    async def test_missing_api_key_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = await web_search_impl("test query")
        assert result.startswith("Error:")
        assert "TAVILY_API_KEY" in result

    async def test_http_error_returns_readable_message(
        self,
        httpx_mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        httpx_mock.add_response(url=TAVILY_SEARCH_URL, method="POST", status_code=503)

        result = await web_search_impl("test query")
        assert result.startswith("Error:")
        assert "503" in result

    async def test_timeout_returns_readable_message(
        self,
        httpx_mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        httpx_mock.add_exception(
            httpx.TimeoutException("timed out"),
            url=TAVILY_SEARCH_URL,
            method="POST",
        )

        result = await web_search_impl("test query")
        assert result.startswith("Error:")
        assert "timed out" in result.lower()


@pytest.mark.unit
@pytest.mark.asyncio
class TestFetchUrlImpl:
    async def test_rejects_localhost_without_network(self) -> None:
        result = await fetch_url_impl("http://127.0.0.1/secret")
        assert result.startswith("Error:")
        assert "security" in result.lower()

    async def test_happy_path_extracts_text(
        self,
        httpx_mock,
    ) -> None:
        html = """
        <html><head><title>Test Page</title></head>
        <body><article><p>Hello from the test page content.</p></article></body>
        </html>
        """
        httpx_mock.add_response(
            url="https://example.com/article",
            method="GET",
            text=html,
        )

        result = await fetch_url_impl("https://example.com/article")

        assert result.startswith("# ")
        assert "Source: https://example.com/article" in result
        assert "Hello from the test page content" in result
        assert not result.startswith("Error:")

    async def test_http_error_returns_readable_message(
        self,
        httpx_mock,
    ) -> None:
        httpx_mock.add_response(
            url="https://example.com/missing",
            method="GET",
            status_code=404,
        )

        result = await fetch_url_impl("https://example.com/missing")
        assert result.startswith("Error:")
        assert "404" in result

    async def test_raw_text_content_is_not_passed_through_trafilatura(
        self,
        httpx_mock,
    ) -> None:
        markdown = "# README\n\nThis is raw markdown content from GitHub."
        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/org/repo/main/README.md",
            method="GET",
            text=markdown,
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )

        result = await fetch_url_impl(
            "https://raw.githubusercontent.com/org/repo/main/README.md",
        )

        assert "This is raw markdown content from GitHub." in result
        assert "# README" in result
        assert not result.startswith("Error:")

    async def test_github_blob_url_is_rewritten_to_raw(
        self,
        httpx_mock,
    ) -> None:
        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/org/repo/main/docs/guide.md",
            method="GET",
            text="Guide content from raw URL.",
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )

        result = await fetch_url_impl("https://github.com/org/repo/blob/main/docs/guide.md")

        assert "Guide content from raw URL." in result
        assert "Source: https://raw.githubusercontent.com/org/repo/main/docs/guide.md" in result

    async def test_truncates_long_content_with_omitted_marker(
        self,
        httpx_mock,
        monkeypatch: pytest.MonkeyPatch,
        clear_settings_cache: None,
    ) -> None:
        monkeypatch.setenv("ADS_FETCH_URL_MAX_CHARS", "500")
        long_body = ("Paragraph one.\n\n" * 10) + ("Paragraph two.\n\n" * 40)
        httpx_mock.add_response(
            url="https://example.com/long",
            method="GET",
            text=long_body,
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )

        result = await fetch_url_impl("https://example.com/long")

        assert "[... omitted" in result
        assert "characters ...]" in result
        assert "Paragraph one." in result
        assert "Paragraph two." in result

    async def test_rejects_responses_larger_than_byte_limit(
        self,
        httpx_mock,
        monkeypatch: pytest.MonkeyPatch,
        clear_settings_cache: None,
    ) -> None:
        monkeypatch.setenv("ADS_FETCH_URL_MAX_RESPONSE_BYTES", "10000")
        httpx_mock.add_response(
            url="https://example.com/huge",
            method="GET",
            text="x" * 20_000,
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )

        result = await fetch_url_impl("https://example.com/huge")

        assert result.startswith("Error:")
        assert "maximum size" in result.lower()

    async def test_html_fallback_extracts_markdown_body(
        self,
        httpx_mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        html = """
        <html><head><title>Ignored</title></head>
        <body>
          <nav>Navigation noise</nav>
          <article class="markdown-body"><p>Fallback markdown body text.</p></article>
        </body>
        </html>
        """
        httpx_mock.add_response(
            url="https://example.com/readme",
            method="GET",
            text=html,
            headers={"Content-Type": "text/html; charset=utf-8"},
        )
        monkeypatch.setattr(
            "ads_agent.infrastructure.mcp.extract.trafilatura.extract",
            lambda *args, **kwargs: "",
        )

        result = await fetch_url_impl("https://example.com/readme")

        assert "Fallback markdown body text." in result
        assert "Navigation noise" not in result
