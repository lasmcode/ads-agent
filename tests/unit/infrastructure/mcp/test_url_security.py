# tests/unit/infrastructure/mcp/test_url_security.py
"""SSRF validation tests for fetch_url."""

from __future__ import annotations

import pytest

from ads_agent.infrastructure.mcp.url_validator import URLValidationError, validate_url


@pytest.mark.unit
class TestURLValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/admin",
            "http://127.0.0.1:8080/",
            "http://[::1]/",
            "http://10.0.0.1/internal",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
            "ftp://example.com/file",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://user:pass@example.com/page",
        ],
    )
    def test_rejects_blocked_urls(self, url: str) -> None:
        with pytest.raises(URLValidationError):
            validate_url(url)

    def test_accepts_public_https_url(self) -> None:
        result = validate_url("https://fastapi.tiangolo.com/features/")
        assert result.scheme == "https"
        assert result.hostname == "fastapi.tiangolo.com"

    def test_rejects_empty_url(self) -> None:
        with pytest.raises(URLValidationError, match="empty"):
            validate_url("")

    def test_rejects_missing_hostname(self) -> None:
        with pytest.raises(URLValidationError, match="hostname"):
            validate_url("http:///path")
