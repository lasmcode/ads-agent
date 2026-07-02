# src/ads_agent/infrastructure/mcp/http.py
"""Shared httpx client factory for MCP HTTP I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from ads_agent.core.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def get_http_timeout() -> httpx.Timeout:
    """Return configured timeout for MCP HTTP requests."""
    settings = get_settings()
    return httpx.Timeout(settings.http_timeout)


def create_async_client(**kwargs: object) -> httpx.AsyncClient:
    """Create an httpx AsyncClient with project default timeout."""
    timeout = kwargs.pop("timeout", get_http_timeout())
    return httpx.AsyncClient(timeout=timeout, **kwargs)  # type: ignore[arg-type]


async def get_async_client() -> AsyncIterator[httpx.AsyncClient]:
    """Async context manager yielding a configured httpx client."""
    async with create_async_client() as client:
        yield client
