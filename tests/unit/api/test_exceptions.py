# tests/unit/api/test_exceptions.py
"""Unit tests for API exception handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ads_agent.api.exceptions import DecisionNotFoundError
from ads_agent.core.settings import get_settings


@pytest.mark.unit
@pytest.mark.asyncio
async def test_not_found_returns_404(client) -> None:
    with patch(
        "ads_agent.api.v1.router.decision_service.get_decision",
        AsyncMock(side_effect=DecisionNotFoundError("missing-id")),
    ):
        response = await client.get("/api/v1/decisions/missing-id")
    assert response.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_internal_error_includes_debug_in_development(client, monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    get_settings.cache_clear()

    with patch(
        "ads_agent.api.v1.router.decision_service.create_decision",
        AsyncMock(side_effect=RuntimeError("boom")),
    ):
        response = await client.post(
            "/api/v1/decisions",
            json={"query": "Should I use Redis or Memcached for session storage?"},
        )

    assert response.status_code == 500
    data = response.json()
    assert "debug" in data
    get_settings.cache_clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_internal_error_hides_debug_in_production(client, monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()

    with patch(
        "ads_agent.api.v1.router.decision_service.create_decision",
        AsyncMock(side_effect=RuntimeError("boom")),
    ):
        response = await client.post(
            "/api/v1/decisions",
            json={"query": "Should I use Redis or Memcached for session storage?"},
        )

    assert response.status_code == 500
    data = response.json()
    assert "debug" not in data
    assert "traceback" not in data
    get_settings.cache_clear()
