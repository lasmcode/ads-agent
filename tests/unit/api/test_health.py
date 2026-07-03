# tests/unit/api/test_health.py
"""Unit tests for /health endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ads_agent.api.v1.schemas import DependencyHealthDTO, HealthResponse


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_all_ok(client) -> None:
    healthy = HealthResponse(
        status="healthy",
        postgres=DependencyHealthDTO(ok=True, configured=True),
        langfuse=DependencyHealthDTO(ok=True, configured=False, detail="not configured"),
    )
    with patch(
        "ads_agent.api.v1.router.run_health_checks",
        AsyncMock(return_value=healthy),
    ):
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_postgres_down_returns_503(client) -> None:
    unhealthy = HealthResponse(
        status="unhealthy",
        postgres=DependencyHealthDTO(ok=False, configured=True, detail="connection refused"),
        langfuse=DependencyHealthDTO(ok=True, configured=False),
    )
    with patch(
        "ads_agent.api.v1.router.run_health_checks",
        AsyncMock(return_value=unhealthy),
    ):
        response = await client.get("/health")

    assert response.status_code == 503
    assert response.json()["postgres"]["ok"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_degraded_langfuse_still_200(client) -> None:
    degraded = HealthResponse(
        status="degraded",
        postgres=DependencyHealthDTO(ok=True, configured=True),
        langfuse=DependencyHealthDTO(ok=False, configured=True, detail="timeout"),
    )
    with patch(
        "ads_agent.api.v1.router.run_health_checks",
        AsyncMock(return_value=degraded),
    ):
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
