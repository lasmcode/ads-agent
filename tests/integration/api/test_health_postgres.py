# tests/integration/api/test_health_postgres.py
"""Integration test: /health with real PostgreSQL via docker-compose."""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_with_real_postgres(postgres_client) -> None:
    response = await postgres_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["postgres"]["ok"] is True
    assert data["status"] in ("healthy", "degraded")
