# tests/integration/api/test_api_e2e_mocked.py
"""Integration E2E: full API with mocked LLM (no token cost)."""

from __future__ import annotations

from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver
import pytest

from ads_agent.api.dependencies import get_checkpointer
from ads_agent.api.main import create_app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_e2e_mocked_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run POST → GET report → GET receipt with mocked LLM and MemorySaver."""
    saver = MemorySaver()
    monkeypatch.setattr(
        "ads_agent.api.main.get_postgres_checkpointer",
        AsyncMock(return_value=saver),
    )
    monkeypatch.setattr("ads_agent.api.main.close_checkpointer_pool", AsyncMock())
    monkeypatch.setattr("ads_agent.api.main.close_vector_store_pool", AsyncMock())

    application = create_app()

    async def override_checkpointer() -> MemorySaver:
        return saver

    application.dependency_overrides[get_checkpointer] = override_checkpointer

    transport = ASGITransport(app=application, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=180.0) as client:
        create_resp = await client.post(
            "/api/v1/decisions",
            json={
                "query": "Should I use Redis or Memcached for session storage?",
                "context": "Team of 5, moderate traffic",
            },
        )
        assert create_resp.status_code == 200
        body = create_resp.json()
        request_id = body["request_id"]
        assert body["report"] is not None
        assert body["receipt"]["total_tokens"] >= 0

        report_resp = await client.get(f"/api/v1/decisions/{request_id}")
        assert report_resp.status_code == 200
        assert report_resp.json()["report"]["recommendation"]

        receipt_resp = await client.get(f"/api/v1/decisions/{request_id}/receipt")
        assert receipt_resp.status_code == 200
        assert receipt_resp.json()["request_id"] == request_id

    application.dependency_overrides.clear()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_e2e_mocked_with_postgres(postgres_client) -> None:
    """Full flow persisted in real Postgres checkpoints."""
    create_resp = await postgres_client.post(
        "/api/v1/decisions",
        json={"query": "Should I use Kafka or RabbitMQ for event streaming?"},
    )
    assert create_resp.status_code == 200
    request_id = create_resp.json()["request_id"]

    report_resp = await postgres_client.get(f"/api/v1/decisions/{request_id}")
    assert report_resp.status_code == 200
    assert report_resp.json()["report"] is not None
