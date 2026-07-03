# tests/unit/api/test_decisions.py
"""Unit tests for decision API endpoints (mocked LLM, MemorySaver)."""

from __future__ import annotations

import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_decision_returns_report(client) -> None:
    response = await client.post(
        "/api/v1/decisions",
        json={
            "query": "Should I use pgvector or Qdrant for vector search?",
            "context": "Already running PostgreSQL in production",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"]
    assert data["report"] is not None
    assert data["report"]["recommendation"]
    assert data["receipt"]["agents_run"] >= 1
    assert "X-Request-ID" in response.headers


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_decision_after_create(client) -> None:
    create_resp = await client.post(
        "/api/v1/decisions",
        json={"query": "Should I use Kafka or RabbitMQ for event streaming?"},
    )
    request_id = create_resp.json()["request_id"]

    get_resp = await client.get(f"/api/v1/decisions/{request_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["request_id"] == request_id
    assert data["report"] is not None
    assert data["receipt"]["duration_s"] >= 0
    create_duration = create_resp.json()["receipt"]["duration_s"]
    assert data["receipt"]["duration_s"] == create_duration


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_receipt_after_create(client) -> None:
    create_resp = await client.post(
        "/api/v1/decisions",
        json={"query": "Should I use REST or GraphQL for our public API?"},
    )
    request_id = create_resp.json()["request_id"]

    receipt_resp = await client.get(f"/api/v1/decisions/{request_id}/receipt")
    assert receipt_resp.status_code == 200
    data = receipt_resp.json()
    assert data["request_id"] == request_id
    assert isinstance(data["agents"], list)
    assert data["total_tokens"] >= 0
    assert data["completed_at"] is not None
    assert data["total_duration_seconds"] is not None
    assert data["total_duration_seconds"] >= 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_decision_not_found(client) -> None:
    response = await client.get("/api/v1/decisions/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert "request_id" in response.json()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_decision_validation_error(client) -> None:
    response = await client.post(
        "/api/v1/decisions",
        json={"query": "short"},
    )
    assert response.status_code == 422
