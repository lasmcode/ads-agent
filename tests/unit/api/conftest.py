# tests/unit/api/conftest.py
"""Fixtures for FastAPI unit tests — no live Postgres required."""

from __future__ import annotations

from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver
import pytest

from ads_agent.api.dependencies import get_checkpointer
from ads_agent.api.main import create_app


@pytest.fixture
async def memory_saver() -> MemorySaver:
    return MemorySaver()


@pytest.fixture
async def app(monkeypatch: pytest.MonkeyPatch, memory_saver: MemorySaver):
    monkeypatch.setattr(
        "ads_agent.api.main.get_postgres_checkpointer",
        AsyncMock(return_value=memory_saver),
    )
    monkeypatch.setattr("ads_agent.api.main.close_checkpointer_pool", AsyncMock())
    monkeypatch.setattr("ads_agent.api.main.close_vector_store_pool", AsyncMock())

    application = create_app()

    async def override_checkpointer() -> MemorySaver:
        return memory_saver

    application.dependency_overrides[get_checkpointer] = override_checkpointer
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app) -> AsyncClient:
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
