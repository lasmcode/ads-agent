# tests/integration/api/conftest.py
"""Shared fixtures for API integration tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient
import pytest
from tests.unit.conftest import _mock_complete

from ads_agent.api.dependencies import get_checkpointer
from ads_agent.api.main import create_app
from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.checkpointer import get_postgres_checkpointer


@pytest.fixture(autouse=True)
def mock_api_integration_llm(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Mock LLM/research for integration API tests — skip for real E2E."""
    if request.node.fspath.basename == "test_api_e2e_real.py":
        return

    monkeypatch.setenv("ADS_EVAL_ENABLED", "false")
    get_settings.cache_clear()
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from unittest.mock import AsyncMock

    from ads_agent.agents.research.nodes import ResearchAgentResult

    async def _stub_run_research_agent(query: str, tools=None) -> ResearchAgentResult:
        return ResearchAgentResult(
            output=(
                f"Research for: {query}\n\nKey findings:\n- Option A is fast\n- Option B is simple\n\n"
                "Sources:\n- https://example.com/a\n- https://example.com/b"
            ),
            input_tokens=100,
            output_tokens=50,
            source_urls=["https://example.com/a", "https://example.com/b"],
            retrieved_contexts=[],
        )

    mock = AsyncMock(side_effect=_mock_complete)
    targets = [
        "ads_agent.infrastructure.llm.client.complete",
        "ads_agent.agents.supervisor.nodes.complete",
        "ads_agent.agents.analysis.nodes.complete",
        "ads_agent.agents.writer.nodes.complete",
    ]
    for target in targets:
        monkeypatch.setattr(target, mock)
    monkeypatch.setattr(
        "ads_agent.agents.research.nodes.run_research_agent",
        AsyncMock(side_effect=_stub_run_research_agent),
    )
    monkeypatch.setattr(
        "ads_agent.infrastructure.observability.tracer.get_langfuse_client",
        lambda: None,
    )


@pytest.fixture
async def postgres_app(monkeypatch: pytest.MonkeyPatch):
    """FastAPI app wired to real Postgres checkpointer (requires docker-up)."""
    try:
        checkpointer = await get_postgres_checkpointer()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable — run `make docker-up` first ({exc})")

    monkeypatch.setattr(
        "ads_agent.api.main.get_postgres_checkpointer",
        AsyncMock(return_value=checkpointer),
    )

    application = create_app()

    async def override_checkpointer():
        return checkpointer

    application.dependency_overrides[get_checkpointer] = override_checkpointer
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def postgres_client(postgres_app) -> AsyncClient:
    transport = ASGITransport(app=postgres_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=180.0) as ac:
        yield ac
