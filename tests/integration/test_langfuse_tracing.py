# tests/integration/test_langfuse_tracing.py
"""Integration test: verify traces appear in Langfuse via the public API."""

from __future__ import annotations

import asyncio
import base64
import os

import httpx
from langgraph.checkpoint.memory import MemorySaver
import pytest

from ads_agent.agents.research.nodes import ResearchAgentResult
from ads_agent.agents.supervisor.graph import run_pipeline
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.infrastructure.observability.tracer import flush_traces
from tests.unit.conftest import _mock_complete

_SUFFICIENT_RESEARCH_OUTPUT = (
    "Research findings for session storage:\n\n"
    "Key findings:\n"
    "- Redis supports rich data structures, persistence options, and pub/sub for session invalidation\n"
    "- Memcached is optimized for pure key-value caching with lower memory overhead per item\n"
    "- Redis Cluster enables horizontal scaling; Memcached clients handle sharding explicitly\n"
    "- Both support TTL-based session expiry; Redis adds optional durability for session recovery\n\n"
    "Sources:\n"
    "- https://example.com/redis-sessions\n"
    "- https://example.com/memcached-sessions"
)


def _langfuse_configured() -> bool:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    return bool(public_key and secret_key and not public_key.startswith("pk-lf-test"))


def _langfuse_host() -> str:
    return os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")


def _auth_header() -> str:
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return f"Basic {token}"


async def _fetch_trace(trace_id: str) -> httpx.Response:
    host = _langfuse_host()
    url = f"{host}/api/public/traces/{trace_id}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        return await client.get(url, headers={"Authorization": _auth_header()})


async def _wait_for_trace(trace_id: str, *, attempts: int = 15, delay_s: float = 2.0) -> dict:
    """Poll Langfuse until the trace is ingested and queryable."""
    last_response: httpx.Response | None = None
    for _ in range(attempts):
        last_response = await _fetch_trace(trace_id)
        if last_response.status_code == 200:
            return last_response.json()
        await asyncio.sleep(delay_s)

    status = last_response.status_code if last_response else "no response"
    body = last_response.text if last_response else ""
    msg = f"Trace {trace_id} not found after polling (last status={status}): {body}"
    raise AssertionError(msg)


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.skipif(not _langfuse_configured(), reason="LANGFUSE_PUBLIC_KEY/SECRET_KEY not set")
async def test_trace_appears_in_langfuse_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    End-to-end: run the pipeline and confirm the trace is queryable via Langfuse API.

    Requires real Langfuse Cloud credentials in .env.
    LLM calls are mocked so this test does not need GEMINI_API_KEY.
    """

    async def _stub_research(query: str, tools=None) -> ResearchAgentResult:
        return ResearchAgentResult(
            output=_SUFFICIENT_RESEARCH_OUTPUT,
            input_tokens=50,
            output_tokens=30,
            source_urls=[
                "https://example.com/redis-sessions",
                "https://example.com/memcached-sessions",
            ],
        )

    monkeypatch.setattr(
        "ads_agent.agents.research.nodes.run_research_agent",
        _stub_research,
    )
    monkeypatch.setattr(
        "ads_agent.infrastructure.llm.client.complete",
        _mock_complete,
    )
    for target in (
        "ads_agent.agents.supervisor.nodes.complete",
        "ads_agent.agents.analysis.nodes.complete",
        "ads_agent.agents.writer.nodes.complete",
    ):
        monkeypatch.setattr(target, _mock_complete)

    request = DecisionRequest(
        query="Should I use Redis or Memcached for session storage?",
    )
    _, receipt = await run_pipeline(request, checkpointer=MemorySaver())
    flush_traces()

    assert receipt.trace_id, "Expected a Langfuse trace_id on the receipt"
    assert receipt.circuit_breaker_triggered is False

    payload = await _wait_for_trace(receipt.trace_id)

    observations = payload.get("observations") or []
    observation_names = {obs.get("name") for obs in observations}

    assert "supervisor" in observation_names
    assert "research" in observation_names
    assert "analysis" in observation_names
    assert "writer" in observation_names

    generation_names = {obs.get("name") for obs in observations if obs.get("type") == "GENERATION"}
    assert "analysis-llm" in generation_names
    assert "writer-llm" in generation_names
