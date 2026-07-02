# tests/unit/infrastructure/test_tracer.py
"""Unit tests for Langfuse tracer wrapper — fail-safe observability."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver
import pytest

from ads_agent.agents.supervisor.graph import run_pipeline
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.infrastructure.observability import tracer


@pytest.mark.unit
def test_capture_trace_id_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured Langfuse client returns a non-empty trace ID."""
    mock_client = MagicMock()
    mock_client.get_current_trace_id.return_value = "trace-abc-123"

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-real-key")
    monkeypatch.setattr(tracer, "get_langfuse_client", lambda: mock_client)

    assert tracer.capture_trace_id() == "trace-abc-123"


@pytest.mark.unit
def test_trace_id_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without credentials, capture_trace_id returns None without error."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert tracer.capture_trace_id() is None
    assert tracer.is_tracing_enabled() is False


@pytest.mark.unit
def test_submit_scores_calls_create_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """submit_pipeline_scores sends both quality scores to Langfuse."""
    mock_client = MagicMock()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-real-key")
    monkeypatch.setattr(tracer, "get_langfuse_client", lambda: mock_client)

    tracer.submit_pipeline_scores(
        "trace-xyz",
        has_sources=True,
        trade_offs_count=3,
    )

    assert mock_client.create_score.call_count == 2
    score_names = {call.kwargs["name"] for call in mock_client.create_score.call_args_list}
    assert score_names == {"has_sources", "trade_offs_count"}


@pytest.mark.unit
def test_submit_scores_noop_without_trace_id() -> None:
    """Scores are skipped when trace_id is missing."""
    mock_client = MagicMock()
    with patch.object(tracer, "get_langfuse_client", return_value=mock_client):
        tracer.submit_pipeline_scores(None, has_sources=True, trade_offs_count=3)
    mock_client.create_score.assert_not_called()


@pytest.mark.unit
def test_flush_traces_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """flush_traces must never raise even when the client fails."""
    mock_client = MagicMock()
    mock_client.flush.side_effect = ConnectionError("langfuse unreachable")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-real-key")
    monkeypatch.setattr(tracer, "get_langfuse_client", lambda: mock_client)

    tracer.flush_traces()  # must not raise


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pipeline_succeeds_when_langfuse_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Langfuse failures must never crash the pipeline."""
    mock_client = MagicMock()
    mock_client.get_current_trace_id.return_value = None

    @contextmanager
    def _exploding_observation(**kwargs):
        raise ConnectionError("langfuse down")

    mock_client.start_as_current_observation.side_effect = _exploding_observation
    mock_client.flush.side_effect = ConnectionError("flush failed")

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-real-key")
    monkeypatch.setattr(tracer, "get_langfuse_client", lambda: mock_client)

    request = DecisionRequest(query="Should I use Redis or Memcached for sessions?")
    final_state, receipt = await run_pipeline(request, checkpointer=MemorySaver())

    assert final_state.get("final_report") is not None
    assert receipt.completed_at is not None
    assert receipt.trace_id is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pipeline_assigns_trace_id_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_pipeline stores Langfuse trace_id on the receipt when tracing works."""
    mock_client = MagicMock()
    mock_client.get_current_trace_id.return_value = "trace-pipeline-001"

    @contextmanager
    def _noop_observation(**kwargs):
        yield MagicMock()

    mock_client.start_as_current_observation.side_effect = _noop_observation

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-real-key")
    monkeypatch.setattr(tracer, "get_langfuse_client", lambda: mock_client)

    # propagate_attributes must be a no-op context manager
    monkeypatch.setattr(
        "langfuse.propagate_attributes",
        lambda **kwargs: contextmanager(lambda: (yield))(),
        raising=False,
    )

    request = DecisionRequest(query="Should I use Redis or Memcached for sessions?")
    _, receipt = await run_pipeline(request, checkpointer=MemorySaver())

    assert receipt.trace_id == "trace-pipeline-001"
    mock_client.flush.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pipeline_trace_id_none_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without Langfuse credentials, receipt.trace_id stays None."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    request = DecisionRequest(query="Should I use Redis or Memcached for sessions?")
    _, receipt = await run_pipeline(request, checkpointer=MemorySaver())

    assert receipt.trace_id is None
    assert receipt.completed_at is not None
