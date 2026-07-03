# tests/unit/test_evaluation_fire_and_forget.py
"""Unit tests for fire-and-forget evaluation — must not block or break the pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.memory import MemorySaver
import pytest

from ads_agent.agents.supervisor.graph import run_pipeline
from ads_agent.application.services.evaluation_runner import _run_evaluation, schedule_evaluation
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.core.settings import get_settings


@pytest.fixture
def enable_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADS_EVAL_ENABLED", "true")
    monkeypatch.setenv("ADS_EVAL_SAMPLE_RATE", "1.0")
    get_settings.cache_clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_evaluation_swallows_ragas_timeout(enable_eval: None) -> None:
    """RAGAS timeout must not propagate — quality_score stays None."""
    from ads_agent.core.entities.decision_report import DecisionReport

    report = DecisionReport(
        request_id="req-1",
        query="pgvector vs Qdrant?",
        recommendation="Use pgvector.",
        summary="Summary text.",
    )
    state = {
        "final_report": report,
        "retrieved_contexts": ["chunk about pgvector"],
    }

    with patch(
        "ads_agent.application.services.evaluation_runner.evaluate_report",
        AsyncMock(side_effect=TimeoutError()),
    ):
        await _run_evaluation(state, "trace-123")

    assert report.quality_score is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_evaluation_swallows_ragas_api_error(enable_eval: None) -> None:
    """RAGAS API errors must not propagate."""
    from ads_agent.core.entities.decision_report import DecisionReport

    report = DecisionReport(
        request_id="req-2",
        query="LangGraph vs PydanticAI?",
        recommendation="Use LangGraph.",
        summary="Summary text.",
    )
    state = {
        "final_report": report,
        "retrieved_contexts": [],
    }

    with patch(
        "ads_agent.application.services.evaluation_runner.evaluate_report",
        AsyncMock(side_effect=RuntimeError("RAGAS API unavailable")),
    ):
        await _run_evaluation(state, "trace-456")

    assert report.quality_score is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pipeline_returns_before_background_eval_completes(
    enable_eval: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline must return immediately; evaluation runs in background."""
    eval_started = asyncio.Event()
    eval_release = asyncio.Event()

    async def _slow_eval(*args, **kwargs):
        eval_started.set()
        await eval_release.wait()
        return {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_precision": 0.8,
            "quality_score": 0.86,
        }

    monkeypatch.setattr(
        "ads_agent.application.services.evaluation_runner.evaluate_report",
        AsyncMock(side_effect=_slow_eval),
    )

    request = DecisionRequest(query="Should I use pgvector or Qdrant?")
    final_state, receipt = await run_pipeline(request, checkpointer=MemorySaver())

    assert final_state.get("final_report") is not None
    assert receipt.completed_at is not None
    await asyncio.wait_for(eval_started.wait(), timeout=2.0)
    assert final_state["final_report"].quality_score is None

    eval_release.set()
    await asyncio.sleep(0.05)
    assert final_state["final_report"].quality_score == pytest.approx(0.86)


@pytest.mark.unit
def test_schedule_evaluation_respects_sample_rate_zero(
    enable_eval: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sample rate 0 skips scheduling entirely."""
    monkeypatch.setenv("ADS_EVAL_SAMPLE_RATE", "0.0")
    get_settings.cache_clear()
    with patch("ads_agent.application.services.evaluation_runner.asyncio.create_task") as mock_task:
        schedule_evaluation({"final_report": object()}, "trace-1")
    mock_task.assert_not_called()
