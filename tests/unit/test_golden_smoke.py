# tests/unit/test_golden_smoke.py
"""Fast smoke tests using 2-3 golden dataset entries with mocked pipeline."""

from __future__ import annotations

from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver
import pytest
from tests.fixtures.load_golden_dataset import load_golden_subset

from ads_agent.agents.supervisor.graph import run_pipeline
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.core.settings import get_settings


@pytest.fixture
def enable_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADS_EVAL_ENABLED", "true")
    monkeypatch.setenv("ADS_EVAL_SAMPLE_RATE", "1.0")
    get_settings.cache_clear()


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry_id",
    ["pgvector-vs-qdrant", "langgraph-vs-pydanticai", "jenkins-vs-github-actions"],
)
@pytest.mark.asyncio
async def test_golden_smoke_schedules_evaluation(
    entry_id: str,
    enable_eval: None,
) -> None:
    """Golden smoke: pipeline completes and schedules evaluation for sample queries."""
    entries = load_golden_subset([entry_id])
    assert len(entries) == 1
    entry = entries[0]

    with patch(
        "ads_agent.agents.supervisor.graph.schedule_evaluation",
    ) as mock_schedule:
        request = DecisionRequest(query=entry.query)
        final_state, receipt = await run_pipeline(request, checkpointer=MemorySaver())

    assert final_state.get("final_report") is not None
    assert receipt.completed_at is not None
    mock_schedule.assert_called_once()
    call_args = mock_schedule.call_args
    assert call_args[0][0] is final_state
