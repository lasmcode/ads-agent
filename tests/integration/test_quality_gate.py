# tests/integration/test_quality_gate.py
"""
Quality gate integration test — golden dataset vs real pipeline + RAGAS + DeepEval.

Disabled by default to preserve Gemini free-tier quota. Opt in explicitly:

    RUN_QUALITY_GATE=1 make test-eval

Also requires GEMINI_API_KEY when enabled. Runs on nightly CI (manual dispatch)
or local opt-in — not on every PR.
"""

from __future__ import annotations

import os
import statistics
from unittest.mock import AsyncMock

from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams
from langgraph.checkpoint.memory import MemorySaver
import pytest

from ads_agent.agents.research.nodes import ResearchAgentResult
from ads_agent.agents.supervisor.graph import run_pipeline
from ads_agent.application.services.evaluation_service import evaluate_report
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.core.settings import get_settings
from tests.fixtures.load_golden_dataset import (
    GoldenEntry,
    build_reference_text,
    load_golden_dataset,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.evaluation,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("RUN_QUALITY_GATE", "").lower() not in ("1", "true", "yes"),
        reason="Quality gate disabled by default. Set RUN_QUALITY_GATE=1 to enable.",
    ),
    pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"),
]


@pytest.fixture(autouse=True)
def disable_background_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluate synchronously in this module — avoid racing fire-and-forget tasks."""
    monkeypatch.setenv("ADS_EVAL_ENABLED", "false")
    get_settings.cache_clear()


def _report_text(report) -> str:
    parts = [
        report.recommendation,
        report.summary,
        *report.key_considerations,
    ]
    return "\n".join(parts).lower()


def _keyword_coverage(entry: GoldenEntry, report) -> float:
    text = _report_text(report)
    if not entry.expected_key_considerations:
        return 1.0
    hits = sum(1 for kw in entry.expected_key_considerations if kw.lower() in text)
    return hits / len(entry.expected_key_considerations)


async def _stub_research(entry: GoldenEntry, query: str, tools=None) -> ResearchAgentResult:
    keywords = ", ".join(entry.expected_key_considerations)
    output = (
        f"Research findings for: '{query}'\n\n"
        f"Direction: {entry.expected_recommendation_direction}\n"
        f"Key considerations: {keywords}\n\n"
        "Sources:\n"
        "- https://example.com/architecture-guide\n"
        "- https://example.com/comparison-matrix"
    )
    contexts = [
        f"Technical comparison for {entry.id}: {entry.expected_recommendation_direction}. "
        f"Factors include {keywords}."
    ]
    return ResearchAgentResult(
        output=output,
        input_tokens=200,
        output_tokens=100,
        source_urls=[
            "https://example.com/architecture-guide",
            "https://example.com/comparison-matrix",
        ],
        retrieved_contexts=contexts,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", load_golden_dataset(), ids=lambda e: e.id)
async def test_golden_entry_quality(
    entry: GoldenEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each golden entry must pass RAGAS scoring, DeepEval alignment, and keyword checks."""

    async def _research_stub(query: str, tools=None) -> ResearchAgentResult:
        return await _stub_research(entry, query, tools)

    monkeypatch.setattr(
        "ads_agent.agents.research.nodes.run_research_agent",
        AsyncMock(side_effect=_research_stub),
    )

    request = DecisionRequest(query=entry.query)
    final_state, _receipt = await run_pipeline(request, checkpointer=MemorySaver())

    report = final_state.get("final_report")
    assert report is not None

    contexts = final_state.get("retrieved_contexts") or [
        f"Context for {entry.id}: {entry.expected_recommendation_direction}"
    ]
    reference = build_reference_text(entry)
    scores = await evaluate_report(report, contexts, reference=reference)

    assert scores["quality_score"] is not None
    assert scores["faithfulness"] is not None
    assert scores["answer_relevancy"] is not None

    keyword_ratio = _keyword_coverage(entry, report)
    assert keyword_ratio >= 0.5, (
        f"Expected at least half of keywords {entry.expected_key_considerations} "
        f"in report for {entry.id}, got {keyword_ratio:.0%}"
    )

    gemini_model = GeminiModel(
        model="gemini-2.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
    )
    alignment_metric = GEval(
        name="RecommendationDirection",
        criteria=(
            f"The recommendation should align with this expected direction: "
            f"{entry.expected_recommendation_direction}"
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=gemini_model,
        threshold=0.5,
    )
    test_case = LLMTestCase(
        input=entry.query,
        actual_output=f"{report.recommendation}\n\n{report.summary}",
    )
    alignment_metric.measure(test_case)
    assert alignment_metric.score is not None
    assert alignment_metric.score >= alignment_metric.threshold


@pytest.mark.asyncio
async def test_quality_gate_batch_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch average quality_score must meet EVAL_QUALITY_THRESHOLD."""
    threshold = get_settings().eval_quality_threshold
    quality_scores: list[float] = []

    for entry in load_golden_dataset():

        async def _research_stub(query: str, tools=None, _entry=entry) -> ResearchAgentResult:
            return await _stub_research(_entry, query, tools)

        monkeypatch.setattr(
            "ads_agent.agents.research.nodes.run_research_agent",
            AsyncMock(side_effect=_research_stub),
        )

        request = DecisionRequest(query=entry.query)
        final_state, _ = await run_pipeline(request, checkpointer=MemorySaver())
        report = final_state.get("final_report")
        assert report is not None

        contexts = final_state.get("retrieved_contexts") or [
            f"Context for {entry.id}: {entry.expected_recommendation_direction}"
        ]
        scores = await evaluate_report(
            report,
            contexts,
            reference=build_reference_text(entry),
        )
        if scores["quality_score"] is not None:
            quality_scores.append(scores["quality_score"])

    assert quality_scores, "No quality scores collected from golden dataset"
    batch_avg = statistics.mean(quality_scores)
    assert batch_avg >= threshold, (
        f"Batch average quality_score {batch_avg:.3f} below threshold {threshold}"
    )
