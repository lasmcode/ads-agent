# tests/unit/application/test_evaluation_service.py
"""Unit tests for evaluation_service quality score computation."""

from __future__ import annotations

import pytest

from ads_agent.application.services.evaluation_service import (
    ANSWER_RELEVANCY_WEIGHT,
    CONTEXT_PRECISION_WEIGHT,
    FAITHFULNESS_WEIGHT,
    compute_quality_score,
)


@pytest.mark.unit
class TestComputeQualityScore:
    def test_full_weighted_average(self) -> None:
        """0.9, 0.8, 0.7 → 0.815 with default weights."""
        result = compute_quality_score(0.9, 0.8, 0.7)
        expected = (
            0.9 * FAITHFULNESS_WEIGHT
            + 0.8 * ANSWER_RELEVANCY_WEIGHT
            + 0.7 * CONTEXT_PRECISION_WEIGHT
        )
        assert result == pytest.approx(expected)
        assert result == pytest.approx(0.815)

    def test_renormalizes_without_context_precision(self) -> None:
        """Missing context_precision reweights faithfulness and answer_relevancy."""
        result = compute_quality_score(0.9, 0.8, None)
        expected = (0.9 * FAITHFULNESS_WEIGHT + 0.8 * ANSWER_RELEVANCY_WEIGHT) / (
            FAITHFULNESS_WEIGHT + ANSWER_RELEVANCY_WEIGHT
        )
        assert result == pytest.approx(expected)

    def test_returns_none_when_all_missing(self) -> None:
        assert compute_quality_score(None, None, None) is None

    def test_single_metric_available(self) -> None:
        assert compute_quality_score(0.75, None, None) == pytest.approx(0.75)
