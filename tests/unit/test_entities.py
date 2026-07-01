# tests/unit/test_entities.py
"""
Unit tests for domain entities.
Zero external dependencies — pure Python validation.
"""

from __future__ import annotations

import pytest

from ads_agent.core.entities.decision_report import (
    DecisionReport,
    RecommendationStrength,
    TradeOff,
)
from ads_agent.core.entities.decision_request import DecisionComplexity, DecisionRequest
from ads_agent.core.entities.execution_receipt import (
    AgentMetrics,
    AgentStatus,
    ExecutionReceipt,
)

# ---------------------------------------------------------------------------
# DecisionRequest tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDecisionRequest:
    def test_creates_with_required_fields(self) -> None:
        req = DecisionRequest(query="Should I use pgvector or Qdrant for RAG?")
        assert req.query == "Should I use pgvector or Qdrant for RAG?"
        assert req.id is not None
        assert len(req.id) == 36  # UUID4 format

    def test_auto_generates_unique_ids(self) -> None:
        req1 = DecisionRequest(query="Should I use pgvector or Qdrant for RAG?")
        req2 = DecisionRequest(query="Should I use pgvector or Qdrant for RAG?")
        assert req1.id != req2.id

    def test_default_complexity_is_moderate(self) -> None:
        req = DecisionRequest(query="Should I use pgvector or Qdrant for RAG?")
        assert req.complexity == DecisionComplexity.MODERATE

    def test_rejects_query_too_short(self) -> None:
        with pytest.raises(ValueError):
            DecisionRequest(query="short")

    def test_rejects_query_too_long(self) -> None:
        with pytest.raises(ValueError):
            DecisionRequest(query="x" * 2001)

    def test_is_immutable(self) -> None:
        """Frozen model must reject field assignment after creation."""
        req = DecisionRequest(query="Should I use pgvector or Qdrant for RAG?")
        from pydantic import ValidationError

        with pytest.raises((ValidationError, TypeError)):
            req.query = "modified"  # type: ignore[misc]

    def test_accepts_optional_context(self) -> None:
        req = DecisionRequest(
            query="Should I use pgvector or Qdrant for RAG?",
            context="We have 10M vectors and need sub-10ms p99 latency",
        )
        assert req.context is not None


# ---------------------------------------------------------------------------
# ExecutionReceipt tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecutionReceipt:
    def test_creates_with_request_id(self) -> None:
        receipt = ExecutionReceipt(request_id="test-123")
        assert receipt.request_id == "test-123"
        assert receipt.total_tokens == 0
        assert not receipt.circuit_breaker_triggered

    def test_accumulates_agent_metrics(self) -> None:
        receipt = ExecutionReceipt(request_id="test-123")
        m1 = AgentMetrics(
            agent_name="research",
            status=AgentStatus.COMPLETED,
            input_tokens=100,
            output_tokens=200,
        )
        m2 = AgentMetrics(
            agent_name="analysis",
            status=AgentStatus.COMPLETED,
            input_tokens=300,
            output_tokens=150,
        )
        receipt.add_agent_metrics(m1)
        receipt.add_agent_metrics(m2)

        assert len(receipt.agents) == 2
        assert receipt.total_input_tokens == 400
        assert receipt.total_output_tokens == 350
        assert receipt.total_tokens == 750

    def test_mark_completed_sets_timestamp(self) -> None:
        receipt = ExecutionReceipt(request_id="test-123")
        assert receipt.completed_at is None
        receipt.mark_completed()
        assert receipt.completed_at is not None

    def test_duration_computed_correctly(self) -> None:
        receipt = ExecutionReceipt(request_id="test-123")
        receipt.mark_completed()
        duration = receipt.total_duration_seconds
        assert duration is not None
        assert duration >= 0

    def test_summary_returns_dict(self) -> None:
        receipt = ExecutionReceipt(request_id="test-123")
        receipt.mark_completed()
        summary = receipt.to_summary()
        assert "request_id" in summary
        assert "duration_s" in summary
        assert "total_tokens" in summary


# ---------------------------------------------------------------------------
# AgentMetrics tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentMetrics:
    def test_total_tokens_computed(self) -> None:
        m = AgentMetrics(
            agent_name="research",
            status=AgentStatus.COMPLETED,
            input_tokens=500,
            output_tokens=300,
        )
        assert m.total_tokens == 800

    def test_duration_none_when_not_completed(self) -> None:
        m = AgentMetrics(agent_name="research", status=AgentStatus.RUNNING)
        assert m.duration_seconds is None


# ---------------------------------------------------------------------------
# DecisionReport tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDecisionReport:
    def test_creates_valid_report(self) -> None:
        report = DecisionReport(
            request_id="req-123",
            query="pgvector vs Qdrant?",
            recommendation="Use pgvector for simplicity at your scale.",
            summary="Both are solid choices. pgvector wins on ops simplicity.",
        )
        assert report.recommendation_strength == RecommendationStrength.MODERATE

    def test_accepts_trade_offs(self) -> None:
        trade_off = TradeOff(
            dimension="Operational complexity",
            option_a="pgvector: managed via standard Postgres tooling",
            option_b="Qdrant: separate service to maintain",
            winner="pgvector",
        )
        report = DecisionReport(
            request_id="req-123",
            query="pgvector vs Qdrant?",
            recommendation="Use pgvector.",
            summary="Analysis summary.",
            trade_offs=[trade_off],
        )
        assert len(report.trade_offs) == 1
        assert report.trade_offs[0].winner == "pgvector"

    def test_quality_score_bounded(self) -> None:
        with pytest.raises(ValueError):
            DecisionReport(
                request_id="req-123",
                query="pgvector vs Qdrant?",
                recommendation="Use pgvector.",
                summary="Analysis.",
                quality_score=1.5,  # Must be 0.0-1.0
            )
