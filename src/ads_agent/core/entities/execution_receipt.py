# src/ads_agent/core/entities/execution_receipt.py
"""
Domain entity: ExecutionReceipt
Operational metadata generated for every agent execution.
This is the AgentOps/FinOps layer — every run produces a receipt.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class AgentStatus(StrEnum):
    """Execution status of a single agent node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentMetrics(BaseModel):
    """Operational metrics for a single agent node execution."""

    agent_name: str
    status: AgentStatus = AgentStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error_message: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock time the agent took to execute."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by this agent."""
        return self.input_tokens + self.output_tokens


class ExecutionReceipt(BaseModel):
    """
    Full operational receipt for a single agent pipeline execution.
    Analogous to a payment receipt: issued after every transaction.

    Contains timing, token usage, cost estimates, and quality signals
    that make the system auditable and comparable across runs.
    """

    request_id: str = Field(description="Links back to the DecisionRequest.id")
    trace_id: str | None = Field(
        default=None,
        description="Langfuse trace ID for deep-dive observability",
    )
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # Per-agent breakdown
    agents: list[AgentMetrics] = Field(default_factory=list)

    # Aggregate counters (computed from agents list)
    sources_consulted: int = Field(default=0)
    iterations: int = Field(default=0)
    circuit_breaker_triggered: bool = Field(default=False)

    # Cost estimation (USD)
    # Populated by LiteLLM's cost tracking in later phases
    estimated_cost_usd: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_duration_seconds(self) -> float | None:
        """End-to-end wall-clock time for the full pipeline."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_input_tokens(self) -> int:
        return sum(a.input_tokens for a in self.agents)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.agents)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def mark_completed(self) -> None:
        """Seal the receipt with a completion timestamp."""
        self.completed_at = datetime.now(UTC)

    def add_agent_metrics(self, metrics: AgentMetrics) -> None:
        """Register metrics from a completed agent node."""
        self.agents.append(metrics)

    def to_summary(self) -> dict:
        """Human-readable summary for logging and API responses."""
        return {
            "request_id": self.request_id,
            "duration_s": round(self.total_duration_seconds or 0, 2),
            "agents_run": len(self.agents),
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "sources_consulted": self.sources_consulted,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "trace_id": self.trace_id,
        }
