# src/ads_agent/api/v1/schemas.py
"""API DTOs — separate from domain entities (Clean Architecture boundary)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ads_agent.core.entities.decision_report import DecisionReport
    from ads_agent.core.entities.execution_receipt import AgentMetrics, ExecutionReceipt


class CreateDecisionRequestBody(BaseModel):
    """Request body for POST /api/v1/decisions."""

    query: str = Field(
        min_length=10,
        max_length=2000,
        description="Technical decision question",
        examples=["Should I use Redis or Memcached for session storage?"],
    )
    context: str | None = Field(
        default=None,
        max_length=5000,
        description="Optional context: team size, scale, constraints",
        examples=["Team of 5, 10k concurrent users, AWS deployment"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "Should I use Redis or Memcached for session storage?",
                    "context": "Team of 5, 10k concurrent users",
                }
            ]
        }
    )


class TradeOffDTO(BaseModel):
    dimension: str
    option_a: str
    option_b: str
    winner: str | None = None


class DecisionReportDTO(BaseModel):
    request_id: str
    query: str
    recommendation: str
    recommendation_strength: str
    summary: str
    trade_offs: list[TradeOffDTO] = Field(default_factory=list)
    key_considerations: list[str] = Field(default_factory=list)
    when_to_choose_alternative: str | None = None
    sources: list[str] = Field(default_factory=list)
    quality_score: float | None = None

    @classmethod
    def from_domain(cls, report: DecisionReport) -> DecisionReportDTO:
        return cls(
            request_id=report.request_id,
            query=report.query,
            recommendation=report.recommendation,
            recommendation_strength=report.recommendation_strength.value,
            summary=report.summary,
            trade_offs=[
                TradeOffDTO(
                    dimension=t.dimension,
                    option_a=t.option_a,
                    option_b=t.option_b,
                    winner=t.winner,
                )
                for t in report.trade_offs
            ],
            key_considerations=report.key_considerations,
            when_to_choose_alternative=report.when_to_choose_alternative,
            sources=report.sources,
            quality_score=report.quality_score,
        )


def build_langfuse_trace_url(trace_id: str | None, langfuse_host: str) -> str | None:
    if not trace_id:
        return None
    host = langfuse_host.rstrip("/")
    return f"{host}/trace/{trace_id}"


class ReceiptSummaryDTO(BaseModel):
    request_id: str
    duration_s: float
    agents_run: int
    total_tokens: int
    estimated_cost_usd: float | None = None
    sources_consulted: int = 0
    circuit_breaker_triggered: bool = False
    trace_id: str | None = None
    langfuse_trace_url: str | None = None

    @classmethod
    def from_receipt(cls, receipt: ExecutionReceipt, langfuse_host: str) -> ReceiptSummaryDTO:
        summary = receipt.to_summary()
        trace_id = summary.get("trace_id")
        return cls(
            request_id=summary["request_id"],
            duration_s=summary["duration_s"],
            agents_run=summary["agents_run"],
            total_tokens=summary["total_tokens"],
            estimated_cost_usd=summary.get("estimated_cost_usd"),
            sources_consulted=summary.get("sources_consulted", 0),
            circuit_breaker_triggered=summary.get("circuit_breaker_triggered", False),
            trace_id=trace_id,
            langfuse_trace_url=build_langfuse_trace_url(trace_id, langfuse_host),
        )


class DecisionResponse(BaseModel):
    """Response for POST /decisions and GET /decisions/{request_id}."""

    request_id: str
    report: DecisionReportDTO | None = None
    error: str | None = None
    receipt: ReceiptSummaryDTO

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "request_id": "550e8400-e29b-41d4-a716-446655440000",
                    "report": {
                        "request_id": "550e8400-e29b-41d4-a716-446655440000",
                        "query": "Should I use Redis or Memcached?",
                        "recommendation": "Use Redis for session storage.",
                        "recommendation_strength": "moderate",
                        "summary": "Redis offers richer data structures...",
                        "trade_offs": [],
                        "key_considerations": ["Persistence requirements"],
                        "when_to_choose_alternative": None,
                        "sources": ["https://example.com/redis"],
                        "quality_score": None,
                    },
                    "error": None,
                    "receipt": {
                        "request_id": "550e8400-e29b-41d4-a716-446655440000",
                        "duration_s": 18.5,
                        "agents_run": 4,
                        "total_tokens": 4200,
                        "estimated_cost_usd": 0.012,
                        "sources_consulted": 3,
                        "circuit_breaker_triggered": False,
                        "trace_id": "abc123",
                        "langfuse_trace_url": "https://cloud.langfuse.com/trace/abc123",
                    },
                }
            ]
        }
    )


class AgentMetricsDTO(BaseModel):
    agent_name: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_seconds: float | None = None
    error_message: str | None = None

    @classmethod
    def from_domain(cls, metrics: AgentMetrics) -> AgentMetricsDTO:
        return cls(
            agent_name=metrics.agent_name,
            status=metrics.status.value,
            started_at=metrics.started_at,
            completed_at=metrics.completed_at,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            total_tokens=metrics.total_tokens,
            duration_seconds=metrics.duration_seconds,
            error_message=metrics.error_message,
        )


class ExecutionReceiptResponse(BaseModel):
    """Full operational receipt for FinOps/AgentOps dashboards."""

    request_id: str
    trace_id: str | None = None
    langfuse_trace_url: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    total_duration_seconds: float | None = None
    agents: list[AgentMetricsDTO] = Field(default_factory=list)
    sources_consulted: int = 0
    source_urls: list[str] = Field(default_factory=list)
    iterations: int = 0
    circuit_breaker_triggered: bool = False
    estimated_cost_usd: float | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_domain(cls, receipt: ExecutionReceipt, langfuse_host: str) -> ExecutionReceiptResponse:
        return cls(
            request_id=receipt.request_id,
            trace_id=receipt.trace_id,
            langfuse_trace_url=build_langfuse_trace_url(receipt.trace_id, langfuse_host),
            started_at=receipt.started_at,
            completed_at=receipt.completed_at,
            total_duration_seconds=receipt.total_duration_seconds,
            agents=[AgentMetricsDTO.from_domain(a) for a in receipt.agents],
            sources_consulted=receipt.sources_consulted,
            source_urls=receipt.source_urls,
            iterations=receipt.iterations,
            circuit_breaker_triggered=receipt.circuit_breaker_triggered,
            estimated_cost_usd=receipt.estimated_cost_usd,
            total_input_tokens=receipt.total_input_tokens,
            total_output_tokens=receipt.total_output_tokens,
            total_tokens=receipt.total_tokens,
        )


HealthStatus = Literal["healthy", "degraded", "unhealthy"]


class DependencyHealthDTO(BaseModel):
    ok: bool
    configured: bool = True
    detail: str | None = None


class HealthResponse(BaseModel):
    status: HealthStatus
    postgres: DependencyHealthDTO
    langfuse: DependencyHealthDTO
