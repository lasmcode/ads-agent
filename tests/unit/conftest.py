# tests/unit/conftest.py
"""
Shared fixtures for unit tests.

Phase 2 research uses LLM + MCP; Phase 3 adds a Postgres-backed checkpointer
and knowledge store. Phase 4 adds structured LLM for analysis/writer/supervisor.
Graph routing and CLI unit tests mock all of it to stay fast, keyless, and DB-free.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from langgraph.checkpoint.memory import MemorySaver
import pytest

from ads_agent.agents.research.nodes import ResearchAgentResult
from ads_agent.core.entities.decision_report import RecommendationStrength, TradeOff
from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.llm.client import LLMCompletionResult
from ads_agent.infrastructure.llm.schemas import AnalysisOutput, SupervisorDecision, WriterDraft


def _build_mock_llm_result(response_model, parsed) -> LLMCompletionResult:
    return LLMCompletionResult(
        parsed=parsed,
        raw_content=parsed.model_dump_json() if parsed else None,
        input_tokens=100,
        output_tokens=50,
        estimated_cost_usd=0.001,
        model="mock/model",
    )


async def _mock_complete(messages, model, *, response_model=None, receipt=None, **kwargs):
    if response_model is SupervisorDecision:
        parsed = SupervisorDecision(next_agent="analysis")
    elif response_model is AnalysisOutput:
        parsed = AnalysisOutput(
            trade_offs=[
                TradeOff(
                    dimension="Performance",
                    option_a="pgvector: good enough for moderate scale",
                    option_b="Qdrant: optimized for high-throughput vector search",
                    winner="Qdrant",
                ),
                TradeOff(
                    dimension="Operational Complexity",
                    option_a="pgvector: reuses existing PostgreSQL ops",
                    option_b="Qdrant: requires separate cluster management",
                    winner="pgvector",
                ),
                TradeOff(
                    dimension="Ecosystem Integration",
                    option_a="pgvector: native SQL joins with app data",
                    option_b="Qdrant: dedicated API, less SQL coupling",
                    winner="pgvector",
                ),
            ],
        )
    elif response_model is WriterDraft:
        parsed = WriterDraft(
            recommendation="Use pgvector if you already run PostgreSQL; choose Qdrant for dedicated vector scale.",
            recommendation_strength=RecommendationStrength.MODERATE,
            summary="Both options are viable. pgvector minimizes ops overhead when Postgres is already in stack.",
            key_considerations=[
                "Existing PostgreSQL investment reduces pgvector adoption cost",
                "Qdrant excels at dedicated vector search workloads",
            ],
            when_to_choose_alternative="Choose Qdrant when vector query latency at scale is the primary bottleneck.",
        )
    else:
        parsed = None

    result = _build_mock_llm_result(response_model, parsed)
    if receipt is not None:
        current = receipt.estimated_cost_usd or 0.0
        receipt.estimated_cost_usd = current + result.estimated_cost_usd
    return result


@pytest.fixture(autouse=True)
def disable_eval_for_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests fast and offline — evaluation is tested in dedicated modules."""
    monkeypatch.setenv("ADS_EVAL_ENABLED", "false")
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def disable_langfuse_for_unit_tests(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """
    Keep unit tests keyless and offline — Langfuse is tested in test_tracer.py
    and tests/integration/test_langfuse_tracing.py.
    """
    if request.node.fspath.basename == "test_tracer.py":
        return
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(
        "ads_agent.infrastructure.observability.tracer.get_langfuse_client",
        lambda: None,
    )


@pytest.fixture(autouse=True)
def mock_research_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace run_research_agent with a stub for pipeline/CLI unit tests."""

    async def _stub_run_research_agent(query: str, tools=None) -> ResearchAgentResult:
        output = (
            f"Research findings for: '{query}'\n\n"
            "Key findings:\n"
            "- pgvector integrates natively with PostgreSQL and supports ACID transactions\n"
            "- Qdrant offers dedicated vector search performance with horizontal scaling\n"
            "- Both support metadata filtering; pgvector reuses existing Postgres ops\n"
            "- Qdrant requires a separate cluster but optimizes pure vector workloads\n\n"
            "Sources:\n"
            "- https://example.com/pgvector-overview\n"
            "- https://example.com/qdrant-comparison"
        )
        return ResearchAgentResult(
            output=output,
            input_tokens=100,
            output_tokens=50,
            source_urls=[
                "https://example.com/pgvector-overview",
                "https://example.com/qdrant-comparison",
            ],
            retrieved_contexts=[],
        )

    monkeypatch.setattr(
        "ads_agent.agents.research.nodes.run_research_agent",
        AsyncMock(side_effect=_stub_run_research_agent),
    )


@pytest.fixture(autouse=True)
def mock_llm_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace centralized LLM complete() at all agent import sites."""
    mock = AsyncMock(side_effect=_mock_complete)
    targets = [
        "ads_agent.infrastructure.llm.client.complete",
        "ads_agent.agents.supervisor.nodes.complete",
        "ads_agent.agents.analysis.nodes.complete",
        "ads_agent.agents.writer.nodes.complete",
    ]
    for target in targets:
        monkeypatch.setattr(target, mock)


@pytest.fixture(autouse=True)
def mock_postgres_checkpointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace the CLI's Postgres checkpointer with an in-memory one.

    Unit tests invoke `ads_agent.cli.main()` directly; without this, every
    CLI test would require a live PostgreSQL connection just to obtain a
    checkpointer, even though checkpoint persistence itself isn't under
    test here (see tests/integration for that).
    """
    monkeypatch.setattr(
        "ads_agent.cli.get_postgres_checkpointer",
        AsyncMock(return_value=MemorySaver()),
    )
    monkeypatch.setattr("ads_agent.cli.close_checkpointer_pool", AsyncMock())
    monkeypatch.setattr("ads_agent.cli.close_vector_store_pool", AsyncMock())
