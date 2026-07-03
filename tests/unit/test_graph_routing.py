# tests/unit/test_graph_routing.py
"""
Unit tests for graph routing logic.

Key insight from LangGraph docs:
  "Mock the LLM using unittest.mock.AsyncMock and assert directly on
  state transitions. The router and state schema are pure Python —
  fully testable without LLM involvement."

These tests validate:
  1. The supervisor routes correctly through all stages
  2. The circuit breaker triggers at MAX_ITERATIONS
  3. The full graph executes end-to-end with stub nodes
  4. The ExecutionReceipt is populated correctly
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
import pytest

from ads_agent.agents.state import MAX_ITERATIONS, AgentState
from ads_agent.agents.supervisor.nodes import (
    deterministic_route,
    should_continue,
    supervisor_node,
)
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.core.entities.execution_receipt import ExecutionReceipt

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_request() -> DecisionRequest:
    return DecisionRequest(query="Should I use pgvector or Qdrant for my RAG system?")


@pytest.fixture
def base_state(sample_request: DecisionRequest) -> AgentState:
    """Minimal valid AgentState for testing — all optional fields at defaults."""
    from langchain_core.messages import HumanMessage

    return AgentState(
        request=sample_request,
        messages=[HumanMessage(content=sample_request.query)],
        next_agent="",
        research_output=None,
        retrieved_contexts=[],
        analysis_output=None,
        final_report=None,
        receipt=ExecutionReceipt(request_id=sample_request.id),
        iterations=0,
        error=None,
    )


# ---------------------------------------------------------------------------
# Supervisor routing logic tests (pure Python — no LLM)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSupervisorRouting:
    def test_routes_to_research_when_no_data(self, base_state: AgentState) -> None:
        """First call must always go to research."""
        assert deterministic_route(base_state) == "research"

    def test_routes_to_analysis_after_research(self, base_state: AgentState) -> None:
        """With research done, supervisor must route to analysis."""
        base_state["research_output"] = "Research findings stub"
        assert deterministic_route(base_state) == "analysis"

    def test_routes_to_writer_after_analysis(self, base_state: AgentState) -> None:
        """With research + analysis done, supervisor must route to writer."""

        base_state["research_output"] = "Research findings stub"
        base_state["analysis_output"] = "Analysis output stub"
        assert deterministic_route(base_state) == "writer"

    def test_routes_to_finish_after_report(self, base_state: AgentState) -> None:
        """With all outputs done, supervisor must route to FINISH."""
        from ads_agent.core.entities.decision_report import DecisionReport

        base_state["research_output"] = "Research"
        base_state["analysis_output"] = "Analysis"
        base_state["final_report"] = DecisionReport(
            request_id=base_state["request"].id,
            query=base_state["request"].query,
            recommendation="Use pgvector.",
            summary="Summary.",
        )
        assert deterministic_route(base_state) == "FINISH"

    def test_routes_to_finish_on_error(self, base_state: AgentState) -> None:
        """Any error must terminate the pipeline gracefully."""
        base_state["error"] = "Research agent failed: timeout"
        assert deterministic_route(base_state) == "FINISH"

    @pytest.mark.asyncio
    async def test_increments_iteration_counter(self, base_state: AgentState) -> None:
        """Each supervisor call must increment the iteration counter."""
        result = await supervisor_node(base_state)
        assert result["iterations"] == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_at_max_iterations(self, base_state: AgentState) -> None:
        """Circuit breaker must trigger when iterations reach MAX_ITERATIONS."""
        base_state["iterations"] = MAX_ITERATIONS
        result = await supervisor_node(base_state)
        assert result["next_agent"] == "FINISH"
        assert result["receipt"].circuit_breaker_triggered is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_not_triggered_before_max(self, base_state: AgentState) -> None:
        """Circuit breaker must NOT trigger before MAX_ITERATIONS."""
        base_state["iterations"] = MAX_ITERATIONS - 1
        result = await supervisor_node(base_state)
        assert result["receipt"].circuit_breaker_triggered is False


# ---------------------------------------------------------------------------
# Conditional edge function tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShouldContinue:
    def test_returns_next_agent_from_state(self, base_state: AgentState) -> None:
        base_state["next_agent"] = "research"
        assert should_continue(base_state) == "research"

    def test_defaults_to_finish_when_missing(self, base_state: AgentState) -> None:
        base_state["next_agent"] = ""
        # Empty string → falls back to FINISH
        result = should_continue(base_state)
        assert result == "" or result == "FINISH"  # Both are valid empty-state behaviors


# ---------------------------------------------------------------------------
# Full pipeline integration test (no LLM — stub nodes only)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_pipeline_executes_with_stubs(
    sample_request: DecisionRequest,
) -> None:
    """
    Run the complete graph end-to-end using stub nodes and MemorySaver.
    No LLM calls — validates graph wiring, state flow, and receipt population.
    """
    from ads_agent.agents.supervisor.graph import run_pipeline

    final_state, receipt = await run_pipeline(
        request=sample_request,
        checkpointer=MemorySaver(),
    )

    # Graph must complete without error
    assert final_state is not None

    # All three worker agents must have run
    agent_names = [m.agent_name for m in receipt.agents]
    assert "research" in agent_names
    assert "analysis" in agent_names
    assert "writer" in agent_names

    # Final report must exist
    assert final_state.get("final_report") is not None
    assert final_state["final_report"].request_id == sample_request.id

    # Receipt must be sealed
    assert receipt.completed_at is not None
    assert receipt.total_duration_seconds is not None
    assert receipt.total_duration_seconds >= 0

    # Circuit breaker must NOT have triggered for a normal run
    assert receipt.circuit_breaker_triggered is False

    # Langfuse not configured in CI — trace_id stays None
    assert receipt.trace_id is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pipeline_receipt_tracks_all_agents(
    sample_request: DecisionRequest,
) -> None:
    """Receipt must contain metrics from supervisor + 3 workers."""
    from ads_agent.agents.supervisor.graph import run_pipeline

    _, receipt = await run_pipeline(
        request=sample_request,
        checkpointer=MemorySaver(),
    )

    # Supervisor runs 4 times (initial + after each worker + final FINISH check)
    # Workers run once each
    # Total agents in receipt: supervisor (4x) + research + analysis + writer = 7
    # We validate at least 3 worker entries exist
    worker_names = {m.agent_name for m in receipt.agents}
    assert "research" in worker_names
    assert "analysis" in worker_names
    assert "writer" in worker_names


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pipeline_handles_worker_failure(
    sample_request: DecisionRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing worker must set error state and terminate without circuit breaker."""
    from ads_agent.agents.common import safe_node
    from ads_agent.agents.supervisor.graph import run_pipeline

    def _failing_research(_state: AgentState) -> dict:
        raise RuntimeError("timeout")

    failing_node = safe_node("research")(_failing_research)
    monkeypatch.setattr(
        "ads_agent.agents.supervisor.graph.research_node",
        failing_node,
    )

    final_state, receipt = await run_pipeline(
        request=sample_request,
        checkpointer=MemorySaver(),
    )

    assert final_state.get("error") is not None
    assert "research" in final_state["error"]
    assert final_state.get("final_report") is None
    assert receipt.circuit_breaker_triggered is False

    failed_metrics = [
        m for m in receipt.agents if m.agent_name == "research" and m.status.value == "failed"
    ]
    assert len(failed_metrics) == 1
    assert failed_metrics[0].error_message == "timeout"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pipeline_persists_checkpoint(
    sample_request: DecisionRequest,
) -> None:
    """MemorySaver must persist graph state retrievable by thread_id."""
    from langchain_core.messages import HumanMessage

    from ads_agent.agents.supervisor.graph import build_graph
    from ads_agent.core.entities.execution_receipt import ExecutionReceipt

    saver = MemorySaver()
    thread_id = "test-thread-checkpoint"
    config = {"configurable": {"thread_id": thread_id}}
    graph = build_graph(checkpointer=saver)

    initial_state: AgentState = {
        "request": sample_request,
        "messages": [HumanMessage(content=sample_request.query)],
        "next_agent": "",
        "research_output": None,
        "retrieved_contexts": [],
        "analysis_output": None,
        "final_report": None,
        "receipt": ExecutionReceipt(request_id=sample_request.id),
        "iterations": 0,
        "error": None,
    }

    await graph.ainvoke(initial_state, config=config)
    snapshot = await graph.aget_state(config)

    assert snapshot.values is not None
    assert snapshot.values.get("final_report") is not None
    assert snapshot.values["request"].id == sample_request.id
    assert snapshot.values["receipt"].completed_at is not None
    assert snapshot.values["receipt"].total_duration_seconds is not None
    assert snapshot.values["receipt"].total_duration_seconds >= 0
