# src/ads_agent/agents/state.py
"""
Shared graph state for the ADS Agent multi-agent pipeline.

Design principles:
  - TypedDict with Annotated reducers: the LangGraph-recommended pattern
  - add_messages reducer: handles concurrent message appends safely
  - All fields have defaults: no node is forced to set every field
  - Immutable inputs (request) separate from mutable pipeline data

State lifecycle:
  START → supervisor → [research | analysis | writer] → supervisor → END
  The supervisor reads `next_agent` to decide routing.
  Each worker reads messages and writes its output back to messages.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ads_agent.core.entities.decision_report import DecisionReport
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.core.entities.execution_receipt import ExecutionReceipt
from ads_agent.core.settings import get_settings


class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the LangGraph graph.

    LangGraph merges state updates incrementally — each node only needs
    to return the fields it modifies. Fields not returned remain unchanged.

    The `messages` field uses the `add_messages` reducer, which appends
    new messages rather than replacing the list — safe for concurrent nodes.
    """

    # --- Input (set once at graph entry, never modified) ---
    request: DecisionRequest

    # --- Message history (append-only via add_messages reducer) ---
    # Contains the full conversation: human query + agent outputs
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Routing control ---
    # The supervisor sets this to decide which agent runs next.
    # Value must be one of: "research", "analysis", "writer", "FINISH"
    next_agent: str

    # --- Pipeline data (each agent populates its section) ---
    research_output: str | None  # Raw research findings
    analysis_output: str | None  # Structured trade-off analysis
    final_report: DecisionReport | None  # Completed report from writer

    # --- Operational metadata ---
    receipt: ExecutionReceipt  # Running receipt — updated by each node
    iterations: int  # Circuit breaker counter
    error: str | None  # Last error message, if any


# Maximum iterations before the circuit breaker halts the graph.
# Prevents infinite supervisor loops that exhaust API budget.
# Override via ADS_MAX_ITERATIONS in the environment.
MAX_ITERATIONS: int = get_settings().max_iterations
