# src/ads_agent/application/services/decision_service.py
"""Application service for decision pipeline orchestration."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ads_agent.agents.supervisor.graph import run_pipeline
from ads_agent.api.exceptions import DecisionNotFoundError
from ads_agent.api.v1.schemas import (
    CreateDecisionRequestBody,
    DecisionReportDTO,
    DecisionResponse,
    ExecutionReceiptResponse,
    ReceiptSummaryDTO,
)
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.core.settings import AppSettings
from ads_agent.infrastructure.persistence.decision_run_store import get_state

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from ads_agent.agents.state import AgentState
    from ads_agent.core.entities.execution_receipt import ExecutionReceipt


def _map_to_response(
    final_state: AgentState,
    receipt: ExecutionReceipt,
    settings: AppSettings,
) -> DecisionResponse:
    report = final_state.get("final_report")
    return DecisionResponse(
        request_id=receipt.request_id,
        report=DecisionReportDTO.from_domain(report) if report is not None else None,
        error=final_state.get("error"),
        receipt=ReceiptSummaryDTO.from_receipt(receipt, settings.langfuse_host),
    )


async def create_decision(
    body: CreateDecisionRequestBody,
    checkpointer: BaseCheckpointSaver,
    settings: AppSettings,
) -> DecisionResponse:
    request = DecisionRequest(query=body.query, context=body.context)
    final_state, receipt = await asyncio.wait_for(
        run_pipeline(request, checkpointer=checkpointer),
        timeout=settings.api_pipeline_timeout,
    )
    return _map_to_response(final_state, receipt, settings)


async def get_decision(
    request_id: str,
    checkpointer: BaseCheckpointSaver,
    settings: AppSettings,
) -> DecisionResponse:
    state = await get_state(request_id, checkpointer)
    if state is None:
        raise DecisionNotFoundError(request_id)
    receipt = state["receipt"]
    return _map_to_response(state, receipt, settings)


async def get_receipt(
    request_id: str,
    checkpointer: BaseCheckpointSaver,
    settings: AppSettings,
) -> ExecutionReceiptResponse:
    state = await get_state(request_id, checkpointer)
    if state is None:
        raise DecisionNotFoundError(request_id)
    return ExecutionReceiptResponse.from_domain(state["receipt"], settings.langfuse_host)
