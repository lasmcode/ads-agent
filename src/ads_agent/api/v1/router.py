# src/ads_agent/api/v1/router.py
"""API v1 route handlers."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from ads_agent.api.dependencies import CheckpointerDep, SettingsDep
from ads_agent.api.v1.schemas import (
    CreateDecisionRequestBody,
    DecisionResponse,
    ExecutionReceiptResponse,
    HealthResponse,
)
from ads_agent.application.services import decision_service
from ads_agent.infrastructure.health import run_health_checks

router = APIRouter()


@router.post(
    "/decisions",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a decision pipeline",
    description=(
        "Creates a DecisionRequest, executes the full multi-agent pipeline synchronously "
        "(research → analysis → writer → evaluation), and returns the structured report "
        "with an execution receipt summary. Typical latency: 10-30 seconds."
    ),
)
async def create_decision(
    body: CreateDecisionRequestBody,
    checkpointer: CheckpointerDep,
    settings: SettingsDep,
) -> DecisionResponse:
    return await decision_service.create_decision(body, checkpointer, settings)


@router.get(
    "/decisions/{request_id}",
    response_model=DecisionResponse,
    summary="Get a previously generated decision report",
    description=(
        "Retrieves a decision report and receipt summary from LangGraph checkpoints "
        "stored in PostgreSQL. Returns 404 if no run exists for the given request_id."
    ),
)
async def get_decision(
    request_id: str,
    checkpointer: CheckpointerDep,
    settings: SettingsDep,
) -> DecisionResponse:
    return await decision_service.get_decision(request_id, checkpointer, settings)


@router.get(
    "/decisions/{request_id}/receipt",
    response_model=ExecutionReceiptResponse,
    summary="Get full execution receipt",
    description=(
        "Returns the complete ExecutionReceipt for FinOps/AgentOps analysis, "
        "including per-agent token usage, timing, cost estimates, and a Langfuse trace link."
    ),
)
async def get_receipt(
    request_id: str,
    checkpointer: CheckpointerDep,
    settings: SettingsDep,
) -> ExecutionReceiptResponse:
    return await decision_service.get_receipt(request_id, checkpointer, settings)


health_router = APIRouter(tags=["health"])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Verifies connectivity to PostgreSQL (required) and Langfuse (optional). "
        "Returns HTTP 503 when PostgreSQL is unreachable; HTTP 200 with status "
        "'degraded' when Langfuse is configured but unreachable."
    ),
    responses={
        200: {"description": "Service healthy or degraded"},
        503: {"description": "PostgreSQL unreachable"},
    },
)
async def health(settings: SettingsDep, response: Response) -> HealthResponse:
    result = await run_health_checks(settings)
    if not result.postgres.ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
