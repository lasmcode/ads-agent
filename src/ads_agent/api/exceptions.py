# src/ads_agent/api/exceptions.py
"""API exceptions and centralized exception handlers."""

from __future__ import annotations

import asyncio
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import structlog

from ads_agent.core.settings import get_settings

log = structlog.get_logger(__name__)


class DecisionNotFoundError(Exception):
    """Raised when no checkpoint exists for the given request_id."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"Decision run not found: {request_id}")


class PipelineUnavailableError(Exception):
    """Raised when a required dependency (e.g. Postgres) is unavailable."""


def _error_detail(message: str, exc: Exception | None = None) -> dict[str, str | None]:
    settings = get_settings()
    detail: dict[str, str | None] = {"detail": message}
    if settings.app_env != "production" and exc is not None:
        detail["debug"] = str(exc)
        detail["traceback"] = traceback.format_exc()
    return detail


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(DecisionNotFoundError)
    async def decision_not_found_handler(
        _request: Request,
        exc: DecisionNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc), "request_id": exc.request_id},
        )

    @app.exception_handler(PipelineUnavailableError)
    async def pipeline_unavailable_handler(
        _request: Request,
        exc: PipelineUnavailableError,
    ) -> JSONResponse:
        log.error("pipeline_unavailable", error=str(exc))
        return JSONResponse(
            status_code=503,
            content=_error_detail("Service temporarily unavailable", exc),
        )

    @app.exception_handler(asyncio.TimeoutError)
    async def pipeline_timeout_handler(
        _request: Request,
        exc: asyncio.TimeoutError,
    ) -> JSONResponse:
        settings = get_settings()
        log.error("pipeline_timeout", timeout_s=settings.api_pipeline_timeout)
        return JSONResponse(
            status_code=504,
            content={
                "detail": f"Pipeline exceeded timeout of {settings.api_pipeline_timeout}s",
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def generic_handler(_request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", error=str(exc))
        return JSONResponse(
            status_code=500,
            content=_error_detail("Internal server error", exc),
        )
