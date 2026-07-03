# src/ads_agent/api/main.py
"""FastAPI application entry point for the ADS Agent gateway."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog
import uvicorn

from ads_agent.api.exceptions import register_exception_handlers
from ads_agent.api.middleware import RequestLoggingMiddleware
from ads_agent.api.v1.router import health_router, router
from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.checkpointer import close_checkpointer_pool, get_postgres_checkpointer
from ads_agent.infrastructure.observability.tracer import flush_traces
from ads_agent.infrastructure.vector_store.connection import close_pool as close_vector_store_pool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup: open Postgres checkpointer pool. Shutdown: flush Langfuse and close pools."""
    log.info("api_startup")
    await get_postgres_checkpointer()
    yield
    flush_traces()
    await close_checkpointer_pool()
    await close_vector_store_pool()
    log.info("api_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ADS Agent API",
        description=(
            "Architecture Decision Support — multi-agent pipeline exposed as a REST API. "
            "Runs research, analysis, and writer agents to produce structured technical "
            "decision reports with full execution receipts."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()


def run() -> None:
    """Console script entry point: uvicorn ads-agent-api."""
    settings = get_settings()
    uvicorn.run(
        "ads_agent.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
