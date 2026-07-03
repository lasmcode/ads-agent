# src/ads_agent/infrastructure/health.py
"""Health checks for external dependencies."""

from __future__ import annotations

import base64
import os
from typing import Any, cast

import httpx
import structlog

from ads_agent.api.v1.schemas import DependencyHealthDTO, HealthResponse, HealthStatus
from ads_agent.core.settings import AppSettings, get_settings
from ads_agent.infrastructure.checkpointer import get_postgres_checkpointer
from ads_agent.infrastructure.observability.tracer import is_tracing_enabled

log = structlog.get_logger(__name__)


async def check_postgres(settings: AppSettings | None = None) -> DependencyHealthDTO:
    """Verify PostgreSQL connectivity via the checkpointer pool."""
    _ = settings  # reserved for future per-request settings override
    try:
        checkpointer = await get_postgres_checkpointer()
        pool = cast(Any, checkpointer.conn)  # noqa: TC006
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
        return DependencyHealthDTO(ok=True, configured=True, detail=None)
    except Exception as exc:
        log.warning("postgres_health_failed", error=str(exc))
        return DependencyHealthDTO(ok=False, configured=True, detail=str(exc))


def _langfuse_auth_header() -> str | None:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not public_key or not secret_key:
        return None
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return f"Basic {token}"


async def check_langfuse(settings: AppSettings | None = None) -> DependencyHealthDTO:
    """Verify Langfuse connectivity when credentials are configured."""
    app_settings = settings or get_settings()

    if not is_tracing_enabled():
        return DependencyHealthDTO(
            ok=True,
            configured=False,
            detail="Langfuse credentials not configured",
        )

    auth = _langfuse_auth_header()
    if auth is None:
        return DependencyHealthDTO(
            ok=True,
            configured=False,
            detail="Langfuse credentials incomplete",
        )

    host = app_settings.langfuse_host.rstrip("/")
    url = f"{host}/api/public/health"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers={"Authorization": auth})
        if response.status_code == 200:
            return DependencyHealthDTO(ok=True, configured=True, detail=None)
        return DependencyHealthDTO(
            ok=False,
            configured=True,
            detail=f"Langfuse health returned HTTP {response.status_code}",
        )
    except Exception as exc:
        log.warning("langfuse_health_failed", error=str(exc))
        return DependencyHealthDTO(ok=False, configured=True, detail=str(exc))


async def run_health_checks(settings: AppSettings | None = None) -> HealthResponse:
    """Run all dependency health checks and compute overall status."""
    app_settings = settings or get_settings()
    postgres = await check_postgres(app_settings)
    langfuse = await check_langfuse(app_settings)

    status: HealthStatus
    if not postgres.ok:
        status = "unhealthy"
    elif langfuse.configured and not langfuse.ok:
        status = "degraded"
    else:
        status = "healthy"

    return HealthResponse(status=status, postgres=postgres, langfuse=langfuse)
