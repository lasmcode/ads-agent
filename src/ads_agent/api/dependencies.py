# src/ads_agent/api/dependencies.py
"""FastAPI dependency injection providers."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from ads_agent.core.settings import AppSettings, get_settings
from ads_agent.infrastructure.checkpointer import get_postgres_checkpointer
from ads_agent.infrastructure.observability.tracer import get_langfuse_client


async def get_checkpointer() -> AsyncPostgresSaver:
    """Provide the process-wide AsyncPostgresSaver (pool opened at lifespan startup)."""
    return await get_postgres_checkpointer()


def get_app_settings() -> AppSettings:
    return get_settings()


def get_langfuse() -> Any | None:
    return get_langfuse_client()


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


SettingsDep = Annotated[AppSettings, Depends(get_app_settings)]
CheckpointerDep = Annotated[AsyncPostgresSaver, Depends(get_checkpointer)]
LangfuseDep = Annotated[Any | None, Depends(get_langfuse)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
