# src/ads_agent/infrastructure/observability/tracer.py
"""
Langfuse v4 tracer wrapper.

Langfuse v4 (released March 2026) rewrote its SDK over OpenTelemetry.
Key facts verified from official docs:
  - get_client() returns the singleton Langfuse client
  - @observe decorator creates traces/spans automatically
  - start_as_current_observation() is the context manager API for manual spans
  - Trace attributes: user_id, session_id, metadata, tags

We wrap these APIs here so the rest of the codebase never imports
Langfuse directly — making it swappable if needed.
"""

from __future__ import annotations

from functools import wraps
import os
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

log = structlog.get_logger(__name__)


def _is_langfuse_configured() -> bool:
    """Check whether Langfuse env vars are set to real (non-test) values."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    return bool(public_key) and not public_key.startswith("pk-lf-test")


def get_langfuse_client() -> Any | None:
    """
    Return a configured Langfuse client, or None in test/unconfigured environments.
    Returning None instead of raising keeps agent nodes resilient:
    observability failure must never crash the pipeline.
    """
    if not _is_langfuse_configured():
        log.debug("langfuse_disabled", reason="no valid credentials configured")
        return None

    try:
        from langfuse import get_client

        client = get_client()
        log.info("langfuse_connected", host=os.getenv("LANGFUSE_HOST"))
        return client
    except Exception as exc:
        # Observability must never crash the agent pipeline
        log.warning("langfuse_init_failed", error=str(exc))
        return None


def traced(name: str | None = None, as_type: str = "span") -> Callable:
    """
    Decorator that wraps a function in a Langfuse observation span.
    Falls back to a no-op if Langfuse is not configured.

    Usage:
        @traced(name="research-agent", as_type="span")
        async def run_research(state: AgentState) -> dict:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            client = get_langfuse_client()

            if client is None:
                # No-op: run function without tracing
                return await fn(*args, **kwargs)

            span_name = name or fn.__name__
            with client.start_as_current_observation(
                as_type=as_type,
                name=span_name,
            ):
                return await fn(*args, **kwargs)

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            client = get_langfuse_client()

            if client is None:
                return fn(*args, **kwargs)

            span_name = name or fn.__name__
            with client.start_as_current_observation(
                as_type=as_type,
                name=span_name,
            ):
                return fn(*args, **kwargs)

        # Return the correct wrapper based on whether the function is async
        import asyncio

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


def flush_traces() -> None:
    """
    Flush pending Langfuse traces.
    Must be called before process exit in short-lived applications.
    In long-running FastAPI apps, the SDK flushes automatically.
    """
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            log.warning("langfuse_flush_failed", error=str(exc))
