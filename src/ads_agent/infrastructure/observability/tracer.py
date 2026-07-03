# src/ads_agent/infrastructure/observability/tracer.py
"""
Langfuse v4 tracer wrapper.

Langfuse v4 (released March 2026) rewrote its SDK over OpenTelemetry.
Key facts verified from official docs:
  - get_client() returns the singleton Langfuse client
  - start_as_current_observation() is the context manager API for manual spans
  - propagate_attributes() sets user_id, session_id, metadata on child observations
  - get_current_trace_id() links traces back to ExecutionReceipt

We wrap these APIs here so the rest of the codebase never imports
Langfuse directly — making it swappable if needed.

Dashboard design note (Phase 7 — not implemented yet):
  GET /api/public/v2/metrics?query=<url-encoded JSON>
  Example daily cost aggregation:
    {
      "view": "observations",
      "metrics": [{"measure": "totalCost", "aggregation": "sum"}],
      "timeDimension": {"granularity": "day"},
      "fromTimestamp": "2026-06-01T00:00:00Z",
      "toTimestamp": "2026-06-30T23:59:59Z"
    }
  Auth: HTTP Basic with LANGFUSE_PUBLIC_KEY:LANGFUSE_SECRET_KEY
  Metrics API v2 is Cloud-only; self-hosted use GET /api/public/metrics/daily.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Generator

log = structlog.get_logger(__name__)

_PIPELINE_TRACE_NAME = "ads-agent-pipeline"


def is_tracing_enabled() -> bool:
    """Return True when Langfuse credentials are configured for real ingestion."""
    return _is_langfuse_configured()


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
        log.warning("langfuse_init_failed", error=str(exc))
        return None


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a Langfuse SDK call; log and swallow errors."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.warning("langfuse_call_failed", fn=getattr(fn, "__name__", str(fn)), error=str(exc))
        return None


@contextmanager
def pipeline_trace(
    request_id: str,
    *,
    session_id: str | None = None,
) -> Generator[None]:
    """
    Root observation for a full pipeline run.

    Sets session_id and request_id metadata on all child observations.
    """
    client = get_langfuse_client()
    if client is None:
        yield
        return

    try:
        from langfuse import propagate_attributes

        with client.start_as_current_observation(
            as_type="span",
            name=_PIPELINE_TRACE_NAME,
            input={"request_id": request_id},
        ) as root_span:
            with propagate_attributes(
                session_id=session_id,
                metadata={"request_id": request_id},
            ):
                yield
            _safe_call(root_span.update, output={"request_id": request_id})
    except Exception as exc:
        log.warning("langfuse_pipeline_trace_failed", error=str(exc))
        yield


@contextmanager
def agent_span(
    name: str,
    *,
    iteration: int | None = None,
) -> Generator[None]:
    """Per-node span inside the active pipeline trace."""
    client = get_langfuse_client()
    if client is None:
        yield
        return

    metadata: dict[str, Any] = {}
    if iteration is not None:
        metadata["iteration"] = iteration

    try:
        with client.start_as_current_observation(
            as_type="span",
            name=name,
            metadata=metadata or None,
        ):
            yield
    except Exception as exc:
        log.warning("langfuse_agent_span_failed", span=name, error=str(exc))
        yield


@contextmanager
def llm_generation(
    name: str,
    model: str,
    input_messages: Any,
) -> Generator[Any]:
    """
    Nested generation observation for an LLM call.

    Yields the Langfuse observation object so callers can update output and usage.
    Yields None when tracing is disabled (callers should no-op on .update).
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    try:
        with client.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input_messages,
        ) as generation:
            yield generation
    except Exception as exc:
        log.warning("langfuse_generation_failed", generation=name, error=str(exc))
        yield None


def update_generation(
    generation: Any,
    *,
    output: Any,
    input_tokens: int,
    output_tokens: int,
    model: str | None = None,
) -> None:
    """Update a generation observation with output and token usage."""
    if generation is None:
        return

    usage_details = {
        "input": input_tokens,
        "output": output_tokens,
        "total": input_tokens + output_tokens,
    }
    kwargs: dict[str, Any] = {
        "output": output,
        "usage_details": usage_details,
    }
    if model is not None:
        kwargs["model"] = model
    _safe_call(generation.update, **kwargs)


def capture_trace_id() -> str | None:
    """Return the active Langfuse trace ID, or None when unavailable."""
    client = get_langfuse_client()
    if client is None:
        return None

    trace_id = _safe_call(client.get_current_trace_id)
    if isinstance(trace_id, str) and trace_id:
        return trace_id
    return None


def submit_pipeline_scores(
    trace_id: str | None,
    *,
    has_sources: bool,
    trade_offs_count: int,
) -> None:
    """
    Send post-execution quality scores to Langfuse.

    Scores:
      - has_sources: 1 if sources were consulted, 0 otherwise
      - trade_offs_count: number of trade-offs in the analysis (depth proxy)
    """
    if not trace_id:
        return

    client = get_langfuse_client()
    if client is None:
        return

    _safe_call(
        client.create_score,
        name="has_sources",
        value=1.0 if has_sources else 0.0,
        trace_id=trace_id,
        data_type="NUMERIC",
    )
    _safe_call(
        client.create_score,
        name="trade_offs_count",
        value=float(trade_offs_count),
        trace_id=trace_id,
        data_type="NUMERIC",
    )


def submit_evaluation_scores(
    trace_id: str | None,
    *,
    faithfulness: float | None,
    answer_relevancy: float | None,
    context_precision: float | None,
    quality_score: float | None,
) -> None:
    """
    Send RAGAS evaluation scores to Langfuse.

    Scores: faithfulness, answer_relevancy, context_precision, quality_score.
    """
    if not trace_id:
        return

    client = get_langfuse_client()
    if client is None:
        return

    score_values: dict[str, float | None] = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "quality_score": quality_score,
    }
    for name, value in score_values.items():
        if value is None:
            continue
        _safe_call(
            client.create_score,
            name=name,
            value=value,
            trace_id=trace_id,
            data_type="NUMERIC",
        )


def compute_pipeline_scores(final_state: dict[str, Any] | Any) -> tuple[bool, int]:
    """Derive score inputs from the final pipeline state."""
    receipt = final_state.get("receipt")
    has_sources = bool(receipt and receipt.source_urls)

    trade_offs_count = 0
    final_report = final_state.get("final_report")
    if final_report is not None and hasattr(final_report, "trade_offs"):
        trade_offs_count = len(final_report.trade_offs)

    return has_sources, trade_offs_count


def flush_traces() -> None:
    """
    Flush pending Langfuse traces.
    Must be called before process exit in short-lived applications.
    In long-running FastAPI apps, the SDK flushes automatically.
    """
    client = get_langfuse_client()
    if client is not None:
        _safe_call(client.flush)
