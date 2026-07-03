# src/ads_agent/application/services/evaluation_runner.py
"""
Fire-and-forget evaluation scheduler — runs RAGAS scoring off the critical path.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING

import structlog

from ads_agent.application.services.evaluation_service import evaluate_report
from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.observability.tracer import flush_traces, submit_evaluation_scores

if TYPE_CHECKING:
    from ads_agent.agents.state import AgentState

log = structlog.get_logger(__name__)

_background_tasks: set[asyncio.Task[None]] = set()


def schedule_evaluation(state: AgentState, trace_id: str | None) -> None:
    """
    Schedule background RAGAS evaluation without blocking the caller.

    Respects ADS_EVAL_ENABLED and ADS_EVAL_SAMPLE_RATE.
    """
    settings = get_settings()
    if not settings.eval_enabled:
        return

    if random.random() >= settings.eval_sample_rate:
        log.debug("evaluation_skipped_by_sample_rate", sample_rate=settings.eval_sample_rate)
        return

    final_report = state.get("final_report")
    if final_report is None:
        return

    task = asyncio.create_task(_run_evaluation(state, trace_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run_evaluation(state: AgentState, trace_id: str | None) -> None:
    """Execute evaluation, populate quality_score, and push scores to Langfuse."""
    settings = get_settings()
    final_report = state.get("final_report")
    if final_report is None:
        return

    contexts = state.get("retrieved_contexts") or []
    request_id = final_report.request_id

    try:
        scores = await asyncio.wait_for(
            evaluate_report(final_report, contexts),
            timeout=settings.eval_timeout_seconds,
        )
    except TimeoutError:
        log.warning("evaluation_timed_out", request_id=request_id)
        return
    except Exception as exc:
        log.warning("evaluation_failed", request_id=request_id, error=str(exc))
        return
    else:
        final_report.quality_score = scores.get("quality_score")
        submit_evaluation_scores(
            trace_id,
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("answer_relevancy"),
            context_precision=scores.get("context_precision"),
            quality_score=scores.get("quality_score"),
        )
        log.info(
            "evaluation_completed",
            request_id=request_id,
            quality_score=scores.get("quality_score"),
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("answer_relevancy"),
            context_precision=scores.get("context_precision"),
        )
    finally:
        flush_traces()
