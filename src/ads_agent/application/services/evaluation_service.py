# src/ads_agent/application/services/evaluation_service.py
"""
RAGAS-based evaluation for DecisionReport quality scoring.

Weighted quality_score formula (production default):
  faithfulness      x 0.40  — evidence grounding; hallucination is the #1 risk
  answer_relevancy  x 0.35  — must answer the concrete architecture question
  context_precision x 0.25  — retrieval quality; improvable independently of writer

When context_precision is unavailable (no retrieved contexts), weights are
renormalized over the remaining metrics only.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings
from langchain_litellm import ChatLiteLLM
import litellm
import structlog

from ads_agent.core.settings import get_settings

if TYPE_CHECKING:
    from ads_agent.core.entities.decision_report import DecisionReport

log = structlog.get_logger(__name__)

FAITHFULNESS_WEIGHT = 0.40
ANSWER_RELEVANCY_WEIGHT = 0.35
CONTEXT_PRECISION_WEIGHT = 0.25


class _LiteLLMEmbeddings(Embeddings):
    """Thin LangChain Embeddings adapter over LiteLLM for RAGAS metrics."""

    def __init__(self, model: str, dimensions: int) -> None:
        self._model = model
        self._dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        settings = get_settings()
        response = litellm.embedding(
            model=self._model,
            input=[text],
            dimensions=settings.embedding_dimensions,
        )
        return list(response.data[0]["embedding"])

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [await self.aembed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        settings = get_settings()
        response = await litellm.aembedding(
            model=self._model,
            input=[text],
            dimensions=settings.embedding_dimensions,
        )
        return list(response.data[0]["embedding"])


def _format_report_answer(report: DecisionReport) -> str:
    """Combine recommendation and summary for RAGAS answer scoring."""
    return f"{report.recommendation}\n\n{report.summary}".strip()


def compute_quality_score(
    faithfulness: float | None,
    answer_relevancy: float | None,
    context_precision: float | None,
) -> float | None:
    """
    Compute weighted quality_score from individual RAGAS metrics.

    Renormalizes weights when context_precision is None (no RAG contexts).
    Returns None when no metric values are available.
    """
    components: list[tuple[float, float]] = []
    if faithfulness is not None:
        components.append((faithfulness, FAITHFULNESS_WEIGHT))
    if answer_relevancy is not None:
        components.append((answer_relevancy, ANSWER_RELEVANCY_WEIGHT))
    if context_precision is not None:
        components.append((context_precision, CONTEXT_PRECISION_WEIGHT))

    if not components:
        return None

    total_weight = sum(weight for _, weight in components)
    if total_weight <= 0:
        return None

    return sum(value * weight for value, weight in components) / total_weight


@lru_cache
def _get_ragas_wrappers() -> tuple[Any, Any]:
    """Build cached RAGAS LLM and embedding wrappers."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    settings = get_settings()
    llm = LangchainLLMWrapper(ChatLiteLLM(model=settings.eval_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        _LiteLLMEmbeddings(settings.embedding_model, settings.embedding_dimensions),
    )
    return llm, embeddings


async def _score_metric(metric: Any, sample: Any) -> float | None:
    """Score a single RAGAS metric; return None on failure."""
    try:
        result = await metric.single_turn_ascore(sample)
    except Exception as exc:
        log.warning(
            "ragas_metric_failed",
            metric=getattr(metric, "name", type(metric).__name__),
            error=str(exc),
        )
        return None

    if result is None:
        return None
    return float(result)


async def evaluate_report(
    report: DecisionReport,
    contexts: list[str],
    *,
    reference: str | None = None,
) -> dict[str, float | None]:
    """
    Evaluate a DecisionReport with RAGAS metrics.

    Args:
        report: Completed pipeline report to score.
        contexts: Raw retrieved chunk contents from hybrid_search.
        reference: Optional ground-truth text for ContextPrecision (golden dataset).

    Returns:
        Dict with faithfulness, answer_relevancy, context_precision, quality_score.
    """
    from ragas import SingleTurnSample
    from ragas.metrics import ContextPrecision as ContextPrecisionWithReference
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )

    settings = get_settings()
    llm, embeddings = _get_ragas_wrappers()

    answer = _format_report_answer(report)
    sample = SingleTurnSample(
        user_input=report.query,
        response=answer,
        retrieved_contexts=contexts or [],
        reference=reference,
    )

    faithfulness_metric = Faithfulness(llm=llm)
    relevancy_metric = ResponseRelevancy(llm=llm, embeddings=embeddings)

    faithfulness, answer_relevancy = await asyncio.gather(
        _score_metric(faithfulness_metric, sample),
        _score_metric(relevancy_metric, sample),
    )

    context_precision: float | None = None
    if contexts:
        if reference:
            context_precision = await _score_metric(
                ContextPrecisionWithReference(llm=llm),
                sample,
            )
        else:
            context_precision = await _score_metric(
                LLMContextPrecisionWithoutReference(llm=llm),
                sample,
            )
    else:
        log.info("evaluation_skipped_context_precision", reason="no_retrieved_contexts")

    quality_score = compute_quality_score(faithfulness, answer_relevancy, context_precision)

    thresholds = {
        "faithfulness": settings.eval_faithfulness_threshold,
        "answer_relevancy": settings.eval_answer_relevancy_threshold,
        "context_precision": settings.eval_context_precision_threshold,
    }
    for metric_name, value in (
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
        ("context_precision", context_precision),
    ):
        if value is not None and value < thresholds[metric_name]:
            log.warning(
                "evaluation_below_threshold",
                metric=metric_name,
                value=value,
                threshold=thresholds[metric_name],
                request_id=report.request_id,
            )

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "quality_score": quality_score,
    }
