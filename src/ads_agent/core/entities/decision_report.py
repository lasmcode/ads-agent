# src/ads_agent/core/entities/decision_report.py
"""
Domain entity: DecisionReport
The primary output of the agent pipeline — structured analysis with recommendation.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class RecommendationStrength(StrEnum):
    """Confidence level of the agent's recommendation."""

    STRONG = "strong"  # Clear winner — recommend without hesitation
    MODERATE = "moderate"  # Lean toward one option with caveats
    CONDITIONAL = "conditional"  # Depends heavily on context
    INCONCLUSIVE = "inconclusive"  # Insufficient data to decide


class TradeOff(BaseModel):
    """A single trade-off between two options."""

    dimension: str = Field(description="e.g. 'Performance', 'Operational Complexity'")
    option_a: str = Field(description="How option A scores on this dimension")
    option_b: str = Field(description="How option B scores on this dimension")
    winner: str | None = Field(default=None, description="Which option wins, if clear")


class DecisionReport(BaseModel):
    """
    Structured output of the full agent analysis pipeline.
    Designed to be rendered as a document, not just a string.
    """

    request_id: str
    query: str = Field(description="Original user question — for context")

    # Executive summary
    recommendation: str = Field(description="Clear, direct recommendation in 1-3 sentences")
    recommendation_strength: RecommendationStrength = RecommendationStrength.MODERATE

    # Structured analysis
    summary: str = Field(description="2-3 paragraph analysis of the decision space")
    trade_offs: list[TradeOff] = Field(
        default_factory=list,
        description="Structured comparison across key dimensions",
    )
    key_considerations: list[str] = Field(
        default_factory=list,
        description="Bullet points of critical factors to consider",
    )
    when_to_choose_alternative: str | None = Field(
        default=None,
        description="Conditions under which the non-recommended option is better",
    )

    # Evidence trail
    sources: list[str] = Field(
        default_factory=list,
        description="URLs and references used in the analysis",
    )

    # Quality signal — populated by Evaluation Engine in Phase 6
    quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="0.0-1.0 quality score from evaluation engine",
    )
