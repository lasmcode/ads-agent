# src/ads_agent/infrastructure/llm/schemas.py
"""Pydantic schemas for LLM structured output via function-calling."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ads_agent.core.entities.decision_report import RecommendationStrength, TradeOff


class SupervisorDecision(BaseModel):
    """Routing decision returned by the supervisor LLM."""

    next_agent: Literal["research", "analysis", "writer", "FINISH"] = Field(
        description="The next agent to invoke, or FINISH to end the pipeline.",
    )


class AnalysisOutput(BaseModel):
    """Structured trade-off analysis from the analysis agent."""

    trade_offs: list[TradeOff] = Field(
        description="3-6 trade-offs comparing the options across key dimensions.",
    )

    @field_validator("trade_offs")
    @classmethod
    def validate_trade_off_count(cls, value: list[TradeOff]) -> list[TradeOff]:
        count = len(value)
        if count < 3:
            msg = f"Analysis must include at least 3 trade-offs, got {count}"
            raise ValueError(msg)
        if count > 6:
            msg = f"Analysis must include at most 6 trade-offs, got {count}"
            raise ValueError(msg)
        return value


class WriterDraft(BaseModel):
    """Narrative fields for the final DecisionReport (sources assigned programmatically)."""

    recommendation: str = Field(description="Clear, direct recommendation in 1-3 sentences.")
    recommendation_strength: RecommendationStrength = Field(
        default=RecommendationStrength.MODERATE,
        description="Confidence level of the recommendation.",
    )
    summary: str = Field(description="2-3 paragraph analysis of the decision space.")
    key_considerations: list[str] = Field(
        default_factory=list,
        description="Bullet points of critical factors to consider.",
    )
    when_to_choose_alternative: str | None = Field(
        default=None,
        description="Conditions under which the non-recommended option is better.",
    )
