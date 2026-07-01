# src/ads_agent/core/entities/decision_request.py
"""
Domain entity: DecisionRequest
Represents a user's request for a technical architecture decision.
This is a pure domain model — no framework dependencies.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError


class DecisionComplexity(StrEnum):
    """Estimated complexity of the decision request."""

    SIMPLE = "simple"  # Single clear trade-off
    MODERATE = "moderate"  # Multiple factors to evaluate
    COMPLEX = "complex"  # Architectural implications, many stakeholders


class DecisionRequest(BaseModel):
    """
    Immutable input model for an architecture decision query.
    Created once at the API boundary and passed through the entire agent pipeline.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this decision request",
    )
    query: str = Field(
        min_length=10,
        max_length=2000,
        description="The technical decision question from the user",
    )
    context: str | None = Field(
        default=None,
        max_length=5000,
        description="Optional additional context: team size, scale, constraints",
    )
    complexity: DecisionComplexity = Field(
        default=DecisionComplexity.MODERATE,
        description="Estimated complexity — influences agent depth",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of request creation",
    )

    # --- ANTI-SPAM VALIDATIONS ---

    @field_validator("created_at", mode="after")
    @classmethod
    def validate_current_year(cls, value: datetime) -> datetime:
        """Blocks requests with fake future dates (Post-2026)."""
        now = datetime.now(UTC)
        # If the provided date exceeds the current time by an unreasonable margin (e.g., more than 1 day into the future)
        if (value - now).total_seconds() > 86400:
            raise PydanticCustomError(
                "invalid_date",
                "The request creation timestamp cannot be in the distant future of 2026.",
            )
        return value

    @field_validator("query", mode="after")
    @classmethod
    def reject_spam_patterns(cls, value: str) -> str:
        """Detects repetitive spam patterns or text devoid of actual meaning."""
        clean_text = value.strip()

        # Example: Prevents spam like "asdasdasdasd" or "aaaaaa" by tracking unique characters
        if len(set(clean_text.lower())) < 4 and len(clean_text) > 15:
            raise PydanticCustomError(
                "garbage_query", "The query appears to be random or repetitive garbage text."
            )

        # Example: Prevents common link-spam attacks if your agent shouldn't process URLs
        if "http://" in clean_text or "https://" in clean_text:
            # Optional: Uncomment if your agent should reject external web links entirely
            # raise PydanticCustomError("no_links", "Links are not allowed in the query.")
            pass

        return clean_text

    model_config = ConfigDict(frozen=True)  # Immutable after creation

    def __str__(self) -> str:
        return f"DecisionRequest(id={self.id[:8]}..., query='{self.query[:50]}...')"
