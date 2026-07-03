# tests/unit/api/test_schemas.py
"""Unit tests for API DTO validation."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from ads_agent.api.v1.schemas import CreateDecisionRequestBody


@pytest.mark.unit
def test_create_decision_body_valid() -> None:
    body = CreateDecisionRequestBody(
        query="Should I use Redis or Memcached for sessions?",
        context="Small team, moderate traffic",
    )
    assert body.query.startswith("Should")


@pytest.mark.unit
def test_create_decision_body_query_too_short() -> None:
    with pytest.raises(ValidationError):
        CreateDecisionRequestBody(query="too short")


@pytest.mark.unit
def test_create_decision_body_context_too_long() -> None:
    with pytest.raises(ValidationError):
        CreateDecisionRequestBody(
            query="Should I use Redis or Memcached for sessions?",
            context="x" * 5001,
        )
