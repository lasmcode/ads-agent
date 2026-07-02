# tests/unit/infrastructure/test_llm_client.py
"""Unit tests for the centralized LiteLLM client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.exceptions import RateLimitError
import pytest

from ads_agent.core.entities.execution_receipt import ExecutionReceipt
from ads_agent.infrastructure.llm.client import (
    LLMCompletionResult,
    _inline_json_schema_refs,
    _is_gemini_model,
    accumulate_token_cost,
    complete,
    record_llm_usage,
)
from ads_agent.infrastructure.llm.schemas import AnalysisOutput, SupervisorDecision


def _make_json_content_response(
    payload: str,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    response = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = []
    response.choices[0].message.content = payload
    return response


@pytest.mark.unit
@pytest.mark.asyncio
async def test_complete_parses_structured_output() -> None:
    with (
        patch(
            "ads_agent.infrastructure.llm.client._acompletion_with_retry",
            new_callable=AsyncMock,
            return_value=_make_json_content_response('{"next_agent": "analysis"}'),
        ),
        patch(
            "ads_agent.infrastructure.llm.client._extract_cost",
            return_value=0.002,
        ),
    ):
        result = await complete(
            [{"role": "user", "content": "route"}],
            "gemini/gemini-2.5-flash",
            response_model=SupervisorDecision,
        )

    assert isinstance(result.parsed, SupervisorDecision)
    assert result.parsed.next_agent == "analysis"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.estimated_cost_usd == 0.002


@pytest.mark.unit
def test_inline_json_schema_resolves_nested_defs() -> None:
    schema = AnalysisOutput.model_json_schema()
    assert "$defs" in schema

    inlined = _inline_json_schema_refs(schema)
    assert "$defs" not in inlined
    assert "$ref" not in json.dumps(inlined)


@pytest.mark.unit
def test_is_gemini_model() -> None:
    assert _is_gemini_model("gemini/gemini-2.5-flash") is True
    assert _is_gemini_model("gpt-4o") is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_complete_gemini_uses_json_schema_response_format() -> None:
    payload = json.dumps({"next_agent": "analysis"})
    with (
        patch(
            "ads_agent.infrastructure.llm.client._acompletion_with_retry",
            new_callable=AsyncMock,
            return_value=_make_json_content_response(payload),
        ) as mock_completion,
        patch(
            "ads_agent.infrastructure.llm.client._extract_cost",
            return_value=0.002,
        ),
    ):
        result = await complete(
            [{"role": "user", "content": "route"}],
            "gemini/gemini-2.5-flash",
            response_model=SupervisorDecision,
        )

    call_kwargs = mock_completion.await_args.kwargs
    assert "response_format" in call_kwargs
    assert "tools" not in call_kwargs
    assert isinstance(result.parsed, SupervisorDecision)
    assert result.parsed.next_agent == "analysis"


@pytest.mark.unit
def test_record_llm_usage_accumulates_cost() -> None:
    receipt = ExecutionReceipt(request_id="req-1")
    record_llm_usage(
        receipt,
        LLMCompletionResult(
            parsed=None,
            raw_content=None,
            input_tokens=1,
            output_tokens=1,
            estimated_cost_usd=0.003,
            model="test",
        ),
    )
    record_llm_usage(
        receipt,
        LLMCompletionResult(
            parsed=None,
            raw_content=None,
            input_tokens=1,
            output_tokens=1,
            estimated_cost_usd=0.002,
            model="test",
        ),
    )
    assert receipt.estimated_cost_usd == pytest.approx(0.005)


@pytest.mark.unit
def test_accumulate_token_cost_updates_receipt() -> None:
    receipt = ExecutionReceipt(request_id="req-1")
    with patch(
        "ads_agent.infrastructure.llm.client.estimate_token_cost",
        return_value=0.001,
    ):
        cost = accumulate_token_cost(receipt, "gemini/gemini-2.5-flash", 100, 50)
    assert cost == 0.001
    assert receipt.estimated_cost_usd == 0.001


@pytest.mark.unit
@pytest.mark.asyncio
async def test_complete_retries_on_rate_limit() -> None:
    success = _make_json_content_response('{"next_agent": "analysis"}')
    with (
        patch(
            "ads_agent.infrastructure.llm.client.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=[RateLimitError("429 rate limit", "openai", "gpt-4"), success],
        ),
        patch(
            "ads_agent.infrastructure.llm.client._extract_cost",
            return_value=0.0,
        ),
    ):
        result = await complete(
            [{"role": "user", "content": "route"}],
            "gemini/gemini-2.5-flash",
            response_model=SupervisorDecision,
        )

    assert isinstance(result.parsed, SupervisorDecision)
