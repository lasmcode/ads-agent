# src/ads_agent/infrastructure/llm/client.py
"""Centralized async LiteLLM wrapper with usage tracking and retry."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

import litellm
from litellm.exceptions import RateLimitError
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ads_agent.core.entities.execution_receipt import ExecutionReceipt

log = structlog.get_logger(__name__)


@dataclass
class LLMCompletionResult:
    """Result of a single LiteLLM completion call."""

    parsed: BaseModel | None
    raw_content: str | None
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    model: str


def _is_rate_limit(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "resource exhausted" in message


def _is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()


def _inline_json_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve Pydantic $ref/$defs into a flat JSON Schema.

    Gemini rejects tool parameters that contain unresolved $ref pointers.
    """
    root = copy.deepcopy(schema)
    defs = root.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                if ref.startswith("#/$defs/"):
                    def_name = ref.removeprefix("#/$defs/")
                    if def_name not in defs:
                        msg = f"Unresolved JSON schema reference: {ref}"
                        raise ValueError(msg)
                    return resolve(copy.deepcopy(defs[def_name]))
                return node
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    resolved = resolve(root)
    if isinstance(resolved, dict):
        resolved.pop("$defs", None)
        resolved.pop("title", None)
    return resolved


def _pydantic_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    return _inline_json_schema_refs(model.model_json_schema())


def _pydantic_to_tool(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to an OpenAI-compatible function tool definition."""
    return {
        "type": "function",
        "function": {
            "name": model.__name__,
            "description": (model.__doc__ or f"Structured output: {model.__name__}").strip(),
            "parameters": _pydantic_json_schema(model),
        },
    }


def _pydantic_to_response_format(model: type[BaseModel]) -> dict[str, Any]:
    """Gemini-native structured output via JSON schema (no OpenAI-style tools)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": _pydantic_json_schema(model),
            "strict": True,
        },
    }


def _response_debug_summary(response: Any) -> str:
    try:
        if hasattr(response, "model_dump"):
            return json.dumps(response.model_dump(), default=str)[:2000]
        if isinstance(response, dict):
            return json.dumps(response, default=str)[:2000]
    except Exception:
        pass
    return repr(response)[:2000]


def _extract_response_message(response: Any) -> Any:
    """Return the first assistant message from a LiteLLM completion response."""
    choices = getattr(response, "choices", None) or []
    if choices:
        return choices[0].message

    msg = (
        "LLM returned no choices — likely schema rejection or provider error. "
        f"Response summary: {_response_debug_summary(response)}"
    )
    raise ValueError(msg)


def _parse_json_payload(raw: Any, response_model: type[BaseModel]) -> BaseModel:
    if isinstance(raw, dict):
        return response_model.model_validate(raw)
    if isinstance(raw, str) and raw.strip():
        return response_model.model_validate_json(raw)
    msg = f"No parseable structured payload for {response_model.__name__}"
    raise ValueError(msg)


def _parse_structured_output(response: Any, response_model: type[BaseModel]) -> BaseModel:
    """Parse structured output from tool calls or JSON content."""
    message = _extract_response_message(response)
    tool_calls = getattr(message, "tool_calls", None) or []

    if tool_calls:
        arguments = tool_calls[0].function.arguments
        return _parse_json_payload(arguments, response_model)

    content = getattr(message, "content", None)
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        content = "\n".join(text_parts)

    return _parse_json_payload(content, response_model)


def _extract_usage(response: Any) -> tuple[int, int]:
    """Extract input/output token counts from a LiteLLM response."""
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", None) or usage.get("prompt_tokens", 0)
        completion = getattr(usage, "completion_tokens", None) or usage.get("completion_tokens", 0)
        return int(prompt or 0), int(completion or 0)

    return 0, 0


def _extract_cost(response: Any, model: str) -> float:
    """Estimate USD cost for a completion using LiteLLM pricing tables."""
    try:
        cost = litellm.completion_cost(completion_response=response, model=model)
        return float(cost or 0.0)
    except Exception:
        log.debug("completion_cost_unavailable", model=model)
        return 0.0


def _parse_tool_call(response: Any, response_model: type[BaseModel]) -> BaseModel:
    """Parse structured output from an LLM response."""
    return _parse_structured_output(response, response_model)


def record_llm_usage(
    receipt: ExecutionReceipt | None,
    result: LLMCompletionResult,
) -> None:
    """Accumulate estimated cost on the receipt from an LLM call."""
    if receipt is None:
        return
    current = receipt.estimated_cost_usd or 0.0
    receipt.estimated_cost_usd = current + result.estimated_cost_usd


def estimate_token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token counts when no raw completion response exists."""
    if input_tokens == 0 and output_tokens == 0:
        return 0.0
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        return float(prompt_cost + completion_cost)
    except Exception:
        log.debug("token_cost_unavailable", model=model)
        return 0.0


def accumulate_token_cost(
    receipt: ExecutionReceipt | None,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate and accumulate token-based cost on the receipt."""
    cost = estimate_token_cost(model, input_tokens, output_tokens)
    if receipt is not None and cost > 0:
        current = receipt.estimated_cost_usd or 0.0
        receipt.estimated_cost_usd = current + cost
    return cost


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_rate_limit),
    reraise=True,
)
async def _acompletion_with_retry(**kwargs: Any) -> Any:
    return await litellm.acompletion(**kwargs)


async def complete(
    messages: list[dict[str, Any]],
    model: str,
    *,
    response_model: type[BaseModel] | None = None,
    receipt: ExecutionReceipt | None = None,
    agent_name: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> LLMCompletionResult:
    """
    Run an async LiteLLM completion with optional structured output via function-calling.

    When response_model is provided, forces a tool call matching the Pydantic schema.
    Automatically records estimated cost on the receipt when provided.
    """
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **kwargs,
    }

    if response_model is not None:
        if _is_gemini_model(model):
            call_kwargs["response_format"] = _pydantic_to_response_format(response_model)
        else:
            tool = _pydantic_to_tool(response_model)
            call_kwargs["tools"] = [tool]
            call_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": response_model.__name__},
            }

    log.debug(
        "llm_completion_start",
        model=model,
        agent=agent_name,
        structured=response_model is not None,
    )

    response = await _acompletion_with_retry(**call_kwargs)

    input_tokens, output_tokens = _extract_usage(response)
    estimated_cost = _extract_cost(response, model)

    parsed: BaseModel | None = None
    raw_content: str | None = None

    if response_model is not None:
        parsed = _parse_tool_call(response, response_model)
        raw_content = parsed.model_dump_json()
    else:
        message = _extract_response_message(response)
        raw_content = getattr(message, "content", None)

    result = LLMCompletionResult(
        parsed=parsed,
        raw_content=raw_content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost,
        model=model,
    )

    record_llm_usage(receipt, result)

    log.info(
        "llm_completion_done",
        model=model,
        agent=agent_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(estimated_cost, 6),
    )

    return result
