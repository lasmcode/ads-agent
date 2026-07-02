# src/ads_agent/agents/research/nodes.py
"""
Research Agent node — Phase 2: MCP-backed ReAct agent.

Uses create_react_agent with MCP tools (web search, doc retrieval, URL fetch).
The function signature and return contract remain unchanged across phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_litellm import ChatLiteLLM
from langgraph.prebuilt import create_react_agent
import structlog

from ads_agent.agents.common import safe_node
from ads_agent.agents.research.prompts import RESEARCH_SYSTEM_PROMPT
from ads_agent.core.entities.execution_receipt import AgentMetrics, AgentStatus
from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.mcp.client import get_mcp_tools

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from ads_agent.agents.state import AgentState

log = structlog.get_logger(__name__)

_URL_PATTERN = re.compile(r"https?://[^\s\)\]>\"']+")


@dataclass
class ResearchAgentResult:
    """Output from the research ReAct agent invocation."""

    output: str
    input_tokens: int
    output_tokens: int
    source_urls: list[str]


def _extract_urls_from_messages(messages: list[BaseMessage]) -> list[str]:
    """Collect unique URLs from tool outputs and AI messages."""
    found: list[str] = []
    seen: set[str] = set()

    for message in messages:
        content = message.content
        if isinstance(content, str):
            texts = [content]
        elif isinstance(content, list):
            texts = [
                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                for block in content
            ]
        else:
            texts = [str(content)]

        for text in texts:
            for match in _URL_PATTERN.findall(text):
                url = match.rstrip(".,;:")
                if url not in seen:
                    seen.add(url)
                    found.append(url)

    return found


def _extract_token_usage(messages: list[BaseMessage]) -> tuple[int, int]:
    """Sum input/output tokens from AI message metadata."""
    input_tokens = 0
    output_tokens = 0

    for message in messages:
        if not isinstance(message, AIMessage):
            continue

        usage = message.usage_metadata
        if usage:
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            continue

        meta = message.response_metadata or {}
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        if token_usage:
            input_tokens += int(
                token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
            )
            output_tokens += int(
                token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
            )

    if input_tokens == 0 and output_tokens == 0:
        log.warning("research_token_usage_unavailable")

    return input_tokens, output_tokens


async def run_research_agent(
    query: str, tools: list[BaseTool] | None = None
) -> ResearchAgentResult:
    """
    Run the research ReAct agent with MCP tools.

    Extracted for testability — graph/CLI unit tests mock this function.
    """
    settings = get_settings()
    mcp_tools = tools if tools is not None else await get_mcp_tools()

    model = ChatLiteLLM(model=settings.research_model, temperature=0)
    agent = create_react_agent(model, mcp_tools, prompt=RESEARCH_SYSTEM_PROMPT)

    result: dict[str, Any] = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
    )

    messages: list[BaseMessage] = result.get("messages") or []
    if not messages:
        msg = "Research agent returned no messages"
        raise RuntimeError(msg)

    final_message = messages[-1]
    if isinstance(final_message.content, str):
        output = final_message.content
    else:
        output = str(final_message.content)

    input_tokens, output_tokens = _extract_token_usage(messages)
    source_urls = _extract_urls_from_messages(messages)

    return ResearchAgentResult(
        output=output,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        source_urls=source_urls,
    )


@safe_node("research")
async def research_node(state: AgentState) -> dict:
    """
    Research Agent: gathers evidence for the technical decision.

    Phase 2: ReAct agent with MCP tools (web search, docs, URL fetch).
    """
    log.info("research_node_started", request_id=state["request"].id)

    started_at = datetime.now(UTC)
    query = state["request"].query

    agent_result = await run_research_agent(query)
    research_output = agent_result.output

    completed_at = datetime.now(UTC)

    metrics = AgentMetrics(
        agent_name="research",
        status=AgentStatus.COMPLETED,
        started_at=started_at,
        completed_at=completed_at,
        input_tokens=agent_result.input_tokens,
        output_tokens=agent_result.output_tokens,
    )

    receipt = state.get("receipt")
    if receipt:
        receipt.add_agent_metrics(metrics)
        if agent_result.source_urls:
            receipt.add_consulted_sources(agent_result.source_urls)

    log.info(
        "research_node_completed",
        duration_s=metrics.duration_seconds,
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        sources=len(agent_result.source_urls),
    )

    return {
        "research_output": research_output,
        "messages": [AIMessage(content=research_output, name="research")],
        "receipt": receipt,
    }
