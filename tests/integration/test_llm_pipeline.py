# tests/integration/test_llm_pipeline.py
"""
Optional integration tests for Phase 4 LLM agents.

Requires GEMINI_API_KEY in the environment. Skipped when absent.
"""

from __future__ import annotations

import os

import pytest

from ads_agent.agents.analysis.nodes import run_analysis_agent
from ads_agent.agents.writer.nodes import run_writer_agent
from ads_agent.infrastructure.llm.schemas import AnalysisOutput


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_real_llm_analysis_and_writer() -> None:
    research_output = (
        "pgvector integrates with PostgreSQL and is suitable for moderate vector workloads. "
        "Qdrant is a dedicated vector database with strong performance at scale. "
        "Sources: https://example.com/pgvector and https://example.com/qdrant"
    )
    query = "Should I use pgvector or Qdrant?"

    analysis, analysis_result = await run_analysis_agent(query, research_output)
    assert isinstance(analysis, AnalysisOutput)
    assert 3 <= len(analysis.trade_offs) <= 6
    assert analysis_result.input_tokens > 0 or analysis_result.output_tokens > 0

    draft, writer_result = await run_writer_agent(query, research_output, analysis)
    assert draft.recommendation
    assert draft.summary
    assert writer_result.input_tokens > 0 or writer_result.output_tokens > 0
