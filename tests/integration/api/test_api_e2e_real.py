# tests/integration/api/test_api_e2e_real.py
"""
Manual E2E test with real Gemini LLM — costs tokens.

Run only when explicitly requested:
  GEMINI_API_KEY=... uv run pytest tests/integration/api/test_api_e2e_real.py -m "integration and e2e" -v
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_api_e2e_real_llm(postgres_client) -> None:
    """POST /decisions with real pipeline — manual only, consumes Gemini quota."""
    response = await postgres_client.post(
        "/api/v1/decisions",
        json={
            "query": "Should I use Redis or Memcached for session storage?",
            "context": "Small startup, 5k concurrent users",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["report"] is not None
    assert data["report"]["recommendation"]
    assert data["receipt"]["total_tokens"] > 0

    request_id = data["request_id"]
    receipt_resp = await postgres_client.get(f"/api/v1/decisions/{request_id}/receipt")
    assert receipt_resp.status_code == 200
