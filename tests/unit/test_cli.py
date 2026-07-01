# tests/unit/test_cli.py
"""Unit tests for the CLI entry point."""

from __future__ import annotations

import json

import pytest

from ads_agent.cli import main


@pytest.mark.unit
def test_cli_run_text_output(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI run command prints a readable report and receipt summary."""
    exit_code = main(
        [
            "run",
            "Should I use pgvector or Qdrant for my RAG system?",
            "--output",
            "text",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "=== ADS Agent — Decision Report ===" in captured.out
    assert "Recommendation:" in captured.out
    assert "=== Execution Receipt ===" in captured.out
    assert "Request ID:" in captured.out


@pytest.mark.unit
def test_cli_run_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI run command supports JSON output."""
    exit_code = main(
        [
            "run",
            "Should I use pgvector or Qdrant for my RAG system?",
            "--output",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    json_start = captured.out.find("{")
    assert json_start != -1
    payload = json.loads(captured.out[json_start:])
    assert payload["report"] is not None
    assert payload["error"] is None
    assert "request_id" in payload["receipt"]
