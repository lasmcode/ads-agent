# src/ads_agent/cli.py
"""Command-line interface for the ADS Agent pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

from ads_agent.agents.supervisor.graph import run_pipeline
from ads_agent.core.entities.decision_request import DecisionRequest
from ads_agent.infrastructure.asyncio_compat import run as run_async
from ads_agent.infrastructure.checkpointer import close_checkpointer_pool, get_postgres_checkpointer
from ads_agent.infrastructure.vector_store.connection import close_pool as close_vector_store_pool

if TYPE_CHECKING:
    from ads_agent.agents.state import AgentState
    from ads_agent.core.entities.execution_receipt import ExecutionReceipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ads-agent",
        description="Architecture Decision Support — multi-agent pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the decision pipeline for a query")
    run_parser.add_argument(
        "query",
        help="Technical decision question (min 10 characters)",
    )
    run_parser.add_argument(
        "--thread-id",
        default=None,
        help="Checkpoint thread ID for resumable runs",
    )
    run_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    return parser


def _format_text_output(
    final_state: AgentState,
    receipt: ExecutionReceipt,
) -> str:
    report = final_state.get("final_report")
    lines = [
        "=== ADS Agent — Decision Report ===",
        "",
    ]

    if report is not None:
        lines.extend(
            [
                f"Recommendation: {report.recommendation}",
                f"Strength: {report.recommendation_strength.value}",
                "",
                f"Summary: {report.summary}",
                "",
            ]
        )
    elif final_state.get("error"):
        lines.extend(
            [
                f"Pipeline error: {final_state['error']}",
                "",
            ]
        )

    summary = receipt.to_summary()
    lines.extend(
        [
            "=== Execution Receipt ===",
            f"Request ID:    {summary['request_id']}",
            f"Duration (s):  {summary['duration_s']}",
            f"Agents run:    {summary['agents_run']}",
            f"Total tokens:  {summary['total_tokens']}",
            f"Circuit break: {summary['circuit_breaker_triggered']}",
        ]
    )
    if summary.get("trace_id"):
        lines.append(f"Trace ID:      {summary['trace_id']}")
    return "\n".join(lines)


def _format_json_output(
    final_state: AgentState,
    receipt: ExecutionReceipt,
) -> str:
    report = final_state.get("final_report")
    payload = {
        "report": report.model_dump(mode="json") if report is not None else None,
        "error": final_state.get("error"),
        "receipt": receipt.to_summary(),
    }
    return json.dumps(payload, indent=2)


async def _run_pipeline(args: argparse.Namespace) -> int:
    request = DecisionRequest(query=args.query)

    try:
        checkpointer = await get_postgres_checkpointer()
        final_state, receipt = await run_pipeline(
            request=request,
            thread_id=args.thread_id,
            checkpointer=checkpointer,
        )
    finally:
        # A CLI invocation is a single process/one-shot run — release both
        # pools before exiting instead of leaving connections open until
        # the interpreter tears down (which would print pool warnings).
        await close_checkpointer_pool()
        await close_vector_store_pool()

    if args.output == "json":
        print(_format_json_output(final_state, receipt))
    else:
        print(_format_text_output(final_state, receipt))

    return 1 if final_state.get("error") else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_async(_run_pipeline(args))

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
