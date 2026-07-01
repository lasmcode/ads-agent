# src/ads_agent/infrastructure/mcp/logging_config.py
"""Logging setup for the MCP server subprocess."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_stdio_logging() -> None:
    """
    Route all MCP server logs to stderr.

    In stdio transport, stdout is reserved exclusively for the MCP JSON-RPC
    wire protocol. Any structlog output to stdout breaks tool calls.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=logging.WARNING,
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
