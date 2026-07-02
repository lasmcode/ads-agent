# tests/conftest.py
"""Shared pytest configuration for all test suites."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest import Config, Item


def pytest_asyncio_loop_factories(
    config: Config,
    item: Item,
) -> dict[str, asyncio.SelectorEventLoop] | None:
    """
    Use SelectorEventLoop on Windows so psycopg async tests can connect.

    Replaces the deprecated import-time ``set_event_loop_policy`` workaround
    that previously lived in connection.py / checkpointer.py.
    """
    if sys.platform == "win32":
        return {"default": asyncio.SelectorEventLoop}
    return None
