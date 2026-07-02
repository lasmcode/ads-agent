# src/ads_agent/infrastructure/asyncio_compat.py
"""Windows-compatible asyncio.run helper for psycopg async I/O."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
import sys
from typing import Any

type _Coroutine[T] = Coroutine[Any, Any, T]


def run[T](coro: _Coroutine[T], *, debug: bool | None = None) -> T:
    """
    Run *coro* to completion.

    On Windows, psycopg's async driver requires a selector-based event loop;
    the OS default (ProactorEventLoop) raises InterfaceError on connect().
    Python 3.14 deprecates global ``set_event_loop_policy`` — pass
    ``loop_factory=`` to ``asyncio.run`` instead.
    """
    if sys.platform == "win32":
        return asyncio.run(coro, debug=debug, loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(coro, debug=debug)
