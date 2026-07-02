# src/ads_agent/infrastructure/mcp/client.py
"""LangChain MCP client — loads ADS Agent tools for LangGraph agents."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from langchain_mcp_adapters.client import MultiServerMCPClient
import structlog

from ads_agent.core.settings import get_settings

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

log = structlog.get_logger(__name__)


def _mcp_subprocess_env() -> dict[str, str]:
    """
    Environment for the MCP stdio subprocess.

    Suppresses FastMCP banner/logging (stdout is the MCP wire protocol on stdio)
    and forces UTF-8 on Windows to avoid encoding issues in Git Bash/PowerShell.
    """
    env = {k: v for k, v in os.environ.items() if isinstance(v, str)}
    env["FASTMCP_SHOW_SERVER_BANNER"] = "false"
    env["FASTMCP_LOG_ENABLED"] = "false"
    env["FASTMCP_CHECK_FOR_UPDATES"] = "off"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _build_server_config() -> dict:
    """Build MultiServerMCPClient config from application settings."""
    settings = get_settings()

    if settings.mcp_transport == "streamable-http":
        url = f"http://{settings.mcp_http_host}:{settings.mcp_http_port}/mcp"
        return {
            "ads-agent": {
                "transport": "http",
                "url": url,
            }
        }

    return {
        "ads-agent": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "ads_agent.infrastructure.mcp.server"],
            "env": _mcp_subprocess_env(),
        }
    }


async def get_mcp_tools() -> list[BaseTool]:
    """
    Load MCP tools from the ADS Agent FastMCP server as LangChain tools.

    Uses stdio transport locally (spawns server subprocess) or HTTP when configured.
    """
    config = _build_server_config()
    log.info("mcp_client_loading_tools", transport=get_settings().mcp_transport)

    client = MultiServerMCPClient(config)
    tools = await client.get_tools()

    log.info("mcp_client_tools_loaded", tool_count=len(tools), tool_names=[t.name for t in tools])
    return tools
