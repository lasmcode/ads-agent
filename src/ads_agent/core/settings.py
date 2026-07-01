# src/ads_agent/core/settings.py
"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

McpTransport = Literal["stdio", "streamable-http"]


class AppSettings(BaseSettings):
    """Central configuration for the ADS Agent pipeline."""

    max_iterations: int = Field(
        default=5,
        ge=1,
        description="Maximum supervisor iterations before the circuit breaker triggers",
    )
    log_level: str = Field(default="INFO", description="Logging level for structlog")
    http_timeout: float = Field(
        default=15.0,
        gt=0,
        description="HTTP timeout in seconds for MCP tool I/O",
    )
    mcp_transport: McpTransport = Field(
        default="stdio",
        description="MCP server transport: stdio for local dev, streamable-http for deployment",
    )
    mcp_http_host: str = Field(
        default="127.0.0.1",
        description="Host to bind when MCP transport is streamable-http",
    )
    mcp_http_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port to bind when MCP transport is streamable-http",
    )
    research_model: str = Field(
        default="gemini/gemini-2.5-flash",
        description="LiteLLM model identifier for the research ReAct agent",
    )

    model_config = SettingsConfigDict(
        env_prefix="ADS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> AppSettings:
    """Return cached application settings."""
    return AppSettings()
