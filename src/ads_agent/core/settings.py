# src/ads_agent/core/settings.py
"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Central configuration for the ADS Agent pipeline."""

    max_iterations: int = Field(
        default=5,
        ge=1,
        description="Maximum supervisor iterations before the circuit breaker triggers",
    )
    log_level: str = Field(default="INFO", description="Logging level for structlog")

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
