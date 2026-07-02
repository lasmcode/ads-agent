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
    fetch_url_max_chars: int = Field(
        default=12_000,
        ge=500,
        le=100_000,
        description="Maximum characters returned by fetch_url after extraction",
    )
    fetch_url_max_response_bytes: int = Field(
        default=2_000_000,
        ge=10_000,
        le=20_000_000,
        description="Maximum HTTP response size in bytes for fetch_url downloads",
    )
    tech_docs_max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum Tavily results for search_tech_docs",
    )

    # --- RAG / vector store (Phase 3) ---
    database_url: str = Field(
        default="postgresql://adsagent:adsagent@localhost:5432/adsagent",
        description=(
            "PostgreSQL connection string (psycopg conninfo format). "
            "Override via ADS_DATABASE_URL in .env — must match POSTGRES_* "
            "when using local docker-compose."
        ),
    )
    embedding_model: str = Field(
        default="gemini/gemini-embedding-001",
        description=(
            "LiteLLM embedding model identifier. NOTE: 'gemini/text-embedding-004' was "
            "the originally specified model but Google decommissioned it (returns HTTP 404 "
            "as of this phase) — 'gemini-embedding-001' is its verified, supported successor."
        ),
    )
    embedding_dimensions: int = Field(
        default=768,
        ge=1,
        le=3072,
        description="Output vector width requested from the embedding model; must match "
        "the pgvector column width (vector(768)) in schema.sql",
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks hybrid_search returns to the research agent",
    )
    rag_score_threshold: float = Field(
        default=0.02,
        ge=0.0,
        description=(
            "Minimum fused RRF score for a chunk to be treated as high-confidence "
            "context. With the standard RRF constant k=60, a chunk ranked #1 in a "
            "single retrieval list scores 1/61≈0.0164; ranked #1 in BOTH lists scores "
            "2/61≈0.0328. 0.02 requires a chunk to rank near the top of at least one "
            "list — retune against real corpus relevance judgments, not intuition."
        ),
    )
    rag_chunk_min_tokens: int = Field(
        default=256,
        ge=1,
        description="Lower bound for chunk size (in tokens) produced by the chunker",
    )
    rag_chunk_max_tokens: int = Field(
        default=512,
        ge=1,
        description="Upper bound for chunk size (in tokens) produced by the chunker",
    )
    rag_chunk_overlap_tokens: int = Field(
        default=50,
        ge=0,
        description="Token overlap between consecutive chunks, preserves cross-boundary context",
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
