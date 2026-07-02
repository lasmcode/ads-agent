"""Central configuration for search_tech_docs sources and domains."""

from __future__ import annotations

from typing import Literal, get_args

TechDocSource = Literal["langgraph", "mcp", "postgres", "fastapi", "oracle"]

TECH_DOC_SOURCE_DOMAINS: dict[TechDocSource, list[str]] = {
    "langgraph": ["langchain.com", "docs.langchain.com"],
    "mcp": ["modelcontextprotocol.io"],
    "postgres": ["postgresql.org"],
    "fastapi": ["fastapi.tiangolo.com"],
    "oracle": ["oracle.com", "docs.oracle.com"],
}

if set(TECH_DOC_SOURCE_DOMAINS.keys()) != set(get_args(TechDocSource)):
    msg = "TECH_DOC_SOURCE_DOMAINS keys must match TechDocSource literal values"
    raise ValueError(msg)


def tech_doc_sources() -> tuple[TechDocSource, ...]:
    """Return supported technical documentation source identifiers."""
    return get_args(TechDocSource)


def tech_doc_sources_description(*, separator: str = ", ") -> str:
    """Human-readable list of supported sources for tool and prompt descriptions."""
    return separator.join(tech_doc_sources())


def tech_doc_domains(source: TechDocSource) -> list[str]:
    """Return allowed search domains for a documentation source."""
    return list(TECH_DOC_SOURCE_DOMAINS[source])
