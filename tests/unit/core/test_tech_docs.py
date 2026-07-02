# tests/unit/core/test_tech_docs.py
"""Tests for centralized technical documentation source configuration."""

from __future__ import annotations

from typing import get_args

import pytest

from ads_agent.core.tech_docs import (
    TECH_DOC_SOURCE_DOMAINS,
    TechDocSource,
    tech_doc_domains,
    tech_doc_sources,
    tech_doc_sources_description,
)


@pytest.mark.unit
class TestTechDocsConfig:
    def test_literal_matches_domain_registry(self) -> None:
        assert set(TECH_DOC_SOURCE_DOMAINS.keys()) == set(get_args(TechDocSource))

    def test_tech_doc_sources_returns_all_literals(self) -> None:
        assert set(tech_doc_sources()) == set(get_args(TechDocSource))

    def test_tech_doc_sources_description_lists_all_sources(self) -> None:
        description = tech_doc_sources_description()
        for source in tech_doc_sources():
            assert source in description

    def test_tech_doc_domains_returns_copy(self) -> None:
        domains = tech_doc_domains("fastapi")
        assert domains == ["fastapi.tiangolo.com"]
        domains.append("example.com")
        assert TECH_DOC_SOURCE_DOMAINS["fastapi"] == ["fastapi.tiangolo.com"]
