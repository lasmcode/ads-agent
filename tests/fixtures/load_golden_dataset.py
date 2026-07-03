# tests/fixtures/load_golden_dataset.py
"""Load the versioned golden evaluation dataset."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from pydantic import BaseModel, Field

_GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


class GoldenEntry(BaseModel):
    """Single golden dataset question with expected direction and keywords."""

    id: str
    query: str
    expected_recommendation_direction: str
    expected_key_considerations: list[str] = Field(default_factory=list)


class GoldenDataset(BaseModel):
    """Versioned golden dataset container."""

    version: str
    entries: list[GoldenEntry]


@lru_cache
def load_golden_dataset() -> list[GoldenEntry]:
    """Return all golden dataset entries."""
    raw = json.loads(_GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    return GoldenDataset.model_validate(raw).entries


def load_golden_subset(entry_ids: list[str]) -> list[GoldenEntry]:
    """Return golden entries matching the given ids."""
    by_id = {entry.id: entry for entry in load_golden_dataset()}
    return [by_id[entry_id] for entry_id in entry_ids if entry_id in by_id]


def build_reference_text(entry: GoldenEntry) -> str:
    """Compose a reference string for RAGAS ContextPrecision from golden metadata."""
    keywords = ", ".join(entry.expected_key_considerations)
    return f"{entry.expected_recommendation_direction}. Key considerations: {keywords}"
