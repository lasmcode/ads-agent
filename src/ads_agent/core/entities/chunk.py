# src/ads_agent/core/entities/chunk.py
"""
Domain entity: Chunk
A retrieved fragment of ingested knowledge (a slice of a source document).
This is a pure domain model — no psycopg, litellm, or SQL leaks into this layer.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """
    A single unit of retrieved knowledge returned by a VectorStorePort.

    Instances are produced by adapters (e.g. the pgvector-backed retriever)
    and consumed by agents — the domain layer never depends on how the
    chunk was scored or stored.
    """

    id: str = Field(description="Stable identifier of the chunk (adapter-defined, e.g. a UUID)")
    source_url: str = Field(description="URL of the document this chunk was extracted from")
    title: str = Field(default="", description="Section/document title, if available")
    content: str = Field(description="The chunk's text content")
    score: float = Field(
        default=0.0,
        description=(
            "Relevance score in the adapter's own scale. For hybrid_search "
            "results this is the fused Reciprocal Rank Fusion score."
        ),
    )
    metadata: dict = Field(default_factory=dict, description="Adapter-specific extra metadata")

    model_config = ConfigDict(frozen=True)

    def __str__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Chunk(source={self.source_url}, score={self.score:.4f}, content='{preview}...')"
