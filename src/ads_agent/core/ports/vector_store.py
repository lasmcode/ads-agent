# src/ads_agent/core/ports/vector_store.py
"""
Port: VectorStorePort

Defines the contract a knowledge-store adapter must fulfill, independent of
the underlying storage technology (pgvector, Qdrant, Pinecone, ...).

Clean Architecture rationale:
    The `agents/` layer and any future use-case code depend only on this
    ABC, never on `infrastructure.vector_store.*` directly. Swapping pgvector
    for Qdrant later means writing a new adapter that implements this port —
    zero changes to `research/nodes.py` or any other consumer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ads_agent.core.entities.chunk import Chunk


class VectorStorePort(ABC):
    """Abstract contract for a hybrid (lexical + semantic) knowledge store."""

    @abstractmethod
    async def setup_schema(self) -> None:
        """
        Create the underlying storage schema if it does not already exist
        (tables, indexes, extensions). Idempotent — safe to call on every
        application startup.
        """
        raise NotImplementedError

    @abstractmethod
    async def ingest(self, url: str) -> int:
        """
        Fetch, chunk, embed, and store a document at `url`.

        Idempotent: re-ingesting an unchanged URL must not create duplicate
        rows. Returns the number of chunks written (inserted or updated).
        """
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: str, top_k: int = 10) -> list[Chunk]:
        """
        Retrieve the `top_k` most relevant chunks for `query`.

        Adapters are free to combine multiple retrieval strategies (lexical,
        semantic, hybrid) internally — callers only see the final ranked list.
        """
        raise NotImplementedError
