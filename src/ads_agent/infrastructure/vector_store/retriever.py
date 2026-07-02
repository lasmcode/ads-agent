# src/ads_agent/infrastructure/vector_store/retriever.py
"""
Hybrid search: full-text (BM25-style ts_rank_cd) + dense vector similarity,
fused with Reciprocal Rank Fusion (RRF).

Why hybrid instead of vector-only:
    Pure semantic search misses exact-term matches (API names, error codes,
    config keys) that lexical search catches, and vice versa — lexical
    search misses paraphrased/conceptual queries that embeddings catch.
    Verified market data (May 2026) shows hybrid search + RRF reaches 66.4%
    MRR vs 56.7% for semantic-only — a 9+ point improvement — which is why
    both retrieval paths run for every query instead of picking one based
    on heuristics.

RRF formula (Cormack & Clarke, 2009):
    RRF(d) = sum_i  1 / (k + rank_i(d))
    summed over every ranked list i that contains document d, where
    rank_i(d) is d's 1-based rank in that list and k is a damping constant
    (60 — the de facto standard, matching Elasticsearch's own RRF) that
    prevents a single list from dominating the fused order just by placing
    a document 1st. This is implemented exactly as stated, not approximated
    with e.g. a weighted average of normalized scores.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, cast

import litellm
import numpy as np
import structlog

from ads_agent.core.entities.chunk import Chunk
from ads_agent.core.ports.vector_store import VectorStorePort
from ads_agent.core.settings import get_settings
from ads_agent.infrastructure.vector_store.connection import get_pool, setup_schema
from ads_agent.infrastructure.vector_store.ingestion import ingest_document

log = structlog.get_logger(__name__)

_RRF_K = 60
# Oversample each retrieval list before fusing — RRF needs enough candidates
# per list to reorder correctly; capping at top_k directly would starve a
# document that ranks e.g. #2 lexically but #15 semantically.
_CANDIDATE_MULTIPLIER = 4
_MIN_CANDIDATES = 20

_FTS_SQL = """
    SELECT id, source_url, title, content, metadata
    FROM knowledge_chunks
    WHERE tsv @@ websearch_to_tsquery('english', %(query)s)
    ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %(query)s)) DESC
    LIMIT %(limit)s
"""

# `<=>` is pgvector's cosine distance operator — the same metric the HNSW
# index (vector_cosine_ops) was built with, so this query can use the index.
_VECTOR_SQL = """
    SELECT id, source_url, title, content, metadata
    FROM knowledge_chunks
    ORDER BY embedding <=> %(query_embedding)s
    LIMIT %(limit)s
"""


async def _embed_query(query: str) -> np.ndarray:
    """
    Embed the search query. Returned as a numpy array (not a plain list) so
    pgvector's registered psycopg adapter — set up in connection.py's pool
    `configure` hook — sends it as a `vector` literal; a plain Python list
    is adapted as a generic array and pgvector's `<=>` operator has no
    overload for `vector <=> float8[]`.
    """
    settings = get_settings()
    response = await litellm.aembedding(
        model=settings.embedding_model,
        input=[query],
        dimensions=settings.embedding_dimensions,
    )
    return np.array(response.data[0]["embedding"], dtype=np.float32)


async def _fts_search(query: str, limit: int) -> list[dict[str, Any]]:
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(_FTS_SQL, {"query": query, "limit": limit})
        return cast("list[dict[str, Any]]", await cur.fetchall())


async def _vector_search(query_embedding: np.ndarray, limit: int) -> list[dict[str, Any]]:
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(_VECTOR_SQL, {"query_embedding": query_embedding, "limit": limit})
        return cast("list[dict[str, Any]]", await cur.fetchall())


def reciprocal_rank_fusion(*ranked_id_lists: list[str], k: int = _RRF_K) -> dict[str, float]:
    """
    Fuse N ranked ID lists into a single relevance score per ID using RRF.

    RRF(d) = sum_i 1 / (k + rank_i(d)), summed over every list that contains d.
    A document absent from a list contributes 0 for that list — it is not
    penalized beyond simply not receiving that list's term.

    Exposed as a public, side-effect-free function (independent of the
    database) so its math can be unit-tested with synthetic ranked lists.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return dict(scores)


async def hybrid_search(query: str, top_k: int = 10) -> list[Chunk]:
    """
    Retrieve the top_k most relevant chunks for `query` using hybrid search.

    Runs full-text (lexical) and vector (semantic) search concurrently via
    asyncio.gather, then fuses their rankings with Reciprocal Rank Fusion.
    Returns an empty list if the knowledge base has no matching content —
    callers (e.g. the research agent) should treat that as "fall back to
    web search", not as an error.
    """
    if not query.strip():
        return []

    candidate_limit = max(top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES)
    query_embedding = await _embed_query(query)

    fts_rows, vector_rows = await asyncio.gather(
        _fts_search(query, candidate_limit),
        _vector_search(query_embedding, candidate_limit),
    )

    rows_by_id: dict[str, dict[str, Any]] = {
        str(row["id"]): row for row in (*fts_rows, *vector_rows)
    }
    fts_ids = [str(row["id"]) for row in fts_rows]
    vector_ids = [str(row["id"]) for row in vector_rows]

    fused_scores = reciprocal_rank_fusion(fts_ids, vector_ids)
    ranked_ids = sorted(fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True)

    results = [
        Chunk(
            id=doc_id,
            source_url=rows_by_id[doc_id]["source_url"],
            title=rows_by_id[doc_id]["title"],
            content=rows_by_id[doc_id]["content"],
            score=fused_scores[doc_id],
            metadata=rows_by_id[doc_id]["metadata"] or {},
        )
        for doc_id in ranked_ids[:top_k]
    ]

    log.info(
        "hybrid_search_completed",
        query_preview=query[:80],
        fts_candidates=len(fts_rows),
        vector_candidates=len(vector_rows),
        fused_results=len(results),
    )
    return results


class PgVectorStoreAdapter(VectorStorePort):
    """
    Concrete pgvector-backed implementation of VectorStorePort.

    Delegates to the module-level functions above — kept as free functions
    (rather than adapter-only methods) so unit tests can exercise the RRF
    math and the ingestion pipeline directly without instantiating this
    class. Migrating to Qdrant/Pinecone later means writing a new class
    here implementing the same three methods; no other layer changes.
    """

    async def setup_schema(self) -> None:
        await setup_schema()

    async def ingest(self, url: str) -> int:
        return await ingest_document(url)

    async def search(self, query: str, top_k: int = 10) -> list[Chunk]:
        return await hybrid_search(query, top_k=top_k)
