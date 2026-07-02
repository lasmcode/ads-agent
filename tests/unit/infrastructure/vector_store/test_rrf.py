# tests/unit/infrastructure/vector_store/test_rrf.py
"""
Unit tests for Reciprocal Rank Fusion — pure math, no DB, no network.

RRF(d) = sum_i 1/(k + rank_i(d)) over every ranked list i containing d.
These tests verify the formula against hand-computed values (not just that
"a sensible-looking number comes out"), plus the merge/tie-breaking behavior
callers (hybrid_search) depend on.
"""

from __future__ import annotations

import pytest

from ads_agent.infrastructure.vector_store.retriever import reciprocal_rank_fusion


@pytest.mark.unit
class TestReciprocalRankFusion:
    def test_single_list_matches_hand_computed_formula(self) -> None:
        """RRF over one list is just 1/(k + rank) per document — verify exactly."""
        scores = reciprocal_rank_fusion(["a", "b", "c"], k=60)

        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 62)
        assert scores["c"] == pytest.approx(1 / 63)

    def test_document_in_both_lists_sums_both_contributions(self) -> None:
        """A doc ranked #1 in list A and #2 in list B: RRF = 1/(k+1) + 1/(k+2)."""
        scores = reciprocal_rank_fusion(["doc1", "doc2"], ["doc2", "doc1"], k=60)

        assert scores["doc1"] == pytest.approx(1 / 61 + 1 / 62)
        assert scores["doc2"] == pytest.approx(1 / 62 + 1 / 61)
        # Symmetric ranks (1st in one list, 2nd in the other) tie exactly.
        assert scores["doc1"] == pytest.approx(scores["doc2"])

    def test_document_absent_from_a_list_only_scores_the_list_it_is_in(self) -> None:
        """A doc missing from one list is not penalized beyond simply not scoring there."""
        scores = reciprocal_rank_fusion(["only_in_a", "shared"], ["shared", "only_in_b"], k=60)

        assert scores["only_in_a"] == pytest.approx(1 / 61)
        assert scores["only_in_b"] == pytest.approx(1 / 62)
        # "shared" is rank 2 in list A and rank 1 in list B.
        assert scores["shared"] == pytest.approx(1 / 62 + 1 / 61)
        assert scores["shared"] > scores["only_in_a"]
        assert scores["shared"] > scores["only_in_b"]

    def test_rank_one_in_both_lists_beats_rank_one_in_a_single_list(self) -> None:
        """This is the whole point of hybrid search: agreement across signals wins."""
        scores = reciprocal_rank_fusion(["consensus", "lexical_only"], ["consensus"], k=60)

        assert scores["consensus"] > scores["lexical_only"]
        assert scores["consensus"] == pytest.approx(2 / 61)
        assert scores["lexical_only"] == pytest.approx(1 / 62)

    def test_empty_lists_produce_empty_scores(self) -> None:
        assert reciprocal_rank_fusion([], [], k=60) == {}
        assert reciprocal_rank_fusion([], k=60) == {}

    def test_k_constant_dampens_high_rank_dominance(self) -> None:
        """A larger k shrinks every term, narrowing the gap between rank 1 and rank 2."""
        low_k = reciprocal_rank_fusion(["a", "b"], k=1)
        high_k = reciprocal_rank_fusion(["a", "b"], k=1000)

        low_k_gap = low_k["a"] - low_k["b"]
        high_k_gap = high_k["a"] - high_k["b"]
        assert low_k_gap > high_k_gap

    def test_default_k_is_the_de_facto_standard_sixty(self) -> None:
        """No explicit k should default to 60 — the Elasticsearch/Cormack-Clarke standard."""
        scores = reciprocal_rank_fusion(["only"])
        assert scores["only"] == pytest.approx(1 / 61)

    def test_ties_are_resolved_deterministically(self) -> None:
        """Two docs with a genuinely identical fused score keep a stable, reproducible order."""
        # "a" (rank 1, list A only) and "b" (rank 1, list B only) both score
        # exactly 1/61 — an honest tie, not just two different-but-close values.
        scores = reciprocal_rank_fusion(["a"], ["b"], k=60)
        assert scores["a"] == pytest.approx(scores["b"])

        order_first_run = sorted(scores, key=lambda d: (-scores[d], d))
        order_second_run = sorted(
            reciprocal_rank_fusion(["a"], ["b"], k=60),
            key=lambda d: (-scores[d], d),
        )
        assert order_first_run == order_second_run == ["a", "b"]
