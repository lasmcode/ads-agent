# tests/unit/infrastructure/vector_store/test_chunker.py
"""
Unit tests for the structure-aware chunker — no DB, no network.

Covers: header-based splitting (with breadcrumb titles), token-size bounds
for both undersized (merged) and oversized (windowed) sections, and overlap
being carried between windows of an oversized section.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from ads_agent.infrastructure.vector_store.chunker import chunk_document

_CHARS_PER_TOKEN = 4  # mirrors chunker._CHARS_PER_TOKEN


def _paragraph(word: str, token_count: int) -> str:
    """Build a paragraph of roughly `token_count` tokens using `word` filler."""
    chars_needed = token_count * _CHARS_PER_TOKEN
    unit = f"{word} "
    repeats = max(1, chars_needed // len(unit))
    return (unit * repeats).strip()


@pytest.mark.unit
class TestHeaderSplitting:
    def test_splits_on_headers_with_breadcrumb_titles(self) -> None:
        text = (
            "# Persistence\n\n"
            "Intro paragraph about persistence.\n\n"
            "## Checkpointer vs. store\n\n"
            "Checkpointers persist thread-scoped state; stores persist cross-thread data.\n\n"
            "## Next steps\n\n"
            "Read the checkpointer guide.\n"
        )

        chunks = chunk_document(text, min_tokens=1, max_tokens=512, overlap_tokens=10)

        titles = [c.title for c in chunks]
        assert any(t.startswith("Persistence") for t in titles)
        assert any("Checkpointer vs. store" in t for t in titles)
        joined_content = "\n".join(c.content for c in chunks)
        assert "Checkpointers persist thread-scoped state" in joined_content
        assert "Read the checkpointer guide." in joined_content

    def test_nested_headers_build_breadcrumb_path(self) -> None:
        text = (
            "# Troubleshooting\n\n"
            "## PostgresSaver issues\n\n"
            "### thread_id too long\n\n"
            "Keep thread_id under 255 characters.\n"
        )

        chunks = chunk_document(text, min_tokens=1, max_tokens=512, overlap_tokens=10)

        assert any(
            c.title == "Troubleshooting > PostgresSaver issues > thread_id too long" for c in chunks
        )

    def test_text_with_no_headers_becomes_a_single_untitled_section(self) -> None:
        text = "Just a plain paragraph with no markdown headers at all."

        chunks = chunk_document(text, min_tokens=1, max_tokens=512, overlap_tokens=10)

        assert len(chunks) == 1
        assert chunks[0].title == ""
        assert chunks[0].content == text

    def test_empty_text_produces_no_chunks(self) -> None:
        assert chunk_document("", min_tokens=1, max_tokens=512, overlap_tokens=10) == []
        assert chunk_document("   \n\n  ", min_tokens=1, max_tokens=512, overlap_tokens=10) == []


@pytest.mark.unit
class TestTokenBounds:
    def test_merges_undersized_sibling_sections_up_to_min_tokens(self) -> None:
        """Several tiny subsections under one heading should combine into one chunk."""
        text = "# FAQ\n\n" + "".join(f"## Question {i}\n\nShort answer {i}.\n\n" for i in range(10))

        chunks = chunk_document(text, min_tokens=40, max_tokens=512, overlap_tokens=10)

        # All ten tiny answers share the same top-level "FAQ" heading, so they
        # should be merged into far fewer chunks than there are subsections.
        assert len(chunks) < 10
        assert all(len(c.content) // _CHARS_PER_TOKEN <= 512 for c in chunks)

    def test_does_not_merge_across_different_top_level_headings(self) -> None:
        """Sections under unrelated top-level headings must stay separate."""
        text = "# Topic A\n\nShort A content.\n\n# Topic B\n\nShort B content.\n"

        chunks = chunk_document(text, min_tokens=1000, max_tokens=2000, overlap_tokens=10)

        titles = {c.title for c in chunks}
        assert "Topic A" in titles
        assert "Topic B" in titles
        assert len(chunks) == 2

    def test_splits_oversized_section_into_windows_within_max_tokens(self) -> None:
        big_paragraphs = "\n\n".join(_paragraph(f"word{i}", 100) for i in range(10))
        text = f"# Big Section\n\n{big_paragraphs}\n"

        chunks = chunk_document(text, min_tokens=50, max_tokens=300, overlap_tokens=50)

        assert len(chunks) > 1
        for chunk in chunks:
            estimated_tokens = len(chunk.content) // _CHARS_PER_TOKEN
            # Small slack: paragraph packing stops *before* exceeding max_tokens,
            # but a single oversized paragraph can still land exactly at the edge.
            assert estimated_tokens <= 300 + 10

    def test_overlap_is_carried_between_windows(self) -> None:
        """The tail of one window should reappear at the head of the next."""
        paragraphs = [_paragraph(f"marker{i}", 80) for i in range(6)]
        text = "# Section\n\n" + "\n\n".join(paragraphs) + "\n"

        chunks = chunk_document(text, min_tokens=50, max_tokens=200, overlap_tokens=80)

        assert len(chunks) >= 2
        # The paragraph that ends window N (its tail marker) should also open
        # window N+1, proving the ~overlap_tokens carry-over logic ran.
        for earlier, later in pairwise(chunks):
            tail_marker = earlier.content.strip().split()[-2]
            assert tail_marker in later.content

    def test_single_paragraph_larger_than_max_tokens_is_sliced(self) -> None:
        """A pathological giant paragraph (e.g. one huge code block) still gets split."""
        giant_paragraph = _paragraph("code_token", 1000)
        text = f"# Big Code Block\n\n{giant_paragraph}\n"

        chunks = chunk_document(text, min_tokens=50, max_tokens=300, overlap_tokens=20)

        assert len(chunks) > 1
        for chunk in chunks:
            estimated_tokens = len(chunk.content) // _CHARS_PER_TOKEN
            assert estimated_tokens <= 300 + 5
