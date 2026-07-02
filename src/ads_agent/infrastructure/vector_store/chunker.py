# src/ads_agent/infrastructure/vector_store/chunker.py
"""
Structure-aware chunking for technical documentation.

Why structure-aware instead of fixed-size naive splitting:
    Our ingestion corpus is technical documentation (LangGraph, MCP,
    PostgreSQL docs, etc.) — content that is already organized by the
    authors into semantically coherent sections under markdown headers.
    Splitting on a fixed character/token window ignores those boundaries
    and routinely cuts a section in half, so a similarity search can match
    "half an explanation" without its conclusion (or vice versa). Splitting
    on headers first, and only falling back to paragraph-window splitting
    for oversized sections, keeps each chunk topically self-contained.

Why 256-512 tokens with ~50 token overlap:
    - Technical doc sections (a "## Heading" + a few paragraphs) typically
      fall in this range already — most sections need no further splitting.
    - Below ~256 tokens, chunks lose enough surrounding context that cosine
      similarity starts matching on a single sentence in isolation, which
      hurts precision for "how do I..." style queries.
    - Above ~512 tokens, embeddings start to blur multiple sub-topics into
      one vector (the "lost in the middle" effect), which hurts recall for
      queries about a specific paragraph within a long section.
    - ~50 tokens of overlap (roughly one paragraph) prevents losing context
      at a chunk boundary when a section *does* need to be split further.

Token counting:
    We approximate tokens as `len(text) // 4`, the standard rule of thumb
    for English text under BPE tokenizers (GPT/cl100k-style). This avoids
    pulling in a heavyweight tokenizer dependency (tiktoken) purely for
    chunk-sizing — we don't need exact counts, only a size in the right
    ballpark, since pgvector/embedding-model limits are far above our
    512-token chunk ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from ads_agent.core.settings import get_settings

_CHARS_PER_TOKEN = 4

_HEADER_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk produced by the chunker, not yet embedded or persisted."""

    title: str
    content: str


def _estimate_tokens(text: str) -> int:
    """Approximate token count using the ~4-chars-per-token heuristic."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """
    Split markdown text into (breadcrumb_title, content) sections.

    Each section's content spans from just after a header line to the start
    of the next header line (of any level) — i.e. the leaf text directly
    under that header. The title is a " > "-joined breadcrumb built from
    the current heading stack, e.g. "Persistence > Checkpointer vs. store".
    """
    matches = list(_HEADER_PATTERN.finditer(text))
    if not matches:
        stripped = text.strip()
        return [("", stripped)] if stripped else []

    sections: list[tuple[str, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    stack: list[tuple[int, str]] = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        # Pop any stack entries at the same or deeper level before pushing —
        # this is what turns a flat list of headers into a breadcrumb trail.
        stack = [(lvl, h) for lvl, h in stack if lvl < level]
        stack.append((level, heading))
        title = " > ".join(h for _, h in stack)

        if content:
            sections.append((title, content))

    return sections


def _split_paragraphs(content: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT_PATTERN.split(content) if p.strip()]


def _slice_oversized_paragraph(paragraph: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Last-resort character-window slicing for a single paragraph that alone
    exceeds max_tokens (e.g. a large code block with no blank lines).
    """
    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN
    step = max(max_chars - overlap_chars, 1)
    return [paragraph[i : i + max_chars].strip() for i in range(0, len(paragraph), step)]


def _carry_overlap(paragraphs: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Keep trailing paragraphs (from the end) up to ~overlap_tokens for the next window."""
    carried: list[str] = []
    carried_tokens = 0
    for para in reversed(paragraphs):
        tokens = _estimate_tokens(para)
        if carried and carried_tokens + tokens > overlap_tokens:
            break
        carried.insert(0, para)
        carried_tokens += tokens
    return carried, carried_tokens


def _windowed_chunks(paragraphs: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Pack paragraphs into ~max_tokens windows, carrying ~overlap_tokens between them."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        if para_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(_slice_oversized_paragraph(para, max_tokens, overlap_tokens))
            continue

        if current and current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current, current_tokens = _carry_overlap(current, overlap_tokens)

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_document(
    text: str,
    *,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[ChunkDraft]:
    """
    Split markdown-formatted document text into structure-aware chunks.

    Strategy:
        1. Split on markdown headers into leaf sections with breadcrumb titles.
        2. Oversized sections (> max_tokens) are further split by paragraph,
           packed into ~max_tokens windows with ~overlap_tokens carried over.
        3. Undersized sections (< min_tokens) are merged with sibling
           sections that share the same top-level heading, so a document
           with many short subsections doesn't explode into dozens of tiny,
           low-context chunks. Sections under different top-level headings
           are never merged, to avoid blending unrelated topics.

    Args:
        text: Markdown text (headers as '#'..'######'), e.g. from
            trafilatura's markdown output format.
        min_tokens/max_tokens/overlap_tokens: Override AppSettings defaults —
            mainly useful for tests.

    Returns:
        Ordered list of ChunkDraft; empty if `text` has no content.
    """
    settings = get_settings()
    min_tokens = min_tokens if min_tokens is not None else settings.rag_chunk_min_tokens
    max_tokens = max_tokens if max_tokens is not None else settings.rag_chunk_max_tokens
    overlap_tokens = (
        overlap_tokens if overlap_tokens is not None else settings.rag_chunk_overlap_tokens
    )

    sections = _split_by_headers(text)
    drafts: list[ChunkDraft] = []

    buffer_title = ""
    buffer_parts: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer_parts, buffer_tokens
        if buffer_parts:
            drafts.append(ChunkDraft(title=buffer_title, content="\n\n".join(buffer_parts)))
        buffer_parts, buffer_tokens = [], 0

    for title, content in sections:
        tokens = _estimate_tokens(content)

        if tokens > max_tokens:
            flush()
            for window in _windowed_chunks(_split_paragraphs(content), max_tokens, overlap_tokens):
                drafts.append(ChunkDraft(title=title, content=window))
            continue

        top_level_heading = title.split(" > ", 1)[0]
        buffer_top_level = buffer_title.split(" > ", 1)[0]
        if buffer_parts and top_level_heading != buffer_top_level:
            flush()

        if not buffer_parts:
            buffer_title = title

        buffer_parts.append(content)
        buffer_tokens += tokens

        if buffer_tokens >= min_tokens:
            flush()

    flush()
    return drafts
