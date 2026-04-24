"""Semantic chunking pipeline — three-layer: structural → semantic → recursive.

This module provides:
- ChunkConfig / ChunkResult dataclasses
- load_tokenizer() for real token counting via tokenizers library
- _split_sentences() for sentence segmentation
- _detect_semantic_boundaries() for embedding-based topic shift detection
- _sentences_to_segments() for splitting sentences at boundary indices
- _merge_runts() for merging undersized chunks
- chunk_section() — the main three-layer pipeline
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    """Configuration for the semantic chunking pipeline."""

    chunk_size: int = 500
    overlap_ratio: float = 0.0
    similarity_percentile: float = 95.0
    min_distance_floor: float = 0.1
    min_sentences_for_semantic: int = 10
    min_chunk_tokens: int = 50
    enable_semantic: bool = True
    tokenizer_path: Optional[str] = None


@dataclass
class ChunkResult:
    """A single chunk produced by the semantic chunking pipeline."""

    text: str
    section_title: str
    chunk_index: int
    token_count: int
    has_heading_context: bool


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def load_tokenizer(path: Optional[str] = None) -> Callable[[str], int]:
    """Load a real tokenizer and return a token-counting callable.

    Resolution order:
      1. Explicit *path* argument
      2. ``TOKENIZER_JSON`` environment variable

    Returns a ``Callable[[str], int]`` that counts tokens for a given string.

    Raises:
        ValueError: If no tokenizer path is configured.
        FileNotFoundError: If the resolved path does not exist.
    """
    from tokenizers import Tokenizer  # lazy import — heavy dependency

    _DEFAULT_TOKENIZER = "tokenizer.json"
    resolved = path or os.getenv("TOKENIZER_JSON", "") or _DEFAULT_TOKENIZER
    if not resolved:
        raise ValueError(
            "No tokenizer path configured. Set the TOKENIZER_JSON environment "
            "variable or pass a path to load_tokenizer()."
        )

    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Tokenizer file not found: {resolved}"
        )

    tokenizer = Tokenizer.from_file(resolved)

    def count_tokens(text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(tokenizer.encode(text).ids)

    return count_tokens


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Abbreviations that shouldn't trigger sentence breaks
_ABBREVS = frozenset([
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "vs", "etc", "inc", "ltd", "corp", "dept", "univ",
    "vol", "fig", "eq", "approx", "al", "e.g", "i.e",
])

_SENTENCE_END_RE = re.compile(
    r"""
    (?<=[.!?])       # lookbehind: sentence-ending punctuation
    (?<!\b\w\.)      # not after single-letter abbreviation (e.g. "U.")
    \s+              # one or more whitespace chars
    (?=[A-Z"\'\(])   # lookahead: next sentence starts with uppercase, quote, or paren
    """,
    re.VERBOSE,
)

# Titles / abbreviations that end with "." but aren't sentence endings
_TITLE_RE = re.compile(
    r"^(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Vs|Rev|Gen|Gov|Sgt|Cpl|Pvt|Capt|Lt|Col|Maj)"
    r"\.$",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex.

    Handles common edge cases: abbreviations, decimal numbers, URLs.
    Returns list of non-empty sentence strings.
    """
    if not text or not text.strip():
        return []

    # Split on sentence-ending punctuation followed by whitespace + uppercase
    parts = _SENTENCE_END_RE.split(text)
    parts = [s.strip() for s in parts if s.strip()]

    # Rejoin fragments that end with a title abbreviation (Dr. / Mr. / etc.)
    merged: List[str] = []
    for part in parts:
        if merged and _TITLE_RE.search(merged[-1].split()[-1] if merged[-1] else ""):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    return merged


# ---------------------------------------------------------------------------
# Semantic boundary detection
# ---------------------------------------------------------------------------

def _detect_semantic_boundaries(
    sentences: List[str],
    embeddings: List[List[float]],
    percentile: float = 95.0,
    min_distance_floor: float = 0.1,
) -> List[int]:
    """Find sentence indices where topic shifts occur.

    Computes cosine similarity between consecutive sentence embeddings,
    derives distances as ``1 - similarity``, then returns indices where
    distance exceeds ``max(percentile_threshold, min_distance_floor)``.

    Args:
        sentences: List of sentence strings (len >= 2).
        embeddings: Matching list of embedding vectors (same length).
        percentile: Percentile threshold over distance distribution.
        min_distance_floor: Minimum cosine distance to flag a boundary.

    Returns:
        Sorted list of boundary indices. Index *i* means split *after*
        sentence *i*. All indices in ``[0, len(sentences) - 2]``.
        Empty list if no meaningful boundaries found.
    """
    import numpy as np

    if len(sentences) < 2 or len(embeddings) < 2:
        return []

    emb_array = np.array(embeddings, dtype=np.float64)

    # Normalize to unit vectors
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = emb_array / norms

    # Pairwise cosine similarity between consecutive sentences
    similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)
    distances = 1.0 - similarities

    # Adaptive threshold: percentile of distance distribution, floored
    threshold = float(np.percentile(distances, percentile))
    threshold = max(threshold, min_distance_floor)

    boundaries = [
        int(i) for i, d in enumerate(distances) if d >= threshold
    ]

    return sorted(boundaries)


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def _sentences_to_segments(
    sentences: List[str],
    boundary_indices: List[int],
) -> List[str]:
    """Split sentence list into segments at boundary indices.

    Each boundary index *i* means: split *after* sentence *i*.
    Sentences within each segment are joined with a single space.
    """
    segments: List[str] = []
    prev = 0
    for idx in sorted(boundary_indices):
        seg = sentences[prev : idx + 1]
        if seg:
            segments.append(" ".join(seg))
        prev = idx + 1
    # Remaining sentences after last boundary
    if prev < len(sentences):
        segments.append(" ".join(sentences[prev:]))
    return [s for s in segments if s.strip()]


def _merge_runts(
    chunks: List[str],
    min_chunk_tokens: int,
    token_counter: Callable[[str], int],
) -> List[str]:
    """Merge chunks smaller than *min_chunk_tokens* into adjacent chunks.

    Strategy: walk forward; if a chunk is too small, merge it into the
    previous chunk. If it's the first chunk, merge into the next one.
    Preserves ordering.
    """
    if len(chunks) <= 1:
        return chunks

    merged: List[str] = [chunks[0]]
    for chunk in chunks[1:]:
        if token_counter(chunk) < min_chunk_tokens:
            # Merge into previous
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    # Check if the first chunk is now a runt (could happen if original first
    # chunk was small and nothing merged into it yet)
    if len(merged) > 1 and token_counter(merged[0]) < min_chunk_tokens:
        merged[1] = merged[0] + " " + merged[1]
        merged.pop(0)

    return merged


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------

def chunk_section(
    title: str,
    content: str,
    config: ChunkConfig,
    token_counter: Callable[[str], int],
    embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> List[ChunkResult]:
    """Three-layer semantic chunking pipeline.

    Layer 1: Section arrives pre-split by heading (structural boundary).
    Layer 2: Semantic boundary detection on original section sentences.
    Layer 3: Recursive sub-splitting via semchunk within each segment.

    Then: merge runts, prepend heading context, emit ChunkResults.
    """
    import semchunk

    if not content or not content.strip():
        return [
            ChunkResult(
                text=title or "",
                section_title=title,
                chunk_index=0,
                token_count=token_counter(title or ""),
                has_heading_context=False,
            )
        ]

    # ── Short content early return (task 9.2) ──
    content_tokens = token_counter(content)
    if content_tokens < config.min_chunk_tokens:
        heading_prefix = (
            f"## {title}\n\n" if title and title != "(no title)" else ""
        )
        final_text = heading_prefix + content
        return [
            ChunkResult(
                text=final_text,
                section_title=title,
                chunk_index=0,
                token_count=token_counter(final_text),
                has_heading_context=bool(heading_prefix),
            )
        ]

    # ── Layer 2: Semantic boundary detection on ORIGINAL content ──
    sentences = _split_sentences(content)
    segments = [content]  # default: whole section = one segment

    if (
        config.enable_semantic
        and embedding_fn is not None
        and len(sentences) >= config.min_sentences_for_semantic
    ):
        try:
            embeddings = embedding_fn(sentences)
            boundary_indices = _detect_semantic_boundaries(
                sentences,
                embeddings,
                config.similarity_percentile,
                config.min_distance_floor,
            )
            if boundary_indices:
                segments = _sentences_to_segments(sentences, boundary_indices)
        except (ConnectionError, OSError, Exception) as exc:
            # Task 9.1: fallback — skip semantic, use recursive only
            logger.warning(
                "Embedding server error during semantic boundary detection "
                "for section '%s': %s. Falling back to recursive splitting.",
                title,
                exc,
            )

    # ── Layer 3: Recursive sub-splitting via semchunk per segment ──
    overlap = int(config.chunk_size * config.overlap_ratio)
    chunker = semchunk.chunkerify(token_counter, config.chunk_size)

    sub_chunks: List[str] = []
    for segment in segments:
        if token_counter(segment) <= config.chunk_size:
            sub_chunks.append(segment)
        else:
            sub_chunks.extend(chunker(segment, overlap=overlap))

    # ── Merge runt chunks ──
    if sub_chunks:
        sub_chunks = _merge_runts(sub_chunks, config.min_chunk_tokens, token_counter)

    # Guarantee at least one chunk
    if not sub_chunks:
        sub_chunks = [content]

    # ── Heading-as-context-bridge ──
    heading_prefix = (
        f"## {title}\n\n" if title and title != "(no title)" else ""
    )

    results: List[ChunkResult] = []
    for i, chunk_text in enumerate(sub_chunks):
        final_text = heading_prefix + chunk_text
        results.append(
            ChunkResult(
                text=final_text,
                section_title=title,
                chunk_index=i,
                token_count=token_counter(final_text),
                has_heading_context=bool(heading_prefix),
            )
        )

    return results
