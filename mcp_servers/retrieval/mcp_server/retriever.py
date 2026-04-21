"""Retrieval layer: search Qdrant, group by chapter/book, assemble context."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.embedder import Embedder
from src.storage import Storage
from mcp_server.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """A single search result with metadata."""
    score: float
    text: str
    book_title: str
    section_title: str
    source_file: str
    chapter_index: int
    section_index: int
    chunk_index: int
    vector: Optional[List[float]] = None


@dataclass
class GroupedResult:
    """A group of chunks grouped by chapter or book."""
    group_key: str
    group_label: str
    book_title: str
    source_file: str
    chunk_index: int
    results: List[ChunkResult]
    best_score: float
    avg_score: float


@dataclass
class EvidenceBundle:
    """Assembled evidence for LLM consumption."""
    query: str
    groups: List[GroupedResult]
    total_chunks: int
    prompt_context: str


class Retriever:
    """Retrieval layer: search → group → assemble evidence."""

    def __init__(
        self,
        storage: Optional[Storage] = None,
        embedder: Optional[Embedder] = None,
        collection: Optional[str] = None,
    ):
        self._storage = storage or Storage()
        self._embedder = embedder or Embedder(
            settings.OLLAMA_URL, settings.EMBEDDING_MODEL
        )
        self._collection = collection or settings.QDRANT_COLLECTION
        self._top_k = settings.RETRIEVAL_TOP_K
        self._context_radius = settings.RETRIEVAL_CONTEXT_RADIUS
        self._group_by = settings.RETRIEVAL_GROUP_BY

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        group_by: Optional[str] = None,
    ) -> EvidenceBundle:
        """Search and group results by chapter or book.

        Args:
            query: Search query string.
            top_k: Override default top-k count.
            group_by: Override default grouping (chapter | book).

        Returns:
            EvidenceBundle with grouped results and prompt context.
        """
        top_k = top_k or self._top_k
        group_by = group_by or self._group_by

        # 1. Retrieve top-k chunks from Qdrant
        raw_results = self._storage.search(
            self._collection, query, top_k=top_k
        )

        if not raw_results:
            return EvidenceBundle(
                query=query,
                groups=[],
                total_chunks=0,
                prompt_context="",
            )

        # 2. Convert to ChunkResult objects
        chunk_results = []
        for r in raw_results:
            chunk_results.append(ChunkResult(
                score=r.get("score", 0),
                text=r.get("text", ""),
                book_title=r.get("book_title", ""),
                section_title=r.get("section_title", ""),
                source_file=r.get("source_file", ""),
                chapter_index=r.get("chapter_index", 0),
                section_index=r.get("section_index", 0),
                chunk_index=r.get("chunk_index", 0),
            ))

        # 3. Expand with context (surrounding chunks from same book/chapter)
        expanded_results = self._expand_with_context(chunk_results)

        # 4. Group by chapter or book
        groups = self._group_results(expanded_results, group_by)

        # 5. Assemble prompt context
        prompt_context = self._build_prompt_context(groups)

        return EvidenceBundle(
            query=query,
            groups=groups,
            total_chunks=len(expanded_results),
            prompt_context=prompt_context,
        )

    def search_raw(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """Return raw search results without grouping.

        Args:
            query: Search query string.
            top_k: Number of results.

        Returns:
            List of dicts with score, text, and metadata.
        """
        top_k = top_k or self._top_k
        raw_results = self._storage.search(self._collection, query, top_k=top_k)

        output = []
        for r in raw_results:
            output.append({
                "score": round(r.get("score", 0), 4),
                "text": r.get("text", ""),
                "book_title": r.get("book_title", ""),
                "section_title": r.get("section_title", ""),
                "source_file": r.get("source_file", ""),
                "chapter_index": r.get("chapter_index", 0),
                "section_index": r.get("section_index", 0),
                "chunk_index": r.get("chunk_index", 0),
            })
        return output

    def get_context(
        self,
        source_file: str,
        section_title: str,
        radius: Optional[int] = None,
    ) -> EvidenceBundle:
        """Get surrounding chunks around a specific section.

        Args:
            source_file: EPUB filename.
            section_title: Chapter/section title.
            radius: Number of surrounding chunks per side.

        Returns:
            EvidenceBundle with context window.
        """
        radius = radius or self._context_radius

        # Query all chunks from this file + section
        top_k = (radius * 2 + 1) * 5  # generous top-k, filter below
        raw_results = self._storage.search(
            self._collection, section_title, top_k=top_k
        )

        # Filter to matching file + section, sorted by indices
        filtered = []
        for r in raw_results:
            if (
                r.get("source_file") == source_file
                and r.get("section_title") == section_title
            ):
                filtered.append(ChunkResult(
                    score=r.get("score", 0),
                    text=r.get("text", ""),
                    book_title=r.get("book_title", ""),
                    section_title=r.get("section_title", ""),
                    source_file=r.get("source_file", ""),
                    chapter_index=r.get("chapter_index", 0),
                    section_index=r.get("section_index", 0),
                    chunk_index=r.get("chunk_index", 0),
                ))

        # Sort by section_index, then chunk_index
        filtered.sort(key=lambda x: (x.section_index, x.chunk_index))

        # Take ±radius around the middle
        center = len(filtered) // 2
        start = max(0, center - radius)
        end = min(len(filtered), center + radius + 1)
        window = filtered[start:end]

        groups = [
            GroupedResult(
                group_key=f"{source_file}::{section_title}",
                group_label=section_title,
                book_title=window[0].book_title if window else "",
                source_file=source_file,
                chunk_index=0,
                results=window,
                best_score=window[0].score if window else 0,
                avg_score=(sum(r.score for r in window) / len(window)) if window else 0,
            )
        ]

        prompt_context = self._build_prompt_context(groups)

        return EvidenceBundle(
            query=f"context for {section_title}",
            groups=groups,
            total_chunks=len(window),
            prompt_context=prompt_context,
        )

    def _expand_with_context(
        self,
        results: List[ChunkResult],
    ) -> List[ChunkResult]:
        """Add surrounding chunks for each unique (file, chapter) pair."""
        # Index existing results by key
        existing_keys = set()
        for r in results:
            key = (r.source_file, r.chapter_index, r.section_index, r.chunk_index)
            existing_keys.add(key)

        # For each unique (file, chapter), fetch additional context
        file_chapters: Dict[tuple, List[ChunkResult]] = {}
        for r in results:
            key = (r.source_file, r.chapter_index)
            file_chapters.setdefault(key, []).append(r)

        expanded = list(results)  # start with original results
        expanded_keys = set(existing_keys)

        for (source_file, chapter_idx), chunks in file_chapters.items():
            # Get min/max section indices
            min_sec = min(c.section_index for c in chunks)
            max_sec = max(c.section_index for c in chunks)

            # Query for broader context
            context_top_k = (min_sec + self._context_radius + 1) * 10
            context_results = self._storage.search(
                self._collection,
                f"file:{source_file}",
                top_k=context_top_k,
            )

            for r in context_results:
                r_chunk = ChunkResult(
                    score=r.get("score", 0),
                    text=r.get("text", ""),
                    book_title=r.get("book_title", ""),
                    section_title=r.get("section_title", ""),
                    source_file=r.get("source_file", ""),
                    chapter_index=r.get("chapter_index", 0),
                    section_index=r.get("section_index", 0),
                    chunk_index=r.get("chunk_index", 0),
                )
                r_key = (
                    r_chunk.source_file,
                    r_chunk.chapter_index,
                    r_chunk.section_index,
                    r_chunk.chunk_index,
                )
                if r_key not in expanded_keys:
                    # Only add if within context radius of any existing result
                    matching = [
                        c
                        for c in chunks
                        if c.source_file == r_chunk.source_file
                        and abs(c.section_index - r_chunk.section_index) <= self._context_radius
                    ]
                    if matching:
                        expanded.append(r_chunk)
                        expanded_keys.add(r_key)

        # Deduplicate by (file, chapter, section, chunk)
        seen = set()
        unique = []
        for r in expanded:
            key = (r.source_file, r.chapter_index, r.section_index, r.chunk_index)
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    def _group_results(
        self,
        results: List[ChunkResult],
        group_by: str,
    ) -> List[GroupedResult]:
        """Group results by chapter or book."""
        groups: Dict[str, List[ChunkResult]] = {}

        for r in results:
            if group_by == "book":
                key = r.book_title or r.source_file
                label = r.book_title or r.source_file
            else:
                # Default: group by chapter
                key = f"{r.book_title}::ch{r.chapter_index}"
                label = f"Chapter {r.chapter_index}"

            groups.setdefault(key, []).append(r)

        # Build GroupedResult objects, sorted by best score descending
        grouped = []
        for key, chunks in groups.items():
            chunks.sort(key=lambda x: x.score, reverse=True)
            grouped.append(GroupedResult(
                group_key=key,
                group_label=label,
                book_title=chunks[0].book_title,
                source_file=chunks[0].source_file,
                chunk_index=0,
                results=chunks,
                best_score=chunks[0].score,
                avg_score=sum(c.score for c in chunks) / len(chunks),
            ))

        grouped.sort(key=lambda x: x.best_score, reverse=True)
        return grouped

    def _build_prompt_context(self, groups: List[GroupedResult]) -> str:
        """Build a prompt-ready context string from grouped results.

        Returns formatted text suitable for passing to an LLM.
        """
        if not groups:
            return ""

        parts = []
        for i, group in enumerate(groups, 1):
            parts.append(f"\n=== {group.group_label} ({group.book_title}) ===")
            for chunk in group.results:
                parts.append(f"[score:{chunk.score:.4f}] {chunk.text}")
            parts.append("")

        return "\n".join(parts).strip()