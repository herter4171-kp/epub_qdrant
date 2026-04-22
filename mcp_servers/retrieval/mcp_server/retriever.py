"""Retrieval layer: search Qdrant, group by section/book, assemble context.

Supports single-collection targeting and cross-collection search across all
configured collections in the MCP server.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from qdrant_client.models import FieldCondition, MatchValue

from src.embedder import Embedder
from src.storage import Storage
from mcp_server.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """A single search result with unified metadata."""
    score: float
    text: str
    source_file: str
    # Unified fields (work across EPUB + paper schemas)
    title: str = ""
    section: str = ""
    doc_type: str = ""
    # Legacy fields (backward compat)
    book_title: str = ""
    section_title: str = ""
    chapter_index: int = 0
    section_index: int = 0
    chunk_index: int = 0
    token_count: int = 0
    publisher: Optional[str] = None
    language: Optional[str] = None
    isbn: Optional[str] = None
    arxiv_id: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    publish_date: Optional[str] = None
    authors: Optional[list] = None
    year: Optional[int] = None
    vector: Optional[List[float]] = None


@dataclass
class GroupedResult:
    """A group of chunks grouped by section or book."""
    group_key: str
    group_label: str
    title: str
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
    collections_queried: List[str] = field(default_factory=list)


class Retriever:
    """Retrieval layer: search → group → assemble evidence.

    Supports single-collection targeting (via the ``collection`` parameter)
    and cross-collection search (via ``search_collections``).
    """

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
        # Resolve to a single collection name; fall back to default or first configured
        target = collection or settings.DEFAULT_COLLECTION
        if not target:
            target = settings.QDRANT_COLLECTION  # legacy single-col fallback
        self._collection = target
        self._top_k = settings.RETRIEVAL_TOP_K
        self._context_radius = settings.RETRIEVAL_CONTEXT_RADIUS
        self._group_by = settings.RETRIEVAL_GROUP_BY

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        group_by: Optional[str] = None,
        collection: Optional[str] = None,
        filter_by: Optional[Dict[str, str]] = None,
    ) -> EvidenceBundle:
        """Search and group results within a single collection.

        Args:
            query: Search query string.
            top_k: Override default top-k count.
            group_by: Override default grouping (section | book).
            collection: Target a specific collection (overrides init default).
            filter_by: Optional metadata key->value pre-filter.

        Returns:
            EvidenceBundle with grouped results and prompt context.
        """
        top_k = top_k or self._top_k
        group_by = group_by or self._group_by
        col = collection or self._collection

        # 1. Retrieve top-k chunks from Qdrant
        if filter_by:
            raw_results = self._storage.search_with_filter(col, query, top_k=top_k, filter_by=filter_by)
        else:
            raw_results = self._storage.search(col, query, top_k=top_k)

        if not raw_results:
            return EvidenceBundle(
                query=query, groups=[], total_chunks=0, prompt_context="",
            )

        # 2. Convert to ChunkResult objects (unified)
        chunk_results = []
        for r in raw_results:
            chunk_results.append(ChunkResult(
                score=r.get("score", 0),
                text=r.get("text", ""),
                source_file=r.get("source_file", ""),
                title=r.get("title", "") or r.get("book_title", "") or r.get("section_title", ""),
                section=r.get("section", "") or r.get("section_title", ""),
                doc_type=r.get("doc_type", ""),
                book_title=r.get("book_title", ""),
                section_title=r.get("section_title", ""),
                chapter_index=r.get("chapter_index", 0),
                section_index=r.get("section_index", 0),
                chunk_index=r.get("chunk_index", 0),
                token_count=r.get("token_count", 0),
                publisher=r.get("publisher"),
                language=r.get("language"),
                isbn=r.get("isbn"),
                arxiv_id=r.get("arxiv_id"),
                category=r.get("category"),
                subcategory=r.get("subcategory"),
                publish_date=r.get("publish_date"),
                authors=r.get("authors"),
                year=r.get("year"),
            ))

        # 3. Expand with context
        expanded_results = self._expand_with_context(chunk_results)

        # 4. Group by section or book
        groups = self._group_results(expanded_results, group_by)

        # 5. Assemble prompt context
        prompt_context = self._build_prompt_context(groups)

        return EvidenceBundle(
            query=query, groups=groups, total_chunks=len(expanded_results),
            prompt_context=prompt_context,
        )

    def search_collections(
        self,
        query: str,
        top_k: Optional[int] = None,
        group_by: Optional[str] = None,
        collections: Optional[List[str]] = None,
        filter_by: Optional[Dict[str, str]] = None,
    ) -> EvidenceBundle:
        """Search across ALL configured collections and merge results.

        Args:
            query: Search query string.
            top_k: Total results per collection (merged total = top_k * num_collections).
            group_by: Override default grouping.
            collections: Override which collections to search. Defaults to configured list.
            filter_by: Optional metadata key->value pre-filter.

        Returns:
            EvidenceBundle with cross-collection grouped results.
        """
        top_k = top_k or self._top_k
        group_by = group_by or self._group_by
        target_collections = collections or settings.collections

        if not target_collections:
            return EvidenceBundle(
                query=query, groups=[], total_chunks=0, prompt_context="",
            )

        all_results: List[ChunkResult] = []
        for col in target_collections:
            try:
                if filter_by:
                    raw = self._storage.search_with_filter(col, query, top_k=top_k, filter_by=filter_by)
                else:
                    raw = self._storage.search(col, query, top_k=top_k)
                for r in raw:
                    all_results.append(ChunkResult(
                        score=r.get("score", 0),
                        text=r.get("text", ""),
                        source_file=r.get("source_file", ""),
                        title=r.get("title", "") or r.get("book_title", "") or r.get("section_title", ""),
                        section=r.get("section", "") or r.get("section_title", ""),
                        doc_type=r.get("doc_type", ""),
                        book_title=r.get("book_title", ""),
                        section_title=r.get("section_title", ""),
                        chapter_index=r.get("chapter_index", 0),
                        section_index=r.get("section_index", 0),
                        chunk_index=r.get("chunk_index", 0),
                        token_count=r.get("token_count", 0),
                        publisher=r.get("publisher"),
                        language=r.get("language"),
                        isbn=r.get("isbn"),
                        arxiv_id=r.get("arxiv_id"),
                        category=r.get("category"),
                        subcategory=r.get("subcategory"),
                        publish_date=r.get("publish_date"),
                        authors=r.get("authors"),
                        year=r.get("year"),
                    ))
            except Exception as e:
                logger.warning(f"search_collections failed for '{col}': {e}")

        if not all_results:
            return EvidenceBundle(
                query=query, groups=[], total_chunks=0, prompt_context="",
            )

        # Sort globally by score, then group
        all_results.sort(key=lambda x: x.score, reverse=True)
        groups = self._group_results(all_results, group_by)
        prompt_context = self._build_prompt_context(groups)

        return EvidenceBundle(
            query=query, groups=groups, total_chunks=len(all_results),
            prompt_context=prompt_context, collections_queried=target_collections,
        )

    def search_raw(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> List[dict]:
        """Return raw search results without grouping."""
        top_k = top_k or self._top_k
        col = collection or self._collection
        raw_results = self._storage.search(col, query, top_k=top_k)

        output = []
        for r in raw_results:
            output.append({
                "score": round(r.get("score", 0), 4),
                "text": r.get("text", ""),
                "doc_type": r.get("doc_type", ""),
                "title": r.get("title", "") or r.get("book_title", ""),
                "section": r.get("section", "") or r.get("section_title", ""),
                "source_file": r.get("source_file", ""),
                "chunk_index": r.get("chunk_index", 0),
            })
        return output

    def get_context(
        self,
        source_file: str,
        section_title: str,
        radius: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> EvidenceBundle:
        """Get surrounding chunks around a specific section."""
        radius = radius or self._context_radius
        col = collection or self._collection

        top_k = (radius * 2 + 1) * 5
        raw_results = self._storage.search(col, section_title, top_k=top_k)

        filtered = []
        for r in raw_results:
            if r.get("source_file") == source_file:
                # Support both legacy and unified field names
                st = r.get("section_title", "") or r.get("section", "")
                if st == section_title:
                    filtered.append(ChunkResult(
                        score=r.get("score", 0),
                        text=r.get("text", ""),
                        source_file=r.get("source_file", ""),
                        title=r.get("title", "") or r.get("book_title", ""),
                        section=r.get("section", "") or st,
                        doc_type=r.get("doc_type", ""),
                        book_title=r.get("book_title", ""),
                        section_title=st,
                        chapter_index=r.get("chapter_index", 0),
                        section_index=r.get("section_index", 0),
                        chunk_index=r.get("chunk_index", 0),
                    ))

        filtered.sort(key=lambda x: (x.section_index, x.chunk_index))
        if not filtered:
            return EvidenceBundle(
                query=f"context for {section_title}", groups=[], total_chunks=0, prompt_context="",
            )

        center = len(filtered) // 2
        start = max(0, center - radius)
        end = min(len(filtered), center + radius + 1)
        window = filtered[start:end]

        groups = [
            GroupedResult(
                group_key=f"{source_file}::{section_title}",
                group_label=section_title,
                title=window[0].title if window else "",
                source_file=source_file,
                chunk_index=0,
                results=window,
                best_score=window[0].score if window else 0,
                avg_score=(sum(r.score for r in window) / len(window)) if window else 0,
            )
        ]

        prompt_context = self._build_prompt_context(groups)
        return EvidenceBundle(
            query=f"context for {section_title}", groups=groups,
            total_chunks=len(window), prompt_context=prompt_context,
        )

    def _expand_with_context(
        self,
        results: List[ChunkResult],
    ) -> List[ChunkResult]:
        """Add surrounding chunks for each unique (file, section) pair."""
        existing_keys = set()
        for r in results:
            key = (r.source_file, r.section_index, r.chunk_index)
            existing_keys.add(key)

        file_sections: Dict[tuple, List[ChunkResult]] = {}
        for r in results:
            key = (r.source_file, r.section_index)
            file_sections.setdefault(key, []).append(r)

        expanded = list(results)
        expanded_keys = set(existing_keys)

        for (source_file, sec_idx), chunks in file_sections.items():
            min_sec = min(c.section_index for c in chunks)
            max_sec = max(c.section_index for c in chunks)
            context_top_k = (max_sec - min_sec + self._context_radius + 1) * 10
            context_results = self._storage.search(
                self._collection, f"file:{source_file}", top_k=context_top_k,
            )

            for r in context_results:
                st = r.get("section_title", "") or r.get("section", "")
                sec_idx_r = r.get("section_index", 0)
                ci = r.get("chunk_index", 0)
                r_key = (r.get("source_file", ""), sec_idx_r, ci)
                if r_key not in expanded_keys:
                    matching = [
                        c for c in chunks
                        if c.source_file == r.get("source_file", "")
                        and abs(c.section_index - sec_idx_r) <= self._context_radius
                    ]
                    if matching:
                        expanded.append(ChunkResult(
                            score=r.get("score", 0), text=r.get("text", ""),
                            source_file=r.get("source_file", ""),
                            book_title=r.get("book_title", ""),
                            section_title=st, section_index=sec_idx_r, chunk_index=ci,
                        ))
                        expanded_keys.add(r_key)

        seen = set()
        unique = []
        for r in expanded:
            key = (r.source_file, r.section_index, r.chunk_index)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def _group_results(
        self,
        results: List[ChunkResult],
        group_by: str,
    ) -> List[GroupedResult]:
        """Group results by section or book."""
        groups: Dict[str, List[ChunkResult]] = {}

        for r in results:
            title = r.title or r.book_title or r.source_file
            section = r.section or r.section_title or ""
            if group_by == "book":
                key = title
                label = title
            else:
                # Default: group by section/title
                if section:
                    key = f"{title}::{section}"
                    label = section
                else:
                    key = title
                    label = title

            groups.setdefault(key, []).append(r)

        grouped = []
        for key, chunks in groups.items():
            chunks.sort(key=lambda x: x.score, reverse=True)
            grouped.append(GroupedResult(
                group_key=key, group_label=label,
                title=chunks[0].title or chunks[0].book_title,
                source_file=chunks[0].source_file, chunk_index=0,
                results=chunks,
                best_score=chunks[0].score,
                avg_score=sum(c.score for c in chunks) / len(chunks),
            ))

        grouped.sort(key=lambda x: x.best_score, reverse=True)
        return grouped

    def search_with_metadata_filter(
        self,
        query: str,
        filter_by: Optional[Dict[str, str]] = None,
        top_k: Optional[int] = None,
    ) -> EvidenceBundle:
        """Search with metadata filtering support."""
        return self.search(query, top_k=top_k, filter_by=filter_by)

    def _build_prompt_context(self, groups: List[GroupedResult]) -> str:
        """Build a prompt-ready context string from grouped results."""
        if not groups:
            return ""

        parts = []
        for i, group in enumerate(groups, 1):
            parts.append(f"\n=== {group.group_label} ({group.title}) ===")
            for chunk in group.results:
                meta_parts = []
                if chunk.doc_type:
                    meta_parts.append(f"type:{chunk.doc_type}")
                if chunk.publisher:
                    meta_parts.append(f"publisher:{chunk.publisher}")
                if chunk.language:
                    meta_parts.append(f"lang:{chunk.language}")
                if chunk.isbn:
                    meta_parts.append(f"isbn:{chunk.isbn}")
                if chunk.arxiv_id:
                    meta_parts.append(f"arxiv:{chunk.arxiv_id}")
                if chunk.category:
                    meta_parts.append(f"cat:{chunk.category}")
                meta_str = " ".join(meta_parts)
                meta_prefix = f" [{meta_str}]" if meta_str else ""
                parts.append(f"[score:{chunk.score:.4f}] {chunk.text}{meta_prefix}")
            parts.append("")

        return "\n".join(parts).strip()