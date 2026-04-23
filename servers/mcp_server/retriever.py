"""Retrieval layer: search Qdrant, group by section/book, assemble context.

Supports single-collection targeting and cross-collection search across all
configured collections in the MCP server.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from qdrant_client.models import FieldCondition, MatchValue, SparseVector

from servers.embedding_server.client import get_dense_vectors, get_sparse_vectors
from src.storage import Storage
from servers.mcp_server.config import settings

logger = logging.getLogger(__name__)

# Common section-title variants that should be normalized or signal fallback.
# Maps lowercase canonical names → a normalized form; None means "skip, use semantic".
_NORMALIZED_TITLES: Dict[str, Optional[str]] = {
    "front matter": None,
    "frontmatter": None,
    "preface": "preface",
    "introduction": "introduction",
    "foreword": "foreword",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "copyright": None,
    "(no title)": None,
    "no title": None,
}


@dataclass
class Source:
    """A bibliographic source with assigned citation id."""
    id: int
    authors: str
    title: str
    year: str = ""
    arxiv_id: Optional[str] = None
    source_file: str = ""
    publisher: Optional[str] = None

    def format(self) -> str:
        """Format this source as a numbered bibliographic entry."""
        parts = []
        if self.authors:
            parts.append(self.authors)
        parts.append(self.title)
        if self.arxiv_id:
            parts.append(f"arXiv:{self.arxiv_id},")
        elif self.publisher:
            parts.append(f"{self.publisher},")
        if self.year:
            parts.append(self.year)
        elif hasattr(self, "publish_date") and self.publish_date:
            parts.append(self.publish_date)
        return f"[{self.id}] {' '.join(parts)}."

    @property
    def citation_tag(self) -> str:
        """Return [Source: n] tag for inline citation."""
        return f"[Source: {self.id}]"


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
    sources: List[Source] = field(default_factory=list)


class Retriever:
    """Retrieval layer: search → group → assemble evidence.

    Supports single-collection targeting (via the ``collection`` parameter)
    and cross-collection search (via ``search_collections``).
    """

    def __init__(
        self,
        storage: Optional[Storage] = None,
        collection: Optional[str] = None,
    ):
        self._storage = storage or Storage()
        # Resolve to a single collection name; fall back to default or first configured
        target = collection or settings.DEFAULT_COLLECTION
        if not target:
            target = settings.QDRANT_COLLECTIONS.split(",")[0].strip()  # legacy single-col fallback
        self._collection = target
        self._top_k = settings.RETRIEVAL_TOP_K
        self._context_radius = settings.RETRIEVAL_CONTEXT_RADIUS
        self._group_by = settings.RETRIEVAL_GROUP_BY

    def _embed(self, text: str) -> List[float]:
        """Get dense embedding for a single text string."""
        return get_dense_vectors([text])[0]

    def _embed_sparse(self, text: str) -> Dict:
        """Get sparse MiniCOIL embedding for a single text string (query mode).

        Returns a dict with 'indices' (List[int]) and 'values' (List[float]) keys.
        """
        return get_sparse_vectors([text], is_query=True)[0]

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

        # 4. Build bibliography from unique sources
        sources = self._build_bibliography(expanded_results)

        # 5. Group by section or book
        groups = self._group_results(expanded_results, group_by)

        # 6. Assemble prompt context with inline citations
        prompt_context = self._build_prompt_context(groups, sources)

        return EvidenceBundle(
            query=query, groups=groups, total_chunks=len(expanded_results),
            prompt_context=prompt_context, sources=sources,
        )

    def hybrid_search(
        self,
        query: str,
        collection: str,
        top_k: int = 20,
        filter_by: Optional[Dict[str, str]] = None,
        sparse_weight: float = 0.25,
    ) -> List[ChunkResult]:
        """Search with dense + sparse vectors, fuse via Reciprocal Rank Fusion.

        Targets the -named collections (books-named, papers-named) which have
        both "dense" and "sparse" named vectors.

        Args:
            query: Search query string.
            collection: Collection name (should be a -named collection).
            top_k: Number of results to return after RRF fusion.
            filter_by: Optional metadata pre-filter.
            sparse_weight: Multiplier for sparse vector in RRF fusion (default 0.25).
                Set to 0 for dense-only, increase to give sparse more weight.

        Returns:
            List of ChunkResult sorted by RRF score.
        """
        # Build Qdrant filter from filter_by dict
        qdrant_filter = None
        if filter_by:
            from qdrant_client.models import FieldCondition, MatchValue, Filter
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_by.items()
            ]
            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Generate dense query vector via unified embedding server
        query_dense = get_dense_vectors([query])[0]

        # Generate sparse query vector via unified embedding server
        sparse_query = get_sparse_vectors([query], is_query=True)[0]

        # Dense search (top k*2 for fusion headroom)
        # Use using="dense" + raw list[float] — NOT NamedVector
        dense_hits = self._client.query_points(
            collection_name=collection,
            query=query_dense,
            using="dense",
            limit=top_k * 2,
            query_filter=qdrant_filter,
        )

        # Sparse search (top k*2 for fusion headroom)
        # Pass SparseVector directly — it's accepted by query_points natively
        sparse_hits = self._client.query_points(
            collection_name=collection,
            query=SparseVector(
                indices=sparse_query["indices"],
                values=sparse_query["values"],
            ),
            using="sparse",
            limit=top_k * 2,
            query_filter=qdrant_filter,
        )

        # Reciprocal Rank Fusion
        rrf_scores = defaultdict(float)
        k_rrf = 60

        for rank, hit in enumerate(dense_hits.points):
            rrf_scores[hit.id] += 1.0 / (k_rrf + rank + 1)
        for rank, hit in enumerate(sparse_hits.points):
            rrf_scores[hit.id] += sparse_weight * (1.0 / (k_rrf + rank + 1))

        # Merge: build a lookup from id → full hit (prefer dense for metadata)
        all_points = list(dense_hits.points) + list(sparse_hits.points)
        id_to_point = {}
        for p in all_points:
            id_to_point[p.id] = p

        # Sort by RRF score descending, take top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:top_k]

        results = []
        for point_id in sorted_ids:
            point = id_to_point[point_id]
            results.append(ChunkResult(
                score=rrf_scores[point_id],
                text=point.payload.get("text", ""),
                source_file=point.payload.get("source_file", ""),
                title=point.payload.get("title", "") or point.payload.get("book_title", "") or point.payload.get("section_title", ""),
                section=point.payload.get("section", "") or point.payload.get("section_title", ""),
                doc_type=point.payload.get("doc_type", ""),
                book_title=point.payload.get("book_title", ""),
                section_title=point.payload.get("section_title", ""),
                chapter_index=point.payload.get("chapter_index", 0),
                section_index=point.payload.get("section_index", 0),
                chunk_index=point.payload.get("chunk_index", 0),
                token_count=point.payload.get("token_count", 0),
                publisher=point.payload.get("publisher"),
                language=point.payload.get("language"),
                isbn=point.payload.get("isbn"),
                arxiv_id=point.payload.get("arxiv_id"),
                category=point.payload.get("category"),
                subcategory=point.payload.get("subcategory"),
                publish_date=point.payload.get("publish_date"),
                authors=point.payload.get("authors"),
                year=point.payload.get("year"),
            ))

        return results

    def search_collections(
        self,
        query: str,
        top_k: Optional[int] = None,
        group_by: Optional[str] = None,
        collections: Optional[List[str]] = None,
        filter_by: Optional[Dict[str, str]] = None,
        sparse_weight: float = 0.25,
    ) -> EvidenceBundle:
        """Search across ALL configured collections with hybrid (dense+sparse) search.

        Uses RRF fusion of dense semantic + sparse keyword vectors when MiniCOIL
        client is available. Falls back to pure dense + z-score normalization
        if MiniCOIL is unavailable.

        For -named collections (books-named, papers-named): uses hybrid_search().
        For original collections (books, papers): uses pure dense + z-score.

        Args:
            query: Search query string.
            top_k: Total results per collection (merged total = top_k * num_collections).
            group_by: Override default grouping.
            collections: Override which collections to search. Defaults to configured list.
            filter_by: Optional metadata key->value pre-filter.
            sparse_weight: Multiplier for sparse vector in RRF fusion (default 0.25).

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

        # Use collection names exactly as provided by the caller — no silent redirects.
        effective_collections = list(target_collections)

        # Collect raw results per collection
        collection_results: Dict[str, List[ChunkResult]] = {}
        q_lower = query.lower()

        for col in effective_collections:
            try:
                if "-named" in col:
                    # Hybrid search with RRF fusion
                    raw = self.hybrid_search(query, col, top_k=top_k * 2, filter_by=filter_by, sparse_weight=sparse_weight)
                elif filter_by:
                    raw = self._storage.search_with_filter(col, query, top_k=top_k, filter_by=filter_by)
                else:
                    raw = self._storage.search(col, query, top_k=top_k)

                # hybrid_search() already returns List[ChunkResult], so just use as-is.
                # For raw dict results (non-hybrid), convert to ChunkResult.
                if "-named" in col and raw and isinstance(raw[0], ChunkResult):
                    collection_results[col] = raw
                else:
                    collection_results[col] = [
                        ChunkResult(
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
                        )
                        for r in raw
                    ]
            except Exception as e:
                logger.warning(f"search_collections failed for '{col}': {e}")

        if not collection_results:
            return EvidenceBundle(
                query=query, groups=[], total_chunks=0, prompt_context="",
            )

        # For non-hybrid results (original collections), apply z-score normalization.
        # Hybrid results already have RRF scores that are comparable.
        all_results: List[ChunkResult] = []
        for col, results in collection_results.items():
            if "-named" in col:
                # Hybrid RRF scores are already normalized — use as-is with metadata boost
                for r in results:
                    boost = self._compute_metadata_boost(r, q_lower)
                    all_results.append(ChunkResult(
                        score=r.score + boost,
                        text=r.text,
                        source_file=r.source_file,
                        title=r.title,
                        section=r.section,
                        doc_type=r.doc_type,
                        book_title=r.book_title,
                        section_title=r.section_title,
                        chapter_index=r.chapter_index,
                        section_index=r.section_index,
                        chunk_index=r.chunk_index,
                        token_count=r.token_count,
                        publisher=r.publisher,
                        language=r.language,
                        isbn=r.isbn,
                        arxiv_id=r.arxiv_id,
                        category=r.category,
                        subcategory=r.subcategory,
                        publish_date=r.publish_date,
                        authors=r.authors,
                        year=r.year,
                    ))
            else:
                # Z-score normalization for original collections
                scores = np.array([r.score for r in results])
                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores)) + 1e-8

                for r in results:
                    z_score = (r.score - mean_score) / std_score
                    boost = self._compute_metadata_boost(r, q_lower)
                    all_results.append(ChunkResult(
                        score=z_score + boost,
                        text=r.text,
                        source_file=r.source_file,
                        title=r.title,
                        section=r.section,
                        doc_type=r.doc_type,
                        book_title=r.book_title,
                        section_title=r.section_title,
                        chapter_index=r.chapter_index,
                        section_index=r.section_index,
                        chunk_index=r.chunk_index,
                        token_count=r.token_count,
                        publisher=r.publisher,
                        language=r.language,
                        isbn=r.isbn,
                        arxiv_id=r.arxiv_id,
                        category=r.category,
                        subcategory=r.subcategory,
                        publish_date=r.publish_date,
                        authors=r.authors,
                        year=r.year,
                    ))

        # Sort globally by normalized/boosted score, then group
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Build bibliography from unique sources
        sources = self._build_bibliography(all_results)

        groups = self._group_results(all_results, group_by)
        prompt_context = self._build_prompt_context(groups, sources)

        return EvidenceBundle(
            query=query, groups=groups, total_chunks=len(all_results),
            prompt_context=prompt_context, collections_queried=effective_collections,
            sources=sources,
        )

    @property
    def _client(self):
        """Lazy access to the Qdrant client from storage."""
        return self._storage._client

    def _compute_metadata_boost(self, chunk: ChunkResult, q_lower: str) -> float:
        """Boost score if the query explicitly references the chunk's metadata.

        Boost values are hardcoded for now; consider making configurable via
        RETRIEVAL_BOOST_* settings in a future phase.

        Returns:
            Boost amount in [0, 0.15].
        """
        boost = 0.0

        # Publisher mention (highest boost — often the most discriminative field)
        if chunk.publisher and chunk.publisher.lower() in q_lower:
            boost += 0.15

        # Category mention (papers)
        if chunk.category and chunk.category.lower() in q_lower:
            boost += 0.10

        # Subcategory mention (papers)
        if chunk.subcategory and chunk.subcategory.lower() in q_lower:
            boost += 0.10

        # Doc_type mention (e.g. "book", "paper")
        if chunk.doc_type and chunk.doc_type.lower() in q_lower:
            boost += 0.05

        return boost

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

    def _normalize_title(self, title: str) -> Optional[str]:
        """Normalize a section title. Returns None if it signals semantic fallback."""
        if not title:
            return None
        lower = title.strip().lower()
        normalized = _NORMALIZED_TITLES.get(lower)
        return normalized  # None means "skip exact match, use semantic"

    def _semantic_anchor(
        self,
        source_file: str,
        anchor_text: str,
        top_k: int = 30,
    ) -> Optional[ChunkResult]:
        """Perform a vector search scoped to a single file to find the best-matching section.

        Returns the top ChunkResult whose source_file matches, or None if nothing found.
        """
        # Search using the anchor_text with a file filter
        if top_k > 50:
            top_k = 50
        raw_results = self._storage.search(
            self._collection, anchor_text, top_k=top_k,
        )
        for r in raw_results:
            if r.get("source_file") == source_file:
                st = r.get("section_title", "") or r.get("section", "")
                return ChunkResult(
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
                )
        return None

    def _find_section_chunks(
        self,
        source_file: str,
        section_title: str,
        radius: int,
    ) -> List[ChunkResult]:
        """Try exact match first; if it fails or is a known-bad title, fall back to semantic.

        Returns a sorted list of matching ChunkResults for the identified section,
        or an empty list if nothing is found.
        """
        normalized = self._normalize_title(section_title)

        # Step 1: Exact match lookup
        top_k = (radius * 2 + 1) * 5
        raw_results = self._storage.search(
            self._collection, section_title, top_k=top_k,
        )

        exact_matches = []
        for r in raw_results:
            if r.get("source_file") != source_file:
                continue
            st = r.get("section_title", "") or r.get("section", "")
            if st == section_title:
                exact_matches.append(ChunkResult(
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

        if exact_matches and normalized is not None:
            # We got exact matches and the title is not a known-bad sentinel
            exact_matches.sort(key=lambda x: (x.section_index, x.chunk_index))
            return exact_matches

        # Step 2: Semantic fallback — search within source_file for the best anchor
        best = self._semantic_anchor(source_file, section_title, top_k=30)
        if best:
            sec_idx = best.section_index
            # Fetch all chunks from the same section (and nearby sections)
            context_top_k = (radius * 2 + 3) * 5
            ctx_results = self._storage.search(
                self._collection, f"file:{source_file}", top_k=context_top_k,
            )
            window = []
            for r in ctx_results:
                if r.get("source_file") != source_file:
                    continue
                st = r.get("section_title", "") or r.get("section", "")
                si = r.get("section_index", 0)
                # Include chunks from the matched section and within radius
                if abs(si - sec_idx) <= radius:
                    window.append(ChunkResult(
                        score=r.get("score", 0),
                        text=r.get("text", ""),
                        source_file=r.get("source_file", ""),
                        title=r.get("title", "") or r.get("book_title", ""),
                        section=r.get("section", "") or st,
                        doc_type=r.get("doc_type", ""),
                        book_title=r.get("book_title", ""),
                        section_title=st,
                        chapter_index=r.get("chapter_index", 0),
                        section_index=si,
                        chunk_index=r.get("chunk_index", 0),
                    ))
            if window:
                window.sort(key=lambda x: (x.section_index, x.chunk_index))
                return window

        return []

    def get_context(
        self,
        source_file: str,
        section_title: Optional[str] = None,
        query: Optional[str] = None,
        radius: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> EvidenceBundle:
        """Get surrounding chunks around a specific section or topic.

        Resolves the anchor via a three-tier strategy:
        1. Exact ``section_title`` match on the given ``source_file``.
        2. Semantic intra-file fallback if exact match fails or title is a
           known-bad sentinel (``"(no title)"``, ``"front matter"``, etc.).
        3. If a natural-language ``query`` is provided instead of (or alongside)
           ``section_title``, use it as the anchor text for semantic search.

        Args:
            source_file: The filename to scope the search within.
            section_title: Optional section title for exact-match lookup.
            query: Optional natural-language query to use as semantic anchor.
            radius: Surrounding chunks per side.
            collection: Target collection (overrides init default).

        Returns:
            EvidenceBundle with the window of chunks around the anchor.
        """
        radius = radius or self._context_radius
        col = collection or self._collection

        # Determine the best anchor text.
        anchor = section_title
        if not anchor and query:
            anchor = query
        if not anchor:
            return EvidenceBundle(
                query=f"context for {source_file}", groups=[], total_chunks=0,
                prompt_context="",
            )

        # Step 1: Try exact match + semantic fallback chain.
        window = self._find_section_chunks(source_file, anchor, radius)

        if not window:
            # Last-resort: use the query to find any relevant chunks in file.
            fallback = self._semantic_anchor(source_file, anchor, top_k=radius * 4 + 1)
            if fallback:
                window = [fallback]
            else:
                return EvidenceBundle(
                    query=f"context for {source_file}", groups=[], total_chunks=0,
                    prompt_context="",
                )

        # Step 2: If the window is smaller than expected, expand via _expand_with_context.
        if len(window) < (radius * 2 + 1):
            window = self._expand_with_context(window)

        label = section_title or anchor
        sources = self._build_bibliography(window)

        groups = [
            GroupedResult(
                group_key=f"{source_file}::{label}",
                group_label=label,
                title=window[0].title if window else "",
                source_file=source_file,
                chunk_index=0,
                results=window,
                best_score=window[0].score if window else 0,
                avg_score=(sum(r.score for r in window) / len(window)) if window else 0,
            )
        ]

        prompt_context = self._build_prompt_context(groups, sources)
        return EvidenceBundle(
            query=f"context for {source_file}", groups=groups,
            total_chunks=len(window), prompt_context=prompt_context, sources=sources,
        )

    def _build_bibliography(
        self,
        results: List[ChunkResult],
    ) -> List[Source]:
        """Build a deduplicated bibliography from chunk results.

        Assigns sequential [n] ids and formats citations based on document type.
        Papers (with arxiv_id) get arXiv format; EPUBs get book format.
        """
        # Deduplicate by (title, source_file), keeping the richest metadata
        seen: Dict[str, ChunkResult] = {}
        for r in results:
            key = (r.title or r.book_title, r.source_file)
            if key not in seen:
                seen[key] = r
            else:
                # Prefer entries with more metadata
                existing = seen[key]
                if r.arxiv_id and not existing.arxiv_id:
                    seen[key] = r
                elif r.authors and not existing.authors:
                    seen[key] = r
                elif r.year and not existing.year:
                    seen[key] = r

        sources = []
        for i, (key, chunk) in enumerate(seen.items(), 1):
            authors = ""
            if chunk.authors:
                if isinstance(chunk.authors, list):
                    authors = ", ".join(chunk.authors)
                else:
                    authors = str(chunk.authors)

            title = chunk.title or chunk.book_title or ""
            year = str(chunk.year) if chunk.year else ""
            arxiv_id = chunk.arxiv_id or None
            publisher = chunk.publisher or None
            source_file = chunk.source_file

            sources.append(Source(
                id=i,
                authors=authors,
                title=title,
                year=year,
                arxiv_id=arxiv_id,
                source_file=source_file,
                publisher=publisher,
            ))

        return sources

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

    def _build_prompt_context(
        self,
        groups: List[GroupedResult],
        sources: List[Source],
    ) -> str:
        """Build a prompt-ready context string from grouped results with citations.

        Each chunk gets a [Source: n] inline tag, and a Sources bibliography
        section is appended at the end.
        """
        if not groups:
            return ""

        # Build a mapping from source_file -> Source id for inline citation tags
        source_file_to_id: Dict[str, int] = {}
        for src in sources:
            source_file_to_id[src.source_file] = src.id

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

                # Look up source citation tag
                citation_tag = ""
                src_id = source_file_to_id.get(chunk.source_file)
                if src_id:
                    citation_tag = f" [Source: {src_id}]"

                parts.append(f"[score:{chunk.score:.4f}] {chunk.text}{meta_prefix}{citation_tag}")
            parts.append("")

        # Append bibliography section
        if sources:
            parts.append("\n**Sources:**")
            for src in sources:
                parts.append(src.format())

        return "\n".join(parts).strip()