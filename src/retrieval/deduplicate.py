"""Deduplication of query results by dense chunk ID."""

from dataclasses import dataclass
from typing import List


@dataclass
class RetrievalResult:
    """A dense chunk returned to the LLM."""
    id: str
    text: str
    source_url: str
    document_id: str
    section_title: str
    score: float


def deduplicate(
    dense_results: List[RetrievalResult],
    sparse_resolved: List[RetrievalResult],
) -> List[RetrievalResult]:
    """Return dense results plus sparse-resolved results not already present.

    Args:
        dense_results: Dense chunks returned directly by dense query,
            ordered by score descending.
        sparse_resolved: Dense chunks fetched by resolving sparse hit
            dense_chunk_ids. May contain duplicates of dense_results.

    Returns:
        All dense_results, then any sparse_resolved chunks whose ID
        does not already appear in dense_results. Order within each
        group is preserved.
    """
    seen_ids = {r.id for r in dense_results}
    additional = [r for r in sparse_resolved if r.id not in seen_ids]
    return dense_results + additional
