"""Query pipeline for hybrid dense + sparse retrieval.

Both retrieval paths return dense chunks to the LLM. The sparse path
resolves hits to dense chunks via stored dense_chunk_ids.
The LLM never sees raw sparse text.
"""

from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.retrieval.collection_config import CollectionConfig
from src.retrieval.retrieval_config import RetrievalConfig
from src.retrieval.deduplicate import RetrievalResult, deduplicate


def retrieve(
    query: str,
    config: RetrievalConfig,
    n_total: int = 6,
    sparse_frac: float = 0.33,
    qdrant_client: Optional[QdrantClient] = None,
) -> List[RetrievalResult]:
    """Query both collections and return deduplicated dense chunks.

    Args:
        query: The user query string.
        config: RetrievalConfig specifying both collections.
        n_total: Total number of result slots before deduplication.
        sparse_frac: Fraction of slots allocated to sparse query.
        qdrant_client: Qdrant client instance.

    Returns:
        Deduplicated list of RetrievalResult (dense chunks only).
        Dense results appear first. Sparse-resolved results follow.
    """
    if qdrant_client is None:
        qdrant_client = QdrantClient("localhost", port=6333)

    n_sparse = round(n_total * sparse_frac)
    n_dense = n_total - n_sparse

    dense_results = _query_dense(config.dense, query, n_dense, qdrant_client)
    sparse_resolved = _query_sparse_resolved(
        config.sparse, config.dense, query, n_sparse, qdrant_client
    )

    return deduplicate(dense_results, sparse_resolved)


def _query_dense(
    config: CollectionConfig, query: str, n: int, client: QdrantClient
) -> List[RetrievalResult]:
    """Query dense collection, return top-n as RetrievalResult list."""
    # Placeholder for actual Qdrant dense search
    # In a real implementation, this would use client.search(...)
    return []


def _query_sparse_resolved(
    sparse_config: CollectionConfig,
    dense_config: CollectionConfig,
    query: str,
    n: int,
    client: QdrantClient,
) -> List[RetrievalResult]:
    """Query sparse collection, resolve hits to dense chunks.

    1. Query sparse collection for top-n hits.
    2. For each hit, read dense_chunk_ids from payload.
    3. Fetch those IDs from the dense collection.
    4. Return as RetrievalResult list.
    5. If dense_chunk_ids is empty for a hit, log and skip.
    """
    # Placeholder for actual Qdrant sparse search and ID fetch
    return []
