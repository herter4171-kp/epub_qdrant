"""Tool execution against Qdrant — embed + search."""

import logging
from typing import Dict, List, Optional, Set

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, HasIdCondition, SparseVector

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 2
_RETRY_DELAY = 1.0


def _embed_dense(url: str, text: str) -> List[float]:
    """Embed a single text → 768-d dense vector."""
    resp = requests.post(
        f"{url}/embed_dense",
        json={"texts": [text]},
        timeout=(10, 300),
    )
    resp.raise_for_status()
    vectors = resp.json()["vectors"]
    return vectors[0] if vectors else []


def _embed_sparse(url: str, text: str, sparse_only: bool = False) -> Dict:
    """Embed a single text → sparse vector (indices, values)."""
    resp = requests.post(
        f"{url}/embed_sparse",
        json={"texts": [text], "is_query": True},
        timeout=(10, 300),
    )
    resp.raise_for_status()
    vectors = resp.json()["vectors"]
    vec = vectors[0] if vectors else {"indices": [], "values": []}
    # Convert SparseVector models to dicts
    if hasattr(vec, "model_dump"):
        return vec.model_dump()
    if hasattr(vec, "dict"):
        return vec.dict()
    return dict(vec)


def _embed_sparse_sae(url: str, text: str) -> Dict:
    """Embed a single text → SAE sparse vector (indices, values)."""
    resp = requests.post(
        f"{url}/embed_sae",
        json={"texts": [text], "is_query": True},
        timeout=(10, 300),
    )
    resp.raise_for_status()
    vectors = resp.json()["vectors"]
    vec = vectors[0] if vectors else {"indices": [], "values": []}
    if hasattr(vec, "model_dump"):
        return vec.model_dump()
    if hasattr(vec, "dict"):
        return vec.dict()
    return dict(vec)


def execute_search(
    query: str,
    *,
    embed_url: str,
    qdrant_url: str,
    dense_collection: str,
    sparse_collection: str,
    dense_vector_name: str = "",
    dense_k: int = 4,
    sparse_k: int = 2,
    sparse_only: bool = False,
    excluded_dense_ids: Optional[Set] = None,
    excluded_sparse_ids: Optional[Set] = None,
) -> List[dict]:
    """Execute search against Qdrant — the tool the model calls.

    1. Embed the query (dense + sparse).
    2. Run dense search against dense_collection, sparse search against
       sparse_collection, excluding any previously-seen IDs per collection.
    3. Concatenate results in rank order: dense results first, then sparse.
    4. Return a list of dicts, one per result.
    """
    # Embed the query
    dense_vec = _embed_dense(embed_url, query) if dense_k > 0 else []
    sparse_vec: Dict = {"indices": [], "values": []}
    if sparse_k > 0:
        if sparse_only:
            sparse_vec = _embed_sparse_sae(embed_url, query)
        else:
            sparse_vec = _embed_sparse(embed_url, query, sparse_only=False)

    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url)

    # Build exclusion filters
    dense_filter = (
        Filter(must_not=[HasIdCondition(has_id=list(excluded_dense_ids))])
        if excluded_dense_ids else None
    )
    sparse_filter = (
        Filter(must_not=[HasIdCondition(has_id=list(excluded_sparse_ids))])
        if excluded_sparse_ids else None
    )

    # Dense search
    dense_hits = []
    if dense_k > 0:
        dense_kw: Dict = {
            "collection_name": dense_collection,
            "query": dense_vec,
            "limit": dense_k,
            "with_payload": True,
        }
        if dense_vector_name:
            dense_kw["using"] = dense_vector_name
        if dense_filter:
            dense_kw["query_filter"] = dense_filter
        dense_hits = client.query_points(**dense_kw).points

    # Sparse search
    sparse_hits = []
    if sparse_k > 0:
        sparse_query = SparseVector(
            indices=sparse_vec.get("indices", []),
            values=sparse_vec.get("values", []),
        )
        sparse_kw: Dict = {
            "collection_name": sparse_collection,
            "query": sparse_query,
            "using": "sparse",
            "limit": sparse_k,
            "with_payload": True,
        }
        if sparse_filter:
            sparse_kw["query_filter"] = sparse_filter
        sparse_hits = client.query_points(**sparse_kw).points

    # Build result list: dense first, then sparse.
    results: List[dict] = []

    for i, p in enumerate(dense_hits):
        token_count = p.payload.get("token_count", 0)
        results.append({
            "rank": i + 1,
            "source": "dense",
            "score": p.score if p.score else 0.0,
            "id": p.id,
            "title": p.payload.get("title", ""),
            "text": p.payload.get("text", ""),
            "token_count": token_count,
        })

    for i, p in enumerate(sparse_hits):
        token_count = p.payload.get("token_count", 0)
        results.append({
            "rank": len(dense_hits) + i + 1,
            "source": "sparse",
            "score": p.score if p.score else 0.0,
            "id": p.id,
            "title": p.payload.get("title", ""),
            "text": p.payload.get("text", ""),
            "token_count": token_count,
        })

    logger.info(
        "execute_search: query='%s' | dense_k=%d sparse_k=%d | total=%d | excl_dense=%d excl_sparse=%d",
        query[:80], dense_k, sparse_k, len(results),
        len(excluded_dense_ids) if excluded_dense_ids else 0,
        len(excluded_sparse_ids) if excluded_sparse_ids else 0,
    )
    return results


def format_chunks_for_model(chunks: List[dict]) -> str:
    """Render chunk results as plain text for the model.

    One block per chunk, --- separated. No truncation.
    """
    parts = []
    for c in chunks:
        block = (
            f"[{c['rank']}] title: {c.get('title', '')}\n"
            f"source: {c.get('source', '')}  score: {c.get('score', 0.0):.4f}\n"
            f"{c.get('text', '')}"
        )
        parts.append(block)
    return "\n---\n".join(parts)