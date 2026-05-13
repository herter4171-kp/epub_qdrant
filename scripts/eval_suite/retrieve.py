"""Retrieve from two collections (dense + sparse) and merge into dense chunks.

Sparse hits are resolved to their dense chunks via the ``dense_chunk_ids``
payload field. When a sparse hit's payload contains multiple
``dense_chunk_ids`` (boundary case), all referenced dense chunks are
fetched in one batch and combined into a single MergedChunk so the LLM
sees the full surrounding context, not just one side of the boundary.
"""

import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from .schemas import MergedChunk, Prompt, RetrievalSet

logger = logging.getLogger(__name__)


def _gen_docket_id() -> str:
    return secrets.token_hex(4)[:7]


def _coerce_id(raw: Any) -> Any:
    """Accept str or int. Sparse payloads stored IDs as str(p.id); coerce to int when possible."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return raw


def _query_dense(
    client: QdrantClient,
    collection: str,
    dense_vector_name: str,
    query_vec: List[float],
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    kw: Dict[str, Any] = {
        "collection_name": collection,
        "query": query_vec,
        "limit": limit,
        "with_payload": True,
    }
    if dense_vector_name:
        kw["using"] = dense_vector_name
    hits = client.query_points(**kw).points
    return [
        {
            "id": p.id,
            "score": p.score if p.score else 0.0,
            "token_count": p.payload.get("token_count", 0),
            "text": p.payload.get("text", ""),
            "title": p.payload.get("title", ""),
        }
        for p in hits
    ]


def _query_sparse_and_resolve(
    client: QdrantClient,
    sparse_collection: str,
    dense_collection: str,
    sparse_vec: Dict[str, Any],
    limit: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Query sparse, batch-fetch referenced dense chunks, combine per sparse hit.

    Returns (sparse_raw, sparse_resolved):
      sparse_raw     — trace of sparse hits with dense_chunk_ids payload
      sparse_resolved — one entry per sparse hit, each carrying combined
                        dense-chunk text. Multi-id sparse hits collapse
                        N dense chunks into one combined entry.
    """
    if limit <= 0:
        return [], []

    sparse_query = SparseVector(
        indices=sparse_vec.get("indices", []),
        values=sparse_vec.get("values", []),
    )
    hits = client.query_points(
        collection_name=sparse_collection,
        query=sparse_query,
        using="sparse",
        limit=limit,
        with_payload=True,
    ).points

    sparse_raw: List[Dict[str, Any]] = []
    sparse_resolved: List[Dict[str, Any]] = []
    ids_to_fetch: set = set()
    hit_id_lists: List[Tuple[Any, float, List[Any]]] = []

    for p in hits:
        raw_ids = p.payload.get("dense_chunk_ids", []) or []
        coerced = [_coerce_id(x) for x in raw_ids]
        sparse_raw.append({
            "id": p.id,
            "score": p.score if p.score else 0.0,
            "text": p.payload.get("text", ""),
            "section_title": p.payload.get("section_title", ""),
            "dense_chunk_ids": coerced,
        })
        if not coerced:
            # Use the sparse hit's own text as the resolved chunk
            sparse_resolved.append({
                "id": p.id,
                "score": p.score if p.score else 0.0,
                "token_count": p.payload.get("token_count", 0),
                "text": p.payload.get("text", ""),
                "title": p.payload.get("section_title", ""),
                "constituent_ids": [],
                "originating_sparse_id": p.id,
            })
            continue
        for i in coerced:
            ids_to_fetch.add(i)
        hit_id_lists.append((p.id, p.score if p.score else 0.0, coerced))

    if not ids_to_fetch:
        return sparse_raw, []

    fetched = client.retrieve(
        collection_name=dense_collection,
        ids=list(ids_to_fetch),
        with_payload=True,
        with_vectors=False,
    )
    id_to_point = {pt.id: pt for pt in fetched}

    sparse_resolved: List[Dict[str, Any]] = []
    for sparse_hit_id, sparse_score, dense_ids in hit_id_lists:
        parts: List[str] = []
        constituent_ids: List[Any] = []
        total_tokens = 0
        title = ""
        for did in dense_ids:
            pt = id_to_point.get(did)
            if pt is None:
                logger.warning(
                    "sparse hit id=%s references missing dense id=%s in '%s'",
                    sparse_hit_id, did, dense_collection,
                )
                continue
            parts.append(pt.payload.get("text", ""))
            constituent_ids.append(pt.id)
            total_tokens += pt.payload.get("token_count", 0)
            if not title and pt.payload.get("title"):
                title = pt.payload["title"]
        if not parts:
            continue
        if len(parts) == 1:
            combined_id: Any = constituent_ids[0]
            combined_text = parts[0]
        else:
            combined_id = "+".join(str(i) for i in sorted(constituent_ids, key=str))
            combined_text = "\n\n---\n\n".join(parts)

        sparse_resolved.append({
            "id": combined_id,
            "score": sparse_score,
            "token_count": total_tokens,
            "text": combined_text,
            "title": title,
            "constituent_ids": constituent_ids,
            "originating_sparse_id": sparse_hit_id,
        })

    return sparse_raw, sparse_resolved


def retrieve(
    prompt: Prompt,
    embeddings: Dict[str, Any],
    dense_k: int,
    sparse_k: int,
    qdrant_url: str,
    dense_collection: str,
    sparse_collection: str,
    dense_vector_name: str,
    topk: int,
) -> RetrievalSet:
    dense_vec = embeddings["dense"]
    sparse_vec = embeddings["sparse"]

    client = QdrantClient(url=qdrant_url)

    dense_raw = _query_dense(
        client, dense_collection, dense_vector_name, dense_vec, dense_k,
    )
    sparse_raw, sparse_resolved = _query_sparse_and_resolve(
        client, sparse_collection, dense_collection, sparse_vec, sparse_k,
    )

    seen_ids: set = set()
    id_to_chunk: Dict[Any, MergedChunk] = {}
    merged: List[MergedChunk] = []

    def _add(item: Dict[str, Any], source: str) -> None:
        ident = item["id"]
        if ident in seen_ids:
            # Already present — just record this additional source so relevance
            # scores are attributed to both paths.
            if ident in id_to_chunk and source not in id_to_chunk[ident].sources:
                id_to_chunk[ident].sources.append(source)
            return
        # For combined (composite-id) entries, also dedupe if any constituent
        # already covered by a direct dense hit.
        constituents = item.get("constituent_ids") or []
        if constituents and all(c in seen_ids for c in constituents):
            return
        seen_ids.add(ident)
        for c in constituents:
            seen_ids.add(c)
        chunk = MergedChunk(
            rank=0,
            id=ident,
            source=source,
            sources=[source],
            token_count=item.get("token_count", 0),
            text=item.get("text", ""),
            title=item.get("title", ""),
            docket_id="",
            constituent_ids=list(constituents),
            originating_sparse_id=item.get("originating_sparse_id"),
        )
        id_to_chunk[ident] = chunk
        merged.append(chunk)

    for item in dense_raw:
        _add(item, "dense")
    for item in sparse_resolved:
        _add(item, "sparse_resolved")

    if len(merged) < topk:
        for item in dense_raw:
            _add(item, "dense")
            if len(merged) >= topk:
                break

    used: set = set()
    for m in merged:
        did = _gen_docket_id()
        while did in used:
            did = _gen_docket_id()
        used.add(did)
        m.docket_id = did

    for i, m in enumerate(merged):
        m.rank = i + 1

    return RetrievalSet(
        prompt_index=prompt.index,
        prompt_text=prompt.text,
        topk=topk,
        sparse_k=sparse_k,
        dense_k=dense_k,
        sparse_fraction=f"{sparse_k / topk:.2f}" if topk > 0 else "0.00",
        dense_collection=dense_collection,
        sparse_collection=sparse_collection,
        dense_vector_name=dense_vector_name,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        dense_raw=dense_raw,
        sparse_raw=sparse_raw,
        merged=merged,
    )
