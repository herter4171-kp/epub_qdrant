"""Retrieve from Qdrant via two single-vector queries + dedupe."""

import logging
import secrets
from typing import Any, Dict, List
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from .schemas import MergedChunk, Prompt, RetrievalSet

logger = logging.getLogger(__name__)


def _gen_judge_id() -> str:
    """Random 7-char alnum hash, commit-style."""
    return secrets.token_hex(4)[:7]


def retrieve(
    prompt: Prompt,
    embeddings: Dict[str, Any],
    dense_k: int,
    sparse_k: int,
    qdrant_url: str,
    collection: str,
    topk: int,
) -> RetrievalSet:
    """Two queries (dense + sparse), concatenate, dedupe by id (dense priority).

    Returns a RetrievalSet with dense_raw, sparse_raw, and merged.
    """
    dense_vec = embeddings["dense"]
    sparse_vec = embeddings["sparse"]

    client = QdrantClient(url=qdrant_url)

    dense_raw: List[Dict[str, Any]] = []
    sparse_raw: List[Dict[str, Any]] = []

    # Dense query
    if dense_k > 0:
        dense_hits = client.query_points(
            collection_name=collection,
            query=dense_vec,
            using="dense",
            limit=dense_k,
        ).points
        for p in dense_hits:
            dense_raw.append({
                "id": p.id,
                "score": p.score if p.score else 0.0,
                "token_count": p.payload.get("token_count", 0),
                "text": p.payload.get("text", ""),
                "title": p.payload.get("title", ""),
            })

    # Sparse query — Qdrant needs SparseVector(indices, values), not raw dict
    if sparse_k > 0:
        sparse_query = SparseVector(
            indices=sparse_vec.get("indices", []),
            values=sparse_vec.get("values", []),
        )
        sparse_hits = client.query_points(
            collection_name=collection,
            query=sparse_query,
            using="sparse",
            limit=sparse_k,
        ).points
        for p in sparse_hits:
            sparse_raw.append({
                "id": p.id,
                "score": p.score if p.score else 0.0,
                "token_count": p.payload.get("token_count", 0),
                "text": p.payload.get("text", ""),
                "title": p.payload.get("title", ""),
            })

    # Dedupe: dense priority
    seen_ids: set = set()
    merged: List[MergedChunk] = []

    def _add(item: Dict[str, Any], source: str) -> None:
        if item["id"] in seen_ids:
            return
        seen_ids.add(item["id"])
        merged.append(MergedChunk(
            rank=0,
            id=item["id"],
            source=source,
            token_count=item["token_count"],
            text=item["text"],
            title=item.get("title", ""),
            judge_id="",
        ))

    for item in dense_raw:
        _add(item, "dense")
    for item in sparse_raw:
        _add(item, "sparse")

    # Pad if needed: pull from the other list's tail
    if len(merged) < topk:
        if dense_k > 0:
            for item in dense_raw:
                _add(item, "dense")
                if len(merged) >= topk:
                    break
        if sparse_k > 0 and len(merged) < topk:
            for item in sparse_raw:
                _add(item, "sparse")
                if len(merged) >= topk:
                    break

    # Mint judge_ids; ensure uniqueness within set
    used: set = set()
    for m in merged:
        jid = _gen_judge_id()
        while jid in used:
            jid = _gen_judge_id()
        used.add(jid)
        m.judge_id = jid

    # Assign ranks (1-indexed)
    for i, m in enumerate(merged):
        m.rank = i + 1

    return RetrievalSet(
        prompt_index=prompt.index,
        prompt_text=prompt.text,
        topk=topk,
        sparse_k=sparse_k,
        dense_k=dense_k,
        sparse_fraction=f"{sparse_k / topk:.2f}" if topk > 0 else "0.00",
        collection=collection,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        dense_raw=dense_raw,
        sparse_raw=sparse_raw,
        merged=merged,
    )