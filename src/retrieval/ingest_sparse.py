"""Sparse-only collection builder.

Pulls text payloads from an existing dense collection, splits each into
256-token sparse chunks, embeds via SPLADE, and writes them to a new
sparse-only Qdrant collection. Each sparse point's payload carries
``dense_chunk_ids = [originating_dense_id]`` so query-time can resolve
sparse hits back to dense chunks.

Run:
    python3 -m src.retrieval.ingest_sparse \\
        --dense-collection papers \\
        --sparse-collection papers-sparse-256 \\
        [--limit N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
)

from servers.embedding_server.client import (
    get_sparse_vectors,
    health_check,
)
from src.config import settings
from src.ingestion.semantic_chunker import (
    ChunkConfig,
    chunk_section,
    load_tokenizer,
)

logger = logging.getLogger(__name__)

SPARSE_CHUNK_SIZE = 256
MIN_CHUNK_TOKENS = 50
SPARSE_EMBED_BATCH = 32
SCROLL_BATCH = 256

PROTECTED = frozenset({"books", "books-named", "papers", "papers-named"})

INDEX_FIELDS = ("doc_type", "arxiv_id", "title", "section_title", "category")


def _ensure_sparse_collection(client: QdrantClient, name: str) -> None:
    if name in PROTECTED:
        raise ValueError(f"Refusing to overwrite protected collection '{name}'")
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config={},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    for field in INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema="keyword",
            )
        except Exception:
            pass


def _derive_source_url(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    canonical = arxiv_id.replace("_", ".")
    return f"https://arxiv.org/abs/{canonical}"


def _build_sparse_payload(
    dense_payload: Dict,
    dense_id,
    sparse_chunk_idx: int,
    chunk_text: str,
    token_count: int,
) -> Dict:
    arxiv_id = dense_payload.get("arxiv_id", "")
    return {
        "text": chunk_text,
        "doc_type": dense_payload.get("doc_type", "paper"),
        "arxiv_id": arxiv_id,
        "title": dense_payload.get("title", ""),
        "authors": dense_payload.get("authors", ""),
        "category": dense_payload.get("category", ""),
        "subcategory": dense_payload.get("subcategory", ""),
        "publish_date": dense_payload.get("publish_date", ""),
        "section_title": dense_payload.get("section_title", ""),
        "source_url": _derive_source_url(arxiv_id),
        "chunk_index": sparse_chunk_idx,
        "token_count": token_count,
        "dense_chunk_ids": [str(dense_id)],
    }


def run(
    dense_collection: str,
    sparse_collection: str,
    limit: Optional[int] = None,
) -> None:
    if not health_check():
        logger.error("Embedding server not healthy at %s", settings.EMBEDDING_SERVER_URL)
        sys.exit(1)
    logger.info("Embedding server: OK")

    client = QdrantClient(url=settings.QDRANT_URL)
    existing = {c.name for c in client.get_collections().collections}
    if dense_collection not in existing:
        logger.error("Dense collection '%s' does not exist.", dense_collection)
        sys.exit(1)

    _ensure_sparse_collection(client, sparse_collection)
    logger.info("Sparse collection '%s' ready.", sparse_collection)

    token_counter = load_tokenizer(settings.TOKENIZER_JSON or None)
    chunk_cfg = ChunkConfig(
        chunk_size=SPARSE_CHUNK_SIZE,
        overlap_ratio=0.0,
        min_chunk_tokens=MIN_CHUNK_TOKENS,
        enable_semantic=False,
        tokenizer_path=settings.TOKENIZER_JSON or None,
    )

    dense_info = client.get_collection(dense_collection)
    total_dense = dense_info.points_count or 0
    logger.info("Scrolling up to %s dense points from '%s'",
                limit if limit is not None else total_dense, dense_collection)

    try:
        sparse_info = client.get_collection(sparse_collection)
        next_sparse_id = sparse_info.points_count or 0
    except Exception:
        next_sparse_id = 0

    processed_dense = 0
    written_sparse = 0
    offset = None

    while True:
        page_limit = SCROLL_BATCH
        if limit is not None:
            remaining = limit - processed_dense
            if remaining <= 0:
                break
            page_limit = min(SCROLL_BATCH, remaining)

        pts, next_offset = client.scroll(
            collection_name=dense_collection,
            limit=page_limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break

        batch_texts: List[str] = []
        batch_meta: List[Dict] = []

        for p in pts:
            payload = p.payload or {}
            text = (payload.get("text") or "").strip()
            processed_dense += 1
            if not text:
                continue

            results = chunk_section(
                title=payload.get("section_title", ""),
                content=text,
                config=chunk_cfg,
                token_counter=token_counter,
                embedding_fn=None,
            )
            for cr in results:
                meta = _build_sparse_payload(
                    dense_payload=payload,
                    dense_id=p.id,
                    sparse_chunk_idx=cr.chunk_index,
                    chunk_text=cr.text,
                    token_count=cr.token_count,
                )
                batch_texts.append(cr.text)
                batch_meta.append(meta)

        if batch_texts:
            all_vecs: List[Dict] = []
            for b in range(0, len(batch_texts), SPARSE_EMBED_BATCH):
                all_vecs.extend(get_sparse_vectors(
                    batch_texts[b:b + SPARSE_EMBED_BATCH], is_query=False,
                ))

            points = []
            for vec, meta in zip(all_vecs, batch_meta):
                points.append(PointStruct(
                    id=next_sparse_id,
                    vector={"sparse": SparseVector(
                        indices=list(vec["indices"]),
                        values=list(vec["values"]),
                    )},
                    payload=meta,
                ))
                next_sparse_id += 1

            client.upsert(collection_name=sparse_collection, points=points)
            written_sparse += len(points)
            logger.info("  upserted %d sparse points (dense scrolled: %d)",
                        len(points), processed_dense)

        if next_offset is None:
            break
        offset = next_offset

    logger.info("Done. Dense scrolled: %d, sparse written: %d",
                processed_dense, written_sparse)


def main():
    parser = argparse.ArgumentParser(
        description="Build sparse-only Qdrant collection from existing dense collection.",
    )
    parser.add_argument("--dense-collection", required=True,
                        help="Source dense collection (read-only).")
    parser.add_argument("--sparse-collection", required=True,
                        help="Target sparse-only collection (created if missing).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max dense points to scroll (smoke testing).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run(args.dense_collection, args.sparse_collection, args.limit)


if __name__ == "__main__":
    main()
