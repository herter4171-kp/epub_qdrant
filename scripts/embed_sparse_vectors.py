#!/usr/bin/env python3
"""Two-pass hybrid migration: scroll original Qdrant collections, embed dense
and sparse vectors via the unified embedding server, and upsert into new
-hybrid collections (books-hybrid, papers-hybrid).

Pass 1 (dense): scroll original collection → embed text via /embed_dense →
    upsert into -hybrid collection with dense named vector + original payload.
Pass 2 (sparse): scroll the -hybrid collection itself → embed text via
    /embed_sparse (with 512-word chunking + max-aggregation) → upsert same
    point IDs with sparse named vector (Qdrant merges automatically).

Both passes are idempotent — they check point counts before running.

Run from Mac:
    python3 scripts/embed_sparse_vectors.py

Connects to:
    Qdrant on GPU box:       192.168.68.75:6333
    Embedding server:        EMBEDDING_SERVER_URL (default http://localhost:8100)
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict

# Ensure the project root is on sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("embed_sparse")

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    PointVectors,
    SparseVector,
    VectorParams,
    SparseVectorParams,
    Distance,
    Modifier,
)

from servers.embedding_server.client import get_dense_vectors, get_sparse_vectors, health_check


# ── Configuration ────────────────────────────────────────────────────────────

QDRANT_HOST = "192.168.68.75"
QDRANT_PORT = 6333

BATCH_SIZE = 128          # Points fetched per Qdrant scroll
MINICOIL_BATCH = 32       # Chunks sent per sparse embedding request
MAX_WORDS_PER_CHUNK = 512 # Words per chunk — caps attention matrix size

COLLECTION_MAP = {
    "books": "books-hybrid",
    "papers": "papers-hybrid",
}


# ── Text chunking + sparse aggregation ───────────────────────────────────────

def chunk_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> List[str]:
    """Split text into fixed word-count chunks."""
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def aggregate_sparse_vecs(vecs: List[Dict]) -> Dict:
    """Aggregate sparse vectors from multiple chunks by taking max value per index.

    If a term appears anywhere in the document with high weight, it's represented.
    This is how production sparse pipelines handle long documents (SPLADE et al).
    """
    aggregated: Dict[int, float] = {}
    for vec in vecs:
        for idx, val in zip(vec["indices"], vec["values"]):
            if idx not in aggregated or val > aggregated[idx]:
                aggregated[idx] = val
    return {
        "indices": list(aggregated.keys()),
        "values": list(aggregated.values()),
    }


def embed_document(text: str) -> Dict:
    """Chunk a document and return a single aggregated sparse vector."""
    chunks = chunk_text(text)
    if not chunks:
        return {"indices": [], "values": []}

    # Sub-batch chunks to MINICOIL_BATCH to keep request sizes bounded
    all_vecs = []
    for i in range(0, len(chunks), MINICOIL_BATCH):
        sub_chunks = chunks[i:i + MINICOIL_BATCH]
        sub_vecs = get_sparse_vectors(sub_chunks, is_query=False)
        all_vecs.extend(sub_vecs)

    return aggregate_sparse_vecs(all_vecs)


# ── Collection management ─────────────────────────────────────────────────────

def create_target_collections(client: QdrantClient) -> None:
    """Create books-hybrid and papers-hybrid with named vector configs."""
    existing = [c.name for c in client.get_collections().collections]

    for src, dst in COLLECTION_MAP.items():
        if dst in existing:
            logger.info(f"Collection '{dst}' already exists, skipping creation.")
            continue

        logger.info(f"Creating collection: {dst}")
        client.create_collection(
            collection_name=dst,
            vectors_config={
                "dense": VectorParams(size=768, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF)
            },
        )

        index_fields = [
            "source_file", "book_title", "section_title",
            "publisher", "language", "isbn", "arxiv_id",
            "category", "title", "doc_type",
        ]
        for field in index_fields:
            try:
                client.create_payload_index(
                    collection_name=dst,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception as e:
                logger.warning(f"Failed to index field '{field}' in '{dst}': {e}")


# ── Two-pass scroll + embed + upsert ─────────────────────────────────────────

def scroll_and_embed(
    client: QdrantClient,
    src_collection: str,
    dst_collection: str,
) -> int:
    """Two-pass migration: dense first, then sparse into the same -hybrid collection.

    Pass 1 (dense): scroll the ORIGINAL collection (read-only), extract text,
        embed via get_dense_vectors, upsert into -hybrid with dense vector + payload.
    Pass 2 (sparse): scroll the -hybrid collection we just populated, extract text,
        embed via embed_document (chunked sparse), upsert same point IDs with
        sparse vector only (Qdrant merges automatically).

    Both passes are idempotent — skip if target point count already matches source.
    """
    src_info = client.get_collection(src_collection)
    dst_info = client.get_collection(dst_collection)
    src_count = src_info.points_count if hasattr(src_info, "points_count") else 0
    dst_count = dst_info.points_count if hasattr(dst_info, "points_count") else 0

    # ── Pass 1: Dense ────────────────────────────────────────────────────
    if src_count > 0 and dst_count == src_count:
        logger.info(
            f"Pass 1 (dense) skipped for {src_collection} → {dst_collection}: "
            f"already fully populated ({dst_count}/{src_count} points)"
        )
    else:
        logger.info(
            f"\nPass 1 (dense): {src_collection} → {dst_collection} "
            f"({src_count} source, {dst_count} already in target)"
        )

        offset = None
        total = 0
        skipped = 0

        while True:
            points, next_offset = client.scroll(
                collection_name=src_collection,
                limit=BATCH_SIZE,
                offset=offset,
                with_vectors=False,
                with_payload=True,
            )
            if not points:
                break

            valid = [p for p in points if p.payload.get("text", "").strip()]
            skipped += len(points) - len(valid)

            if valid:
                texts = [p.payload["text"] for p in valid]
                dense_vecs = get_dense_vectors(texts)

                upsert_points = []
                for p, dvec in zip(valid, dense_vecs):
                    upsert_points.append(
                        PointStruct(
                            id=p.id,
                            vector={"dense": dvec},
                            payload=p.payload,
                        )
                    )

                client.upsert(collection_name=dst_collection, points=upsert_points)
                total += len(upsert_points)
                print(f"  [dense] Upserted {total} points into {dst_collection}...", end="\r")

            if next_offset is None:
                break
            offset = next_offset

        logger.info(
            f"\n  Pass 1 done: {total} points with dense vectors in {dst_collection} "
            f"({skipped} skipped for empty text)"
        )

    # ── Pass 2: Sparse ───────────────────────────────────────────────────
    # Refresh counts after Pass 1
    dst_info = client.get_collection(dst_collection)
    dst_count = dst_info.points_count if hasattr(dst_info, "points_count") else 0

    logger.info(
        f"\nPass 2 (sparse): scrolling {dst_collection} "
        f"({dst_count} points to embed sparse)"
    )

    offset = None
    total = 0
    skipped = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=dst_collection,
            limit=BATCH_SIZE,
            offset=offset,
            with_vectors=False,
            with_payload=True,
        )
        if not points:
            break

        valid = [p for p in points if p.payload.get("text", "").strip()]
        skipped += len(points) - len(valid)

        if valid:
            upsert_points = []
            for p in valid:
                text = p.payload["text"]
                sv = embed_document(text)
                upsert_points.append(
                    PointVectors(
                        id=p.id,
                        vector={
                            "sparse": SparseVector(
                                indices=sv["indices"],
                                values=sv["values"],
                            ),
                        },
                    )
                )

            client.update_vectors(collection_name=dst_collection, points=upsert_points)
            total += len(upsert_points)
            print(f"  [sparse] Upserted {total} points into {dst_collection}...", end="\r")

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        f"\n  Pass 2 done: {total} points with sparse vectors in {dst_collection} "
        f"({skipped} skipped for empty text)"
    )

    return dst_count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not health_check():
        logger.error("Embedding server is not reachable. Aborting.")
        sys.exit(1)
    logger.info("Embedding server health check passed.")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

    existing = [c.name for c in client.get_collections().collections]
    logger.info(f"Existing collections: {existing}")

    create_target_collections(client)

    grand_total = 0
    for src, dst in COLLECTION_MAP.items():
        if src not in existing:
            logger.warning(f"Source collection '{src}' not found. Skipping.")
            continue
        count = scroll_and_embed(client, src, dst)
        grand_total += count

    logger.info(f"\nTwo-pass hybrid migration complete. Total points: {grand_total}")


if __name__ == "__main__":
    main()
