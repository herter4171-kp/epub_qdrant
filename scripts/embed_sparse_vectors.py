#!/usr/bin/env python3
"""Phase 2: Scroll existing Qdrant collections, compute MiniCOIL sparse vectors,
upsert into new named-vector collections (books-named, papers-named).

Run from Mac:
    python3 scripts/embed_sparse_vectors.py

Connects to:
    Qdrant on GPU box: 192.168.68.75:6333
    MiniCOIL server:    192.168.68.75:9000

Long documents are chunked into MAX_WORDS_PER_CHUNK word windows and their
sparse vectors are aggregated by taking the max value per token index across
all chunks. This preserves full document coverage without blowing up the
ONNX attention matrix.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict

# Ensure the project root and MCP server paths are on sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_mcp_server_dir = _project_root / "servers" / "retrieval_mcp"
if str(_mcp_server_dir) not in sys.path:
    sys.path.insert(0, str(_mcp_server_dir))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("embed_sparse")

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector,
    VectorParams,
    SparseVectorParams,
    Distance,
    Modifier,
)

from src.embedding.client import get_sparse_vectors


# ── Configuration ────────────────────────────────────────────────────────────

QDRANT_HOST = "192.168.68.75"
QDRANT_PORT = 6333

BATCH_SIZE = 128          # Points fetched per Qdrant scroll
MINICOIL_BATCH = 32       # Chunks sent per MiniCOIL request — fixed size, no OOM
MAX_WORDS_PER_CHUNK = 512 # Words per chunk — caps attention matrix size

COLLECTION_MAP = {
    "books": "books-named",
    "papers": "papers-named",
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
    """Create books-named and papers-named with named vector configs."""
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


# ── Scroll + embed + upsert ───────────────────────────────────────────────────

def scroll_and_embed(
    client: QdrantClient,
    src_collection: str,
    dst_collection: str,
) -> int:
    """Scroll src_collection, embed via MiniCOIL, upsert into dst_collection.

    Idempotent: skips if dst already has the same point count as src.
    Long documents are chunked and their sparse vectors aggregated.
    """
    src_info = client.get_collection(src_collection)
    dst_info = client.get_collection(dst_collection)
    src_count = src_info.points_count if hasattr(src_info, "points_count") else 0
    dst_count = dst_info.points_count if hasattr(dst_info, "points_count") else 0

    if src_count > 0 and dst_count == src_count:
        logger.info(
            f"Skipping {src_collection} → {dst_collection}: "
            f"already fully embedded ({dst_count}/{src_count} points)"
        )
        return dst_count

    logger.info(
        f"\nProcessing {src_collection} → {dst_collection} "
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
            with_vectors=True,
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
                    PointStruct(
                        id=p.id,
                        vector={
                            "dense": p.vector,  # raw list[float] from unnamed collection
                            "sparse": SparseVector(
                                indices=sv["indices"],
                                values=sv["values"],
                            ),
                        },
                        payload=p.payload,
                    )
                )

            client.upsert(collection_name=dst_collection, points=upsert_points)
            total += len(upsert_points)
            print(f"  Upserted {total} points into {dst_collection}...", end="\r")

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        f"\n  Done: {total} points migrated to {dst_collection} "
        f"({skipped} skipped for empty text)"
    )
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from src.embedding.client import health_check
    if not health_check():
        logger.error("MiniCOIL server is not reachable. Aborting.")
        sys.exit(1)
    logger.info("MiniCOIL server health check passed.")

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

    logger.info(f"\nPhase 2 complete. Total points embedded: {grand_total}")


if __name__ == "__main__":
    main()
