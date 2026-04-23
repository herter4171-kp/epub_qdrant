#!/usr/bin/env python3
"""Ingest books (EPUBs) and papers (PDFs) into *-fresh named-vector collections.

Uses a polymorphic DocumentLoader so the embedding pipeline is written once.
Two-pass approach per the spec:
  Pass 1: embed dense → upsert with {"dense": vec} + payload
  Pass 2: scroll back → embed sparse → upsert same point IDs with {"sparse": ...}

Collections created:
  books-fresh   — EPUBs from ./test_books/
  papers-fresh  — PDFs  from ./downloads/

Usage:
    .venv/bin/python scripts/ingest_fresh.py                     # all books + papers
    .venv/bin/python scripts/ingest_fresh.py --books-only        # just EPUBs
    .venv/bin/python scripts/ingest_fresh.py --papers-only       # just PDFs
    .venv/bin/python scripts/ingest_fresh.py --limit 1           # one file per type
    .venv/bin/python scripts/ingest_fresh.py --cleanup           # delete -fresh after
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Import config FIRST so load_dotenv sets EMBEDDING_SERVER_URL before the
# client module reads os.getenv at import time.
from src.config import settings  # noqa: E402 — triggers load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    PointVectors,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from servers.embedding_server.client import (
    get_dense_vectors,
    get_sparse_vectors,
    health_check,
)
from src.config import settings
from src.ingestion.loader import DocumentChunk, DocumentLoader

# ── Paths ─────────────────────────────────────────────────────────────────────

BOOKS_DIR = _PROJECT_ROOT / "test_books"
PAPERS_DIR = _PROJECT_ROOT / "downloads"

PROTECTED = {"books", "books-named", "papers", "papers-named"}

DENSE_BATCH = 128
SPARSE_BATCH = 32
SCROLL_BATCH = 256
MAX_SPARSE_WORDS = 512


# ── Collection setup ──────────────────────────────────────────────────────────

INDEX_FIELDS = [
    "doc_type", "source_file", "book_title", "section_title",
    "publisher", "language", "isbn",
    "arxiv_id", "category", "title",
]


def ensure_named_collection(client: QdrantClient, name: str) -> None:
    """Create a collection with dense + sparse named vectors if it doesn't exist."""
    assert name not in PROTECTED, f"Refusing to touch protected collection '{name}'"
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        log.info("Collection '%s' already exists — reusing.", name)
        return

    log.info("Creating collection: %s", name)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    for field in INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema="keyword",
            )
        except Exception as e:
            log.warning("Index '%s' on '%s': %s", field, name, e)


# ── Pass 1: load files → chunk → embed dense → upsert ────────────────────────

def pass1_dense(
    client: QdrantClient,
    collection: str,
    files: List[Path],
) -> int:
    """Load documents, chunk, embed dense, upsert with named dense vector."""
    log.info("Pass 1 (dense): %d files → %s", len(files), collection)

    # Check what's already ingested
    try:
        info = client.get_collection(collection)
        point_offset = info.points_count or 0
    except Exception:
        point_offset = 0

    total = 0
    for i, fpath in enumerate(files, 1):
        loader = DocumentLoader.for_path(fpath)
        chunks = loader.load(fpath)
        if not chunks:
            log.warning("  [%d/%d] %s — 0 chunks, skipping", i, len(files), fpath.name)
            continue

        # Embed dense in batches
        texts = [c.text for c in chunks]
        all_vecs: List[List[float]] = []
        for b in range(0, len(texts), DENSE_BATCH):
            batch = texts[b : b + DENSE_BATCH]
            all_vecs.extend(get_dense_vectors(batch))

        # Build points
        points: List[PointStruct] = []
        for idx, (chunk, vec) in enumerate(zip(chunks, all_vecs)):
            pid = point_offset + total + idx
            points.append(PointStruct(
                id=pid,
                vector={"dense": vec},
                payload={"text": chunk.text, **chunk.metadata},
            ))

        client.upsert(collection_name=collection, points=points)
        total += len(points)
        log.info(
            "  [%d/%d] %s — %d chunks upserted (running total: %d)",
            i, len(files), fpath.name, len(points), total,
        )

    log.info("Pass 1 done: %d points with dense vectors in '%s'", total, collection)
    return total


# ── Pass 2: scroll collection → embed sparse → upsert same IDs ───────────────

def _chunk_text_for_sparse(text: str) -> List[str]:
    """Split text into ≤512-word windows for sparse embedding."""
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i : i + MAX_SPARSE_WORDS])
        for i in range(0, len(words), MAX_SPARSE_WORDS)
    ]


def _aggregate_sparse(vecs: List[dict]) -> dict:
    """Max-pool sparse vectors across chunks."""
    agg: dict[int, float] = {}
    for v in vecs:
        for idx, val in zip(v["indices"], v["values"]):
            if idx not in agg or val > agg[idx]:
                agg[idx] = val
    return {"indices": list(agg.keys()), "values": list(agg.values())}


def pass2_sparse(client: QdrantClient, collection: str) -> int:
    """Scroll the collection, embed sparse, upsert same point IDs."""
    info = client.get_collection(collection)
    count = info.points_count or 0
    log.info("Pass 2 (sparse): scrolling %d points in '%s'", count, collection)

    offset = None
    total = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=False,
            with_payload=["text"],
        )
        if not points:
            break

        upsert_points: List[PointStruct] = []
        for p in points:
            text = (p.payload.get("text") or "").strip()
            if not text:
                continue

            windows = _chunk_text_for_sparse(text)
            all_sparse: List[dict] = []
            for b in range(0, len(windows), SPARSE_BATCH):
                sub = windows[b : b + SPARSE_BATCH]
                all_sparse.extend(get_sparse_vectors(sub, is_query=False))

            sv = _aggregate_sparse(all_sparse)
            upsert_points.append(PointVectors(
                id=p.id,
                vector={"sparse": SparseVector(indices=sv["indices"], values=sv["values"])},
            ))

        if upsert_points:
            client.update_vectors(collection_name=collection, points=upsert_points)
            total += len(upsert_points)
            log.info("  [sparse] %d / %d", total, count)

        if next_offset is None:
            break
        offset = next_offset

    log.info("Pass 2 done: %d points with sparse vectors in '%s'", total, collection)
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest books + papers into -fresh collections.")
    parser.add_argument("--books-only", action="store_true")
    parser.add_argument("--papers-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Max files per type")
    parser.add_argument("--cleanup", action="store_true", help="Delete -fresh collections after")
    args = parser.parse_args()

    # Pre-flight
    if not health_check():
        log.error("Embedding server not healthy at %s", settings.EMBEDDING_SERVER_URL)
        sys.exit(1)
    log.info("Embedding server OK")

    client = QdrantClient(host="192.168.68.75", port=6333)
    log.info("Qdrant OK — existing: %s",
             [c.name for c in client.get_collections().collections])

    do_books = not args.papers_only
    do_papers = not args.books_only

    # ── Books ─────────────────────────────────────────────────────────
    if do_books:
        book_files = sorted(BOOKS_DIR.glob("*.epub"))
        if args.limit:
            book_files = book_files[: args.limit]
        if book_files:
            ensure_named_collection(client, "books-fresh")
            pass1_dense(client, "books-fresh", book_files)
            pass2_sparse(client, "books-fresh")
        else:
            log.warning("No EPUBs found in %s", BOOKS_DIR)

    # ── Papers ────────────────────────────────────────────────────────
    if do_papers:
        paper_files = sorted(PAPERS_DIR.glob("*.pdf"))
        if args.limit:
            paper_files = paper_files[: args.limit]
        if paper_files:
            ensure_named_collection(client, "papers-fresh")
            pass1_dense(client, "papers-fresh", paper_files)
            pass2_sparse(client, "papers-fresh")
        else:
            log.warning("No PDFs found in %s", PAPERS_DIR)

    # ── Cleanup ───────────────────────────────────────────────────────
    if args.cleanup:
        for name in ("books-fresh", "papers-fresh"):
            try:
                client.delete_collection(name)
                log.info("Deleted '%s'", name)
            except Exception:
                pass

    log.info("Done.")


if __name__ == "__main__":
    main()
