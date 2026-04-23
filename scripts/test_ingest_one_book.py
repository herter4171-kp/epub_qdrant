#!/usr/bin/env python3
"""End-to-end test: ingest ONE EPUB from test_books/ into a throwaway Qdrant collection.

This validates the full pipeline:
  1. Parse EPUB → extract metadata + sections
  2. Chunk sections
  3. Embed chunks via the unified embedding server
  4. Upsert into a NEW test collection (never touches production collections)
  5. Run a sample search to verify retrieval works

Usage:
    python scripts/test_ingest_one_book.py
    python scripts/test_ingest_one_book.py --book masteringretrieval-augmentedgeneration
    python scripts/test_ingest_one_book.py --cleanup   # delete the test collection after

The test collection is named 'test-ingest-<timestamp>' to avoid collisions.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import settings
from src.ingestion.epub_parser import parse_epub
from src.ingestion.chunker import chunk_section, Chunk
from servers.embedding_server.client import get_dense_vectors, health_check
from src.storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BOOKS_DIR = _PROJECT_ROOT / "test_books"

# Production collections we must NEVER touch
PROTECTED_COLLECTIONS = {"books", "books-named", "papers", "papers-named"}


def pick_book(name_hint: str = None) -> Path:
    """Pick one EPUB from test_books/. If name_hint given, fuzzy-match it."""
    epubs = sorted(BOOKS_DIR.glob("*.epub"))
    if not epubs:
        log.error("No .epub files found in %s", BOOKS_DIR)
        sys.exit(1)

    if name_hint:
        for ep in epubs:
            if name_hint.lower() in ep.stem.lower():
                return ep
        log.error("No EPUB matching '%s' in %s", name_hint, BOOKS_DIR)
        sys.exit(1)

    # Default: pick the first one alphabetically
    return epubs[0]


def main():
    parser = argparse.ArgumentParser(description="Test ingest one EPUB into a throwaway collection.")
    parser.add_argument("--book", default=None, help="Substring to match an EPUB filename")
    parser.add_argument("--cleanup", action="store_true", help="Delete the test collection after verification")
    args = parser.parse_args()

    # ── 0. Pre-flight checks ──────────────────────────────────────────
    log.info("Checking embedding server health...")
    if not health_check():
        log.error("Embedding server is not healthy. Is it running at %s?", settings.EMBEDDING_SERVER_URL)
        sys.exit(1)
    log.info("Embedding server OK (dense + sparse models loaded)")

    storage = Storage()
    existing = set(storage.list_collections())
    log.info("Existing Qdrant collections: %s", existing)

    # ── 1. Pick a book ────────────────────────────────────────────────
    epub_path = pick_book(args.book)
    log.info("Selected book: %s", epub_path.name)

    # ── 2. Parse EPUB ─────────────────────────────────────────────────
    log.info("Parsing EPUB...")
    book = parse_epub(str(epub_path))
    log.info("  Title:     %s", book.title)
    log.info("  Creator:   %s", book.creator)
    log.info("  Publisher:  %s", book.publisher)
    log.info("  Date:       %s", book.publication_date)
    log.info("  Language:   %s", book.language)
    log.info("  ISBN:       %s", book.isbn)
    log.info("  Sections:   %d", len(book.sections))

    if not book.sections:
        log.error("No sections extracted — EPUB may be DRM-protected or empty.")
        sys.exit(1)

    # ── 3. Chunk ──────────────────────────────────────────────────────
    log.info("Chunking sections (chunk_size=%d, overlap=%d)...", settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    all_chunks: list[Chunk] = []
    for section in book.sections:
        chunks = chunk_section(
            section,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            book_title=book.title,
            publisher=book.publisher,
            language=book.language,
            isbn=book.isbn,
        )
        all_chunks.extend(chunks)

    log.info("  Total chunks: %d", len(all_chunks))
    if not all_chunks:
        log.error("No chunks generated. Something is wrong with the EPUB content.")
        sys.exit(1)

    # Show a sample chunk
    sample = all_chunks[0]
    log.info("  Sample chunk [0]: section='%s', tokens=%d, text='%s...'",
             sample.section_title, sample.token_count, sample.text[:120].replace('\n', ' '))

    # ── 4. Embed ──────────────────────────────────────────────────────
    log.info("Embedding %d chunks via unified embedding server...", len(all_chunks))
    texts = [c.text for c in all_chunks]

    BATCH = 128
    all_vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        log.info("  Embedding batch %d-%d / %d", i, min(i + BATCH, len(texts)), len(texts))
        vectors = get_dense_vectors(batch)
        all_vectors.extend(vectors)

    for chunk, vec in zip(all_chunks, all_vectors):
        chunk.vector = vec

    embedded_count = sum(1 for c in all_chunks if c.vector is not None)
    log.info("  Embedded: %d / %d chunks", embedded_count, len(all_chunks))

    # ── 5. Upsert into a NEW test collection ──────────────────────────
    ts = int(time.time())
    test_collection = f"test-ingest-{ts}"
    assert test_collection not in PROTECTED_COLLECTIONS, "BUG: test collection name collides with production!"

    log.info("Upserting into test collection: %s", test_collection)
    count = storage.upsert_file(str(epub_path), all_chunks, collection_name=test_collection)
    log.info("  Upserted %d chunks", count)

    # ── 6. Verify with a search ───────────────────────────────────────
    log.info("Running verification search...")
    query = "What is retrieval augmented generation?"
    results = storage.search(test_collection, query, top_k=3)

    if results:
        log.info("  Search returned %d results for '%s':", len(results), query)
        for i, r in enumerate(results):
            log.info("    [%d] score=%.4f section='%s' text='%s...'",
                     i, r["score"], r.get("section_title", ""), r["text"][:100].replace('\n', ' '))
    else:
        log.warning("  Search returned 0 results — something may be wrong.")

    # ── 7. Summary ────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("TEST INGEST SUMMARY")
    log.info("=" * 60)
    log.info("  Book:        %s", book.title)
    log.info("  Sections:    %d", len(book.sections))
    log.info("  Chunks:      %d", len(all_chunks))
    log.info("  Embedded:    %d", embedded_count)
    log.info("  Upserted:    %d", count)
    log.info("  Collection:  %s", test_collection)
    log.info("  Search OK:   %s", "YES" if results else "NO")
    log.info("=" * 60)

    # ── 8. Cleanup (optional) ─────────────────────────────────────────
    if args.cleanup:
        log.info("Cleaning up: deleting test collection '%s'...", test_collection)
        storage.delete_collection(test_collection)
        log.info("  Deleted.")
    else:
        log.info("Test collection '%s' left in Qdrant. Use --cleanup to remove it.", test_collection)


if __name__ == "__main__":
    main()
