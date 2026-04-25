#!/usr/bin/env python3
"""Bulk ingest MinerU JSON files (content_list_v2.json) into a Qdrant collection.

Walks the MinerU output tree, discovers all content_list_v2.json files,
parses them, chunks sections, and runs the full two-pass dense + sparse
embedding pipeline — identical to ingest_fresh.py's pattern.

Usage:
    python scripts/ingest_mineru_json.py --collection papers-semantic \\
        [--base-dir /tank/scraps/mineru_output] \\
        [--metadata-dir ./downloads] \\
        [--limit 10] \\
        [--arxiv-id 2603.07444]
"""

import argparse
import glob
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
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
from src.ingestion.mineru_json_parser import parse_content_list, resolve_json_path
from src.ingestion.semantic_chunker import ChunkConfig, chunk_section, load_tokenizer

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MINERU_OUTPUT_DIR = "./mineru_output"
DEFAULT_METADATA_DIR = "./downloads"

DENSE_BATCH = 128
SPARSE_BATCH = 32
SCROLL_BATCH = 256
MAX_SPARSE_WORDS = 512

INDEX_FIELDS = [
    "doc_type", "source_file", "arxiv_id", "category", "title",
    "section_title", "chunk_index", "chunk_count",
]

# ── Collection setup ─────────────────────────────────────────────────────────

PROTECTED = {"books", "books-named", "papers", "papers-named"}


def ensure_collection(client: QdrantClient, name: str) -> None:
    """Create a named-vector collection if it doesn't exist."""
    if name in PROTECTED:
        raise ValueError(f"Refusing to overwrite protected collection '{name}'")
    existing = {c.name for c in client.get_collections().collections}
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


# ── JSON discovery ────────────────────────────────────────────────────────────

def discover_json_files(base_dir: str) -> Dict[str, Path]:
    """Discover all content_list_v2.json files.

    Returns {arxiv_id_underscored: json_path} deduplicated.
    Tries tree layout first, then flat layout.
    """
    results: Dict[str, Path] = {}

    # Tree layout: {base}/**/vlm/*_content_list_v2.json
    tree_pattern = str(Path(base_dir) / "**" / "vlm" / "*_content_list_v2.json")
    for p in glob.glob(tree_pattern, recursive=True):
        pp = Path(p)
        arxiv_id = pp.stem.replace("_content_list_v2", "")
        results[arxiv_id] = pp

    # Flat layout: {base}/*_content_list_v2.json (only if arxiv not already found)
    flat_pattern = str(Path(base_dir) / "*_content_list_v2.json")
    for p in glob.glob(flat_pattern, recursive=True):
        pp = Path(p)
        arxiv_id = pp.stem.replace("_content_list_v2", "")
        if arxiv_id not in results:
            results[arxiv_id] = pp

    log.info("Discovered %d JSON files in %s", len(results), base_dir)
    return results


def read_sidecar(metadata_dir: str, arxiv_id: str) -> Dict[str, str]:
    """Read sidecar metadata JSON for an arxiv ID.

    The metadata file uses the "metadataAttributes" list format from the
    _read_sidecar helper.  We look for {arxiv_id_underscored}.json in the
    metadata directory.
    """
    meta_path = Path(metadata_dir) / f"{arxiv_id}.json"
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        result: Dict[str, str] = {}
        for attr in data.get("metadataAttributes", []):
            if ": " in attr:
                k, v = attr.split(": ", 1)
                result[k] = v
            elif ":" in attr:
                k, v = attr.split(":", 1)
                result[k.strip()] = v.strip()
        return result
    except Exception as e:
        log.warning("Failed to parse %s: %s", meta_path.name, e)
        return {}


# ── Two-pass embedding ───────────────────────────────────────────────────────

def _chunk_text_for_sparse(text: str) -> List[str]:
    """Split text into <=512-word windows for sparse embedding."""
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i : i + MAX_SPARSE_WORDS])
        for i in range(0, len(words), MAX_SPARSE_WORDS)
    ]


def _aggregate_sparse(vecs: List[dict]) -> dict:
    """Max-pool sparse vectors across chunks."""
    agg: Dict[int, float] = {}
    for v in vecs:
        for idx, val in zip(v["indices"], v["values"]):
            if idx not in agg or val > agg[idx]:
                agg[idx] = val
    return {"indices": list(agg.keys()), "values": list(agg.values())}


def ingest_paper(
    client: QdrantClient,
    collection: str,
    arxiv_id: str,
    json_path: Path,
    metadata_dir: str,
    token_counter,
    config: ChunkConfig,
) -> Optional[tuple]:
    """Parse one JSON, chunk sections, embed dense, upsert.

    Returns (sections, chunks, tokens) on success, or None on failure.
    """
    # Parse JSON → sections
    try:
        json_sections = parse_content_list(json_path)
    except Exception as e:
        log.error("JSON parse failed for %s: %s", json_path, e)
        return None

    if not json_sections:
        log.warning("JSON %s produced 0 sections — skipping", json_path)
        return (0, 0, 0)

    # Read sidecar metadata
    meta = read_sidecar(metadata_dir, arxiv_id)

    # Bug 3 fix: prefer dot-format arxiv_id from sidecar (e.g. "2201.11903")
    # over the underscore directory key (e.g. "2201_11903").
    arxiv_id_dot = meta.get("arxiv_id", arxiv_id.replace("_", "."))

    # Bug 1 fix: when sidecar is empty, title falls back to underscore arxiv_id.
    # Use a cleaned title that's more human-readable.
    title = meta.get("title", arxiv_id)
    category = meta.get("category", "")
    subcategory = meta.get("subcategory", "")
    authors = meta.get("authors", "")
    publish_date = meta.get("publish_date", "")

    # Chunk each section
    all_chunks: List[tuple] = []  # (text, metadata)
    for js in json_sections:
        results = chunk_section(
            title=js.title,
            content=js.content,
            config=config,
            token_counter=token_counter,
            embedding_fn=None,  # dense embedding done in batch below
        )
        chunk_count = len(results)
        for cr in results:
            all_chunks.append((
                cr.text,
                {
                    "doc_type": "paper",
                    "source_file": f"{arxiv_id}.pdf",
                    "title": title or "",
                    "arxiv_id": arxiv_id_dot,
                    "category": category or "",
                    "subcategory": subcategory or "",
                    "authors": authors or "",
                    "publish_date": publish_date or "",
                    "section_title": cr.section_title or "",
                    "chunk_index": cr.chunk_index,
                    "chunk_count": chunk_count,
                    "token_count": cr.token_count,
                    "has_heading_context": cr.has_heading_context,
                    "heading_level": js.heading_level,
                },
            ))

    if not all_chunks:
        return (len(json_sections), 0, 0)

    # Embed dense in batches
    texts = [c[0] for c in all_chunks]
    all_vecs: List[List[float]] = []
    for b in range(0, len(texts), DENSE_BATCH):
        batch = texts[b : b + DENSE_BATCH]
        all_vecs.extend(get_dense_vectors(batch))

    # Upsert with dense vectors
    # Point IDs: query current collection count so IDs are globally unique across papers
    try:
        point_offset = client.get_collection(collection).points_count or 0
    except Exception:
        point_offset = 0

    points: List[PointStruct] = []
    for idx, ((chunk_text, metadata), vec) in enumerate(zip(all_chunks, all_vecs)):
        points.append(PointStruct(
            id=point_offset + idx,
            vector={"dense": vec},
            payload={"text": chunk_text, **metadata},
        ))

    client.upsert(collection_name=collection, points=points)

    total_tokens = sum(c[1]["token_count"] for c in all_chunks)
    return (len(json_sections), len(all_chunks), total_tokens)


def pass2_sparse_for_paper(
    client: QdrantClient,
    collection: str,
    arxiv_id: str,
) -> int:
    """Scroll this paper's points, embed sparse, upsert same IDs."""
    f = Filter(must=[FieldCondition(
        key="arxiv_id",
        match=MatchValue(value=arxiv_id),
    )])

    offset = None
    total = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=False,
            with_payload=["text"],
            query_filter=f,
        )
        if not points:
            break

        upsert_points: List[PointVectors] = []
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

        if next_offset is None:
            break
        offset = next_offset

    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bulk ingest MinerU JSON files into a Qdrant collection.",
    )
    parser.add_argument("--collection", required=True, help="Target Qdrant collection name")
    parser.add_argument(
        "--base-dir", default=None,
        help="MinerU output directory (overrides MINERU_OUTPUT_DIR env). Defaults to ./mineru_output.",
    )
    parser.add_argument(
        "--metadata-dir", default=DEFAULT_METADATA_DIR,
        help="Directory containing sidecar metadata JSON files. Defaults to ./downloads.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max papers to process")
    parser.add_argument("--arxiv-id", default=None, help="Process a single paper by ID")
    args = parser.parse_args()

    # Pre-flight: health check
    if not health_check():
        log.error("Embedding server not healthy at %s", settings.EMBEDDING_SERVER_URL)
        sys.exit(1)
    log.info("Embedding server OK")

    # Pre-flight: Qdrant
    client = QdrantClient(url=settings.QDRANT_URL)
    log.info("Qdrant OK — existing: %s",
             [c.name for c in client.get_collections().collections])

    # Resolve base directory
    import os
    env = os.environ.get("MINERU_OUTPUT_DIR")
    if args.base_dir:
        base_dir = args.base_dir
    elif env:
        base_dir = env
    else:
        base_dir = DEFAULT_MINERU_OUTPUT_DIR

    # Ensure collection
    ensure_collection(client, args.collection)

    # Chunk config + tokenizer
    token_counter = load_tokenizer(settings.TOKENIZER_JSON or None)
    config = ChunkConfig(
        chunk_size=settings.CHUNK_SIZE,
        overlap_ratio=settings.CHUNK_OVERLAP_RATIO,
        similarity_percentile=settings.SIMILARITY_PERCENTILE,
        min_distance_floor=settings.MIN_DISTANCE_FLOOR,
        min_sentences_for_semantic=settings.MIN_SENTENCES_FOR_SEMANTIC,
        min_chunk_tokens=settings.MIN_CHUNK_TOKENS,
        enable_semantic=settings.SEMANTIC_CHUNKING_ENABLED,
        tokenizer_path=settings.TOKENIZER_JSON or None,
    )

    # Discover JSONs
    jsons = discover_json_files(base_dir)

    # Filter by arxiv_id if specified
    if args.arxiv_id:
        normalized = args.arxiv_id.replace(".", "_")
        if normalized not in jsons:
            log.error("No JSON found for arxiv_id=%s", args.arxiv_id)
            sys.exit(1)
        jsons = {normalized: jsons[normalized]}
        log.info("Single paper mode: %s", args.arxiv_id)

    # Apply limit
    arxiv_ids = sorted(jsons.keys())
    if args.limit:
        arxiv_ids = arxiv_ids[:args.limit]

    # Process each paper
    total_processed = 0
    total_chunks = 0
    total_skipped = 0
    total_failed = 0

    for i, arxiv_id in enumerate(arxiv_ids, 1):
        json_path = jsons[arxiv_id]

        # Idempotency: check if already ingested
        try:
            info = client.get_collection(args.collection)
        except Exception:
            info = None

        if info is not None:
            # Count this paper's points
            f = Filter(must=[FieldCondition(
                key="arxiv_id", match=MatchValue(value=arxiv_id),
            )])
            try:
                scroll_pts, _ = client.scroll(
                    collection_name=args.collection,
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                    query_filter=f,
                )
                if scroll_pts:
                    # Already has this paper — count total points for it
                    count = 0
                    offset2 = None
                    while True:
                        pts, next_off = client.scroll(
                            collection_name=args.collection,
                            limit=SCROLL_BATCH,
                            offset=offset2,
                            with_payload=False,
                            with_vectors=False,
                            query_filter=f,
                        )
                        if not pts:
                            break
                        count += len(pts)
                        if next_off is None:
                            break
                        offset2 = next_off
                    total_skipped += count
                    log.info("  [%d/%d] %s — already ingested (%d chunks), skipping",
                             i, len(arxiv_ids), arxiv_id, count)
                    continue
            except Exception:
                pass

        # Ingest this paper
        result = ingest_paper(
            client, args.collection, arxiv_id, json_path,
            args.metadata_dir, token_counter, config,
        )

        if result is None:
            log.error("  [%d/%d] %s — FAILED", i, len(arxiv_ids), arxiv_id)
            total_failed += 1
            continue

        sections, chunks, tokens = result
        total_processed += 1
        total_chunks += chunks

        # Pass 2: sparse embedding for this paper's points
        pass2_sparse_for_paper(client, args.collection, arxiv_id)

        log.info("  [%d/%d] %s — %d sections, %d chunks, %d tokens",
                 i, len(arxiv_ids), arxiv_id, sections, chunks, tokens)

    # Final summary
    log.info("=" * 60)
    log.info("SUMMARY:")
    log.info("  Processed: %d", total_processed)
    log.info("  Chunks:    %d", total_chunks)
    log.info("  Skipped:   %d", total_skipped)
    log.info("  Failed:    %d", total_failed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()

