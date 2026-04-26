#!/usr/bin/env python3
"""Bulk ingest MinerU JSON files (content_list_v2.json) into a Qdrant collection.

Walks the MinerU output tree, discovers all content_list_v2.json files,
parses them, chunks sections, and runs the full two-pass dense + sparse
embedding pipeline — identical to ingest_fresh.py's pattern.

Two-phase approach (GPU-efficient):
  Phase 1: chunk ALL papers → embed ALL dense in big batches → upsert all
  Phase 2: scroll entire collection → embed ALL sparse in big batches → upsert all

Single-doc mode (--arxiv-id): per-paper dense-then-sparse for debugging.

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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ── Sparse helpers ────────────────────────────────────────────────────────────

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


# ── Single-paper helpers (for --arxiv-id mode) ────────────────────────────────

def _chunk_paper_to_points(
    arxiv_id: str,
    json_path: Path,
    metadata_dir: str,
    token_counter,
    config: ChunkConfig,
    point_offset: int,
) -> Optional[Tuple[List[PointStruct], int, int, int]]:
    """Parse one JSON, chunk sections, embed dense, upsert.

    Returns (points, sections, chunks, tokens) or None on failure.
    """
    # Parse JSON → sections
    try:
        json_sections = parse_content_list(json_path)
    except Exception as e:
        log.error("JSON parse failed for %s: %s", json_path, e)
        return None

    if not json_sections:
        log.warning("JSON %s produced 0 sections — skipping", json_path)
        return ([], 0, 0, 0)

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

    # Chunk each section (no embedding_fn — dense done in batch below)
    all_chunks: List[Tuple[str, Dict]] = []  # (text, metadata)
    for js in json_sections:
        results = chunk_section(
            title=js.title,
            content=js.content,
            config=config,
            token_counter=token_counter,
            embedding_fn=None,
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
        return ([], len(json_sections), 0, 0)

    # Embed dense in batches
    texts = [c[0] for c in all_chunks]
    all_vecs: List[List[float]] = []
    for b in range(0, len(texts), DENSE_BATCH):
        batch = texts[b : b + DENSE_BATCH]
        all_vecs.extend(get_dense_vectors(batch))

    # Build points
    points: List[PointStruct] = []
    for idx, ((chunk_text, metadata), vec) in enumerate(zip(all_chunks, all_vecs)):
        points.append(PointStruct(
            id=point_offset + idx,
            vector={"dense": vec},
            payload={"text": chunk_text, **metadata},
        ))

    total_tokens = sum(c[1]["token_count"] for c in all_chunks)
    return (points, len(json_sections), len(all_chunks), total_tokens)


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
        pts, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=False,
            with_payload=["text"],
            query_filter=f,
        )
        if not pts:
            break

        upsert_points: List[PointVectors] = []
        for p in pts:
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

    # Filter by arxiv_id if specified — single-paper mode (per-doc dense+sparse)
    if args.arxiv_id:
        normalized = args.arxiv_id.replace(".", "_")
        if normalized not in jsons:
            log.error("No JSON found for arxiv_id=%s", args.arxiv_id)
            sys.exit(1)

        arxiv_id = normalized
        json_path = jsons[arxiv_id]

        # Get current point offset for unique IDs
        try:
            point_offset = client.get_collection(args.collection).points_count or 0
        except Exception:
            point_offset = 0

        # Chunk + embed dense for this single paper
        result = _chunk_paper_to_points(
            arxiv_id, json_path, args.metadata_dir,
            token_counter, config, point_offset,
        )
        if result is None:
            log.error("Single paper %s — FAILED", args.arxiv_id)
            sys.exit(1)

        points, sections, chunks, tokens = result
        client.upsert(collection_name=args.collection, points=points)
        log.info("  dense: %d points upserted", len(points))

        # Pass 2: sparse for this paper
        sparse_count = pass2_sparse_for_paper(client, args.collection, arxiv_id)
        log.info("  [%s] %d sections, %d chunks, %d tokens, %d sparse vectors",
                 args.arxiv_id, sections, chunks, tokens, sparse_count)
        return

    # ── Multi-paper mode: two-phase (GPU-efficient) ──────────────────────

    # Apply limit
    arxiv_ids = sorted(jsons.keys())
    if args.limit:
        arxiv_ids = arxiv_ids[:args.limit]

    log.info("Multi-paper mode: %d papers, two-phase embedding", len(arxiv_ids))

    # ── Phase 1: chunk ALL → embed ALL dense → upsert ALL ────────────────

    # First, check which papers are already ingested (skip them entirely)
    to_process: List[str] = []
    total_skipped = 0

    for arxiv_id in arxiv_ids:
        try:
            f = Filter(must=[FieldCondition(
                key="arxiv_id", match=MatchValue(value=arxiv_id),
            )])
            scroll_pts, _ = client.scroll(
                collection_name=args.collection,
                limit=1,
                with_payload=False,
                with_vectors=False,
                query_filter=f,
            )
            if scroll_pts:
                # Count points for this paper
                count = 0
                off = None
                while True:
                    pts2, noff = client.scroll(
                        collection_name=args.collection,
                        limit=SCROLL_BATCH,
                        offset=off,
                        with_payload=False,
                        with_vectors=False,
                        query_filter=f,
                    )
                    if not pts2:
                        break
                    count += len(pts2)
                    if noff is None:
                        break
                    off = noff
                total_skipped += count
                log.info("  [%d/%d] %s — already ingested (%d chunks), skipping",
                         len(to_process) + 1, len(arxiv_ids), arxiv_id, count)
                continue
        except Exception:
            pass
        to_process.append(arxiv_id)

    log.info("Phase 1: chunking %d papers (skipped %d)", len(to_process), total_skipped)

    # Step 1a: chunk ALL papers into (text, metadata) tuples
    all_chunks: List[Tuple[str, Dict]] = []       # (text, metadata)
    all_ids: List[Tuple[str, int, int]] = []       # (arxiv_id, chunk_start_idx, chunk_count)
    paper_info: Dict[str, dict] = {}               # arxiv_id → {title, category, ...}
    paper_sections: Dict[str, int] = {}             # arxiv_id → section_count

    # Track how many points are already in the collection for ID assignment
    try:
        point_offset = client.get_collection(args.collection).points_count or 0
    except Exception:
        point_offset = 0

    current_chunk_idx = 0

    for i, arxiv_id in enumerate(to_process, 1):
        json_path = jsons[arxiv_id]

        # Parse JSON → sections
        try:
            json_sections = parse_content_list(json_path)
        except Exception as e:
            log.error("  [%d/%d] JSON parse failed for %s: %s", i, len(to_process), json_path, e)
            continue

        if not json_sections:
            log.warning("  [%d/%d] JSON %s produced 0 sections — skipping", i, len(to_process), json_path)
            continue

        # Read sidecar metadata
        meta = read_sidecar(args.metadata_dir, arxiv_id)
        arxiv_id_dot = meta.get("arxiv_id", arxiv_id.replace("_", "."))
        title = meta.get("title", arxiv_id)
        category = meta.get("category", "")
        subcategory = meta.get("subcategory", "")
        authors = meta.get("authors", "")
        publish_date = meta.get("publish_date", "")

        paper_info[arxiv_id] = {
            "arxiv_id": arxiv_id_dot,
            "title": title,
            "category": category,
            "subcategory": subcategory,
            "authors": authors,
            "publish_date": publish_date,
        }

        # Chunk each section
        chunks_for_paper: List[Tuple[str, Dict]] = []
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
                chunks_for_paper.append((
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

        if not chunks_for_paper:
            paper_sections[arxiv_id] = 0
            continue

        all_ids.append((arxiv_id, current_chunk_idx, len(chunks_for_paper)))
        current_chunk_idx += len(chunks_for_paper)
        all_chunks.extend(chunks_for_paper)
        paper_sections[arxiv_id] = len(json_sections)

        log.info("  [%d/%d] %s — %d sections, %d chunks buffered",
                 i, len(to_process), arxiv_id, len(json_sections), len(chunks_for_paper))

    total_buffered = len(all_chunks)
    log.info("Phase 1a: %d total chunks buffered across %d papers", total_buffered, len(to_process))

    if total_buffered == 0:
        log.info("No chunks to embed. Done.")
        return

    # Step 1b: embed ALL dense in batches
    all_texts = [c[0] for c in all_chunks]
    log.info("Phase 1b: embedding %d texts dense...", total_buffered)

    all_vecs: List[List[float]] = []
    for b in range(0, total_buffered, DENSE_BATCH):
        batch = all_texts[b : b + DENSE_BATCH]
        all_vecs.extend(get_dense_vectors(batch))

    log.info("Phase 1b: dense embedding done (%d vectors)", len(all_vecs))

    # Step 1c: upsert ALL points with dense vectors
    log.info("Phase 1c: upserting %d points with dense vectors", total_buffered)

    points: List[PointStruct] = []
    for idx, ((chunk_text, metadata), vec) in enumerate(zip(all_chunks, all_vecs)):
        points.append(PointStruct(
            id=point_offset + idx,
            vector={"dense": vec},
            payload={"text": chunk_text, **metadata},
        ))

    client.upsert(collection_name=args.collection, points=points)
    log.info("Phase 1c: %d points upserted", len(points))

    # ── Phase 2: scroll entire collection → embed ALL sparse → upsert ALL ──

    info = client.get_collection(args.collection)
    total_points = info.points_count or 0
    log.info("Phase 2: scrolling %d points for sparse embedding", total_points)

    sparse_offset = None
    total_sparse = 0
    sparse_points: List[PointVectors] = []

    while True:
        pts, next_offset = client.scroll(
            collection_name=args.collection,
            limit=SCROLL_BATCH,
            offset=sparse_offset,
            with_vectors=False,
            with_payload=["text"],
        )
        if not pts:
            break

        for p in pts:
            text = (p.payload.get("text") or "").strip()
            if not text:
                continue

            windows = _chunk_text_for_sparse(text)
            all_sparse: List[dict] = []
            for b in range(0, len(windows), SPARSE_BATCH):
                sub = windows[b : b + SPARSE_BATCH]
                all_sparse.extend(get_sparse_vectors(sub, is_query=False))

            sv = _aggregate_sparse(all_sparse)
            sparse_points.append(PointVectors(
                id=p.id,
                vector={"sparse": SparseVector(indices=sv["indices"], values=sv["values"])},
            ))

        # Batch upsert sparse vectors for this page
        if sparse_points:
            client.update_vectors(collection_name=args.collection, points=sparse_points)
            total_sparse += len(sparse_points)
            log.info("  [sparse] %d / %d", total_sparse, total_points)
            sparse_points = []

        if next_offset is None:
            break
        sparse_offset = next_offset

    # Flush any remaining sparse points
    if sparse_points:
        client.update_vectors(collection_name=args.collection, points=sparse_points)
        total_sparse += len(sparse_points)

    log.info("Phase 2: %d points with sparse vectors upserted", total_sparse)

    # ── Summary ──────────────────────────────────────────────────────────

    total_chunks_upserted = sum(count for _, _, count in all_ids)
    total_tokens = 0
    for arxiv_id, start, count in all_ids:
        for chunk_text, metadata in all_chunks[start:start+count]:
            total_tokens += metadata.get("token_count", 0)

    log.info("=" * 60)
    log.info("SUMMARY:")
    log.info("  Papers processed: %d", len(to_process))
    log.info("  Papers skipped:   %d", total_skipped)
    log.info("  Total chunks:     %d", total_chunks_upserted)
    log.info("  Total tokens:     %d", total_tokens)
    log.info("  Dense points:     %d", total_buffered)
    log.info("  Sparse vectors:   %d", total_sparse)
    log.info("=" * 60)


if __name__ == "__main__":
    main()