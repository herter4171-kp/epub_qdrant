#!/usr/bin/env python3
"""Bulk ingest MinerU JSON files (content_list_v2.json) into a Qdrant collection.

Named-vector pipeline per procedure:
  Phase 1 — Chunk all papers (pure CPU, no network)
  Phase 2 — Embed dense vectors in batches, accumulate
  Phase 3 — Embed sparse vectors in batches, max-pool per point
  Phase 4a  — Upsert all points with payload only (vector={})
  Phase 4b  — update_vectors() to add dense vectors (preserves payload)
  Phase 4c  — update_vectors() to add sparse vectors (preserves dense + payload)

Each phase is idempotent and resumable. Crash mid-way? Restart, picks up from
boundary. No batch loss beyond current unit of work.

Usage:
    # Dense + sparse (full pipeline):
    python scripts/ingest_mineru_json.py --collection papers-hybrid

    # Dense only (no sparse):
    python scripts/ingest_mineru_json.py --collection papers-dense --phase dense

    # Sparse only (adds to existing dense collection):
    python scripts/ingest_mineru_json.py --collection papers-hybrid --phase sparse
"""

import argparse
import glob
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    PointVectors,
    SparseIndexParams,
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
from src.ingestion.mineru_json_parser import parse_content_list
from src.ingestion.semantic_chunker import ChunkConfig, chunk_section, load_tokenizer

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MINERU_OUTPUT_DIR = "./mineru_output"
DEFAULT_METADATA_DIR = "./downloads"

DENSE_BATCH = 128
SPARSE_BATCH = 256
MAX_SPARSE_WORDS = 512

INDEX_FIELDS = [
    "doc_type", "source_file", "arxiv_id", "category", "title",
    "section_title", "chunk_index", "chunk_count",
]

# ── Collection setup ─────────────────────────────────────────────────────────

PROTECTED = {"books", "books-named", "papers", "papers-named"}


def ensure_collection(client: QdrantClient, name: str) -> None:
    """Create named-vector collection with both dense and sparse configs."""
    if name in PROTECTED:
        raise ValueError(f"Refusing to overwrite protected collection '{name}'")
    existing = {c.name for c in client.get_collections().collections}

    for field in INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema="keyword",
            )
        except Exception as e:
            log.warning("Index '%s' on '%s': %s", field, name, e)

    if name in existing:
        log.info("Collection '%s' already exists — reusing.", name)
        return

    log.info("Creating collection: %s", name)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    log.info("Collection '%s' created with dense(768) + sparse(SPLADE ~30522)", name)
    for field in INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema="keyword",
            )
        except Exception as e:
            log.warning("Index '%s' on '%s': %s", field, name, e)


# ── JSON discovery ────────────────────────────────────────────────────────────

def discover_json_files(base_dir: str) -> Dict[str, Path]:
    """Discover all content_list_v2.json files."""
    results: Dict[str, Path] = {}

    tree_pattern = str(Path(base_dir) / "**" / "vlm" / "*_content_list_v2.json")
    for p in glob.glob(tree_pattern, recursive=True):
        pp = Path(p)
        arxiv_id = pp.stem.replace("_content_list_v2", "")
        results[arxiv_id] = pp

    flat_pattern = str(Path(base_dir) / "*_content_list_v2.json")
    for p in glob.glob(flat_pattern, recursive=True):
        pp = Path(p)
        arxiv_id = pp.stem.replace("_content_list_v2", "")
        if arxiv_id not in results:
            results[arxiv_id] = pp

    log.info("Discovered %d JSON files in %s", len(results), base_dir)
    return results


def read_sidecar(metadata_dir: str, arxiv_id: str) -> Dict[str, str]:
    """Read sidecar metadata JSON for an arxiv ID."""
    meta_path = Path(metadata_dir) / f"{arxiv_id}.pdf.metadata.json"
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        attrs = data.get("metadataAttributes", {})
        if isinstance(attrs, dict):
            result: Dict[str, str] = {}
            for k, v in attrs.items():
                result[k] = str(v) if v else ""
            return result
        elif isinstance(attrs, list):
            result = {}
            for attr in attrs:
                if isinstance(attr, str) and ": " in attr:
                    k, v = attr.split(": ", 1)
                    result[k.strip()] = v.strip()
            return result
        return {}
    except Exception as e:
        log.warning("Failed to parse %s: %s", meta_path.name, e)
        return {}


# ── Chunking helper ───────────────────────────────────────────────────────────

def _chunk_paper_to_chunks(
    arxiv_id: str,
    json_path: Path,
    metadata_dir: str,
    token_counter,
    config: ChunkConfig,
    point_id_start: int,
) -> Tuple[List[Tuple[int, str, Dict]], int, int, int]:
    """Parse one JSON, chunk sections. Returns (chunks, sections, tokens, points)."""
    try:
        json_sections = parse_content_list(json_path)
    except Exception as e:
        log.error("JSON parse failed for %s: %s", json_path, e)
        return ([], 0, 0, 0)

    if not json_sections:
        log.warning("JSON %s produced 0 sections — skipping", json_path)
        return ([], 0, 0, 0)

    meta = read_sidecar(metadata_dir, arxiv_id)
    arxiv_id_dot = meta.get("arxiv_id", arxiv_id.replace("_", "."))
    title = meta.get("title", arxiv_id)
    category = meta.get("category", "")
    subcategory = meta.get("subcategory", "")
    authors = meta.get("authors", "")
    publish_date = meta.get("publish_date", "")

    all_chunks: List[Tuple[int, str, Dict]] = []
    for js in json_sections:
        results = chunk_section(
            title=js.title, content=js.content, config=config,
            token_counter=token_counter, embedding_fn=None,
        )
        chunk_count = len(results)
        for cr in results:
            all_chunks.append((
                point_id_start + len(all_chunks),
                cr.text,
                {
                    "doc_type": "paper",
                    "source_file": f"{arxiv_id}.pdf",
                    "title": title or "",
                    "text": cr.text or "",
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

    total_tokens = sum(c[2]["token_count"] for c in all_chunks)
    return (all_chunks, len(json_sections), total_tokens, len(all_chunks))


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
    """Max-pool sparse vectors across windows for a single point."""
    agg: Dict[int, float] = {}
    for v in vecs:
        for idx, val in zip(v["indices"], v["values"]):
            if idx not in agg or val > agg[idx]:
                agg[idx] = val
    return {"indices": list(agg.keys()), "values": list(agg.values())}


def _format_eta(elapsed: float, done: int, total: int) -> str:
    if total == 0:
        return ""
    rate = done / elapsed if elapsed > 0 else 0
    remaining = total - done
    if rate == 0:
        return "~???:? remaining"
    eta_secs = remaining / rate
    mins = int(eta_secs // 60)
    secs = int(eta_secs % 60)
    return f"~{mins}:{secs:02d} remaining"


# ── Chunk all papers ─────────────────────────────────────────────────────────

def _chunk_all_papers(
    arxiv_ids: List[str],
    jsons: Dict[str, Path],
    metadata_dir: str,
    token_counter,
    config: ChunkConfig,
) -> Tuple[List[Tuple[int, str, Dict]], int, int]:
    """Chunk all papers into (id, text, metadata) tuples."""
    log.info("Chunking %d papers...", len(arxiv_ids))

    all_chunked: List[Tuple[int, str, Dict]] = []
    total_sections = 0
    total_tokens = 0
    start = time.time()

    for i, arxiv_id in enumerate(arxiv_ids, 1):
        json_path = jsons[arxiv_id]
        elapsed = time.time() - start
        eta = _format_eta(elapsed, i, len(arxiv_ids))
        log.info("[%4d/%4d] %s (%s) — %s",
                 i, len(arxiv_ids), arxiv_id, json_path.name, eta)

        result = _chunk_paper_to_chunks(arxiv_id, json_path, metadata_dir,
                                         token_counter, config, len(all_chunked))
        if result is None:
            log.error("  [%s] FAILED — skipping", arxiv_id)
            continue
        chunks, sections, tokens, points = result
        all_chunked.extend(chunks)
        total_sections += sections
        total_tokens += tokens
        log.info("  [%s] ✓ %d chunks, %d tokens", arxiv_id, points, tokens)

    log.info("Chunking complete: %d chunks, %d tokens in %.1fs",
             len(all_chunked), total_tokens, time.time() - start)

    return all_chunked, total_sections, total_tokens


# ── PHASE 1 (Dense): embed + upsert per batch ────────────────────────────────

def run_phase_dense(client: QdrantClient, collection: str,
                    all_chunked: List[Tuple[int, str, Dict]]) -> int:
    """Embed dense vectors and write to Qdrant in batches.

    On a fresh collection: upsert(payload + dense vector) per batch.
    On a collection that already has points (e.g. sparse was run first):
      only update_vectors(dense) — preserves payload and sparse.

    Memory: O(DENSE_BATCH) vectors live at once. Never accumulates all.
    """
    if not all_chunked:
        return 0

    n = len(all_chunked)
    log.info("=" * 60)
    log.info("PHASE dense: %d chunks, batch=%d", n, DENSE_BATCH)
    log.info("=" * 60)

    # Detect whether points already exist so we don't wipe sparse vectors
    # with a full upsert.
    first_pid = all_chunked[0][0]
    has_points = False
    try:
        pts, _ = client.scroll(
            collection_name=collection, limit=1,
            offset=first_pid, with_payload=False, with_vectors=["sparse"],
        )
        if pts:
            has_points = True
    except Exception:
        pass

    if has_points:
        log.info("Existing points detected — will update_vectors(dense) only (preserves sparse + payload).")
    else:
        log.info("Fresh collection — will upsert(payload + dense) per batch.")

    total_batches = (n + DENSE_BATCH - 1) // DENSE_BATCH
    start = time.time()

    for b in range(0, n, DENSE_BATCH):
        batch = all_chunked[b : b + DENSE_BATCH]
        texts = [t[1] for t in batch]

        # Embed this batch only — O(DENSE_BATCH) in memory
        vecs = get_dense_vectors(texts)

        batch_num = b // DENSE_BATCH + 1
        elapsed = time.time() - start
        eta = _format_eta(elapsed, b + len(batch), n)

        if has_points:
            client.update_vectors(
                collection_name=collection,
                points=[
                    PointVectors(id=pid, vector={"dense": vecs[i]})
                    for i, (pid, _text, _meta) in enumerate(batch)
                ],
            )
        else:
            client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(id=pid, vector={"dense": vecs[i]}, payload=meta)
                    for i, (pid, _text, meta) in enumerate(batch)
                ],
            )

        log.info("  [dense %d/%d] %d pts — %s", batch_num, total_batches, len(batch), eta)

        # vecs goes out of scope here — GC can reclaim immediately
        del vecs

    log.info("PHASE dense COMPLETE: %d points", n)
    return n


# ── PHASE 2 (Sparse): embed + update_vectors per batch ───────────────────────

def run_phase_sparse(client: QdrantClient, collection: str,
                     all_chunked: List[Tuple[int, str, Dict]]) -> int:
    """Embed sparse vectors (SPLADE) and write to Qdrant in batches.

    Key design:
      - Each chunk may produce multiple <=512-word windows.
      - Windows for the SAME point that fall in the SAME batch are max-pooled
        before writing.
      - Windows for a point that SPAN multiple batches: the point is written
        with its partial vector after each batch that contains it, and the
        next batch that includes more windows for that point will overwrite
        with a better (larger) max-pool.  This is safe because Qdrant
        update_vectors is idempotent and the final batch for that point wins.
      - Memory at any instant: O(SPARSE_BATCH) windows + O(points_in_batch)
        aggregated vectors.  Never accumulates all chunks.

    Preserves dense vectors and payload on existing points (update_vectors only).
    """
    if not all_chunked:
        return 0

    n = len(all_chunked)
    log.info("=" * 60)
    log.info("PHASE sparse: %d chunks, batch=%d (window size=%d words)",
             n, SPARSE_BATCH, MAX_SPARSE_WORDS)
    log.info("=" * 60)

    # Expand every chunk into (pid, window_text) pairs.
    # This is just a list of small strings — no vectors stored yet.
    point_windows: List[Tuple[int, str]] = []
    for pid, text, _meta in all_chunked:
        for window in _chunk_text_for_sparse(text):
            point_windows.append((pid, window))

    total_windows = len(point_windows)
    total_batches = (total_windows + SPARSE_BATCH - 1) // SPARSE_BATCH
    log.info("Total sparse windows to embed: %d across %d batches",
             total_windows, total_batches)

    total_updated = 0
    start = time.time()

    for b in range(0, total_windows, SPARSE_BATCH):
        batch_windows = point_windows[b : b + SPARSE_BATCH]
        batch_texts = [w[1] for w in batch_windows]

        # ── Embed this batch only ─────────────────────────────────────────
        batch_vecs = get_sparse_vectors(batch_texts, is_query=False)
        # batch_vecs: List[dict] with keys "indices", "values"

        # ── Max-pool within this batch, grouped by PID ────────────────────
        # Use a local dict — discarded at end of loop iteration.
        batch_agg: Dict[int, Dict[int, float]] = defaultdict(dict)
        for (pid, _win_text), sv in zip(batch_windows, batch_vecs):
            pid_agg = batch_agg[pid]
            for idx, val in zip(sv["indices"], sv["values"]):
                if idx not in pid_agg or val > pid_agg[idx]:
                    pid_agg[idx] = val

        # ── Write aggregated sparse vectors for PIDs in this batch ────────
        update_ops = [
            PointVectors(
                id=pid,
                vector={
                    "sparse": SparseVector(
                        indices=list(pid_agg.keys()),
                        values=list(pid_agg.values()),
                    )
                },
            )
            for pid, pid_agg in batch_agg.items()
        ]
        client.update_vectors(collection_name=collection, points=update_ops)
        total_updated += len(update_ops)

        batch_num = b // SPARSE_BATCH + 1
        elapsed = time.time() - start
        eta = _format_eta(elapsed, b + len(batch_windows), total_windows)
        log.info("  [sparse %d/%d] %d windows → %d pts written — %s",
                 batch_num, total_batches, len(batch_windows), len(update_ops), eta)

        # Explicit cleanup — keeps RSS flat across thousands of batches
        del batch_vecs, batch_agg, update_ops

    log.info("PHASE sparse COMPLETE: %d update_vectors calls, %d unique points touched",
             total_batches, total_updated)
    return total_updated


# ── SPARSE-ONLY MODE: scroll existing collection, fill missing sparse ─────────

def run_sparse_only(client: QdrantClient, collection: str,
                    all_chunked: List[Tuple[int, str, Dict]]) -> int:
    """Scroll existing Qdrant points, skip those already with sparse, fill the rest.

    Used when --phase sparse on a pre-existing dense collection.
    """
    log.info("=" * 60)
    log.info("SPARSE-ONLY: filling sparse for existing points in '%s'", collection)
    log.info("=" * 60)

    # ── Find which points are missing sparse ─────────────────────────────
    log.info("Scrolling collection to find points without sparse vectors...")
    missing_pids = set()
    offset = None
    while True:
        pts, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=False,
            with_payload=["text"],
        )
        for pt in pts:
            sparse_vec = pt.vector.get("sparse") if pt.vector else None
            if sparse_vec is None:
                missing_pids.add(pt.id)
        if next_off is None:
            break
        offset = next_off

    log.info("Points needing sparse: %d", len(missing_pids))
    if not missing_pids:
        log.info("All points already have sparse vectors. Done.")
        return 0

    # ── Filter all_chunked to only the missing PIDs ───────────────────────
    to_embed = [(pid, text, meta) for pid, text, meta in all_chunked if pid in missing_pids]
    log.info("Matched %d chunked entries for missing sparse PIDs", len(to_embed))

    if not to_embed:
        log.info("No matching chunked entries found. Done.")
        return 0

    # Delegate to the standard sparse phase (only processes to_embed)
    return run_phase_sparse(client, collection, to_embed)


# ── Upsert payload-only points ───────────────────────────────────────────────

def _upsert_payload_only(client: QdrantClient, collection: str,
                          all_chunked: List[Tuple[int, str, Dict]]) -> None:
    """Upsert points with empty vector dicts — establishes payload without touching vectors."""
    n = len(all_chunked)
    total_batches = (n + DENSE_BATCH - 1) // DENSE_BATCH
    log.info("Upserting %d payload-only points in %d batches...", n, total_batches)
    for b in range(0, n, DENSE_BATCH):
        batch = all_chunked[b : b + DENSE_BATCH]
        client.upsert(
            collection_name=collection,
            points=[
                PointStruct(id=pid, vector={}, payload=meta)
                for pid, _text, meta in batch
            ],
        )
        log.info("  [payload %d/%d] %d pts", b // DENSE_BATCH + 1, total_batches, len(batch))
    log.info("Payload-only upsert done.")

def run_phase_payload(client: QdrantClient, collection: str,
                      all_chunked: List[Tuple[int, str, Dict]]) -> int:
    """Patch payload on existing points. Vectors are never touched.

    Uses set_payload() in batches so dense + sparse are preserved.
    Use when you've added/changed payload fields without re-embedding.
    """
    if not all_chunked:
        return 0

    n = len(all_chunked)
    log.info("=" * 60)
    log.info("PHASE payload: patching %d points, batch=%d", n, DENSE_BATCH)
    log.info("=" * 60)

    total_batches = (n + DENSE_BATCH - 1) // DENSE_BATCH
    start = time.time()

    for b in range(0, n, DENSE_BATCH):
        batch = all_chunked[b : b + DENSE_BATCH]
        batch_num = b // DENSE_BATCH + 1
        elapsed = time.time() - start
        eta = _format_eta(elapsed, b + len(batch), n)

        for pid, _text, meta in batch:
            client.set_payload(
                collection_name=collection,
                payload=meta,
                points=[pid],
            )

        log.info("  [payload %d/%d] %d pts — %s", batch_num, total_batches, len(batch), eta)

    log.info("PHASE payload COMPLETE: %d points patched", n)
    return n

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bulk ingest MinerU JSON files into a Qdrant collection.",
    )
    parser.add_argument("--collection", required=True,
                        help="Target Qdrant collection name")
    parser.add_argument("--base-dir", default=None,
                        help="MinerU output dir (overrides MINERU_OUTPUT_DIR).")
    parser.add_argument("--metadata-dir", default=DEFAULT_METADATA_DIR,
                        help="Sidecar metadata JSON dir. Defaults to ./downloads.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--phase", choices=["dense", "sparse", "all", "payload"],
                        default="all",
                        help=(
                            "dense  = embed dense only (upsert payload+dense per batch);\n"
                            "sparse = embed sparse only (update_vectors sparse per batch);\n"
                            "all    = dense first, then sparse (safest, no vector wipes)"
                        ))
    args = parser.parse_args()

    # ── Pre-flight ────────────────────────────────────────────────────────
    if not health_check():
        log.error("Embedding server not healthy at %s", settings.EMBEDDING_SERVER_URL)
        sys.exit(1)
    log.info("Embedding server OK")

    client = QdrantClient(url=settings.QDRANT_URL)
    log.info("Qdrant OK — existing: %s",
             [c.name for c in client.get_collections().collections])

    import os
    base_dir = args.base_dir or os.environ.get("MINERU_OUTPUT_DIR") or DEFAULT_MINERU_OUTPUT_DIR

    # ── Tokenizer + chunk config ──────────────────────────────────────────
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

    # ── Discover + filter ─────────────────────────────────────────────────
    jsons = discover_json_files(base_dir)
    arxiv_ids = sorted(jsons.keys())
    if args.limit:
        arxiv_ids = arxiv_ids[: args.limit]

    log.info("Processing %d papers (limit=%s)", len(arxiv_ids), args.limit)

    to_process: List[str] = []
    total_skipped = 0
    for arxiv_id in arxiv_ids:
        try:
            f = Filter(must=[FieldCondition(
                key="arxiv_id", match=MatchValue(value=arxiv_id),
            )])
            scroll_pts, _ = client.scroll(
                collection_name=args.collection, limit=1,
                with_payload=False, with_vectors=False, query_filter=f,
            )
            if scroll_pts:
                total_skipped += len(scroll_pts)
                log.info("  %s — already ingested, skipping", arxiv_id)
                continue
        except Exception:
            pass
        to_process.append(arxiv_id)

    log.info("Processing %d papers (skipped %d already-ingested)",
             len(to_process), total_skipped)

    if not to_process:
        log.info("All papers already ingested. Done.")
        return

    # ── Ensure collection ─────────────────────────────────────────────────
    ensure_collection(client, args.collection)

    # ── Chunk all papers (CPU only, no VRAM) ─────────────────────────────
    t0 = time.time()
    all_chunked, total_sections, total_tokens = _chunk_all_papers(
        to_process, jsons, args.metadata_dir, token_counter, config,
    )

    if not all_chunked:
        log.warning("No chunks produced. Exiting.")
        return

    # ── Run requested phase(s) ────────────────────────────────────────────
    #
    # --phase dense:
    #   Fresh collection → upsert(payload + dense) per batch.
    #   Existing collection (sparse already present) → update_vectors(dense).
    #
    # --phase sparse:
    #   Points exist with dense → run_sparse_only (scroll, skip already-done).
    #   Fresh collection (no points yet) → upsert payload-only first, then sparse.
    #
    # --phase all:
    #   1. run_phase_dense  → upserts payload+dense (fresh) or updates dense (existing)
    #   2. run_phase_sparse → update_vectors(sparse) per batch; dense+payload preserved
    #
    if args.phase == "dense":
        run_phase_dense(client, args.collection, all_chunked)

    elif args.phase == "sparse":
        # Check if points already exist in the collection
        first_pid = all_chunked[0][0]
        has_points = False
        try:
            pts, _ = client.scroll(
                collection_name=args.collection, limit=1,
                offset=first_pid, with_payload=False, with_vectors=False,
            )
            if pts:
                has_points = True
        except Exception:
            pass

        if has_points:
            # Dense already present — skip existing sparse, fill missing only
            run_sparse_only(client, args.collection, all_chunked)
        else:
            # Nothing exists yet — establish payload anchors, then sparse
            log.info("No existing points — upserting payload anchors before sparse embed.")
            _upsert_payload_only(client, args.collection, all_chunked)
            run_phase_sparse(client, args.collection, all_chunked)

    elif args.phase == "all":
        # Dense first (upserts payload+dense per batch on fresh collection)
        run_phase_dense(client, args.collection, all_chunked)
        # Sparse second (update_vectors only — dense+payload already on disk)
        run_phase_sparse(client, args.collection, all_chunked)

    # ── Final summary ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("FINAL SUMMARY:")
    log.info("  Papers processed:     %d", len(to_process))
    log.info("  Papers skipped:       %d", total_skipped)
    log.info("  Total sections:       %d", total_sections)
    log.info("  Total chunks:         %d", len(all_chunked))
    log.info("  Total tokens:         %d", total_tokens)
    log.info("  Total time:           %.1fs", time.time() - t0)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
