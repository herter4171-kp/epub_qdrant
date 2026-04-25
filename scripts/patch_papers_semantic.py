#!/usr/bin/env python3
"""Patch existing papers-semantic collection with correct metadata.

Scrolls papers-semantic, reads sidecar metadata from downloads/{arxiv_id}.json
for each paper, and updates payload fields in-place:
  - arxiv_id: converts underscore to dot format
  - title: from sidecar
  - authors: from sidecar
  - category: from sidecar
  - subcategory: from sidecar
  - publish_date: from sidecar

Usage:
    python scripts/patch_papers_semantic.py \
        --collection papers-semantic \
        --metadata-dir ./downloads

This does NOT re-embed — only updates payloads. Vectors stay intact.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    PayloadSchemaType,
)

DENSE_BATCH = 256
SCROLL_BATCH = 500


def read_sidecar(metadata_dir: str, arxiv_id: str) -> Dict[str, str]:
    """Read sidecar metadata JSON for an arxiv ID (underscore format)."""
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


def patch_papers_semantic(
    client: QdrantClient,
    collection: str,
    metadata_dir: str,
    dry_run: bool = False,
) -> None:
    """Scroll all points, patch metadata for each paper."""

    # Get total point count
    try:
        info = client.get_collection(collection)
        total_points = info.points_count or 0
    except Exception as e:
        log.error("Cannot get collection '%s': %s", collection, e)
        sys.exit(1)

    log.info("Collection '%s' has %d points. Starting patch...", collection, total_points)

    # First pass: discover all unique arxiv_ids in the collection
    log.info("Discovering unique arxiv_ids...")
    arxiv_ids_seen: Dict[str, int] = {}  # arxiv_id -> count
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=["arxiv_id"],
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            aid = p.payload.get("arxiv_id", "")
            if aid:
                arxiv_ids_seen[aid] = arxiv_ids_seen.get(aid, 0) + 1
        if next_offset is None:
            break
        offset = next_offset

    log.info("Found %d unique arxiv_ids in collection", len(arxiv_ids_seen))

    # Second pass: patch each paper's points
    patched = 0
    skipped = 0
    failed = 0
    total_updates = 0

    for arxiv_id, count in sorted(arxiv_ids_seen.items()):
        # Read sidecar metadata
        meta = read_sidecar(metadata_dir, arxiv_id)

        if not meta:
            log.debug("No sidecar for arxiv_id=%s — skipping", arxiv_id)
            skipped += 1
            continue

        # Compute dot-format arxiv_id from sidecar
        arxiv_id_dot = meta.get("arxiv_id", arxiv_id.replace("_", "."))

        # Get all points for this paper
        f = Filter(must=[FieldCondition(
            key="arxiv_id", match=MatchValue(value=arxiv_id),
        )])

        try:
            pts, _ = client.scroll(
                collection_name=collection,
                limit=count,
                with_payload=True,
                with_vectors=False,
                query_filter=f,
            )
        except Exception as e:
            log.error("Failed to scroll paper arxiv_id=%s: %s", arxiv_id, e)
            failed += 1
            continue

        # Build payload updates
        updates: List[tuple] = []
        for p in pts:
            update: Dict[str, object] = {}

            # arxiv_id: convert to dot format
            current_dot = p.payload.get("arxiv_id", arxiv_id)
            if current_dot != arxiv_id_dot:
                update["arxiv_id"] = arxiv_id_dot

            # Title
            current_title = p.payload.get("title", arxiv_id)
            if meta.get("title") and current_title == arxiv_id:
                update["title"] = meta["title"]

            # Authors
            current_authors = p.payload.get("authors", "")
            if meta.get("authors") and not current_authors:
                update["authors"] = meta["authors"]

            # Category
            current_category = p.payload.get("category", "")
            if meta.get("category") and not current_category:
                update["category"] = meta["category"]

            # Subcategory
            current_subcategory = p.payload.get("subcategory", "")
            if meta.get("subcategory") and not current_subcategory:
                update["subcategory"] = meta["subcategory"]

            # Publish date
            current_date = p.payload.get("publish_date", "")
            if meta.get("publish_date") and not current_date:
                update["publish_date"] = meta["publish_date"]

            if update:
                updates.append((p.id, update))

        if updates:
            if not dry_run:
                for point_id, payload_update in updates:
                    client.set_payload(
                        collection_name=collection,
                        payload=payload_update,
                        points=[point_id],
                    )
                    total_updates += 1
            else:
                total_updates += len(updates)
                log.info("  [DRY RUN] arxiv_id=%s (%s) — %d updates", arxiv_id, arxiv_id_dot, len(updates))
            patched += 1
        else:
            skipped += 1

    log.info("=" * 60)
    log.info("PATCH SUMMARY:")
    log.info("  Papers patched:  %d", patched)
    log.info("  Papers skipped:  %d", skipped)
    log.info("  Papers failed:   %d", failed)
    log.info("  Total updates:   %d", total_updates)
    log.info("=" * 60)

    if dry_run:
        log.info("Dry run — no changes written.")


def main():
    parser = argparse.ArgumentParser(
        description="Patch papers-semantic collection with correct metadata from sidecar files.",
    )
    parser.add_argument(
        "--collection", default="papers-semantic",
        help="Target Qdrant collection name.",
    )
    parser.add_argument(
        "--metadata-dir", default="./downloads",
        help="Directory containing sidecar metadata JSON files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing.",
    )
    args = parser.parse_args()

    client = QdrantClient(url="http://localhost:6333")
    log.info("Connected to Qdrant at http://localhost:6333")

    patch_papers_semantic(client, args.collection, args.metadata_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()