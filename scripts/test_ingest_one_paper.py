#!/usr/bin/env python3
"""Test ingestion of a single paper into a new test collection.

Useful for verifying the full pipeline (JSON → chunk → embed → upsert)
on one paper before running bulk ingestion.

Usage:
    # With sidecar metadata (requires --metadata-dir):
    python scripts/test_ingest_one_paper.py \
        --arxiv-id 2603.07444 \
        --base-dir /path/to/mineru_output \
        --metadata-dir /path/to/downloads

    # Without sidecar (title falls back to arxiv_id):
    python scripts/test_ingest_one_paper.py \
        --arxiv-id 2603.07444 \
        --base-dir /path/to/mineru_output

    # With local test_books JSON:
    python scripts/test_ingest_one_paper.py \
        --arxiv-id 2603.07444 \
        --base-dir . \
        --json examples/2603_07444_content_list_v2.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PointStruct,
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

DENSE_BATCH = 128
SPARSE_BATCH = 32
MAX_SPARSE_WORDS = 512


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


def ingest_test_paper(
    client: QdrantClient,
    collection: str,
    arxiv_id: str,
    json_path: str,
    metadata_dir: Optional[str] = None,
) -> Optional[tuple]:
    """Parse one JSON, chunk sections, embed dense+sparse, upsert.

    Returns (sections, chunks, tokens, arxiv_id_dot) on success, or None.
    """
    path = Path(json_path)
    if not path.exists():
        log.error("JSON file not found: %s", json_path)
        return None

    # Parse JSON → sections
    try:
        json_sections = parse_content_list(path)
    except Exception as e:
        log.error("JSON parse failed for %s: %s", json_path, e)
        return None

    if not json_sections:
        log.warning("JSON %s produced 0 sections — skipping", json_path)
        return (0, 0, 0, arxiv_id)

    # Read sidecar metadata if available
    arxiv_id_underscored = arxiv_id.replace(".", "_")
    title = arxiv_id
    category = ""
    subcategory = ""
    authors = ""
    publish_date = ""
    arxiv_id_dot = arxiv_id_underscored.replace("_", ".")  # fallback to underscore→dot

    if metadata_dir:
        meta_path = Path(metadata_dir) / f"{arxiv_id_underscored}.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                meta: Dict[str, str] = {}
                for attr in data.get("metadataAttributes", []):
                    if ": " in attr:
                        k, v = attr.split(": ", 1)
                        meta[k] = v
                    elif ":" in attr:
                        k, v = attr.split(":", 1)
                        meta[k.strip()] = v.strip()

                # Bug 3 fix: prefer dot-format from sidecar
                arxiv_id_dot = meta.get("arxiv_id", arxiv_id_underscored.replace("_", "."))
                title = meta.get("title", arxiv_id)
                category = meta.get("category", "")
                subcategory = meta.get("subcategory", "")
                authors = meta.get("authors", "")
                publish_date = meta.get("publish_date", "")

                log.info("Loaded sidecar metadata: title=%s, arxiv_id=%s", title[:60], arxiv_id_dot)
            except Exception as e:
                log.warning("Failed to parse sidecar %s: %s", meta_path, e)

    log.info("Parsed %d sections from %s", len(json_sections), path.name)

    # Chunk each section
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

    all_chunks: List[tuple] = []
    for js in json_sections:
        results = chunk_section(
            title=js.title,
            content=js.content,
            config=config,
            token_counter=token_counter,
            embedding_fn=None,  # batch dense below
        )
        chunk_count = len(results)
        for cr in results:
            all_chunks.append((
                cr.text,
                {
                    "doc_type": "paper",
                    "source_file": f"{arxiv_id_underscored}.pdf",
                    "title": title,
                    "arxiv_id": arxiv_id_dot,
                    "category": category,
                    "subcategory": subcategory,
                    "authors": authors,
                    "publish_date": publish_date,
                    "section_title": cr.section_title or "",
                    "chunk_index": cr.chunk_index,
                    "chunk_count": chunk_count,
                    "token_count": cr.token_count,
                    "has_heading_context": cr.has_heading_context,
                    "heading_level": js.heading_level,
                },
            ))

    if not all_chunks:
        return (len(json_sections), 0, 0, arxiv_id_dot)

    # Embed dense in batches
    texts = [c[0] for c in all_chunks]
    all_vecs: List[List[float]] = []
    for b in range(0, len(texts), DENSE_BATCH):
        batch = texts[b : b + DENSE_BATCH]
        all_vecs.extend(get_dense_vectors(batch))

    # Upsert with dense vectors
    points: List[PointStruct] = []
    for idx, ((chunk_text, metadata), vec) in enumerate(zip(all_chunks, all_vecs)):
        points.append(PointStruct(
            id=idx,
            vector={"dense": vec, "sparse": [0.0]},  # placeholder for sparse
            payload={"text": chunk_text, **metadata},
        ))

    client.upsert(collection_name=collection, points=points)
    log.info("Upserted %d dense points", len(points))

    # Embed sparse (max-pool across chunks)
    sparse_windows = []
    for chunk_text, _ in all_chunks:
        windows = _chunk_text_for_sparse(chunk_text)
        sparse_windows.extend(windows)

    all_sparse: List[dict] = []
    for b in range(0, len(sparse_windows), SPARSE_BATCH):
        sub = sparse_windows[b : b + SPARSE_BATCH]
        all_sparse.extend(get_sparse_vectors(sub, is_query=False))

    sv = _aggregate_sparse(all_sparse)

    # Update sparse vectors for all points
    from qdrant_client.models import PointVectors
    sparse_points = [
        PointVectors(
            id=idx,
            vector=SparseVector(indices=sv["indices"], values=sv["values"]),
        )
        for idx in range(len(points))
    ]
    client.update_vectors(collection_name=collection, points=sparse_points)
    log.info("Upserted sparse vectors for %d points", len(sparse_points))

    total_tokens = sum(c[1]["token_count"] for c in all_chunks)
    return (len(json_sections), len(all_chunks), total_tokens, arxiv_id_dot)


def verify_in_collection(client: QdrantClient, collection: str, arxiv_id_dot: str) -> None:
    """Verify the paper is in the collection with correct metadata."""
    f = Filter(must=[FieldCondition(
        key="arxiv_id", match=MatchValue(value=arxiv_id_dot),
    )])

    try:
        pts, _ = client.scroll(
            collection_name=collection,
            limit=5,
            with_payload=True,
            with_vectors=True,
            query_filter=f,
        )
    except Exception as e:
        log.error("Failed to scroll: %s", e)
        return

    if not pts:
        log.error("No points found for arxiv_id=%s", arxiv_id_dot)
        return

    log.info("=" * 60)
    log.info("VERIFICATION:")
    log.info("Collection: %s", collection)
    log.info("arxiv_id: %s", arxiv_id_dot)
    log.info("Found %d chunks", len(pts))
    log.info("")
    for i, p in enumerate(pts[:3]):
        log.info("  Chunk %d:", i)
        log.info("    section_title: %s", p.payload.get("section_title", ""))
        log.info("    title: %s", p.payload.get("title", ""))
        log.info("    authors: %s", p.payload.get("authors", ""))
        log.info("    category: %s", p.payload.get("category", ""))
        log.info("    token_count: %s", p.payload.get("token_count", ""))
        log.info("    has_heading_context: %s", p.payload.get("has_heading_context", ""))
        log.info("    text[:100]: %s", (p.payload.get("text", "")[:100]))
        log.info("")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Test ingestion of a single paper into a new test collection.",
    )
    parser.add_argument(
        "--arxiv-id", required=True,
        help="Arxiv ID of the paper (e.g. 2603.07444)",
    )
    parser.add_argument(
        "--json", default=None,
        help="Direct path to content_list_v2.json (overrides --base-dir)",
    )
    parser.add_argument(
        "--base-dir", default="./mineru_output",
        help="MinerU output directory for JSON discovery. Defaults to ./mineru_output.",
    )
    parser.add_argument(
        "--metadata-dir", default=None,
        help="Directory containing sidecar metadata JSON files.",
    )
    parser.add_argument(
        "--collection", default="papers-semantic-test",
        help="Target Qdrant collection name. Defaults to papers-semantic-test.",
    )
    args = parser.parse_args()

    # Health check
    if not health_check():
        log.error("Embedding server not healthy at %s", settings.EMBEDDING_SERVER_URL)
        sys.exit(1)
    log.info("Embedding server OK")

    # Resolve JSON path
    if args.json:
        json_path = args.json
    else:
        arxiv_id_underscored = args.arxiv_id.replace(".", "_")
        # Try tree layout first
        tree_path = Path(args.base_dir) / arxiv_id_underscored / "vlm" / f"{arxiv_id_underscored}_content_list_v2.json"
        if tree_path.exists():
            json_path = str(tree_path)
        else:
            # Try flat layout
            flat_path = Path(args.base_dir) / f"{arxiv_id_underscored}_content_list_v2.json"
            if flat_path.exists():
                json_path = str(flat_path)
            else:
                log.error("JSON not found for arxiv_id=%s in %s", args.arxiv_id, args.base_dir)
                sys.exit(1)

    log.info("Arxiv ID: %s", args.arxiv_id)
    log.info("JSON path: %s", json_path)
    log.info("Metadata dir: %s", args.metadata_dir or "None")
    log.info("Collection: %s", args.collection)

    # Connect to Qdrant
    client = QdrantClient(url=settings.QDRANT_URL)

    # Ensure collection
    existing = {c.name for c in client.get_collections().collections}
    if args.collection not in existing:
        log.info("Creating collection: %s", args.collection)
        client.create_collection(
            collection_name=args.collection,
            vectors_config={
                "dense": VectorParams(size=768, distance=Distance.COSINE),
                "sparse": SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        for field in ["doc_type", "source_file", "arxiv_id", "category", "title"]:
            try:
                client.create_payload_index(
                    collection_name=args.collection,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                pass
    else:
        # Clear existing points
        log.info("Collection '%s' exists — clearing...", args.collection)
        try:
            client.delete_collection(collection_name=args.collection)
        except Exception:
            pass
        client.create_collection(
            collection_name=args.collection,
            vectors_config={
                "dense": VectorParams(size=768, distance=Distance.COSINE),
                "sparse": SparseVectorParams(modifier=Modifier.IDF),
            },
        )

    # Ingest
    result = ingest_test_paper(
        client, args.collection, args.arxiv_id, json_path, args.metadata_dir,
    )

    if result is None:
        log.error("Ingestion FAILED")
        sys.exit(1)

    sections, chunks, tokens, arxiv_id_dot = result
    log.info("=" * 60)
    log.info("INGESTION COMPLETE:")
    log.info("  Sections:   %d", sections)
    log.info("  Chunks:     %d", chunks)
    log.info("  Tokens:     %d", tokens)
    log.info("  arxiv_id:   %s", arxiv_id_dot)
    log.info("=" * 60)

    # Verify
    verify_in_collection(client, args.collection, arxiv_id_dot)


if __name__ == "__main__":
    main()