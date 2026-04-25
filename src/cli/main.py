"""CLI entry point for the ingestion pipeline.

Commands:
    create-collection       Create a named-vector collection (dense + sparse)
    ingest-dense            Pass 1: load files → chunk → embed dense → upsert
    ingest-sparse           Pass 2: scroll collection → embed sparse → upsert same IDs
    ingest                  Convenience: create + dense + sparse in one shot
    ingest-papers           Bulk ingest MinerU JSON papers (JSON sidecars + PDF metadata)
    search                  Search a collection
    list-collections        List all Qdrant collections
    delete-collection       Delete a collection
    list-books              List books in a collection
    health                  Check embedding server health
"""

import logging
import sys
from pathlib import Path
from typing import List

import click

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Collections we refuse to mutate
PROTECTED = {"books", "books-named", "papers", "papers-named"}

INDEX_FIELDS = [
    "doc_type", "source_file", "book_title", "section_title",
    "publisher", "language", "isbn",
    "arxiv_id", "category", "title",
]

DENSE_BATCH = 128
SPARSE_BATCH = 32
SCROLL_BATCH = 256
MAX_SPARSE_WORDS = 512


def _guard(collection: str) -> None:
    if collection in PROTECTED:
        click.echo(f"Refusing to touch protected collection '{collection}'.", err=True)
        sys.exit(1)


def _find_files(directory: str) -> List[Path]:
    """Find all ingestible files (.epub, .pdf) in a directory."""
    dirpath = Path(directory)
    files = sorted(dirpath.glob("*.epub")) + sorted(dirpath.glob("*.pdf"))
    return files


def _ensure_collection(client, name: str, protected: frozenset) -> None:
    """Create a named-vector collection if it doesn't exist."""
    from qdrant_client.models import Distance, Modifier, VectorParams, SparseVectorParams
    if name in protected:
        raise ValueError(f"Refusing to overwrite protected collection '{name}'")
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
        )


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
@click.option("--tokenizer-json", default=None, envvar="TOKENIZER_JSON",
              help="Path to tokenizer.json for real token counting")
@click.pass_context
def cli(ctx, tokenizer_json):
    """EPUB/PDF → Qdrant ingestion pipeline."""
    ctx.ensure_object(dict)
    if tokenizer_json:
        import os
        os.environ["TOKENIZER_JSON"] = tokenizer_json


# ── health ────────────────────────────────────────────────────────────────────

@cli.command()
def health():
    """Check embedding server health."""
    from servers.embedding_server.client import health_check
    ok = health_check()
    if ok:
        click.echo("Embedding server: OK (dense + sparse loaded)")
    else:
        click.echo("Embedding server: NOT HEALTHY", err=True)
        sys.exit(1)


# ── create-collection ─────────────────────────────────────────────────────────

@cli.command("create-collection")
@click.argument("collection")
def create_collection_cmd(collection: str):
    """Create a named-vector collection with dense + sparse config."""
    from qdrant_client.models import Distance, Modifier, VectorParams, SparseVectorParams
    from src.storage import Storage

    _guard(collection)
    storage = Storage()
    existing = storage.list_collections()

    if collection in existing:
        click.echo(f"Collection '{collection}' already exists.")
        return

    storage.client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    for field in INDEX_FIELDS:
        try:
            storage.client.create_payload_index(
                collection_name=collection, field_name=field, field_schema="keyword",
            )
        except Exception:
            pass
    click.echo(f"Created collection: {collection}")


# ── ingest-dense (Pass 1) ────────────────────────────────────────────────────

@cli.command("ingest-dense")
@click.argument("directory")
@click.option("--collection", required=True, help="Target collection name")
@click.option("--limit", type=int, default=None, help="Max files to process")
@click.option("--tokenizer-json", default=None, help="Path to tokenizer.json")
def ingest_dense_cmd(directory: str, collection: str, limit: int, tokenizer_json: str):
    """Pass 1: load files, chunk, embed dense vectors, upsert."""
    import os
    if tokenizer_json:
        os.environ["TOKENIZER_JSON"] = tokenizer_json
    from qdrant_client.models import PointStruct
    from servers.embedding_server.client import get_dense_vectors
    from src.ingestion.loader import DocumentLoader
    from src.storage import Storage

    _guard(collection)
    storage = Storage()

    if collection not in storage.list_collections():
        click.echo(f"Collection '{collection}' does not exist. Run create-collection first.", err=True)
        sys.exit(1)

    files = _find_files(directory)
    if not files:
        click.echo(f"No .epub or .pdf files found in {directory}.")
        return
    if limit:
        files = files[:limit]

    click.echo(f"Pass 1 (dense): {len(files)} file(s) → {collection}")

    try:
        info = storage.client.get_collection(collection)
        point_offset = info.points_count or 0
    except Exception:
        point_offset = 0

    total = 0
    for i, fpath in enumerate(files, 1):
        loader = DocumentLoader.for_path(fpath)
        chunks = loader.load(fpath)
        if not chunks:
            click.echo(f"  [{i}/{len(files)}] {fpath.name} — 0 chunks, skipping")
            continue

        texts = [c.text for c in chunks]
        all_vecs: list = []
        for b in range(0, len(texts), DENSE_BATCH):
            all_vecs.extend(get_dense_vectors(texts[b:b + DENSE_BATCH]))

        points = []
        for idx, (chunk, vec) in enumerate(zip(chunks, all_vecs)):
            points.append(PointStruct(
                id=point_offset + total + idx,
                vector={"dense": vec},
                payload={"text": chunk.text, **chunk.metadata},
            ))

        storage.client.upsert(collection_name=collection, points=points)
        total += len(points)
        click.echo(f"  [{i}/{len(files)}] {fpath.name} — {len(points)} chunks (total: {total})")

    click.echo(f"Pass 1 done: {total} points with dense vectors in '{collection}'")


# ── ingest-sparse (Pass 2) ───────────────────────────────────────────────────

@cli.command("ingest-sparse")
@click.argument("collection")
def ingest_sparse_cmd(collection: str):
    """Pass 2: scroll collection, embed sparse vectors, update same point IDs."""
    from qdrant_client.models import PointVectors, SparseVector
    from servers.embedding_server.client import get_sparse_vectors
    from src.storage import Storage

    _guard(collection)
    storage = Storage()

    if collection not in storage.list_collections():
        click.echo(f"Collection '{collection}' does not exist.", err=True)
        sys.exit(1)

    info = storage.client.get_collection(collection)
    count = info.points_count or 0
    click.echo(f"Pass 2 (sparse): scrolling {count} points in '{collection}'")

    offset = None
    total = 0

    while True:
        points, next_offset = storage.client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=False,
            with_payload=["text"],
        )
        if not points:
            break

        upsert_points = []
        for p in points:
            text = (p.payload.get("text") or "").strip()
            if not text:
                continue

            # Chunk for sparse
            words = text.split()
            windows = [
                " ".join(words[j:j + MAX_SPARSE_WORDS])
                for j in range(0, max(len(words), 1), MAX_SPARSE_WORDS)
            ] if words else []

            all_sparse = []
            for b in range(0, len(windows), SPARSE_BATCH):
                all_sparse.extend(get_sparse_vectors(windows[b:b + SPARSE_BATCH], is_query=False))

            # Max-pool aggregation
            agg: dict = {}
            for v in all_sparse:
                for idx, val in zip(v["indices"], v["values"]):
                    if idx not in agg or val > agg[idx]:
                        agg[idx] = val

            upsert_points.append(PointVectors(
                id=p.id,
                vector={"sparse": SparseVector(indices=list(agg.keys()), values=list(agg.values()))},
            ))

        if upsert_points:
            storage.client.update_vectors(collection_name=collection, points=upsert_points)
            total += len(upsert_points)
            click.echo(f"  [sparse] {total} / {count}")

        if next_offset is None:
            break
        offset = next_offset

    click.echo(f"Pass 2 done: {total} points with sparse vectors in '{collection}'")


# ── ingest (convenience: create + dense + sparse) ────────────────────────────

@cli.command()
@click.argument("directory")
@click.option("--collection", required=True, help="Target collection name")
@click.option("--limit", type=int, default=None, help="Max files to process")
@click.option("--tokenizer-json", default=None, help="Path to tokenizer.json")
def ingest(directory: str, collection: str, limit: int, tokenizer_json: str):
    """Full ingest: create collection + dense pass + sparse pass."""
    import os
    if tokenizer_json:
        os.environ["TOKENIZER_JSON"] = tokenizer_json
    from click.testing import CliRunner
    runner = CliRunner()

    # Create
    result = runner.invoke(cli, ["create-collection", collection])
    click.echo(result.output, nl=False)
    if result.exit_code != 0:
        sys.exit(result.exit_code)

    # Dense
    args = ["ingest-dense", directory, "--collection", collection]
    if limit:
        args += ["--limit", str(limit)]
    result = runner.invoke(cli, args)
    click.echo(result.output, nl=False)
    if result.exit_code != 0:
        sys.exit(result.exit_code)

    # Sparse
    result = runner.invoke(cli, ["ingest-sparse", collection])
    click.echo(result.output, nl=False)
    if result.exit_code != 0:
        sys.exit(result.exit_code)


# ── ingest-papers (MinerU JSON) ─────────────────────────────────────────────

@cli.command("ingest-papers")
@click.argument("collection")
@click.option("--base-dir", default=None,
              help="MinerU output directory (overrides MINERU_OUTPUT_DIR env). Defaults to ./mineru_output.")
@click.option("--metadata-dir", default="./downloads",
              help="Directory containing sidecar metadata JSON files. Defaults to ./downloads.")
@click.option("--limit", type=int, default=None, help="Max papers to process")
@click.option("--arxiv-id", default=None, help="Process a single paper by ID")
def ingest_papers_cmd(collection: str, base_dir: str, metadata_dir: str,
                      limit: int, arxiv_id: str):
    """Bulk ingest MinerU JSON files (content_list_v2.json) into a Qdrant collection.

    Walks the MinerU output tree, discovers all content_list_v2.json files,
    parses them, chunks sections, and runs the full two-pass dense + sparse
    embedding pipeline.
    """
    import glob as glob_mod
    import json as json_mod
    import os
    import re as re_mod

    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, FieldCondition, Filter, MatchValue,
        Modifier, PointStruct, PointVectors, SparseVector,
        SparseVectorParams, VectorParams,
    )
    from servers.embedding_server.client import (
        get_dense_vectors, get_sparse_vectors, health_check,
    )

    PAPER_PROTECTED = {"books", "books-named", "papers", "papers-named"}
    DENSE_BATCH = 128
    SPARSE_BATCH = 32
    SCROLL_BATCH = 256
    MAX_SPARSE_WORDS = 512

    # Health check
    if not health_check():
        click.echo(f"Embedding server not healthy at {settings.EMBEDDING_SERVER_URL}", err=True)
        sys.exit(1)
    click.echo("Embedding server: OK")

    # Qdrant client
    client = QdrantClient(url=settings.QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]
    click.echo(f"Qdrant OK — existing collections: {existing}")

    # Resolve base directory
    env_base = os.environ.get("MINERU_OUTPUT_DIR")
    if base_dir:
        pass  # use args value
    elif env_base:
        base_dir = env_base
    else:
        base_dir = "./mineru_output"
    click.echo(f"MinerU output dir: {base_dir}")

    # Ensure collection
    try:
        _ensure_collection(client, collection, PAPER_PROTECTED)
    except ValueError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    click.echo(f"Collection '{collection}' ready")

    # Chunk config + tokenizer
    from src.ingestion.semantic_chunker import ChunkConfig, chunk_section, load_tokenizer
    from src.ingestion.mineru_json_parser import parse_content_list

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
    jsons: dict = {}

    # Tree layout
    tree_pattern = str(Path(base_dir) / "**" / "vlm" / "*_content_list_v2.json")
    for p in glob_mod.glob(tree_pattern, recursive=True):
        pp = Path(p)
        aid = pp.stem.replace("_content_list_v2", "")
        jsons[aid] = pp

    # Flat layout
    flat_pattern = str(Path(base_dir) / "*_content_list_v2.json")
    for p in glob_mod.glob(flat_pattern, recursive=True):
        pp = Path(p)
        aid = pp.stem.replace("_content_list_v2", "")
        if aid not in jsons:
            jsons[aid] = pp

    click.echo(f"Discovered {len(jsons)} JSON file(s)")

    if not jsons:
        click.echo("No JSON files found. Exiting.")
        return

    # Filter by arxiv_id
    if arxiv_id:
        normalized = arxiv_id.replace(".", "_")
        if normalized not in jsons:
            click.echo(f"No JSON found for arxiv_id={arxiv_id}", err=True)
            sys.exit(1)
        jsons = {normalized: jsons[normalized]}
        click.echo(f"Single paper mode: {arxiv_id}")

    # Apply limit
    arxiv_ids = sorted(jsons.keys())
    if limit is not None:
        if limit <= 0:
            click.echo(f"Error: --limit must be a positive integer, got {limit}", err=True)
            sys.exit(1)
        arxiv_ids = arxiv_ids[:limit]

    def read_sidecar(mdir: str, aid: str) -> dict:
        meta_path = Path(mdir) / f"{aid}.json"
        if not meta_path.exists():
            return {}
        try:
            raw = meta_path.read_text(encoding="utf-8")
            data = json_mod.loads(raw)
            result = {}
            for attr in data.get("metadataAttributes", []):
                if ": " in attr:
                    k, v = attr.split(": ", 1)
                    result[k] = v
                elif ":" in attr:
                    k, v = attr.split(":", 1)
                    result[k.strip()] = v.strip()
            return result
        except Exception:
            return {}

    def chunk_text_for_sparse(text: str) -> list:
        words = text.split()
        if not words:
            return []
        return [" ".join(words[i:i + MAX_SPARSE_WORDS]) for i in range(0, len(words), MAX_SPARSE_WORDS)]

    def aggregate_sparse(vecs: list) -> dict:
        agg = {}
        for v in vecs:
            for idx, val in zip(v["indices"], v["values"]):
                if idx not in agg or val > agg[idx]:
                    agg[idx] = val
        return {"indices": list(agg.keys()), "values": list(agg.values())}

    # Process each paper
    total_processed = 0
    total_chunks = 0
    total_skipped = 0
    total_failed = 0

    for i, aid in enumerate(arxiv_ids, 1):
        json_path = jsons[aid]

        # Idempotency check
        try:
            f = Filter(must=[FieldCondition(key="arxiv_id", match=MatchValue(value=aid))])
            pts, _ = client.scroll(
                collection_name=collection, limit=1,
                with_payload=False, with_vectors=False, scroll_filter=f,
            )
            if pts:
                # Count total points for this paper
                count = 0
                off = None
                while True:
                    pts2, noff = client.scroll(
                        collection_name=collection, limit=SCROLL_BATCH,
                        offset=off, with_payload=False, with_vectors=False, scroll_filter=f,
                    )
                    if not pts2:
                        break
                    count += len(pts2)
                    if noff is None:
                        break
                    off = noff
                total_skipped += count
                click.echo(f"  [{i}/{len(arxiv_ids)}] {aid} — already ingested ({count} chunks), skipping")
                continue
        except Exception:
            pass

        # Parse JSON → sections
        try:
            json_sections = parse_content_list(json_path)
        except Exception as e:
            click.echo(f"  [{i}/{len(arxiv_ids)}] {aid} — JSON parse failed: {e}", err=True)
            total_failed += 1
            continue

        if not json_sections:
            click.echo(f"  [{i}/{len(arxiv_ids)}] {aid} — 0 sections, skipping")
            total_failed += 1
            continue

        # Read sidecar metadata
        meta = read_sidecar(metadata_dir, aid)
        paper_title = meta.get("title", aid)
        category = meta.get("category", "")
        subcategory = meta.get("subcategory", "")
        authors = meta.get("authors", "")
        publish_date = meta.get("publish_date", "")

        def _build_metadata_prefix(chunk_meta: dict) -> str:
            """Build structured metadata prefix for embedding.

            Per design doc: metadata is prepended to text before embedding
            so that metadata influences the vector space.
            """
            parts = []
            if chunk_meta.get("category"):
                parts.append(f"category:{chunk_meta['category']}")
            if chunk_meta.get("subcategory"):
                parts.append(f"subcategory:{chunk_meta['subcategory']}")
            if chunk_meta.get("publish_date"):
                parts.append(f"date:{chunk_meta['publish_date']}")
            if chunk_meta.get("section_title"):
                parts.append(f"section:{chunk_meta['section_title']}")
            return " ".join(parts)

        # Chunk sections — prepend metadata prefix to text for embedding
        all_chunks = []  # list of (embed_text, payload_dict)
        for js in json_sections:
            results = chunk_section(
                title=js.title, content=js.content, config=config,
                token_counter=token_counter, embedding_fn=None,
            )
            chunk_count = len(results)
            for cr in results:
                chunk_meta = {
                    "doc_type": "paper",
                    "source_file": f"{aid}.pdf",
                    "title": paper_title or "",
                    "arxiv_id": aid,
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
                }
                # Embedding input: metadata prefix + clean text
                meta_prefix = _build_metadata_prefix(chunk_meta)
                if meta_prefix:
                    embed_text = f"{meta_prefix} {cr.text}"
                else:
                    embed_text = cr.text
                all_chunks.append((embed_text, chunk_meta))

        if not all_chunks:
            click.echo(f"  [{i}/{len(arxiv_ids)}] {aid} — 0 chunks, skipping")
            total_failed += 1
            continue

        # Embed dense
        texts = [c[0] for c in all_chunks]  # embed_text = metadata_prefix + text
        all_vecs = []
        for b in range(0, len(texts), DENSE_BATCH):
            all_vecs.extend(get_dense_vectors(texts[b:b + DENSE_BATCH]))

        # Upsert dense
        try:
            point_offset = client.get_collection(collection).points_count or 0
        except Exception:
            point_offset = 0

        points = []
        for idx, ((chunk_text, metadata), vec) in enumerate(zip(all_chunks, all_vecs)):
            points.append(PointStruct(
                id=point_offset + total_processed + idx,
                vector={"dense": vec},
                payload={"text": chunk_text, **metadata},
            ))

        client.upsert(collection_name=collection, points=points)

        total_tokens = sum(c[1]["token_count"] for c in all_chunks)

        # Pass 2: sparse
        click.echo(f"  [{i}/{len(arxiv_ids)}] {aid} — {len(json_sections)} sections, {len(all_chunks)} chunks, {total_tokens} tokens — embedding sparse...", nl=False)
        click.echo()
        sparse_count = 0

        f = Filter(must=[FieldCondition(key="arxiv_id", match=MatchValue(value=aid))])
        soff = None
        while True:
            pts, noff = client.scroll(
                collection_name=collection, limit=SCROLL_BATCH,
                offset=soff, with_vectors=False, with_payload=["text"], scroll_filter=f,
            )
            if not pts:
                break

            upsert_pts = []
            for p in pts:
                text = (p.payload.get("text") or "").strip()
                if not text:
                    continue
                windows = chunk_text_for_sparse(text)
                all_sparse = []
                for b2 in range(0, len(windows), SPARSE_BATCH):
                    all_sparse.extend(get_sparse_vectors(windows[b2:b2 + SPARSE_BATCH], is_query=False))
                sv = aggregate_sparse(all_sparse)
                upsert_pts.append(PointVectors(
                    id=p.id,
                    vector={"sparse": SparseVector(indices=sv["indices"], values=sv["values"])},
                ))

            if upsert_pts:
                client.update_vectors(collection_name=collection, points=upsert_pts)
                sparse_count += len(upsert_pts)

            if noff is None:
                break
            soff = noff

        click.echo(f"sparse: {sparse_count} points")
        total_processed += 1
        total_chunks += len(all_chunks)

    # Summary
    click.echo("")
    click.echo("=" * 60)
    click.echo("SUMMARY:")
    click.echo(f"  Processed: {total_processed}")
    click.echo(f"  Chunks:    {total_chunks}")
    click.echo(f"  Skipped:   {total_skipped}")
    click.echo(f"  Failed:    {total_failed}")
    click.echo("=" * 60)


# ── search ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--collection", help="Collection name", default=None)
@click.option("--top-k", type=int, default=10, help="Number of results")
def search(query: str, collection: str, top_k: int):
    """Search a collection for relevant passages."""
    from src.storage import Storage
    name = collection or settings.QDRANT_COLLECTION
    storage = Storage()
    results = storage.search(name, query, top_k=top_k)

    if not results:
        click.echo(f"No results in '{name}'.")
        return

    click.echo(f"\nResults for '{query}' in '{name}':\n")
    for i, r in enumerate(results, 1):
        click.echo(f"--- {i}. Score: {r['score']:.4f} ---")
        title = r.get('book_title') or r.get('title') or ''
        if title:
            click.echo(f"  Title: {title}")
        section = r.get('section_title') or ''
        if section:
            click.echo(f"  Section: {section}")
        text_preview = r['text'][:200].replace('\n', ' ')
        if len(r['text']) > 200:
            text_preview += "..."
        click.echo(f"  Text: {text_preview}")
        click.echo()


# ── list-collections ──────────────────────────────────────────────────────────

@cli.command("list-collections")
def list_collections_cmd():
    """List all Qdrant collections."""
    from src.storage import Storage
    storage = Storage()
    for c in storage.list_collections():
        click.echo(c)


# ── delete-collection ─────────────────────────────────────────────────────────

@cli.command("delete-collection")
@click.argument("collection")
def delete_collection_cmd(collection: str):
    """Delete a Qdrant collection."""
    _guard(collection)
    from src.storage import Storage
    storage = Storage()
    storage.delete_collection(collection)
    click.echo(f"Deleted: {collection}")


# ── list-books ────────────────────────────────────────────────────────────────

@cli.command("list-books")
@click.option("--collection", default=None)
def list_books_cmd(collection: str):
    """List books in a collection."""
    from src.storage import Storage
    storage = Storage()
    books = storage.list_books(collection_name=collection)
    if not books:
        click.echo("No books found.")
        return

    name = collection or settings.QDRANT_COLLECTION
    click.echo(f"\nBooks in '{name}':")
    click.echo(f"{'Title':<45} {'Publisher':<12} {'Chunks':>6}")
    click.echo("─" * 65)
    for b in books:
        title = (b.get("book_title") or b.get("source_file", ""))[:44].replace('\n', ' ')
        publisher = (b.get("publisher") or "")[:11]
        chunks = b.get("chunk_count", 0)
        click.echo(f"{title:<45} {publisher:<12} {chunks:>6}")
    total = sum(b.get("chunk_count", 0) for b in books)
    click.echo("─" * 65)
    click.echo(f"Total: {len(books)} book(s), {total} chunks")


if __name__ == "__main__":
    cli()
