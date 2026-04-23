"""CLI entry point for the ingestion pipeline.

Commands:
    create-collection  Create a named-vector collection (dense + sparse)
    ingest-dense       Pass 1: load files → chunk → embed dense → upsert
    ingest-sparse      Pass 2: scroll collection → embed sparse → upsert same IDs
    ingest             Convenience: create + dense + sparse in one shot
    search             Search a collection
    list-collections   List all Qdrant collections
    delete-collection  Delete a collection
    list-books         List books in a collection
    health             Check embedding server health
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
