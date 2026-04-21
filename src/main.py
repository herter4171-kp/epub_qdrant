"""CLI entry point for the EPUB-to-Qdrant ingestion pipeline."""

import logging
import sys
from pathlib import Path
from typing import List

import click

from src.config import settings
from src.epub_parser import parse_epub
from src.chunker import chunk_section
from src.embedder import Embedder
from src.storage import Storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _find_epubs(directory: str) -> List[Path]:
    """Find all .epub files in a directory (non-recursive)."""
    dirpath = Path(directory)
    epubs = sorted(dirpath.glob("*.epub"))
    if not epubs:
        logger.warning(f"No .epub files found in {dirpath}")
    return epubs


def _process_book(
    epub_path: str,
    embedder: Embedder,
    storage: Storage,
    collection: str = None,
) -> int:
    """Parse, chunk, embed, and upsert a single EPUB file.

    Returns:
        Number of chunks upserted.
    """
    path = Path(epub_path)
    logger.info(f"Processing: {path.name}")

    # 1. Parse EPUB
    book = parse_epub(epub_path)
    logger.info(
        f"  Title: {book.title} by {book.creator}"
    )
    logger.info(f"  Sections: {len(book.sections)}")

    # 2. Chunk all sections
    all_chunks: List = []
    for section in book.sections:
        chunks = chunk_section(
            section,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        # Override book_title
        for c in chunks:
            c.book_title = book.title
        all_chunks.extend(chunks)

    logger.info(f"  Chunks: {len(all_chunks)}")

    if not all_chunks:
        logger.warning(f"  No chunks generated for {path.name}")
        return 0

    # 3. Embed
    logger.info("  Embedding...")
    all_chunks = embedder.embed_chunks(all_chunks)

    # 4. Upsert to Qdrant
    count = storage.upsert_file(epub_path, all_chunks, collection_name=collection)
    logger.info(f"  Done: {count} chunks stored")
    return count


@click.group()
def cli():
    """EPUB-to-Qdrant ingestion pipeline."""
    pass


@cli.command()
@click.argument("directory")
@click.option("--url", help="Qdrant URL (overrides env)", default=None)
@click.option("--limit", type=int, help="Limit: only process N files", default=None)
@click.option("--collection", help="Qdrant collection name (overrides env)", default=None)
def ingest(directory: str, url: str, limit: int, collection: str) -> None:
    """Ingest all EPUB files from DIRECTORY into Qdrant."""
    epubs = _find_epubs(directory)
    if not epubs:
        click.echo("No EPUB files found. Nothing to do.")
        return

    if limit:
        epubs = epubs[:limit]
        click.echo(f"Limited to {limit} file(s).")

    click.echo(f"Found {len(epubs)} EPUB file(s). Starting ingest...")

    # Setup components
    embedder = Embedder(settings.OLLAMA_URL, settings.EMBEDDING_MODEL)
    storage = Storage(url=url)

    total = 0
    for i, epub in enumerate(epubs, 1):
        click.echo(f"\n[{i}/{len(epubs)}] {epub.name}")
        try:
            count = _process_book(str(epub), embedder, storage, collection)
            total += count
        except Exception as e:
            logger.error(f"Failed to process {epub.name}: {e}")
            continue

    click.echo(f"\nIngest complete. Total chunks stored: {total}")


@cli.command()
@click.argument("query")
@click.option("--collection", help="Qdrant collection name (overrides env)", default=None)
@click.option("--top-k", type=int, default=10, help="Number of results to return")
def search(query: str, collection: str, top_k: int) -> None:
    """Search a Qdrant collection for relevant passages."""
    name = collection or settings.QDRANT_COLLECTION
    storage = Storage()
    results = storage.search(name, query, top_k=top_k)

    if not results:
        click.echo(f"No results found in collection '{name}'.")
        return

    display_name = collection or settings.QDRANT_COLLECTION
    click.echo(f"\nResults for '{query}' in '{display_name}':\n")
    for i, r in enumerate(results, 1):
        click.echo(f"--- {i}. Score: {r['score']:.4f} ---")
        click.echo(f"  Book: {r['book_title']}")
        click.echo(f"  Section: {r['section_title']}")
        click.echo(f"  Chunk: {r['chunk_index']}")
        # Show first 200 chars of text
        text_preview = r['text'][:200].replace('\n', ' ')
        if len(r['text']) > 200:
            text_preview += "..."
        click.echo(f"  Text: {text_preview}")
        click.echo()


@cli.command(name="list-collections")
def list_collections_command() -> None:
    """List all Qdrant collections."""
    storage = Storage()
    collections = storage.list_collections()
    if not collections:
        click.echo("No collections found.")
        return
    for c in collections:
        click.echo(c)


@cli.command()
@click.argument("collection")
def delete_collection_command(collection: str) -> None:
    """Delete a Qdrant collection."""
    storage = Storage()
    storage.delete_collection(collection)
    click.echo(f"Deleted collection: {collection}")


if __name__ == "__main__":
    cli()