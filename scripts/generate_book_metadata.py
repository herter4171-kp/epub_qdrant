#!/usr/bin/env python3
"""Generate *.epub.metadata.json files from EPUB OPF metadata.

For each .epub in test_books/, read OPF metadata and write a sidecar JSON:
  test_books/book.epub.metadata.json

Schema (matches paper convention):
{
  "metadataAttributes": {
    "title": "...",
    "authors": "...",
    "publisher": "...",
    "language": "...",
    "isbn": "...",
    "publish_date": "..."
  }
}
"""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.ingestion.epub_parser import parse_epub


def generate_metadata(epub_path: Path) -> dict:
    """Parse EPUB OPF and return metadataAttributes dict."""
    book = parse_epub(str(epub_path))
    attrs = {}

    if book.title:
        attrs["title"] = book.title
    if book.creator:
        attrs["authors"] = book.creator
    if book.publisher:
        attrs["publisher"] = book.publisher
    if book.language:
        attrs["language"] = book.language
    if book.isbn:
        attrs["isbn"] = book.isbn
    if book.publication_date:
        attrs["publish_date"] = book.publication_date

    return {"metadataAttributes": attrs}


def main():
    books_dir = PROJECT / "test_books"
    epubs = sorted(books_dir.glob("*.epub"))
    if not epubs:
        print(f"No .epub files in {books_dir}")
        sys.exit(1)

    created = 0
    skipped = 0
    errors = 0

    for epub_path in epubs:
        meta_path = epub_path.with_name(epub_path.name + ".metadata.json")

        # Skip if already exists
        if meta_path.exists():
            skipped += 1
            continue

        try:
            meta = generate_metadata(epub_path)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"  Created: {meta_path.name}")
            created += 1
        except Exception as e:
            print(f"  Error parsing {epub_path.name}: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone: {created} created, {skipped} skipped, {errors} errors out of {len(epubs)} EPUBs")


if __name__ == "__main__":
    main()