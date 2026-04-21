#!/usr/bin/env python3
"""Inspect EPUB metadata from test_books directory."""

import epub
from pathlib import Path

BOOKS_DIR = Path(__file__).parent.parent / "test_books"

def inspect_epub(path: str) -> dict:
    book_file = epub.open_epub(path)
    meta = book_file.opf.metadata
    
    result = {
        "title": meta.titles,
        "creator": meta.creators,
        "publisher": meta.publisher,
        "date": meta.dates,
        "subject": meta.subjects,
        "language": meta.languages,
        "rights": meta.right,
        "description": meta.description,
        "contributor": meta.contributors,
        "coverage": meta.coverage,
        "identifier": meta.identifiers,
        "relation": meta.relation,
        "source": meta.source,
        "type": meta.dc_type,
        "format": meta.format,
        "metas": meta.metas,
    }
    
    book_file.close()
    return result

if __name__ == "__main__":
    for epub_path in sorted(BOOKS_DIR.glob("*.epub")):
        print(f"\n{'='*60}")
        print(f"FILE: {epub_path.name}")
        print(f"{'='*60}")
        try:
            meta = inspect_epub(str(epub_path))
            for key, value in meta.items():
                if value:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"  ERROR: {e}")