# src/ingestion — EPUB & Paper Ingestion

Parses EPUB files and PDF papers into text chunks ready for embedding.

## Components

| File | Purpose |
|------|---------|
| `epub_parser.py` | Parses EPUB structure, extracts text by chapter/section with heading detection |
| `paper_loader.py` | Loads PDF text + optional JSON metadata for paper ingestion |
| `chunker.py` | Paragraph-aware recursive chunking with configurable token size and overlap |
| `paper_chunker.py` | Section-aware paper chunking (Abstract, Intro, Methods, Results, etc.) |

## Usage

```python
from src.ingestion.epub_parser import parse_epub
from src.ingestion.chunker import chunk_section

book = parse_epub("my_book.epub")
chunks = []
for section in book.sections:
    chunks.extend(chunk_section(section, chunk_size=500, chunk_overlap=100, book_title=book.title))
```

## Chunking Strategy

- **Books**: Paragraph-aware recursive split. Groups paragraphs into ~500-token chunks with 100-token overlap. Respects paragraph boundaries; splits long paragraphs on sentences.
- **Papers**: Section-aware split. Splits at section headings (Abstract, Introduction, Methods, Results, Discussion, etc.), then chunks each section independently.

## Token Counting

Rough estimate: `len(text) // 4` characters per token. Suitable for chunk size targets; not production-accurate.