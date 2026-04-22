#!/usr/bin/env python3
"""Embed PDF papers from ./downloads into a Qdrant collection called 'papers'.

Scans ./downloads/ for .pdf files, reads matching .json metadata,
extracts text via pypdf, chunks, embeds via Ollama, and upserts
into a new Qdrant collection named by QDRANT_PAPERS_COLLECTION (default: 'papers').

Limits processing to the first PDF in alphanumeric order by default.
Set PAPER_EMBED_ALL=1 to process all PDFs.

Usage:
    python scripts/embed_papers_to_qdrant.py
    PAPER_EMBED_ALL=1 python scripts/embed_papers_to_qdrant.py  # all PDFs
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

# Ensure project root is on sys.path so `src` package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import settings
from src.embedder import Embedder
from src.paper_chunker import PaperChunk, chunk_paper
from src.storage import Storage

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = _PROJECT_ROOT
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"


def _parse_json_metadata(json_path: Path) -> dict:
    """Parse a metadata JSON file and return a dict of key: value pairs.

    JSON format:
    {
      "metadataAttributes": [
        "title: ...",
        "authors: ...",
        ...
      ]
    }
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    result: dict = {}
    for attr in data.get("metadataAttributes", []):
        # Split on first colon only
        if ": " in attr:
            key, value = attr.split(": ", 1)
            result[key] = value
        elif ":" in attr:
            key, value = attr.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)
    return "\n\n".join(texts)


def _find_matching_metadata(pdf_path: Path) -> Optional[dict]:
    """Find and parse the JSON metadata file matching a PDF.

    PDF naming: {arxiv_id_with_underscores}.pdf
    JSON naming: {arxiv_id_with_underscores}.json
    """
    stem = pdf_path.stem  # e.g., "2010_03768"
    json_path = pdf_path.with_suffix(".json")
    if json_path.exists():
        return _parse_json_metadata(json_path)
    logger.warning(f"No metadata JSON found for {pdf_path.name}, looking for fallback...")
    return None


def _is_already_ingested(storage: Storage, arxiv_id: str) -> bool:
    """Check if a paper with the given arxiv_id already exists in the collection."""
    try:
        # Query for any points with this arxiv_id
        from qdrant_client.models import FieldCondition, MatchValue, Filter
        collection = settings.QDRANT_PAPERS_COLLECTION or "papers"
        result = storage._client.query_points(
            collection_name=collection,
            query_filter=Filter(must=[
                FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_id))
            ]),
            limit=1,
        )
        return result.points is not None and len(result.points) > 0
    except Exception as e:
        logger.warning(f"Failed to check existing arxiv_id '{arxiv_id}': {e}")
        return False


def embed_papers(limit_to_first: bool = True, skip_existing: bool = True) -> int:
    """Embed PDF papers into the Qdrant papers collection.

    Args:
        limit_to_first: If True, only process the first PDF in alphanumeric order.
        skip_existing: If True, skip papers that are already ingested.

    Returns:
        Total number of chunks upserted.
    """
    if not DOWNLOADS_DIR.exists():
        logger.error(f"Downloads directory not found: {DOWNLOADS_DIR}")
        return 0

    # Find all PDF files in downloads/ (non-recursive, flat)
    pdf_files = sorted(DOWNLOADS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No .pdf files found in {DOWNLOADS_DIR}")
        return 0

    if limit_to_first:
        pdf_files = [pdf_files[0]]
        logger.info(f"Limited to first PDF: {pdf_files[0].name}")
    else:
        logger.info(f"Found {len(pdf_files)} PDFs in downloads/")

    # Setup components
    embedder = Embedder(settings.OLLAMA_URL, settings.EMBEDDING_MODEL)
    storage = Storage()

    total_chunks = 0
    total_skipped = 0
    total_failed = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

        # Load metadata first to check existing
        metadata = _find_matching_metadata(pdf_path)
        if metadata:
            arxiv_id = metadata.get("arxiv_id", pdf_path.stem.replace("_", "."))
        else:
            arxiv_id = pdf_path.stem.replace("_", ".")

        if skip_existing and _is_already_ingested(storage, arxiv_id):
            logger.info(f"  Skipping {pdf_path.name} - already ingested (arxiv_id={arxiv_id})")
            total_skipped += 1
            continue

        # 1. Extract text from PDF
        try:
            text = _extract_pdf_text(pdf_path)
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path.name}: {e}")
            continue

        if not text or len(text.strip()) < 100:
            logger.warning(f"  Text extraction yielded too little content for {pdf_path.name}, skipping")
            continue

        logger.info(f"  Extracted {len(text)} chars of text")

        # 2. Load metadata from JSON
        if metadata is None:
            metadata = _find_matching_metadata(pdf_path)
        title = metadata.get("title", pdf_path.stem)
        category = metadata.get("category", "unknown")
        subcategory = metadata.get("subcategory", "unknown")
        authors = metadata.get("authors", "")
        publish_date = metadata.get("publish_date", "")
        abstract = metadata.get("abstract", "")

        logger.info(f"  arxiv_id={arxiv_id}, category={category}, subcategory={subcategory}")

        # 3. Chunk the text
        try:
            chunks = chunk_paper(
                text=text,
                arxiv_id=arxiv_id,
                title=title,
                category=category,
                subcategory=subcategory,
                authors=authors,
                publish_date=publish_date,
                abstract=abstract,
                source_file=str(pdf_path.name),
            )
        except Exception as e:
            logger.error(f"  Failed to chunk {pdf_path.name}: {e}")
            continue

        logger.info(f"  Chunks: {len(chunks)}")

        if not chunks:
            logger.warning(f"  No chunks generated for {pdf_path.name}")
            continue

        # 4. Embed
        logger.info("  Embedding...")
        texts = [c.text for c in chunks]
        vectors = embedder.embed_batch(texts)

        for chunk, vec in zip(chunks, vectors):
            if vec:
                chunk.vector = vec
            else:
                logger.warning(f"  Skipping chunk {chunk.id} - no vector generated")

        # 5. Upsert
        try:
            count = storage.upsert_paper_file(
                pdf_path=str(pdf_path),
                chunks=chunks,
            )
            total_chunks += count
        except Exception as e:
            logger.error(f"  Failed to upsert {pdf_path.name}: {e}")
            continue

    logger.info(f"\nEmbedding complete.")
    logger.info(f"  Total chunks stored: {total_chunks}")
    logger.info(f"  Skipped (already ingested): {total_skipped}")
    logger.info(f"  Total PDFs processed: {len(pdf_files)}")
    return total_chunks


if __name__ == "__main__":
    limit = not os.environ.get("PAPER_EMBED_ALL", "").lower() in ("1", "true", "yes")
    embed_papers(limit_to_first=limit)