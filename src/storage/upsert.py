"""Upsert logic for dense+sparse named vectors (EPUB + paper)."""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from qdrant_client.models import PointStruct

from src.ingestion.chunker import Chunk
from src.config import settings
from src.storage.collections import _sanitize_collection_name, _build_qdrant_client, _ensure_collection

logger = logging.getLogger(__name__)


def upsert_file(client, epub_path: str, chunks: List[Chunk],
                collection_name: Optional[str] = None) -> int:
    """Upsert all chunks for a single EPUB file into the collection.

    Args:
        client: QdrantClient instance.
        epub_path: Path to the source EPUB file.
        chunks: List of Chunk objects with vectors populated.
        collection_name: Override the default collection name.

    Returns:
        Number of chunks upserted.
    """
    path = Path(epub_path)
    name = collection_name or settings.QDRANT_COLLECTION
    collection_name = _sanitize_collection_name(name)

    _ensure_collection(client, collection_name)

    file_hash = hashlib.md5(path.name.encode()).hexdigest()[:8]

    try:
        collection_info = client.get_collection(collection_name)
        existing_points = collection_info.points_count
    except Exception:
        existing_points = 0

    points = []
    for idx, chunk in enumerate(chunks):
        if not hasattr(chunk, "vector") or chunk.vector is None:
            logger.warning(f"Skipping chunk - no vector")
            continue

        point_id = existing_points + idx
        point = PointStruct(
            id=point_id,
            vector=chunk.vector,
            payload={
                "text": chunk.text,
                "book_title": chunk.book_title,
                "section_title": chunk.section_title,
                "chapter_index": chunk.chapter_index,
                "section_index": chunk.section_index,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "source_file": str(path.name),
                "publisher": chunk.publisher,
                "language": chunk.language,
                "isbn": chunk.isbn,
            },
        )
        points.append(point)

    if not points:
        logger.warning(f"No valid points to upsert for {collection_name}")
        return 0

    client.upsert(collection_name=collection_name, points=points)

    logger.info(f"Upserted {len(points)} points into collection '{collection_name}'")
    return len(points)


def upsert_paper_file(client, pdf_path: str, chunks: List,
                      collection_name: Optional[str] = None) -> int:
    """Upsert all chunks for a single PDF paper into the papers collection.

    Args:
        client: QdrantClient instance.
        pdf_path: Path to the source PDF file.
        chunks: List of PaperChunk objects with vectors populated.
        collection_name: Override the papers collection name.

    Returns:
        Number of chunks upserted.
    """
    path = Path(pdf_path)
    name = collection_name or settings.QDRANT_PAPERS_COLLECTION
    collection_name = _sanitize_collection_name(name)

    _ensure_collection(
        client, collection_name,
        index_fields=["arxiv_id", "category", "title", "source_file"],
    )

    try:
        collection_info = client.get_collection(collection_name)
        existing_points = collection_info.points_count
    except Exception:
        existing_points = 0

    points = []
    for chunk in chunks:
        if not hasattr(chunk, "vector") or chunk.vector is None:
            logger.warning(f"Skipping chunk {chunk.id} - no vector")
            continue

        point_id = existing_points + chunk.chunk_index

        point = PointStruct(
            id=point_id,
            vector=chunk.vector,
            payload={
                "text": chunk.text,
                "arxiv_id": chunk.arxiv_id,
                "title": chunk.title,
                "category": chunk.category,
                "subcategory": chunk.subcategory,
                "authors": chunk.authors,
                "publish_date": chunk.publish_date,
                "chunk_index": chunk.chunk_index,
                "chunk_count": chunk.chunk_count,
                "token_count": chunk.token_count,
                "source_file": str(path.name),
            },
        )
        points.append(point)

    if not points:
        logger.warning(f"No valid points to upsert for {collection_name}")
        return 0

    client.upsert(collection_name=collection_name, points=points)

    logger.info(f"Upserted {len(points)} paper chunks into collection '{collection_name}'")
    return len(points)