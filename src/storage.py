"""Qdrant storage: create collections, upsert vectors, and query."""

import logging
from pathlib import Path
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)

from src.chunker import Chunk
from src.config import settings

logger = logging.getLogger(__name__)


def _book_title_to_collection(book_title: str) -> str:
    """Convert a book title to a valid Qdrant collection name."""
    safe = book_title.lower().replace(" ", "-").replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c in ("-", "_"))
    return safe[:63]  # Qdrant collection names max 63 chars


class Storage:
    """Manages Qdrant collection lifecycle and operations."""

    def __init__(
        self,
        url: Optional[str] = None,
        host: str = "192.168.68.75",
        port: int = 6333,
    ):
        base = url or settings.QDRANT_URL
        # Parse url into host/port if provided
        if base:
            from urllib.parse import urlparse
            parsed = urlparse(base)
            self._client = QdrantClient(
                host=parsed.hostname or host,
                port=parsed.port or port,
            )
        else:
            self._client = QdrantClient(host=host, port=port)

        self._vector_size = settings.VECTOR_SIZE
        self._distance = Distance(settings.DISTANCE)

    def _ensure_collection(self, collection_name: str) -> None:
        """Create collection if it does not already exist."""
        # Check existing collections
        collections = self._client.get_collections()
        names = [c.name for c in collections.collections]

        if collection_name not in names:
            logger.info(f"Creating collection: {collection_name}")
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                ),
            )
            # Create payload indexes for metadata filtering
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="book_title",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="section_title",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def upsert_file(
        self,
        epub_path: str,
        chunks: List[Chunk],
    ) -> int:
        """Upsert all chunks for a single EPUB file.

        Creates (or reuses) a collection named after the book.
        Uses deterministic point IDs to allow re-ingestion without duplication.

        Args:
            epub_path: Path to the source EPUB file.
            chunks: List of Chunk objects with vectors populated.

        Returns:
            Number of chunks upserted.
        """
        path = Path(epub_path)
        book_name = path.stem
        collection_name = _book_title_to_collection(book_name)

        self._ensure_collection(collection_name)

        points = []
        for chunk in chunks:
            if not hasattr(chunk, "vector") or chunk.vector is None:
                logger.warning(f"Skipping chunk {chunk.id} - no vector")
                continue

            # Qdrant requires unsigned integer or UUID IDs
            point_id = int(chunk.id)
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
                },
            )
            points.append(point)

        if not points:
            logger.warning(f"No valid points to upsert for {collection_name}")
            return 0

        self._client.upsert(
            collection_name=collection_name,
            points=points,
        )

        logger.info(
            f"Upserted {len(points)} points into collection '{collection_name}'"
        )
        return len(points)

    def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 10,
    ) -> List[dict]:
        """Search a collection for text similar to query_text.

        Args:
            collection_name: Qdrant collection to search.
            query_text: Query string.
            top_k: Number of results to return.

        Returns:
            List of dicts with score, text, and payload.
        """
        # Generate query embedding
        from src.embedder import Embedder
        embedder = Embedder(settings.OLLAMA_URL, settings.EMBEDDING_MODEL)
        query_vector = embedder.embed_single(query_text)

        results = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
        )

        output = []
        for point in results.points:
            output.append({
                "score": point.score if hasattr(point, "score") else None,
                "text": point.payload.get("text", ""),
                "book_title": point.payload.get("book_title", ""),
                "section_title": point.payload.get("section_title", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
            })

        return output

    def list_collections(self) -> List[str]:
        """Return list of all collection names."""
        collections = self._client.get_collections()
        return [c.name for c in collections.collections]

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        self._client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted collection: {collection_name}")