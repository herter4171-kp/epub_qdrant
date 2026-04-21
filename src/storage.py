"""Qdrant storage: create collections, upsert vectors, and query."""

import hashlib
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


def _sanitize_collection_name(name: str) -> str:
    """Sanitize a collection name to be Qdrant-compatible."""
    safe = name.lower().strip()
    safe = "".join(c for c in safe if c.isalnum() or c in ("-", "_", "."))
    return safe[:63]


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
        """Create shared collection if it does not already exist."""
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
            for field in ("source_file", "book_title", "section_title"):
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )

    def upsert_file(
        self,
        epub_path: str,
        chunks: List[Chunk],
        collection_name: Optional[str] = None,
    ) -> int:
        """Upsert all chunks for a single EPUB file into the shared collection.

        All files go into one collection. Each chunk gets a deterministic ID
        that combines a filename hash and a global counter to avoid collisions.

        Args:
            epub_path: Path to the source EPUB file.
            chunks: List of Chunk objects with vectors populated.
            collection_name: Override the default collection name.

        Returns:
            Number of chunks upserted.
        """
        path = Path(epub_path)
        name = collection_name or settings.QDRANT_COLLECTION
        collection_name = _sanitize_collection_name(name)

        self._ensure_collection(collection_name)

        # Track a base offset per file to avoid collisions across books
        # Query the highest existing ID in the collection for this collection
        # Simpler approach: use a global counter passed via payload, store file_hash in payload
        file_hash = hashlib.md5(path.name.encode()).hexdigest()[:8]

        # Get existing point count in collection to determine base ID offset
        try:
            collection_info = self._client.get_collection(collection_name)
            existing_points = collection_info.points_count
        except Exception:
            existing_points = 0

        points = []
        for idx, chunk in enumerate(chunks):
            if not hasattr(chunk, "vector") or chunk.vector is None:
                logger.warning(f"Skipping chunk - no vector")
                continue

            # Integer ID: base + per-chunk offset
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