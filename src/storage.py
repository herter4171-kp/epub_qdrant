"""Qdrant storage: create collections, upsert vectors, and query."""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

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
from src.paper_chunker import PaperChunk

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

    def _ensure_collection(
        self,
        collection_name: str,
        index_fields: Optional[list[str]] = None,
    ) -> None:
        """Create collection if it does not already exist.

        Args:
            collection_name: Name of the Qdrant collection.
            index_fields: List of payload field names to index as KEYWORD.
        """
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
            # Default index fields for books
            if index_fields is None:
                index_fields = [
                    "source_file", "book_title", "section_title",
                    "publisher", "language", "isbn",
                ]
            # Create payload indexes for metadata filtering
            for field in index_fields:
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
                    "publisher": chunk.publisher,
                    "language": chunk.language,
                    "isbn": chunk.isbn,
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
                "score": float(point.score) if hasattr(point, "score") else 0.0,
                "text": point.payload.get("text", ""),
                # Unified fields (for cross-collection compat)
                "doc_id": point.payload.get("doc_id", ""),
                "doc_type": point.payload.get("doc_type", ""),
                "title": point.payload.get("title", ""),
                "section": point.payload.get("section", ""),
                "authors": point.payload.get("authors", []),
                "year": point.payload.get("year", 0),
                # Legacy EPUB fields
                "book_title": point.payload.get("book_title", ""),
                "section_title": point.payload.get("section_title", ""),
                "chapter_index": point.payload.get("chapter_index", 0),
                "section_index": point.payload.get("section_index", 0),
                "chunk_index": point.payload.get("chunk_index", 0),
                "token_count": point.payload.get("token_count", 0),
                "source_file": point.payload.get("source_file", ""),
                "publisher": point.payload.get("publisher", ""),
                "language": point.payload.get("language", ""),
                "isbn": point.payload.get("isbn", ""),
                # Paper-specific fields
                "arxiv_id": point.payload.get("arxiv_id", ""),
                "category": point.payload.get("category", ""),
                "subcategory": point.payload.get("subcategory", ""),
                "publish_date": point.payload.get("publish_date", ""),
                "chunk_count": point.payload.get("chunk_count", 0),
            })

        return output

    def list_collections(self) -> List[str]:
        """Return list of all collection names."""
        collections = self._client.get_collections()
        return [c.name for c in collections.collections]

    def list_collections_info(self) -> List[dict]:
        """Return per-collection metadata stats (point count, etc.).

        Returns:
            List of dicts with collection name, point count, and vector config.
        """
        collections = self._client.get_collections()
        result = []
        for c in collections.collections:
            try:
                info = self._client.get_collection(c.name)
                result.append({
                    "name": c.name,
                    "points": info.points_count if hasattr(info, "points_count") else 0,
                    "vector_size": self._vector_size,
                    "distance": info.distance.value if hasattr(info, "distance") else "Cosine",
                })
            except Exception as e:
                logger.warning(f"Could not get info for collection '{c.name}': {e}")
                result.append({"name": c.name, "points": 0, "error": str(e)})
        return result

    def search_with_filter(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 10,
        filter_by: Optional[Dict[str, str]] = None,
    ) -> List[dict]:
        """Search a collection with optional metadata pre-filtering.

        Args:
            collection_name: Qdrant collection to search.
            query_text: Query string.
            top_k: Number of results to return.
            filter_by: Optional dict of metadata key->value pairs for pre-filtering.

        Returns:
            List of dicts with score, text, and payload.
        """
        from src.embedder import Embedder
        from qdrant_client.models import FieldCondition, MatchValue, Filter

        embedder = Embedder(settings.OLLAMA_URL, settings.EMBEDDING_MODEL)
        query_vector = embedder.embed_single(query_text)

        query_filter = None
        if filter_by:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_by.items()
            ]
            if conditions:
                query_filter = Filter(must=conditions)

        results = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

        output = []
        for point in results.points:
            output.append({
                "score": float(point.score) if hasattr(point, "score") else 0.0,
                "text": point.payload.get("text", ""),
                "doc_id": point.payload.get("doc_id", ""),
                "doc_type": point.payload.get("doc_type", ""),
                "title": point.payload.get("title", ""),
                "section": point.payload.get("section", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "source_file": point.payload.get("source_file", ""),
                "token_count": point.payload.get("token_count", 0),
                # Preserve legacy fields for backward compat
                "book_title": point.payload.get("book_title", ""),
                "section_title": point.payload.get("section_title", ""),
                "publisher": point.payload.get("publisher", ""),
                "language": point.payload.get("language", ""),
                "isbn": point.payload.get("isbn", ""),
                "arxiv_id": point.payload.get("arxiv_id", ""),
                "category": point.payload.get("category", ""),
            })

        return output

    def upsert_paper_file(
        self,
        pdf_path: str,
        chunks: List[PaperChunk],
        collection_name: Optional[str] = None,
    ) -> int:
        """Upsert all chunks for a single PDF paper into the shared papers collection.

        Uses integer point IDs generated from arxiv_id hash for Qdrant compatibility.
        The arxiv_id is stored in payload for traceability.

        Args:
            pdf_path: Path to the source PDF file.
            chunks: List of PaperChunk objects with vectors populated.
            collection_name: Override the papers collection name.

        Returns:
            Number of chunks upserted.
        """
        path = Path(pdf_path)
        name = collection_name or settings.QDRANT_PAPERS_COLLECTION
        collection_name = _sanitize_collection_name(name)

        # Use paper-specific index fields
        self._ensure_collection(
            collection_name,
            index_fields=["arxiv_id", "category", "title", "source_file"],
        )

        # Get existing point count to use as base ID for this paper batch
        try:
            collection_info = self._client.get_collection(collection_name)
            existing_points = collection_info.points_count
        except Exception:
            existing_points = 0

        points = []
        for chunk in chunks:
            if not hasattr(chunk, "vector") or chunk.vector is None:
                logger.warning(f"Skipping chunk {chunk.id} - no vector")
                continue

            # Simple sequential integer ID starting from existing_points
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

        self._client.upsert(
            collection_name=collection_name,
            points=points,
        )

        logger.info(
            f"Upserted {len(points)} paper chunks into collection '{collection_name}'"
        )
        return len(points)

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        self._client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted collection: {collection_name}")
