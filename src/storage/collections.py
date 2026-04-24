"""Collection lifecycle management: create, delete, list, migrate.

Also exports the Storage class that wraps all storage operations.
"""

import logging
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

from src.config import settings
from servers.embedding_server.client import get_dense_vectors
from src.ingestion.chunker import Chunk
from src.storage.config import settings as storage_settings

logger = logging.getLogger(__name__)


def _sanitize_collection_name(name: str) -> str:
    """Sanitize a collection name to be Qdrant-compatible."""
    safe = name.lower().strip()
    safe = "".join(c for c in safe if c.isalnum() or c in ("-", "_", "."))
    return safe[:63]


def _parse_url(base: str):
    """Parse a Qdrant URL into (host, port) tuple."""
    from urllib.parse import urlparse
    parsed = urlparse(base)
    return parsed.hostname or "192.168.68.75", parsed.port or 6333


def _build_qdrant_client(url: Optional[str] = None,
                         host: str = "192.168.68.75",
                         port: int = 6333) -> QdrantClient:
    """Build a QdrantClient from URL or host/port."""
    base = url or settings.QDRANT_URL
    if base:
        h, p = _parse_url(base)
        return QdrantClient(host=h, port=p)
    return QdrantClient(host=host, port=port)


def _ensure_collection(client: QdrantClient,
                       collection_name: str,
                       index_fields: Optional[list] = None,
                       vector_size: int = 768,
                       distance: str = "Cosine") -> None:
    """Create collection if it does not already exist.

    Args:
        client: QdrantClient instance.
        collection_name: Name of the Qdrant collection.
        index_fields: List of payload field names to index as KEYWORD.
        vector_size: Dense vector dimensions.
        distance: Distance metric string.
    """
    collections = client.get_collections()
    names = [c.name for c in collections.collections]

    if collection_name not in names:
        logger.info(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance(distance),
            ),
        )
        if index_fields is None:
            index_fields = [
                "source_file", "book_title", "section_title",
                "publisher", "language", "isbn",
            ]
        for field in index_fields:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )


def _get_vector_name(client: QdrantClient, collection_name: str) -> Optional[str]:
    """Check if a collection uses named vectors. Returns vector name or None for unnamed."""
    try:
        info = client.get_collection(collection_name)
        if hasattr(info, "config") and info.config:
            vectors = info.config.params.vectors
            if isinstance(vectors, dict) and len(vectors) > 0:
                return list(vectors.keys())[0]
    except Exception:
        pass
    return None


def list_collections(client: QdrantClient) -> List[str]:
    """Return list of all collection names."""
    collections = client.get_collections()
    return [c.name for c in collections.collections]


def list_collections_info(client: QdrantClient,
                          vector_size: int = 768,
                          distance: str = "Cosine") -> List[dict]:
    """Return per-collection metadata stats (point count, etc.)."""
    collections = client.get_collections()
    result = []
    for c in collections.collections:
        try:
            info = client.get_collection(c.name)
            result.append({
                "name": c.name,
                "points": info.points_count if hasattr(info, "points_count") else 0,
                "vector_size": vector_size,
                "distance": info.distance.value if hasattr(info, "distance") else distance,
            })
        except Exception as e:
            logger.warning(f"Could not get info for collection '{c.name}': {e}")
            result.append({"name": c.name, "points": 0, "error": str(e)})
    return result


def delete_collection(client: QdrantClient, collection_name: str) -> None:
    """Delete a collection."""
    client.delete_collection(collection_name=collection_name)
    logger.info(f"Deleted collection: {collection_name}")


# ── Storage class (wraps all operations) ──────────────────────────


class Storage:
    """Manages Qdrant collection lifecycle and operations.

    This is the main entry point for storage operations. It wraps the
    submodule functions and provides a unified API.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        host: str = "192.168.68.75",
        port: int = 6333,
    ):
        self._client = _build_qdrant_client(url, host, port)
        self._vector_size = settings.VECTOR_SIZE
        self._distance = Distance(settings.DISTANCE)

    @property
    def client(self) -> QdrantClient:
        """Return the underlying QdrantClient."""
        return self._client

    def _ensure_collection(self, collection_name: str,
                           index_fields: Optional[list] = None) -> None:
        """Create collection if it does not already exist."""
        _ensure_collection(self._client, collection_name, index_fields,
                           self._vector_size, self._distance.value)

    def _get_vector_name(self, collection_name: str) -> Optional[str]:
        """Check if a collection uses named vectors."""
        return _get_vector_name(self._client, collection_name)

    def upsert_file(self, epub_path: str, chunks: List[Chunk],
                    collection_name: Optional[str] = None) -> int:
        """Upsert all chunks for a single EPUB file into the collection."""
        from src.storage.upsert import upsert_file as _upsert_file
        return _upsert_file(self._client, epub_path, chunks, collection_name)

    def upsert_paper_file(self, pdf_path: str, chunks: List,
                          collection_name: Optional[str] = None) -> int:
        """Upsert all chunks for a single PDF paper into the papers collection."""
        from src.storage.upsert import upsert_paper_file as _upsert_paper_file
        return _upsert_paper_file(self._client, pdf_path, chunks, collection_name)

    def search(self, collection_name: str, query_text: str,
               top_k: int = 10) -> List[dict]:
        """Search a collection for text similar to query_text.

        Handles both unnamed-vector collections and named-vector collections.
        """
        query_vector = get_dense_vectors([query_text])[0]

        vector_name = self._get_vector_name(collection_name)
        kwargs: dict = {
            "collection_name": collection_name,
            "query": query_vector,
            "limit": top_k,
        }
        if vector_name:
            kwargs["using"] = vector_name

        results = self._client.query_points(**kwargs)

        output = []
        for point in results.points:
            output.append({
                "score": float(point.score) if hasattr(point, "score") else 0.0,
                "text": point.payload.get("text", ""),
                "doc_id": point.payload.get("doc_id", ""),
                "doc_type": point.payload.get("doc_type", ""),
                "title": point.payload.get("title", ""),
                "section": point.payload.get("section", ""),
                "authors": point.payload.get("authors", []),
                "year": point.payload.get("year", 0),
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
                "arxiv_id": point.payload.get("arxiv_id", ""),
                "category": point.payload.get("category", ""),
                "subcategory": point.payload.get("subcategory", ""),
                "publish_date": point.payload.get("publish_date", ""),
                "chunk_count": point.payload.get("chunk_count", 0),
            })
        return output

    def search_with_filter(self, collection_name: str, query_text: str,
                           top_k: int = 10,
                           filter_by: Optional[Dict[str, str]] = None) -> List[dict]:
        """Search a collection with optional metadata pre-filtering."""
        query_vector = get_dense_vectors([query_text])[0]

        query_filter = None
        if filter_by:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_by.items()
            ]
            if conditions:
                query_filter = Filter(must=conditions)

        vector_name = self._get_vector_name(collection_name)
        kwargs = {
            "collection_name": collection_name,
            "query": query_vector,
            "limit": top_k,
            "query_filter": query_filter,
        }
        if vector_name:
            kwargs["using"] = vector_name

        results = self._client.query_points(**kwargs)

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
                "book_title": point.payload.get("book_title", ""),
                "section_title": point.payload.get("section_title", ""),
                "publisher": point.payload.get("publisher", ""),
                "language": point.payload.get("language", ""),
                "isbn": point.payload.get("isbn", ""),
                "arxiv_id": point.payload.get("arxiv_id", ""),
                "category": point.payload.get("category", ""),
            })
        return output

    def list_collections(self) -> List[str]:
        """Return list of all collection names."""
        return list_collections(self._client)

    def list_collections_info(self) -> List[dict]:
        """Return per-collection metadata stats."""
        return list_collections_info(self._client,
                                     self._vector_size, self._distance)

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        delete_collection(self._client, collection_name)

    def list_books(self, collection_name: Optional[str] = None) -> List[dict]:
        """List unique books in a collection based on source_file metadata."""
        from collections import defaultdict
        coll = collection_name or settings.QDRANT_COLLECTION
        coll = _sanitize_collection_name(coll)

        books_map: dict = defaultdict(lambda: {
            "book_title": "",
            "publisher": "",
            "language": "",
            "isbn": "",
            "chunk_count": 0,
        })

        offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=coll,
                limit=500,
                offset=offset,
                with_payload=["source_file", "book_title", "publisher", "language", "isbn"],
                with_vectors=False,
            )
            if not points:
                break

            for p in points:
                sf = p.payload.get("source_file", "unknown")
                entry = books_map[sf]
                entry["chunk_count"] += 1
                if not entry["book_title"]:
                    entry["book_title"] = p.payload.get("book_title", "")
                    entry["publisher"] = p.payload.get("publisher", "")
                    entry["language"] = p.payload.get("language", "")
                    entry["isbn"] = p.payload.get("isbn", "")

            if next_offset is None:
                break
            offset = next_offset

        return [
            {**entry, "source_file": sf}
            for sf, entry in books_map.items()
        ]
