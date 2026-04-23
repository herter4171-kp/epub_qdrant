"""Qdrant storage layer: create collections, upsert vectors, and query.

Split into logical submodules:
- collections.py: Collection lifecycle (create, delete, list, migrate)
- upsert.py: Upsert logic for EPUB and paper vectors
- scroll.py: Paginated scroll for batch processing
- config.py: Re-exports from src.config
"""

from src.storage.collections import (
    Storage,
    _sanitize_collection_name,
    _build_qdrant_client,
    _ensure_collection,
    list_collections,
    list_collections_info,
    delete_collection,
)
from src.storage.upsert import (
    upsert_file,
    upsert_paper_file,
)
from src.storage.scroll import (
    scroll_all,
    scroll_with_filter,
)
from src.storage.config import settings

__all__ = [
    "Storage",
    "_sanitize_collection_name",
    "_build_qdrant_client",
    "_ensure_collection",
    "list_collections",
    "list_collections_info",
    "delete_collection",
    "upsert_file",
    "upsert_paper_file",
    "scroll_all",
    "scroll_with_filter",
    "settings",
]