"""Scroll existing Qdrant collection, extract (point_id, text) pairs."""

from typing import List, Tuple

from src.storage.scroll import scroll_all
from src.config import settings


def scroll_collection(collection_name: str) -> List[Tuple[int, str]]:
    """Scroll all points, return (point_id, text) in scroll order.
    
    Args:
        collection_name: Name of the Qdrant collection to scroll.
        
    Returns:
        List of (point_id, text) tuples. Text is extracted from the
        'text' field in the payload. Empty strings are filtered out.
    """
    from qdrant_client import QdrantClient
    
    client = QdrantClient(url=settings.QDRANT_URL)
    points = scroll_all(
        client, collection_name,
        batch_size=256,
        with_payload=True,
        with_vectors=False,
    )
    
    result = []
    for p in points:
        text = (p.payload.get("text") or "").strip()
        if text:
            result.append((p.id, text))
    
    return result