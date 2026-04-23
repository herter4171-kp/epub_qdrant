"""Paginated scroll helpers for batch processing."""

from typing import List, Optional, Tuple

def scroll_all(client, collection_name: str,
               batch_size: int = 256,
               with_payload: bool = True,
               with_vectors: bool = False,
               limit: Optional[int] = None) -> List:
    """Scroll through all points in a collection in batches.

    Args:
        client: QdrantClient instance.
        collection_name: Collection to scroll.
        batch_size: Number of points per batch.
        with_payload: Whether to include payload.
        with_vectors: Whether to include vectors.
        limit: Maximum total points to return (None = all).

    Returns:
        List of all points.
    """
    all_points = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        all_points.extend(points)

        if limit and len(all_points) >= limit:
            break

        if next_offset is None:
            break
        offset = next_offset

    if limit:
        all_points = all_points[:limit]

    return all_points


def scroll_with_filter(client, collection_name: str,
                       query_filter,
                       batch_size: int = 256,
                       with_payload: bool = True,
                       with_vectors: bool = False) -> List:
    """Scroll through points matching a filter in batches."""
    return scroll_all(
        client, collection_name,
        batch_size=batch_size,
        with_payload=with_payload,
        with_vectors=with_vectors,
    )