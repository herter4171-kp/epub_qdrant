"""Thin client for the MiniCOIL sparse embedding server on the GPU box.

Usage:
    from src.embedding.client import get_sparse_vectors
    vectors = get_sparse_vectors(["hello world"], is_query=False)
"""

import logging
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

# GPU box MiniCOIL server endpoint
MINICOIL_URL = "http://192.168.68.75:9000/embed"


def get_sparse_vectors(
    texts: List[str],
    is_query: bool = False,
    timeout: int = 360,
) -> List[Dict]:
    """Call the MiniCOIL server and return sparse vectors.

    Args:
        texts: List of strings to embed. Batch size up to 1024.
        is_query: True at search time, False during indexing.
                  MiniCOIL uses different embedding paths — do not mix.
        timeout: Request timeout in seconds.

    Returns:
        List of dicts with 'indices' (List[int]) and 'values' (List[float]) keys.

    Raises:
        requests.exceptions.RequestException: If the server request fails.
    """
    if not texts:
        return []

    if len(texts) > 1024:
        raise ValueError(f"Max 1024 texts per request, got {len(texts)}")

    resp = requests.post(
        MINICOIL_URL,
        json={"texts": texts, "is_query": is_query},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["vectors"]


def health_check() -> bool:
    """Check if the MiniCOIL server is reachable and ready."""
    try:
        resp = requests.get("http://192.168.68.75:9000/health", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("status") == "ok" and data.get("ready") is True
    except Exception as e:
        logger.error(f"MiniCOIL health check failed: {e}")
        return False