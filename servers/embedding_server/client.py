"""Thin HTTP client for the unified embedding server.

Usage:
    from servers.embedding_server.client import get_dense_vectors, get_sparse_vectors, health_check

    vectors = get_dense_vectors(["hello world"])
    sparse = get_sparse_vectors(["hello world"], is_query=True)
    ok = health_check()
"""

import logging
import os
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:8100")

# Default timeout in seconds for embedding requests (models can be slow on large batches)
_TIMEOUT = 300


def get_dense_vectors(texts: List[str], batch_size: int = 128) -> List[List[float]]:
    """Embed texts into 768-d dense vectors via the unified embedding server.

    Args:
        texts: List of strings to embed.
        batch_size: Sub-batch size for GPU processing on the server side.

    Returns:
        List of 768-dimensional float vectors, one per input text.

    Raises:
        requests.ConnectionError: If the embedding server is unreachable.
        requests.Timeout: If the request exceeds the timeout.
        requests.HTTPError: If the server returns a non-2xx status.
    """
    url = f"{EMBEDDING_SERVER_URL}/embed_dense"
    try:
        resp = requests.post(
            url,
            json={"texts": texts, "batch_size": batch_size},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["vectors"]
    except requests.ConnectionError:
        raise requests.ConnectionError(
            f"Cannot connect to embedding server at {EMBEDDING_SERVER_URL}. "
            "Is the server running?"
        )
    except requests.Timeout:
        raise requests.Timeout(
            f"Embedding server at {EMBEDDING_SERVER_URL} timed out after {_TIMEOUT}s "
            f"while embedding {len(texts)} texts."
        )


def get_sparse_vectors(texts: List[str], is_query: bool = False) -> List[Dict]:
    """Embed texts into sparse vectors via the unified embedding server.

    Args:
        texts: List of strings to embed.
        is_query: True for query-mode embeddings, False for document-mode.

    Returns:
        List of dicts with 'indices' (List[int]) and 'values' (List[float]) keys.

    Raises:
        requests.ConnectionError: If the embedding server is unreachable.
        requests.Timeout: If the request exceeds the timeout.
        requests.HTTPError: If the server returns a non-2xx status.
    """
    url = f"{EMBEDDING_SERVER_URL}/embed_sparse"
    try:
        resp = requests.post(
            url,
            json={"texts": texts, "is_query": is_query},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["vectors"]
    except requests.ConnectionError:
        raise requests.ConnectionError(
            f"Cannot connect to embedding server at {EMBEDDING_SERVER_URL}. "
            "Is the server running?"
        )
    except requests.Timeout:
        raise requests.Timeout(
            f"Embedding server at {EMBEDDING_SERVER_URL} timed out after {_TIMEOUT}s "
            f"while embedding {len(texts)} texts."
        )


def health_check() -> bool:
    """Check if the embedding server is healthy with both models loaded.

    Returns:
        True if both dense and sparse models are loaded, False otherwise.
    """
    url = f"{EMBEDDING_SERVER_URL}/health"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("dense") is True and data.get("sparse") is True
    except Exception as e:
        logger.error("Embedding server health check failed: %s", e)
        return False
