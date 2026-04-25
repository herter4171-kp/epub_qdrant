"""Thin HTTP client for the unified embedding server.

Usage:
    from servers.embedding_server.client import get_dense_vectors, get_sparse_vectors, health_check

    vectors = get_dense_vectors(["hello world"])
    sparse = get_sparse_vectors(["hello world"], is_query=True)
    ok = health_check()
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Ensure .env is loaded before reading env vars
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

logger = logging.getLogger(__name__)

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:8100")

# Per-request timeout: (connect, read).  Connect should be fast on LAN;
# read can be slow when the GPU is saturated.
_TIMEOUT = (10, 300)

# Retry forever with backoff — the server is alive but busy.
_RETRY = Retry(
    total=None,            # no cap on total retries
    connect=None,          # no cap on connect retries
    read=None,             # no cap on read retries
    backoff_factor=2,      # 2s, 4s, 8s, 16s, … between retries
    status_forcelist=[502, 503, 504],
    allowed_methods=["POST", "GET"],
)

def _session() -> requests.Session:
    """Build a requests Session with unlimited retries."""
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

_sess = _session()


def get_dense_vectors(texts: List[str], batch_size: int = 128) -> List[List[float]]:
    """Embed texts into 768-d dense vectors via the unified embedding server."""
    url = f"{EMBEDDING_SERVER_URL}/embed_dense"
    resp = _sess.post(
        url,
        json={"texts": texts, "batch_size": batch_size},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["vectors"]


def get_sparse_vectors(texts: List[str], is_query: bool = False) -> List[Dict]:
    """Embed texts into sparse vectors via the unified embedding server."""
    url = f"{EMBEDDING_SERVER_URL}/embed_sparse"
    resp = _sess.post(
        url,
        json={"texts": texts, "is_query": is_query},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["vectors"]


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
