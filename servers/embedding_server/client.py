"""Thin HTTP client for the unified embedding server.

Usage:
    from servers.embedding_server.client import (
        get_dense_vectors,
        get_sparse_vectors,
        rewrite_query,
        health_check,
    )

    dense = get_dense_vectors(["hello world"])
    sparse = get_sparse_vectors(["hello world"], is_query=True)
    rewritten = rewrite_query("how do we handle salt leaks?")
    ok = health_check()

The embedding functions do NOT rewrite — they pass texts through verbatim.
Use rewrite_query() at the agent/client layer for query-time reformulation.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

logger = logging.getLogger(__name__)

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:8100")

_TIMEOUT = (10, 300)

# Only retry on gateway errors — 500 is handled manually below because
# urllib3 Retry won't resend POST bodies reliably on 500.
_RETRY = Retry(
    total=None,
    connect=None,
    read=None,
    backoff_factor=2,
    status_forcelist=[502, 503, 504],
    allowed_methods=["POST", "GET"],
)

# Delays between 500 retries — CUDA TDR recovery typically takes 2-5s,
# but give it more room in case the server needs to reload models.
_500_RETRY_DELAYS = [5, 10, 20, 40, 60]  # seconds


def _session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

_sess = _session()


def _post_with_500_retry(url: str, payload: dict) -> dict:
    """POST with manual retry on 500 (CUDA crash / kernel timeout).

    urllib3 Retry won't resend POST bodies on 500, so we handle it here.
    Waits progressively longer between attempts so the server has time to
    recover before we hammer it again.
    """
    for attempt, delay in enumerate(_500_RETRY_DELAYS + [None], start=1):
        resp = _sess.post(url, json=payload, timeout=_TIMEOUT)
        if resp.status_code != 500:
            resp.raise_for_status()
            return resp.json()
        if delay is None:
            resp.raise_for_status()  # exhausted — let it propagate
        logger.warning(
            "500 from %s (attempt %d/%d) — CUDA crash? Waiting %ds...",
            url, attempt, len(_500_RETRY_DELAYS) + 1, delay,
        )
        time.sleep(delay)

    raise RuntimeError(f"Unreachable")  # satisfies type checker


def get_dense_vectors(texts: List[str], batch_size: int = 128) -> List[List[float]]:
    """Embed texts into 768-d dense vectors.

    Slices input into chunks of batch_size and fires one HTTP request per
    chunk — the server never sees more than batch_size texts at once.
    Results are returned in input order.
    """
    if not texts:
        return []

    url = f"{EMBEDDING_SERVER_URL}/embed_dense"
    results: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        data = _post_with_500_retry(url, {
            "texts": texts[i : i + batch_size],
            "batch_size": batch_size,
        })
        results.extend(data["vectors"])
    return results


def get_sparse_vectors(
    texts: List[str],
    is_query: bool = False,
    batch_size: int = 256,
) -> List[Dict]:
    """Embed texts into sparse (SPLADE) vectors.

    Slices input into chunks of batch_size — one HTTP request per chunk.
    On a 500 (CUDA kernel timeout / TDR), waits for server recovery and
    retries only the failed chunk, not the whole list.
    Results are returned in input order.
    """
    if not texts:
        return []

    url = f"{EMBEDDING_SERVER_URL}/embed_sparse"
    results: List[Dict] = []
    for i in range(0, len(texts), batch_size):
        data = _post_with_500_retry(url, {
            "texts": texts[i : i + batch_size],
            "is_query": is_query,
        })
        results.extend(data["vectors"])
    return results


def rewrite_query(query: str) -> str:
    """Reformulate a user query into a precise technical search query."""
    url = f"{EMBEDDING_SERVER_URL}/rewrite"
    return _post_with_500_retry(url, {"query": query})["rewritten"]


def health_check() -> bool:
    """Check if the embedding server is healthy with both models loaded."""
    url = f"{EMBEDDING_SERVER_URL}/health"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("dense") is True and data.get("sparse") is True
    except Exception as e:
        logger.error("Embedding server health check failed: %s", e)
        return False