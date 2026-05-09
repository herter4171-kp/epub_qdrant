"""Embedding helpers — dense + sparse vectors."""

import logging
from typing import Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_RETRY = Retry(total=None, connect=None, read=None, backoff_factor=2,
               status_forcelist=[502, 503, 504], allowed_methods=["POST", "GET"])


def _session(url: str) -> requests.Session:
    s = requests.Session()
    a = HTTPAdapter(max_retries=_RETRY)
    s.mount("http://", a)
    s.mount("https://", a)
    return s


def dense_embed(url: str, texts: List[str], batch_size: int = 128) -> List[List[float]]:
    """Embed texts → 768-d dense vectors. Batched."""
    if not texts:
        return []
    sess = _session(url)
    results: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = sess.post(f"{url}/embed_dense", json={"texts": chunk}, timeout=(10, 300))
        resp.raise_for_status()
        results.extend(resp.json()["vectors"])
    return results


def sparse_embed(url: str, texts: List[str], batch_size: int = 256) -> List[Dict]:
    """Embed texts → sparse vectors (indices, values). Batched."""
    if not texts:
        return []
    sess = _session(url)
    results: List[Dict] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = sess.post(f"{url}/embed_sparse", json={"texts": chunk, "is_query": True},
                         timeout=(10, 300))
        resp.raise_for_status()
        results.extend(resp.json()["vectors"])
    return results


def sparse_embed_sae(url: str, texts: List[str], batch_size: int = 256) -> List[Dict]:
    """Embed texts → SAE sparse vectors (indices, values). Batched.

    Calls POST /embed_sae which runs SPLADE → SAE encoder → topk.
    Output: 61044-dim sparse vectors with 165 non-zeros.
    """
    if not texts:
        return []
    sess = _session(url)
    results: List[Dict] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = sess.post(f"{url}/embed_sae", json={"texts": chunk, "is_query": True},
                         timeout=(10, 300))
        resp.raise_for_status()
        data = resp.json()["vectors"]
        # Convert SparseVector models to dicts
        for v in data:
            if hasattr(v, "model_dump"):
                results.append(v.model_dump())
            elif hasattr(v, "dict"):
                results.append(v.dict())
            else:
                results.append(dict(v))
    return results
