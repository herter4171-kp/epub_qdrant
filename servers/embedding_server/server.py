"""Unified embedding server — dense (Snowflake) + sparse (MiniCOIL) on one port."""

import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from servers.embedding_server.embedder import (
    DENSE_MODEL,
    SPARSE_MODEL,
    DenseEmbedder,
    SparseEmbedder,
)

logger = logging.getLogger(__name__)

# ── Pydantic schemas ─────────────────────────────────────────────────

class DenseEmbedRequest(BaseModel):
    texts: List[str]
    batch_size: int = 128

class DenseEmbedResponse(BaseModel):
    vectors: List[List[float]]

class SparseEmbedRequest(BaseModel):
    texts: List[str]
    is_query: bool = False

class SparseVector(BaseModel):
    indices: List[int]
    values: List[float]

class SparseEmbedResponse(BaseModel):
    vectors: List[SparseVector]

class HealthResponse(BaseModel):
    status: str
    dense: bool
    sparse: bool

class ModelsResponse(BaseModel):
    dense: str
    sparse: str


MAX_TEXTS = 1024

# ── Model singletons ─────────────────────────────────────────────────

_dense: Optional[DenseEmbedder] = None
_sparse: Optional[SparseEmbedder] = None


def _load_models():
    """Load both models into GPU memory."""
    global _dense, _sparse
    try:
        _dense = DenseEmbedder()
    except Exception:
        logger.exception("Failed to load dense model")
    try:
        _sparse = SparseEmbedder()
    except Exception:
        logger.exception("Failed to load sparse model")


# ── FastAPI app ───────────────────────────────────────────────────────

app = FastAPI(title="Unified Embedding Server", version="0.1.0")


@app.on_event("startup")
def startup():
    _load_models()


@app.post("/embed_dense", response_model=DenseEmbedResponse)
def embed_dense(req: DenseEmbedRequest):
    if len(req.texts) > MAX_TEXTS:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_TEXTS} texts per request")
    if not req.texts:
        return DenseEmbedResponse(vectors=[])
    if _dense is None:
        raise HTTPException(status_code=503, detail="Dense model not loaded")
    vectors = _dense.encode(req.texts, batch_size=req.batch_size)
    return DenseEmbedResponse(vectors=vectors)


@app.post("/embed_sparse", response_model=SparseEmbedResponse)
def embed_sparse(req: SparseEmbedRequest):
    if len(req.texts) > MAX_TEXTS:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_TEXTS} texts per request")
    if not req.texts:
        return SparseEmbedResponse(vectors=[])
    if _sparse is None:
        raise HTTPException(status_code=503, detail="Sparse model not loaded")
    raw = _sparse.encode(req.texts, is_query=req.is_query)
    vectors = [SparseVector(**v) for v in raw]
    return SparseEmbedResponse(vectors=vectors)


@app.get("/health", response_model=HealthResponse)
def health():
    ok = _dense is not None and _sparse is not None
    return HealthResponse(
        status="ok" if ok else "error",
        dense=_dense is not None,
        sparse=_sparse is not None,
    )


@app.get("/models", response_model=ModelsResponse)
def models():
    return ModelsResponse(dense=DENSE_MODEL, sparse=SPARSE_MODEL)


# ── Entry point ───────────────────────────────────────────────────────

def main():
    import uvicorn

    port = int(os.getenv("EMBEDDING_SERVER_PORT", "8100"))
    logger.info("Starting embedding server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
