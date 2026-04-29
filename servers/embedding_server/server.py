"""Unified embedding server — dense (Snowflake) + sparse (SAE-SPLADE) on one port."""

import logging
import os
import time
from typing import List, Optional, Dict

from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from servers.embedding_server.embedder import (
    DENSE_MODEL,
    SPLADE_LOCAL_PATH,
    DenseEmbedder,
    QueryRewriter,
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


MAX_TEXTS = 1024

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

class RewriteRequest(BaseModel):
    query: str

class RewriteResponse(BaseModel):
    rewritten: str

class ProfileDenseRequest(BaseModel):
    count: int = 128

class ProfileDenseResponse(BaseModel):
    elapsed_ms: float
    vram_alloc_before: float
    vram_alloc_after: float
    vram_delta: float

class ProfileSparseRequest(BaseModel):
    count: int = 32
    is_query: bool = False

class ProfileSparseResponse(BaseModel):
    elapsed_ms: float
    vram_alloc_before: float
    vram_alloc_after: float
    vram_delta: float

# ── Model singletons ─────────────────────────────────────────────────

_dense: Optional[DenseEmbedder] = None
_sparse: Optional[SparseEmbedder] = None
_rewriter: Optional[QueryRewriter] = None


def _load_models():
    """Load all models into GPU memory."""
    global _dense, _sparse, _rewriter
    try:
        _dense = DenseEmbedder()
    except Exception:
        logger.exception("Failed to load dense model")
    try:
        _sparse = SparseEmbedder()
    except Exception:
        logger.exception("Failed to load sparse model")
    try:
        _rewriter = QueryRewriter()
    except Exception:
        logger.exception("Failed to load query rewriter")


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
async def embed_sparse(req: SparseEmbedRequest):
    if len(req.texts) > MAX_TEXTS:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_TEXTS} texts per request")
    if not req.texts:
        return SparseEmbedResponse(vectors=[])
    if _sparse is None:
        raise HTTPException(status_code=503, detail="Sparse model not loaded")
    raw = await _sparse.encode_async(req.texts, is_query=req.is_query)
    vectors = [SparseVector(**v) for v in raw]
    return SparseEmbedResponse(vectors=vectors)


@app.post("/rewrite", response_model=RewriteResponse)
async def rewrite_query(req: RewriteRequest):
    """Reformulate a user query into a precise technical search query.

    Uses the IT model to rewrite natural-language prompts into queries
    optimized for retrieving AI/ML research papers from the vector store.
    """
    if _rewriter is None:
        raise HTTPException(status_code=503, detail="Query rewriter not loaded")
    rewritten = await _rewriter.rewrite_async(req.query)
    return RewriteResponse(rewritten=rewritten)


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
    from servers.embedding_server.embedder import IT_MODEL_LOCAL_PATH
    return ModelsResponse(dense=DENSE_MODEL, sparse=SPLADE_LOCAL_PATH)


@app.get("/rewrite/status", response_model=bool)
def rewrite_status():
    """Check whether the query rewriter is loaded and ready."""
    return _rewriter is not None


@app.post("/profile/dense", response_model=ProfileDenseResponse)
def profile_dense(req: ProfileDenseRequest):
    """Measure VRAM delta for dense embedding of `count` texts. Runs in server process."""
    global _dense
    if _dense is None:
        raise HTTPException(status_code=503, detail="Dense model not loaded")
    import torch
    before = torch.cuda.memory_allocated(device="cuda")
    t0 = time.time()
    # Generate dummy texts to match real workload length (~150 words each)
    texts = [f"token{i % 100} word {(i+j) % 200}" for i in range(req.count) for j in range(150)]
    _dense.encode(texts)
    delta = time.time() - t0
    after = torch.cuda.memory_allocated(device="cuda")
    return ProfileDenseResponse(
        elapsed_ms=delta * 1000,
        vram_alloc_before=before / 1e9,
        vram_alloc_after=after / 1e9,
        vram_delta=(after - before) / 1e9,
    )


@app.post("/profile/sparse", response_model=ProfileSparseResponse)
def profile_sparse(req: ProfileSparseRequest):
    """Measure VRAM delta for sparse embedding of `count` texts. Runs in server process."""
    global _sparse
    if _sparse is None:
        raise HTTPException(status_code=503, detail="Sparse model not loaded")
    import torch
    import asyncio
    before = torch.cuda.memory_allocated(device="cuda")
    t0 = time.time()
    texts = [f"token{i % 100} word {(i+j) % 200}" for i in range(req.count) for j in range(150)]
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, lambda: _sparse.encode(texts, is_query=req.is_query))
    delta = time.time() - t0
    after = torch.cuda.memory_allocated(device="cuda")
    return ProfileSparseResponse(
        elapsed_ms=delta * 1000,
        vram_alloc_before=before / 1e9,
        vram_alloc_after=after / 1e9,
        vram_delta=(after - before) / 1e9,
    )


# ── Entry point ───────────────────────────────────────────────────────

def main():
    import uvicorn

    port = int(os.getenv("EMBEDDING_SERVER_PORT", "8100"))
    logger.info("Starting embedding server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
