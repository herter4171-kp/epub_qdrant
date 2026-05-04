"""Unified embedding server — dense (Snowflake) + sparse (SAE-SPLADE) on one port."""

import argparse
import logging
import os
import sys
from typing import List, Optional

from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from servers.embedding_server.embedder import (
    DENSE_MODEL,
    BACKBONE_LOCAL_PATH,
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


# ── Model singletons ─────────────────────────────────────────────────

_dense: Optional[DenseEmbedder] = None
_sparse: Optional[SparseEmbedder] = None
_rewriter: Optional[QueryRewriter] = None
_rewriter_enabled: bool = True


def _load_models(skip_rewriter: bool = False):
    """Load all models into GPU memory.

    Args:
        skip_rewriter: If True, skip loading the IT model (saves ~2.6 GB VRAM).
    """
    global _dense, _sparse, _rewriter, _rewriter_enabled
    try:
        _dense = DenseEmbedder()
    except Exception:
        logger.exception("Failed to load dense model")
    try:
        _sparse = SparseEmbedder()
    except Exception:
        logger.exception("Failed to load sparse model")
    if skip_rewriter:
        logger.info("Skipping IT model load (--no-rewrite flag)")
        _rewriter = None
        _rewriter_enabled = False
    else:
        try:
            _rewriter = QueryRewriter()
        except Exception:
            logger.exception("Failed to load query rewriter")


# ── FastAPI app ───────────────────────────────────────────────────────

app = FastAPI(title="Unified Embedding Server", version="0.1.0")


@app.on_event("startup")
def startup():
    _load_models(skip_rewriter=not _rewriter_enabled)


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

    Returns 503 if the rewriter was disabled (--no-rewrite flag).
    """
    if not _rewriter_enabled:
        raise HTTPException(
            status_code=503,
            detail="Query rewriter is disabled. Start server without --no-rewrite to enable.",
        )
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
    return ModelsResponse(dense=DENSE_MODEL, sparse=BACKBONE_LOCAL_PATH)


@app.get("/rewrite/status", response_model=bool)
def rewrite_status():
    """Check whether the query rewriter is loaded and ready."""
    return _rewriter is not None


# ── Entry point ───────────────────────────────────────────────────────

def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Unified embedding server — dense (Snowflake) + sparse (SAE-SPLADE)",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Skip loading the IT model (saves ~2.6 GB VRAM, /rewrite endpoint disabled)",
    )
    return parser.parse_args()


def main():
    import uvicorn

    args = parse_args()
    global _rewriter_enabled
    _rewriter_enabled = not args.no_rewrite

    if args.no_rewrite:
        logger.info("IT model bypassed (--no-rewrite). /rewrite endpoint will return 503.")
    port = int(os.getenv("EMBEDDING_SERVER_PORT", "8100"))
    logger.info("Starting embedding server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
