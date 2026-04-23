"""MiniCOIL sparse embedding server.

GPU Linux box: pip install fastembed-gpu fastapi uvicorn
Run: uvicorn server:app --host 0.0.0.0 --port 9000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastembed import SparseTextEmbedding

app = FastAPI(title="MiniCOIL Embedding Server")
model = None

# TODO: CUDA WITHOUT BLOWING UP VRAM
#import os
#os.environ["ONNXRUNTIME_CUDA_MEMORY_LIMIT"] = "30064771072" 

@app.on_event("startup")
async def startup():
    global model
    model = SparseTextEmbedding(model_name="Qdrant/minicoil-v1", providers=["CUDAExecutionProvider"])

class EmbedRequest(BaseModel):
    texts: List[str]
    is_query: bool = False  # True for query-time, False for document indexing

class SparseVector(BaseModel):
    indices: List[int]
    values: List[float]

class EmbedResponse(BaseModel):
    vectors: List[SparseVector]

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if not request.texts:
        return EmbedResponse(vectors=[])
    if len(request.texts) > 64:
        raise HTTPException(400, "Max 64 texts per request")
    
    if request.is_query:
        embeddings = list(model.query_embed(request.texts))
    else:
        embeddings = list(model.embed(request.texts))
    
    return EmbedResponse(vectors=[
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in embeddings
    ])

@app.get("/health")
async def health():
    return {"status": "ok", "ready": model is not None}