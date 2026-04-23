# Unified Embedding Server

Single FastAPI server serving both dense and sparse embeddings on one port. Runs on a GPU box (RTX 5090).

## Models

| Type | Model | Output |
|------|-------|--------|
| Dense | embeddinggemma-300m (local) | 768-d float vectors |
| Sparse | Qdrant/minicoil-v1 (fastembed-gpu) | `{indices, values}` sparse vectors |

Both models load into VRAM at startup.

## Endpoints

```
POST /embed_dense   {"texts": [...], "batch_size": 128}  → {"vectors": [[...], ...]}
POST /embed_sparse  {"texts": [...], "is_query": false}   → {"vectors": [{"indices": [...], "values": [...]}, ...]}
GET  /health        → {"status": "ok", "dense": true, "sparse": true}
GET  /models        → {"dense": "Snowflake/...", "sparse": "Qdrant/..."}
```

Limits: max 1024 texts per request. Empty texts returns empty vectors.

## Setup

CUDA 12.8 is required (RTX 5090 / Blackwell). Install torch + xformers from the PyTorch cu128 index first — PyPI defaults to CUDA 13.x which won't work:

```bash
pip install torch==2.7.1 torchvision==0.22.1 xformers==0.0.31 \
    --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining deps:

```bash
pip install sentence-transformers fastembed-gpu fastapi "uvicorn[standard]"
```

## Running

```bash
# Default port 8100
uvicorn servers.embedding_server.server:app --host 0.0.0.0 --port 8100

# Or via env var
EMBEDDING_SERVER_PORT=8100 python -m servers.embedding_server.server
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_SERVER_PORT` | `8100` | Port the server listens on |
| `EMBEDDING_SERVER_URL` | `http://localhost:8100` | Used by the client library to reach the server |

## Client Library

`servers/embedding_server/client.py` — thin HTTP client for callers:

```python
from servers.embedding_server.client import get_dense_vectors, get_sparse_vectors, health_check

dense = get_dense_vectors(["hello world"])           # List[List[float]]
sparse = get_sparse_vectors(["hello"], is_query=True) # List[Dict]
ok = health_check()                                   # bool
```

Reads `EMBEDDING_SERVER_URL` from env. Raises on connection/timeout errors.
