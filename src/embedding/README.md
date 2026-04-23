# src/embedding — Embedding Infrastructure

Embedding logic has been consolidated into the **unified embedding server** at `servers/embedding_server/`.

## Current Architecture

All dense and sparse embedding is handled by a single FastAPI server running on the GPU box:

| Component | Location | Purpose |
|-----------|----------|---------|
| Embedding Server | `servers/embedding_server/server.py` | FastAPI app: `/embed_dense`, `/embed_sparse`, `/health`, `/models` |
| Embedding Client | `servers/embedding_server/client.py` | Thin HTTP client: `get_dense_vectors()`, `get_sparse_vectors()`, `health_check()` |
| Dense Model | embeddinggemma-300m (GPU) | 768-d dense vectors |
| Sparse Model | Qdrant/minicoil-v1 via fastembed-gpu | `{indices, values}` sparse vectors |

The client reads `EMBEDDING_SERVER_URL` from the environment (default `http://localhost:8100`).

## Legacy Files

| File | Status |
|------|--------|
| `minicoil_server.py` | Superseded by the unified embedding server's `/embed_sparse` endpoint |

## Usage

```python
# Dense embedding
from servers.embedding_server.client import get_dense_vectors
vecs = get_dense_vectors(["hello world"])  # List[List[float]], 768-d

# Sparse embedding
from servers.embedding_server.client import get_sparse_vectors
sparse = get_sparse_vectors(["hello world"], is_query=False)  # List[Dict]

# Health check
from servers.embedding_server.client import health_check
ok = health_check()  # True if both models loaded
```
