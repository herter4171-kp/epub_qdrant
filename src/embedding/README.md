# src/embedding — Embedding Infrastructure

All embedding logic for dense (semantic) and sparse (keyword) vectors.

## Components

| File | Purpose |
|------|---------|
| `dense_embedder.py` | Dense embedding via Ollama API (placeholder for sentence_transformers migration) |
| `minicoil_server.py` | MiniCOIL sparse embedding HTTP server — runs on GPU box, served via `uvicorn` |
| `client.py` | HTTP client for calling MiniCOIL server to get sparse vectors |

## MiniCOIL Server (GPU Box)

The MiniCOIL server is NOT an MCP server — it's an embedding service that runs on the GPU machine.

### Launch on remote GPU box (192.168.68.75):

```bash
pip install fastembed-gpu fastapi uvicorn
uvicorn src.embedding.minicoil_server:app --host 0.0.0.0 --port 9000
```

The model (`Qdrant/minicoil-v1`, ~15MB) downloads automatically on first startup.

### Health check:
```bash
curl http://192.168.68.75:9000/health
# → {"status":"ok","ready":true}
```

## Usage

```python
# Dense embedding
from src.embedding.dense_embedder import Embedder
embedder = Embedder(ollama_url, model_name)
vec = embedder.embed_single("hello world")

# Sparse embedding (via HTTP to GPU box)
from src.embedding.client import get_sparse_vectors
sparse = get_sparse_vectors(["hello world"], is_query=False)