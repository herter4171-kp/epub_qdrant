# Architecture Mapping — All Servers, All IPs

## The GPU Box (192.168.68.75)

This is THE machine. Everything important runs here.

| Service | Port | What It Does |
|---------|------|-------------|
| **Qdrant** | 6333 | Vector DB. Collections: papers, papers-semantic, papers-2048ctx-SAE, books, books-semantic, books-hybrid, papers-hybrid, etc. |
| **Embedding Server** | 9000 | FastAPI server. Dense (Snowflake via sentence-transformers) + Sparse (SAE-SPLADE via Gemma 3 270M + JumpReLU SAE). Served via `servers/embedding_server/server.py` |
| **Ollama** | 11434 | NOT used for embeddings. (Was previously used for embeddinggemma:300m but the pipeline now uses the unified embedding server) |
| **GPU** | RTX 5090 | VRAM shared between dense model, SAE backbone, SAE weights, IT model |

## The Mac (Dev Machine)

This is the development machine. No servers run here. All tool calls go through MCP.

| Service | Port/Path | Notes |
|---------|-----------|-------|
| **Qdrant client** | connects to 192.168.68.75:6333 | via qdrant-client Python library |
| **Embedding server client** | connects to 192.168.68.75:9000 | via `servers/embedding_server/client.py` — `get_dense_vectors()`, `get_sparse_vectors()` |
| **agent-lookup MCP** | localhost MCP server | Proxy/adapter that connects to Qdrant at 192.168.68.75:6333. This is how we query collections from this conversation. |

## Key Files

### Embedding Server (on GPU box)
- `servers/embedding_server/embedder.py` — DenseEmbedder + SparseEmbedder (SAE-SPLADE) + QueryRewriter (IT model)
- `servers/embedding_server/server.py` — FastAPI app with `/embed` (dense) and `/embed_sparse` (sparse) endpoints
- `servers/embedding_server/client.py` — Client functions: `get_dense_vectors()`, `get_sparse_vectors()`, `health_check()`, `rewrite_query()`

### Retriever (on GPU box, served via MCP)
- `servers/mcp_server/retriever.py` — `hybrid_search()`, `search_collections()`, `_embed_sparse()`, `_has_sparse_vectors()`

### Indexing Scripts (run from Mac, talk to GPU box)
- `scripts/ingest_fresh.py` — full pipeline: create collection → dense embed → sparse embed → upsert
- `scripts/embed_sparse_vectors.py` — two-pass migration to `-hybrid` collections
- `scripts/ingest_mineru_json.py` — MinerU JSON ingestion

## Client Configuration

All client code points to 192.168.68.75. No localhost references should exist.

```python
# Qdrant
QDRANT_URL = "http://192.168.68.75:6333"

# Embedding server
EMBEDDING_SERVER_URL = "http://192.168.68.75:9000"
```

## Collections Summary

| Collection | Points | Vectors | Backend |
|-----------|--------|---------|---------|
| `papers` | ~90K | dense only (768-d) | Original indexing |
| `papers-semantic` | ~117K | dense only (768-d) | Semantic chunking |
| `papers-2048ctx-SAE` | ~78K | dense + sparse (SAE-SPLADE) | 2048-ctx windowing + SAE sparse |
| `books` | ~6K | dense only | Original |
| `books-semantic` | ~9K | dense only | Semantic chunking |
| `books-hybrid` | ~9K | dense + sparse (MiniCOIL) | Two-pass hybrid |
| `papers-hybrid` | ~?K | dense + sparse (MiniCOIL) | Two-pass hybrid |

## Model Paths (GPU box only)

| Model | Path | Purpose |
|-------|------|---------|
| Dense embedding | `/tank/huggingface/embeddinggemma-300m` | SentenceTransformer dense vectors |
| SAE backbone | `/tank/huggingface/gemma-3-270m` | Gemma 3 270M PT — hidden states |
| SAE weights | `/tank/huggingface/gemma-scope-2-270m-pt/resid_post/layer_12_width_65k_l0_medium` | JumpReLU SAE — concept projection |
| IT model | `/tank/huggingface/gemma-3-270m-it` | Query rewriting via instruct-tuned Gemma |

## What Went Wrong in This Conversation

1. I ran `curl localhost:11434` when Ollama isn't relevant to this pipeline
2. I ran `curl localhost:6333` when Qdrant is on 192.168.68.75
3. I ran `curl localhost:9000` when the embedding server is on 192.168.68.75
4. I should have used `agent-lookup MCP` to query collections, which I did correctly
5. The agent-lookup query returned garbage for both `papers-semantic` and `papers-2048ctx-SAE` — this is the real diagnostic result

## NEVER USE LOCALHOST — RULES

1. Qdrant is on 192.168.68.75:6333 — NEVER localhost:6333
2. Embedding server is on 192.168.68.75:9000 — NEVER localhost:9000
3. Ollama is NOT part of this pipeline — ignore port 11434
4. All curl commands MUST use 192.168.68.75 — never localhost
5. Use agent-lookup MCP to query collections — it proxies to 192.168.68.75

## Next Diagnostic Steps

To actually test the embedding pipeline, use the agent-lookup MCP with queries, or run:
```bash
curl http://192.168.68.75:6333/health
curl http://192.168.68.75:9000/health
