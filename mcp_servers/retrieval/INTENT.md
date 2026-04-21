# Intent: EPUB Knowledge Base Retrieval MCP Server

## Purpose

A standalone MCP (Model Context Protocol) server that exposes retrieval tools over Streamable HTTP for a Qdrant-backed knowledge base of EPUB books.

This server is retrieval-only. Ingestion remains a separate CLI pipeline (`epub_qdrant ingest <directory>`). This server reads from the indexed collection and answers questions using retrieved evidence.

## Design Goals

1. **Retrieval-only**: No ingestion, no file writing, no collection management. Only search and answer.
2. **Single collection isolation**: One collection, locked at startup via `QDRANT_COLLECTION`. No runtime switching. Safe isolation — one server instance = one knowledge base.
3. **Streamable HTTP transport**: MCP over Streamable HTTP per the 2025-03-26 spec. Long-running HTTP service.
4. **n8n compatible**: Standard MCP tools with JSON-compatible I/O. Works with n8n's MCP node.
5. **Streaming LLM output**: The `answer` tool streams the LLM response incrementally via MCP content deltas.

## Architecture

```
n8n / MCP Client
     ↓  Streamable HTTP (POST, JSON-RPC 2.0)
epub_qdrant Retrieval MCP Server (:8090)
     ↓
Qdrant — vector search (top-k chunks)
     ↓
Retriever — group by chapter/book, assemble context
     ↓
LiteLLM — streaming ChatCompletion → LLM answer
```

## Tools

| Tool | Description | Args | Returns |
|------|-------------|------|---------|
| `search` | Semantic search → grouped chunks | `query`, `top_k` | Grouped evidence with scores |
| `answer` | Search + LLM answer | `query`, `top_k`, `group_by` | Streaming LLM answer |
| `get_context` | Surrounding chunks around a section | `source_file`, `section_title`, `radius` | Context window of chunks |
| `list_collections` | Read-only: list available collections | (none) | Collection names |

## Configuration

| Env Var | Default | Required | Description |
|---------|---------|----------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | No | Qdrant endpoint |
| `QDRANT_COLLECTION` | — | **Yes** | Single collection name |
| `OLLAMA_URL` | `http://localhost:11434` | No | Ollama embedding endpoint |
| `EMBEDDING_MODEL` | `embeddinggemma:300m` | No | Embedding model name |
| `LITELLM_API_URL` | `https://litellm.twr.church/v1` | No | LiteLLM ChatCompletion endpoint |
| `LITELLM_API_KEY` | — | **Yes** | LiteLLM API key |
| `LITELLM_MODEL` | `meta-llama/llama-3.1-70b-instruct` | No | LLM model for answers |
| `MCP_PORT` | `8090` | No | HTTP listen port |
| `RETRIEVAL_TOP_K` | `20` | No | Default top-k for retrieval |
| `RETRIEVAL_CONTEXT_RADIUS` | `2` | No | Surrounding chunks per side |
| `RETRIEVAL_GROUP_BY` | `chapter` | No | Group results by `chapter` or `book` |

## Dependencies

Shared from the parent `epub_qdrant` project:
- `src.storage` — Qdrant client, search
- `src.embedder` — Ollama embedding generation
- `src.chunker` — Chunk dataclass
- `src.config` — shared settings

Local to this package:
- `mcp` — Python MCP SDK
- `httpx-sse` — SSE client for MCP streaming
- `litellm` — LiteLLM streaming client
- `fastapi` + `uvicorn` — HTTP server

## Running

```bash
cd mcp_servers/retrieval
pip install -e .
export QDRANT_COLLECTION=books
export LITELLM_API_KEY=your-key
uv mcp_server
# or
python -m mcp_server
# Server listens on :8090
```

## Client Endpoint

```
POST http://localhost:8090/mcp
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": { "query": "what is quantum entanglement", "top_k": 10 }
  }
}