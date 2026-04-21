# EPUB-to-Qdrant Ingestion Pipeline

Standalone pipeline that reads EPUB files, generates embeddings via Ollama, and stores vectors in a **shared single Qdrant collection** for semantic search and knowledge-base retrieval.

## Setup

```bash
# Install dependencies
pip install -e .

# Configure (optional)
cp .env.example .env
# Edit .env with your Qdrant/Ollama addresses
```

## Usage

### Ingest EPUBs

```bash
# Ingest all EPUBs from a directory into shared collection
python -m src.main ingest ./my_books

# With progress output
python -m src.main ingest /path/to/epubs --limit 5

# Or via the entry point script
epub_qdrant ingest ./my_books
```

### Search (CLI)

```bash
# Search the shared collection
python -m src.main search <collection-name> "your query here"

# Limit results
python -m src.main search mylibrary "how to use decorators" --top-k 5
```

### List Collections

```bash
python -m src.main list-collections
```

### Delete a Collection

```bash
python -m src.main delete-collection <collection-name>
```

## MCP Retrieval Server

A standalone MCP server provides knowledge-base tools over Streamable HTTP for use by n8n, LLM clients, or any MCP-compatible tool.

### Tools

| Tool | Description | Args | Returns |
|------|-------------|------|---------|
| `search` | Semantic search → grouped chunks | `query`, `top_k` | Grouped evidence with scores |
| `answer` | Search + LLM answer | `query`, `top_k`, `group_by` | Streaming LLM answer |
| `get_context` | Surrounding chunks around a section | `source_file`, `section_title`, `radius` | Context window of chunks |
| `list_collections` | Read-only: list available collections | (none) | Collection names |

### Running

```bash
cd mcp_servers/retrieval
pip install -e .
export QDRANT_COLLECTION=mylibrary
export LITELLM_API_KEY=your-key
uv mcp_server
# or
python -m mcp_server
# Server listens on :8090
```

### Client Example

```
POST http://localhost:8090/mcp
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "answer",
    "arguments": { "query": "what is quantum entanglement" }
  }
}
```

## Configuration

### Ingestion (.env)

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://192.168.68.75:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | (none) | **Shared collection name** for all EPUBs |
| `OLLAMA_URL` | `http://192.168.68.75:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `embeddinggemma:300m` | Ollama embedding model name |
| `CHUNK_SIZE` | `500` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Token overlap between chunks |
| `VECTOR_SIZE` | `768` | Embedding vector dimensions |
| `DISTANCE` | `Cosine` | Vector distance metric |

### Retrieval MCP Server (.env)

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | — | **Required** — single collection name |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama embedding endpoint |
| `EMBEDDING_MODEL` | `embeddinggemma:300m` | Embedding model name |
| `LITELLM_API_URL` | `https://litellm.twr.church/v1` | LiteLLM ChatCompletion endpoint |
| `LITELLM_API_KEY` | — | **Required** — LiteLLM API key |
| `LITELLM_MODEL` | `meta-llama/llama-3.1-70b-instruct` | LLM model for answers |
| `MCP_PORT` | `8090` | HTTP listen port |
| `RETRIEVAL_TOP_K` | `20` | Default top-k for retrieval |
| `RETRIEVAL_CONTEXT_RADIUS` | `2` | Surrounding chunks per side |
| `RETRIEVAL_GROUP_BY` | `chapter` | Group results by `chapter` or `book` |

## Architecture

```
src/
  __init__.py     # Package marker
  config.py       # Settings from env vars
  epub_parser.py  # EPUB text extraction with heading detection
  chunker.py      # Paragraph-aware chunking with overlap
  embedder.py     # Ollama embedding API calls
  storage.py      # Shared Qdrant collection upsert + search
  main.py         # CLI entry point (ingest/search)

mcp_servers/retrieval/
  mcp_server/
    server.py     # MCP server (FastAPI + Streamable HTTP)
    retriever.py  # Retrieval layer: search → group → evidence
    llm_client.py # LiteLLM streaming client
    config.py     # MCP server settings
```

### Ingestion Flow

1. **Parse**: `epub_parser.py` reads EPUB structure, extracts text by chapter/section
2. **Chunk**: `chunker.py` splits text into ~500 token chunks with 100 token overlap, respecting paragraph boundaries
3. **Embed**: `embedder.py` calls Ollama `/api/embed` for each chunk
4. **Store**: `storage.py` upserts vectors + metadata into **one shared Qdrant collection**

All EPUBs go into a **single shared Qdrant collection** (named by `QDRANT_COLLECTION`). Each chunk carries metadata: `source_file`, `book_title`, `section_title`, `chapter_index`, `section_index`, `chunk_index`, `token_count`.

### Retrieval Flow

1. **Query**: client sends natural-language question via MCP or CLI
2. **Embed**: query text embedded via Ollama
3. **Search**: Qdrant returns top-k chunk matches by cosine similarity
4. **Expand**: surrounding chunks from same chapters added for context
5. **Group**: results grouped by chapter or book
6. **Assemble**: evidence bundle formatted as prompt context
7. **Answer**: LiteLLM streams final answer grounded in retrieved evidence

This makes it a knowledge-base retrieval system, not just a vector store dump.