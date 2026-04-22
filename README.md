# EPUB & Paper-to-Qdrant Ingestion Pipeline

Standalone pipeline that reads EPUB files and PDF papers, generates embeddings via Ollama, and stores vectors in **Qdrant collections** for semantic search and knowledge-base retrieval.

## Table of Contents

- [Setup](#setup)
- [EPUB Ingestion](#epub-ingestion)
- [Paper Embedding](#paper-embedding)
- [Search (CLI)](#search-cli)
- [List Collections](#list-collections)
- [List Books](#list-books)
- [Delete a Collection](#delete-a-collection)
- [MCP Retrieval Server](#mcp-retrieval-server)
- [Configuration](#configuration)
- [Architecture](#architecture)

## Setup

```bash
# Install dependencies
pip install -e .

# Configure (optional)
cp .env.example .env
# Edit .env with your Qdrant/Ollama addresses
```

## EPUB Ingestion

```bash
# Ingest all EPUBs from a directory into the configured collection
python -m src.main ingest ./my_books

# With progress output and limit
python -m src.main ingest /path/to/epubs --limit 5

# Or via the entry point script
epub_qdrant ingest ./my_books

# Specify a custom collection name
epub_qdrant ingest ./my_books --collection epub_kb
```

## Paper Embedding

Download and embed academic papers from `ai-agent-papers/` into a Qdrant papers collection.

```bash
# Embed the first PDF found in ./downloads/
python scripts/embed_papers_to_qdrant.py

# Embed ALL PDFs in ./downloads/
PAPER_EMBED_ALL=1 python scripts/embed_papers_to_qdrant.py

# Embed with verbose output
python scripts/embed_papers.py 2>&1 | tee paper_embed.log
```

### Paper Directory Structure

Papers should be downloaded to `./downloads/` preserving the structure from `ai-agent-papers/`:

```
downloads/
  ├── 2010.03768.pdf          # PDF file
  ├── 2010.03768.json         # Metadata JSON (optional)
  └── ...
```

PDF filenames should match their JSON metadata filenames (stem must match). The JSON metadata uses the format:

```json
{
  "metadataAttributes": [
    "title: Agent Memory Survey",
    "authors: Smith, Jones",
    "arxiv_id: 2010.03768",
    "category: capability-papers",
    "subcategory: memory",
    "publish_date: 2020-10-06",
    "abstract: ..."
  ]
}
```

### Search (CLI)

```bash
# Search the default collection
python -m src.main search "your query here"

# Search a specific collection
python -m src.main search papers "transformer attention"

# Limit results
python -m src.main search epub_kb "how to use decorators" --top-k 5
```

### List Collections

```bash
python -m src.main list-collections
```

### List Books

```bash
# List all books in the default collection
python -m src.main list-books

# List books in a specific collection
python -m src.main list-books --collection epub_kb
```

### Delete a Collection

```bash
python -m src.main delete-collection <collection-name>
```

## MCP Retrieval Server

A standalone MCP server provides knowledge-base tools over Streamable HTTP for use by n8n, LLM clients, or any MCP-compatible tool. It supports **multi-collection search** across all configured Qdrant collections.

### Tools

| Tool | Description | Args | Returns |
|------|-------------|------|---------|
| `search` | Semantic search across collections → grouped chunks | `query`, `top_k`, `group_by`, `collection`, `collections`, `filter_by` | Grouped evidence with scores |
| `answer` | Search + LLM answer | `query`, `top_k`, `group_by`, `collection`, `collections`, `filter_by` | Streaming LLM answer |
| `get_context` | Surrounding chunks around a section | `source_file`, `section_title`, `radius`, `collection` | Context window of chunks |
| `list_collections` | Read-only: list all Qdrant collections | (none) | Collection names, point counts, vector config |

### Running

```bash
cd mcp_servers/retrieval
pip install -e .

# Multi-collection mode (recommended)
export QDRANT_COLLECTIONS=epub_kb,papers
export LITELLM_API_KEY=your-key

# Single-collection fallback
export QDRANT_COLLECTION=mylibrary

uv mcp_server
# or
python -m mcp_server.server
# Server listens on :8090
```

### Multi-Collection Search

When `QDRANT_COLLECTIONS=books,papers` is set, `search` and `answer` search **all** collections by default. To target a single collection:

```json
{
  "name": "search",
  "arguments": {
    "query": "transformer attention",
    "collection": "papers"
  }
}
```

To search a specific subset of collections:

```json
{
  "name": "search",
  "arguments": {
    "query": "agent memory",
    "collections": "papers,books"
  }
}
```

With metadata pre-filtering:

```json
{
  "name": "search",
  "arguments": {
    "query": "reasoning",
    "filter_by": "{\"doc_type\": \"paper\"}"
  }
}
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

### Multi-Collection Overview

The system supports **multiple Qdrant collections** (e.g., `epub_kb` for books, `papers` for academic papers). Configure collections using the comma-separated `QDRANT_COLLECTIONS` variable. When not set, the system falls back to legacy single-collection mode.

| Variable | Default | Description |
|---|---|---|
| `QDRANT_COLLECTIONS` | — | **Multi-collection** — comma-separated list (e.g., `epub_kb,papers`). First entry is the default. |
| `QDRANT_COLLECTION` | `books` | **Legacy fallback** — single collection name for EPUBs |
| `QDRANT_PAPERS_COLLECTION` | `papers` | Collection name for PDF papers |

### Ingestion (.env)

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://192.168.68.75:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | (none) | Collection name for all EPUBs (legacy single-col mode) |
| `QDRANT_PAPERS_COLLECTION` | `papers` | Collection name for PDF papers |
| `OLLAMA_URL` | `http://192.168.68.75:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `embeddinggemma:300m` | Ollama embedding model name |
| `CHUNK_SIZE` | `500` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Token overlap between chunks |
| `VECTOR_SIZE` | `768` | Embedding vector dimensions |
| `DISTANCE` | `Cosine` | Vector distance metric |

### Paper Embedding (.env)

| Variable | Default | Description |
|---|---|---|
| `QDRANT_PAPERS_COLLECTION` | `papers` | Qdrant collection for paper chunks |
| `PAPER_EMBED_ALL` | `0` | Set to `1` to embed all PDFs (default: first only) |

### Retrieval MCP Server (.env)

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTIONS` | — | **Required** — comma-separated collection names (e.g., `epub_kb,papers`) |
| `QDRANT_COLLECTION` | `books` | Legacy fallback if `QDRANT_COLLECTIONS` not set |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama embedding endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `LITELLM_API_URL` | `https://litellm.twr.church/v1` | LiteLLM ChatCompletion endpoint |
| `LITELLM_API_KEY` | — | **Required** for `answer` tool — LiteLLM API key |
| `LITELLM_MODEL` | `qwen36` | LLM model for answers (via llama.cpp through LiteLLM) |
| `MCP_PORT` | `8090` | HTTP listen port |
| `MCP_HOST` | `0.0.0.0` | HTTP listen host |
| `RETRIEVAL_TOP_K` | `15` | Default top-k per collection |
| `RETRIEVAL_CONTEXT_RADIUS` | `2` | Surrounding chunks per side |
| `RETRIEVAL_GROUP_BY` | `section` | Group results by `section` or `book` |

## Architecture

```
ai-agent-papers/          # Source papers (markdown index)
downloads/                # Downloaded PDFs + JSON metadata

src/
  __init__.py             # Package marker
  config.py               # Settings from env vars (multi-collection support)
  epub_parser.py          # EPUB text extraction with heading detection
  paper_chunker.py        # PDF paper text extraction and chunking
  chunker.py              # Paragraph-aware chunking with overlap
  embedder.py             # Ollama embedding API calls
  storage.py              # Qdrant collection upsert + search (EPUB + papers)
  main.py                 # CLI entry point (ingest/search/list-books)

scripts/
  embed_papers.py         # Paper download + metadata extraction
  embed_papers_to_qdrant.py  # PDF embedding pipeline (downloads → Qdrant)

mcp_servers/retrieval/
  mcp_server/
    server.py             # MCP server (FastAPI + Streamable HTTP, v0.2.0)
    retriever.py          # Retrieval layer: search → group → evidence
    llm_client.py         # LiteLLM streaming client
    config.py             # MCP server settings
```

### EPUB Ingestion Flow

1. **Parse**: `epub_parser.py` reads EPUB structure, extracts text by chapter/section
2. **Chunk**: `chunker.py` splits text into ~500 token chunks with 100 token overlap, respecting paragraph boundaries
3. **Embed**: `embedder.py` calls Ollama `/api/embed` for each chunk
4. **Store**: `storage.py` upserts vectors + metadata into **Qdrant collection** (named by `QDRANT_COLLECTION` or `QDRANT_COLLECTIONS[0]`)

Each EPUB chunk carries metadata: `source_file`, `book_title`, `section_title`, `chapter_index`, `section_index`, `chunk_index`, `token_count`.

### Paper Embedding Flow

1. **Download**: `scripts/embed_papers.py` downloads PDFs from arxiv and JSON metadata from `ai-agent-papers/`
2. **Extract**: `pypdf` extracts text from each PDF
3. **Chunk**: `paper_chunker.py` splits paper text into sections with metadata
4. **Embed**: `embedder.py` calls Ollama for embeddings
5. **Store**: `storage.py.upsert_paper_file()` upserts into `QDRANT_PAPERS_COLLECTION`

Each paper chunk carries metadata: `arxiv_id`, `title`, `category`, `subcategory`, `authors`, `publish_date`, `chunk_index`, `chunk_count`, `token_count`, `source_file`.

### Multi-Collection Retrieval

The MCP retrieval server searches across **all configured collections** by default:

1. **Query**: client sends natural-language question via MCP tool call
2. **Embed**: query text embedded via Ollama
3. **Search**: each configured collection returns top-k chunk matches by cosine similarity
4. **Merge**: results are globally sorted by score across collections
5. **Expand**: surrounding chunks from same documents added for context
6. **Group**: results grouped by section or book
7. **Assemble**: evidence bundle formatted as prompt context
8. **Answer**: LiteLLM streams final answer grounded in retrieved evidence

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health, shows configured collections |
| `/collections` | GET | List all Qdrant collections with point counts and vector config |
| `/mcp/info` | GET | MCP protocol info, available tools |
| `/mcp` | POST | Main MCP JSON-RPC 2.0 endpoint |
