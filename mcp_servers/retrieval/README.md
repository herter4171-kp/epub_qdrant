# EPUB Knowledge Base Retrieval MCP Server

Standalone MCP (Model Context Protocol) server that provides knowledge-base retrieval tools over Streamable HTTP for a Qdrant-backed vector index spanning multiple collections of EPUB books and academic papers.

## Quick Start

```bash
cd mcp_servers/retrieval
pip install -e .

# Configure
export QDRANT_COLLECTIONS=epub_kb,papers
export QDRANT_URL=http://localhost:6333
export OLLAMA_URL=http://localhost:11434
export EMBEDDING_MODEL=nomic-embed-text
export LITELLM_API_KEY=your-key
export LITELLM_API_URL=https://litellm.twr.church/v1
export LITELLM_MODEL=qwen36

# Run
python -m mcp_server.server
# or
uv mcp_server
# Server listens on :8090
```

## Tools

| Tool | Description | Args | Returns |
|------|-------------|------|---------|
| `search` | Semantic search across configured collections → grouped chunks | `query`, `top_k`, `group_by`, `collection`, `collections`, `filter_by` | Grouped evidence with scores |
| `answer` | Search + LLM answer | `query`, `top_k`, `group_by`, `collection`, `collections`, `filter_by` | Streaming LLM answer |
| `get_context` | Surrounding chunks around a section | `source_file`, `section_title`, `radius`, `collection` | Context window of chunks |
| `list_collections` | Read-only: list all Qdrant collections | (none) | Collection names, point counts, vector config |

### Tool Arguments

#### `search`

Search semantically relevant chunks grouped by section or book with similarity scores.

```json
{
  "name": "search",
  "arguments": {
    "query": "what is quantum entanglement",
    "top_k": 10,
    "group_by": "section"
  }
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | The search query |
| `top_k` | integer | No | 15 | Number of results per collection |
| `group_by` | string | No | "section" | How to group: "section" or "book" |
| `collection` | string | No | default | Target a specific collection |
| `collections` | string | No | all | Comma-separated list of collections to search |
| `filter_by` | string | No | — | JSON metadata filter, e.g. `{"doc_type": "paper"}` |

#### `answer`

Answer a question using the knowledge base. Retrieves relevant chunks, assembles evidence, and generates an LLM answer.

```json
{
  "name": "answer",
  "arguments": {
    "query": "explain transformer attention mechanisms",
    "top_k": 20,
    "group_by": "book"
  }
}
```

Same parameters as `search`. Returns the LLM-generated answer grounded in retrieved evidence.

#### `get_context`

Get surrounding chunks around a specific section. Useful for reading the full context of a known chapter.

```json
{
  "name": "get_context",
  "arguments": {
    "source_file": "quantum_computing.epub",
    "section_title": "Chapter 3: Qubits",
    "radius": 3
  }
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_file` | string | Yes | — | Filename (EPUB or PDF) |
| `section_title` | string | Yes | — | Chapter/section title |
| `radius` | integer | No | 2 | Surrounding chunks per side |
| `collection` | string | No | default | Target a specific collection |

#### `list_collections`

List all available Qdrant collections with metadata.

```json
{
  "name": "list_collections",
  "arguments": {}
}
```

Returns collection names, point counts, and vector configuration.

## Multi-Collection Behavior

### Default: Cross-collection search

When `QDRANT_COLLECTIONS=epub_kb,papers` is configured, `search` and `answer` search **all** collections by default, merge results globally by score, and group them.

### Target a single collection

```json
{
  "name": "search",
  "arguments": {
    "query": "transformer attention",
    "collection": "papers"
  }
}
```

### Override which collections to search

```json
{
  "name": "search",
  "arguments": {
    "query": "agent memory",
    "collections": "papers,epub_kb"
  }
}
```

### Metadata pre-filtering

```json
{
  "name": "search",
  "arguments": {
    "query": "reasoning",
    "filter_by": "{\"doc_type\": \"paper\"}"
  }
}
```

## REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health, shows configured collections |
| `/collections` | GET | List all Qdrant collections with point counts and vector config |
| `/mcp/info` | GET | MCP protocol info, available tools |
| `/mcp` | POST | Main MCP JSON-RPC 2.0 endpoint |

### Health Check

```bash
curl http://localhost:8090/health
```

Returns:
```json
{
  "status": "ok",
  "collections": ["epub_kb", "papers"],
  "default_collection": "epub_kb",
  "version": "0.2.0"
}
```

### MCP Info

```bash
curl http://localhost:8090/mcp/info
```

Returns protocol version, configured collections, and available tools.

## Client Examples

### JSON-RPC 2.0 (recommended)

```bash
curl -X POST http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "search",
      "arguments": {
        "query": "what is quantum entanglement",
        "top_k": 10
      }
    }
  }'
```

### Legacy format

```bash
curl -X POST http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "method": "search",
    "arguments": {
      "query": "what is quantum entanglement",
      "top_k": 10
    }
  }'
```

### Tools list

```bash
curl -X POST http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
  }'
```

## Configuration

| Env Var | Default | Required | Description |
|---------|---------|----------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | No | Qdrant endpoint |
| `QDRANT_COLLECTIONS` | — | **Yes** | Comma-separated collection names (e.g., `epub_kb,papers`) |
| `QDRANT_COLLECTION` | `books` | No | Legacy fallback if `QDRANT_COLLECTIONS` not set |
| `OLLAMA_URL` | `http://localhost:11434` | No | Ollama embedding endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text` | No | Embedding model name |
| `LITELLM_API_URL` | `https://litellm.twr.church/v1` | No | LiteLLM ChatCompletion endpoint |
| `LITELLM_API_KEY` | — | No* | *Required for `answer` tool |
| `LITELLM_MODEL` | `qwen36` | No | LLM model for answers (via llama.cpp through LiteLLM) |
| `MCP_PORT` | `8090` | No | HTTP listen port |
| `MCP_HOST` | `0.0.0.0` | No | HTTP listen host |
| `RETRIEVAL_TOP_K` | `15` | No | Default top-k per collection |
| `RETRIEVAL_CONTEXT_RADIUS` | `2` | No | Surrounding chunks per side |
| `RETRIEVAL_GROUP_BY` | `section` | No | Group results by `section` or `book` |

\* The server starts without `LITELLM_API_KEY` but the `answer` tool will fail gracefully.

## Architecture

```
n8n / MCP Client
     ↓  Streamable HTTP (POST, JSON-RPC 2.0)
epub_qdrant Retrieval MCP Server (:8090)
     ↓
Qdrant — vector search across configured collections
     ↓
Retriever — group by section/book, assemble context
     ↓
LiteLLM — streaming ChatCompletion → LLM answer
```

### Unified Metadata Schema

Both EPUB and paper chunks share common payload fields for cross-collection compatibility:

| Field | Type | EPUB Example | Paper Example |
|-------|------|-------------|---------------|
| `doc_type` | string (KEYWORD) | `epub` | `paper` |
| `title` | string (KEYWORD) | `Book Title` | `Agent Memory Survey` |
| `section` | string (KEYWORD) | `Chapter 3` | `method` |
| `source_file` | string (KEYWORD) | `library.epub` | `2302_01560.pdf` |
| `chunk_index` | integer | `0` | `2` |
| `token_count` | integer | `256` | `512` |
| `authors` | array of strings | `["Author A"]` | `["Smith", "Jones"]` |
| `year` | integer (optional) | `2020` | `2023` |
| `category` | string (KEYWORD) | — | `application-papers` |
| `subcategory` | string (KEYWORD) | — | `deep-reasoning` |
| `arxiv_id` | string (KEYWORD) | — | `2302.01560` |

### Qdrant Payload Index Strategy

- **KEYWORD index** for: `doc_type`, `source_file`, `title`, `section` (exact match filtering)
- **INTEGER index** for: `chunk_index`, `year` (range queries)
- **No index** for high-cardinality fields: `text`, `authors` (free-text search is via vector space)

## File Structure

```
mcp_servers/retrieval/
  pyproject.toml        # Package config (mcp-server dependency)
  mcp_server/
    __init__.py         # Package marker
    server.py           # MCP server (FastAPI + Streamable HTTP)
    retriever.py        # Retrieval layer: search → group → evidence
    llm_client.py       # LiteLLM streaming client
    config.py           # MCP server settings