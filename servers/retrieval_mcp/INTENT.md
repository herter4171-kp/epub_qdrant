# Intent: EPUB Knowledge Base Retrieval MCP Server

## Purpose

A standalone MCP (Model Context Protocol) server that exposes retrieval tools over Streamable HTTP for a Qdrant-backed knowledge base spanning multiple collections of EPUB books and academic papers.

This server is retrieval-only. Ingestion remains a separate CLI pipeline (`epub_qdrant ingest <directory>`, `scripts/embed_papers_to_qdrant.py`). This server reads from the indexed collections and answers questions using retrieved evidence.

See [README.md](../README.md) for full usage documentation, client examples, and REST endpoint details.

## Design Goals

1. **Retrieval-only**: No ingestion, no file writing, no collection management. Only search and answer.
2. **Multi-collection by default**: `QDRANT_COLLECTIONS` is a comma-separated list (e.g. `epub_kb,papers`). The server searches all collections by default, or can target a single one via tool parameters.
3. **Unified metadata schema**: EPUB and paper chunks share common payload fields (`title`, `section`, `doc_type`, `source_file`) so that cross-collection search and metadata filtering work seamlessly.
4. **Streamable HTTP transport**: MCP over Streamable HTTP per the 2025-03-26 spec. Long-running HTTP service.
5. **Streaming LLM output**: The `answer` tool streams the LLM response incrementally via MCP content deltas.
6. **Metadata-aware search**: Optional `filter_by` parameter lets clients pre-filter by metadata (e.g. `{"doc_type": "paper"}`) before semantic search.

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

## Tools

| Tool | Description | Args | Returns |
|------|-------------|------|---------|
| `search` | Semantic search across configured collections → grouped chunks | `query`, `top_k`, `group_by`, `collection` (target single), `collections` (comma-separated override), `filter_by` (JSON metadata filter) | Grouped evidence with scores |
| `answer` | Search + LLM answer | Same as `search` | Streaming LLM answer |
| `get_context` | Surrounding chunks around a section | `source_file`, `section_title`, `radius`, `collection` | Context window of chunks |
| `list_collections` | Read-only: list all Qdrant collections with stats | (none) | Collection names, point counts, vector config |

## Unified Metadata Schema

Both EPUB and paper chunks share these payload fields for cross-collection compatibility:

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
| `publish_date` | string | — | `2023-02-03` |
| `arxiv_id` | string (KEYWORD) | — | `2302.01560` |

### Legacy Fields (backward compatible)

| Field | Source | Notes |
|-------|--------|-------|
| `book_title` | EPUB only | Falls back to `title` in unified schema |
| `section_title` | EPUB only | Falls back to `section` in unified schema |
| `chapter_index` | EPUB only | Used for ordering |
| `section_index` | EPUB only | Used for ordering |
| `publisher` | EPUB only | — |
| `language` | EPUB only | — |
| `isbn` | EPUB only | — |

### Qdrant Payload Index Strategy

- **KEYWORD index** for: `doc_type`, `source_file`, `title`, `section` (exact match filtering)
- **INTEGER index** for: `chunk_index`, `year` (range queries)
- **No index** for high-cardinality fields: `text`, `authors` (free-text search is via vector space)

## Configuration

| Env Var | Default | Required | Description |
|---------|---------|----------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | No | Qdrant endpoint |
| `QDRANT_COLLECTIONS` | — | **Yes** | Comma-separated collection names (e.g. `epub_kb,papers`). Sets `DEFAULT_COLLECTION` to the first entry. |
| `QDRANT_COLLECTION` | `books` | No (legacy fallback) | Single collection name if `QDRANT_COLLECTIONS` is not set. |
| `OLLAMA_URL` | `http://localhost:11434` | No | Ollama embedding endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text` | No | Embedding model name |
| `LITELLM_API_URL` | `https://litellm.twr.church/v1` | No | LiteLLM ChatCompletion endpoint |
| `LITELLM_API_KEY` | — | No* | *Required for `answer` tool |
| `LITELLM_MODEL` | `qwen36` | No | LLM model for answers (via llama.cpp through LiteLLM) |
| `MCP_PORT` | `8090` | No | HTTP listen port |
| `RETRIEVAL_TOP_K` | `15` | No | Default top-k per collection |
| `RETRIEVAL_CONTEXT_RADIUS` | `2` | No | Surrounding chunks per side |
| `RETRIEVAL_GROUP_BY` | `section` | No | Group results by `section` or `book` |

\* The server starts without `LITELLM_API_KEY` but the `answer` tool will fail gracefully.

## Multi-Collection Behavior

### Default: Cross-collection search
When more than one collection is configured (`QDRANT_COLLECTIONS=epub_kb,papers`), `search` and `answer` search **all** collections by default, merge results globally by score, and group them.

### Target a single collection
Pass `collection` in the tool arguments:
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
Pass `collections` (comma-separated) to search a specific subset:
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
Pass `filter_by` as a JSON string:
```json
{
  "name": "search",
  "arguments": {
    "query": "reasoning",
    "filter_by": "{\"doc_type\": \"paper\"}"
  }
}
```

## Running

```bash
cd mcp_servers/retrieval
pip install -e .
export QDRANT_COLLECTIONS=epub_kb,papers
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
    "arguments": {
      "query": "what is quantum entanglement",
      "top_k": 10
    }
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