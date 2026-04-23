# EPUB & Paper-to-Qdrant Ingestion Pipeline

Standalone pipeline that reads EPUB files and PDF papers, generates embeddings via Ollama, and stores vectors in **Qdrant collections** for semantic search and knowledge-base retrieval.

## Table of Contents

- [Setup](#setup)
- [EPUB Ingestion](#epub-ingestion)
- [Paper Embedding](#paper-embedding)
- [Search (CLI)](#search-cli)
- [Benchmark (LLM-as-Judge Evaluation)](#benchmark-llm-as-judge-evaluation)
- [Phase 2: Hybrid Search](#phase-2-hybrid-search)
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

## Benchmark (LLM-as-Judge Evaluation)

Evaluate retrieval quality using **LLM-as-judge pairwise comparison** — no human-labeled test set needed. Compares a baseline method against any new retrieval phase and reports win rates.

### Setup

```bash
# Ensure litellm is installed (for the LLM judge)
pip install litellm

# Configure the judge's LLM endpoint in .env
# OPENAI_API_BASE=http://192.168.68.75:4000/v1  (or your LiteLLM endpoint)
# LITELLM_API_KEY=your-key
```

### Running

```bash
# Run baseline evaluation (compares current retriever vs itself — all ties expected)
python3 scripts/evaluate.py baseline

# Run a new phase and compare against the stored baseline
python3 scripts/evaluate.py phase_2_hybrid

# Use a different baseline for comparison
python3 scripts/evaluate.py phase_2_hybrid --baseline phase_0

# Custom output path
python3 scripts/evaluate.py phase_2_hybrid --output results_new.json
```

### Query Set

30 tailored queries across three categories:

| Category | Count | Example Queries |
|----------|-------|-----------------|
| **Cross-collection** | 10 | "agentic AI patterns", "multi-agent collaboration strategies", "agent memory systems" |
| **Books-focused** | 10 | "Apress books about agentic AI design", "enterprise patterns for generative AI systems" |
| **Papers-focused** | 10 | "arxiv papers on agent framework architecture", "embodied agents research papers" |

### Output

Results are saved to `results.json` (or custom path):

```json
{
  "version": "1.0",
  "model": "openai/qwen36",
  "evaluated_at": "2026-04-22T21:33:53.994071+00:00",
  "baseline_method": "baseline",
  "num_queries": 30,
  "per_query_scores": {
    "agentic AI patterns": {
      "baseline": { "top5_avg_score": 2.1614, "cross_collection_ratio": 1.0 },
      "phase_2_hybrid": { "top5_avg_score": 2.3451, "judge_wins_against_baseline": true },
      "judgment": { "winner": "phase_2_hybrid", "reason": "..." }
    }
  },
  "aggregate": {
    "phase_2_hybrid": {
      "wins_against_baseline": 18,
      "losses_against_baseline": 10,
      "ties": 2,
      "win_rate": 0.60,
      "avg_score_improvement": 0.1234,
      "elapsed_seconds": 185.3
    }
  }
}
```

### Metrics Explained

| Metric | Description |
|--------|-------------|
| **win_rate** | Proportion of queries where the new method won pairwise judgment |
| **avg_score_improvement** | Mean delta in top-5 average score vs baseline |
| **cross_collection_ratio** | Balance of results across collections (1.0 = perfectly balanced) |

### Phases

| Phase | Description | Target Win Rate |
|-------|-------------|-----------------|
| `baseline` | Current flat cosine search (reference) | — |
| `phase_0` | Z-score normalization + metadata boost | >55% |
| `phase_2_hybrid` | MiniCOIL sparse + dense vectors with RRF fusion | >60% |
| `phase_3_filter` | LLM-driven metadata filter extraction | >65% |

## Phase 2: Hybrid Search

Phase 2 adds **MiniCOIL sparse vectors** alongside existing dense (semantic) vectors, enabling **hybrid search** via Reciprocal Rank Fusion (RRF).

### Prerequisites

The MiniCOIL embedding server must be running on the GPU box:

```bash
# On the Linux GPU box (192.168.68.75):
pip install fastembed-gpu fastapi uvicorn
uvicorn mcp_servers.minicoil_server.server:app --host 0.0.0.0 --port 9000
```

Verify from the Mac:

```bash
curl http://192.168.68.75:9000/health
# → {"status":"ok","ready":true}
```

### Sparse Collections

Phase 2 creates new collections with **named vectors** (`dense` + `sparse`):

| Collection | Vectors | Description |
|-----------|---------|-------------|
| `books` | dense only | Original baseline (read-only reference) |
| `books-named` | dense + sparse | Hybrid search target |
| `papers` | dense only | Original baseline (read-only reference) |
| `papers-named` | dense + sparse | Hybrid search target |

### Running the Hybrid Search Test

To verify hybrid search is working and compare dense vs sparse vs hybrid results:

```bash
# Use the project's .venv (Python 3.14.4) — NOT system Python
cd /Users/justinherter/projects/epub_qdrant
.venv/bin/python3 scripts/test_hybrid_search.py
```

This script:
1. Queries collection configs (verifies sparse vectors are stored)
2. Embeds test queries with Ollama (dense) and MiniCOIL (sparse)
3. Runs dense-only, sparse-only, and RRF-fused hybrid search on `books-named`
4. Prints side-by-side top-10 comparisons
5. Reports overlap statistics between dense and sparse results

**Example output:**
```
Dense top 10 IDs:   [0, 62, 322, 1139, 2112, 2176, 2185, 2197, 2484, 3971]
Sparse top 10 IDs:  [58, 62, 86, 310, 2184, 2185, 2198, 2206, 2725, 5414]
Hybrid top 10 IDs:  [0, 62, 86, 322, 2112, 2184, 2185, 2206, 2725, 5414]
Dense intersect Sparse top 20: 6/34
  *** MODERATE OVERLAP - sparse contributes distinct signal ***
```

### Migrating Sparse Vectors to New Collections

If you need to (re)create the `-named` collections:

```bash
cd /Users/justinherter/projects/epub_qdrant
.venv/bin/python3 scripts/embed_sparse_vectors.py
```

This scrolls existing collections, computes MiniCOIL sparse vectors via the embedding server, and upserts into `books-named` / `papers-named` with named vector configs. (~3-6 minutes for all ~96K points on GPU).

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
  embed_sparse_vectors.py  # Phase 2: scroll + re-embed sparse vectors into -named collections
  test_hybrid_search.py    # Phase 2: verify hybrid search (dense vs sparse vs RRF)
  evaluate.py              # LLM-as-judge evaluation harness

mcp_servers/
  minicoil_server/
    server.py             # MiniCOIL sparse embedding server (runs on GPU box :9000)
    client.py             # HTTP client for sparse embeddings (shared by indexing script + retriever)
  retrieval/
    mcp_server/
      server.py           # MCP server (FastAPI + Streamable HTTP, v0.2.0)
      retriever.py        # Retrieval layer: search → group → evidence (hybrid search)
      llm_client.py       # LiteLLM streaming client
      config.py           # MCP server settings
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

### Phase 2: Hybrid Search Flow

1. **Scroll**: `embed_sparse_vectors.py` scrolls original collections in batches
2. **Embed**: MiniCOIL server (`:9000`) computes sparse vectors via GPU (FastEmbed + ONNX Runtime)
3. **Upsert**: New collections get **named vectors** — `dense` (768-d cosine) + `sparse` (variable-d IDF)
4. **Search**: `retriever.hybrid_search()` runs both dense and sparse queries, fuses via RRF (k_rrf=60)
5. **Normalize**: Z-score normalization applied across fused results for cross-collection fairness

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