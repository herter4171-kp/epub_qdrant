# Tests: End-to-End Ingestion + Retrieval Harness

## Overview

Integration tests that validate the full ingestion → embedding → retrieval pipeline for both books (EPUB) and papers (PDF + JSON metadata). All test collections use the `test-` prefix and are cleaned up after the suite.

**Result: 26 tests run, 21 passed, 5 skipped (MCP), 0 failures.**

## Prerequisites

- **Qdrant** running on `192.168.68.75:6333` (or set `QDRANT_HOST`/`QDRANT_PORT` env vars)
- **Ollama** running on `192.168.68.75:11434` (or set `OLLAMA_URL`)
- **MiniCOIL server** running on `192.168.68.75:9000` (sparse embeddings)
- **MCP server** running on `localhost:8090` (optional, for Stage 4 tests)
- Python 3.11+ with `.venv` activated

## Data Sources

| Type | Source | Metadata |
|------|--------|----------|
| **Book** | First EPUB in `test_books/` (any `*.epub`) | Hardcoded in code (publisher="Apress", language="en", isbn) |
| **Paper** | First PDF+JSON pair in `downloads/` (`*.pdf` + `*.json`) | Parsed from JSON `metadataAttributes` array |

## Test Collections

All test collections are prefixed with `test-` and cleaned up after the suite:

| Collection | Contents | Vectors |
|------------|----------|---------|
| `test-books-named` | One EPUB (~3 sections, ~100 chunks max) | dense (768-d) + sparse (MiniCOIL) |
| `test-papers-named` | One PDF (~50 chunks) | dense (768-d) + sparse (MiniCOIL) |

## Test Stages

### Stage 1: Book Ingestion (`TestBookIngestion`) — 6 tests

| Test | What it does |
|------|-------------|
| `test_01_epub_parsed` | Parses EPUB → Book with title, sections, source_file |
| `test_02_chunks_produced` | `chunk_section()` produces 1-100 Chunk objects with text and metadata |
| `test_03_dense_embeds` | Ollama `/api/embed` produces 768-d dense vectors for all chunks |
| `test_04_sparse_embeds` | MiniCOIL server produces sparse vectors (indices+values) for 10 chunks |
| `test_05_upsert` | Upserts all chunks into `test-books-named` with dense + sparse named vectors |
| `test_06_payload_fields` | Each upserted point has all required metadata fields |

### Stage 2: Paper Ingestion (`TestPaperIngestion`) — 6 tests

| Test | What it does |
|------|-------------|
| `test_01_metadata` | JSON metadata parses into flat dict with arxiv_id, title, category |
| `test_02_chunked` | `chunk_paper()` produces 1-100 PaperChunk objects with metadata |
| `test_03_dense` | Ollama produces dense embeddings for 20 chunks |
| `test_04_sparse` | MiniCOIL produces sparse embeddings for 10 chunks |
| `test_05_upsert` | Upserts all chunks into `test-papers-named` with dense + sparse |
| `test_06_payload_fields` | Each point has arxiv_id, title, category, source_file, doc_type |

### Stage 3: Curl Retrieval (`TestCurlRetrieval`) — 5 tests

Tests retrieval against Qdrant directly via the Python client (curl-equivalent HTTP calls).

| Test | What it does |
|------|-------------|
| `test_01_books_vector` | Dense vector search on `test-books-named` returns results with book_title |
| `test_02_papers_vector` | Dense vector search on `test-papers-named` returns results with arxiv_id |
| `test_03_books_hybrid` | Both dense and sparse queries return results for book query |
| `test_04_papers_hybrid` | Both dense and sparse queries return results for paper query |
| `test_05_rrf_fusion` | Reciprocal Rank Fusion merges dense + sparse rankings into top-5 |

### Stage 4: MCP Server Retrieval (`TestMCPRetrieval`) — 5 tests

Tests retrieval via the MCP server JSON-RPC API. Skipped if MCP server is not reachable.

| Test | What it does |
|------|-------------|
| `test_01_health` | MCP server HTTP health check (200 OK) |
| `test_02_query_books` | `tools/call` → `query` on `test-books-named`, checks groups + total_chunks |
| `test_03_query_papers` | `tools/call` → `query` on `test-papers-named` |
| `test_04_list_collections` | `tools/call` → `list_collections` |
| `test_05_cross_collection` | Cross-collection search across both test collections |

### Stage 5: Cleanup (`TestCleanup`) — 4 tests

| Test | What it does |
|------|-------------|
| `test_00_exist` | Verifies `test-books-named` and `test-papers-named` exist before cleanup |
| `test_01_books` | Deletes `test-books-named`, verifies gone |
| `test_02_papers` | Deletes `test-papers-named`, verifies gone |
| `test_03_no_orphans` | Asserts no `test-*` collections remain |

## Running Tests

### Run all tests
```bash
.venv/bin/python tests/test_ingestion_retrieval.py
```

### Run via pytest (if installed)
```bash
.venv/bin/python -m pytest tests/test_ingestion_retrieval.py -v --tb=short
```

### Run specific stage
```bash
.venv/bin/python tests/test_ingestion_retrieval.py 2>&1 | grep -A100 "TestBookIngestion"
```
Or with pytest:
```bash
.venv/bin/python -m pytest tests/test_ingestion_retrieval.py -v -k "BookIngestion"
.venv/bin/python -m pytest tests/test_ingestion_retrieval.py -v -k "CurlRetrieval"
.venv/bin/python -m pytest tests/test_ingestion_retrieval.py -v -k "MCPRetrieval"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `192.168.68.75` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `OLLAMA_URL` | `http://192.168.68.75:11434` | Ollama embedding endpoint |
| `MINICOIL_URL` | `http://192.168.68.75:9000` | MiniCOIL embedding server |
| `MCP_HOST` | `localhost` | MCP server host |
| `MCP_PORT` | `8090` | MCP server port |

## Actual Run Results

```
Ran 26 tests in 13.169s
OK (skipped=5)
```

| Stage | Tests | Status |
|-------|-------|--------|
| Book Ingestion | 6 | All pass |
| Paper Ingestion | 6 | All pass |
| Curl Retrieval | 5 | All pass |
| MCP Retrieval | 5 | All skipped (server not running) |
| Cleanup | 4 | All pass |

## Troubleshooting

- **MiniCOIL not reachable**: Stage 1 sparse tests and Stage 3/4 hybrid tests will fail. Ensure `http://192.168.68.75:9000/health` returns `{"status":"ok","ready":true}`.
- **Ollama not reachable**: Dense embedding tests fail. Ensure `http://192.168.68.75:11434/api/tags` returns a valid response.
- **MCP server not running**: Stage 4 tests are skipped automatically.
- **No EPUB/PDF found**: Tests are skipped with `SkipTest`.