# Ingestion Pipeline Overview

## Architecture

```
test_books/*.epub  ──→  epub_parser  ──→  chunker  ──→  embedding server  ──→  Qdrant
downloads/*.pdf    ──→  paper_loader ──→  chunker  ──→  embedding server  ──→  Qdrant
downloads/*.json   (metadata sidecar, read during paper ingestion)
```

### Components

| Component | Location | Role |
|-----------|----------|------|
| EPUB parser | `src/ingestion/epub_parser.py` | Extracts sections + OPF metadata (title, creator, publisher, date, language, ISBN) from `.epub` files |
| Paper loader | `src/ingestion/paper_loader.py` | Chunks raw PDF text with metadata (arxiv_id, category, authors, etc.) |
| Chunker | `src/ingestion/chunker.py` | Splits sections into overlapping token-window chunks (default 500 tokens, 100 overlap) |
| Embedding server | `servers/embedding_server/` | FastAPI service on GPU box exposing `/embed_dense` (embeddinggemma-300m, 768d) and `/embed_sparse` (MiniCOIL) |
| Embedding client | `servers/embedding_server/client.py` | Thin HTTP client — `get_dense_vectors()`, `get_sparse_vectors()`, `health_check()` |
| Storage | `src/storage/` | Qdrant client wrapper — collection lifecycle, upsert, search, scroll |
| Config | `src/config.py` | Reads `.env` for `QDRANT_URL`, `EMBEDDING_SERVER_URL`, `CHUNK_SIZE`, etc. |

### Entry Points

| Entry point | What it does | Status |
|-------------|-------------|--------|
| `scripts/ingest_fresh.py` | **Primary ingest** — polymorphic loader for EPUBs + PDFs into `-fresh` named-vector collections (dense + sparse, two-pass) | Working |
| `scripts/embed_papers.py` | Download arxiv PDFs + write JSON metadata to `downloads/` | Working |
| `scripts/embed_papers_to_qdrant.py` | Embed PDFs from `downloads/` into Qdrant `papers` collection (dense only, unnamed vectors) | Working |
| `scripts/embed_sparse_vectors.py` | Two-pass migration from existing unnamed collections → `-hybrid` named-vector collections | Working |
| `scripts/test_ingest_one_book.py` | End-to-end test: one EPUB → throwaway collection → verify search → cleanup | Working |
| `epubq ingest <dir>` (CLI) | Ingest EPUBs from a directory | **Broken** — references missing `src/embedding/dense_embedder.py` |

## Qdrant Collections

| Collection | Contents | Points | Status |
|------------|----------|--------|--------|
| `books` | EPUB book chunks (dense vectors, unnamed) | 6,212 | Production — do not touch |
| `books-named` | EPUB book chunks (dense + sparse named vectors) | 6,212 | Production — do not touch |
| `papers` | Arxiv paper chunks (dense vectors, unnamed) | 90,256 | Production — do not touch |
| `papers-named` | Arxiv paper chunks (dense + sparse named vectors) | 90,256 | Production — do not touch |
| `books-fresh` | Created by `ingest_fresh.py` — EPUBs with named dense + sparse | varies | Test/staging |
| `papers-fresh` | Created by `ingest_fresh.py` — PDFs with named dense + sparse | varies | Test/staging |

Test scripts create ephemeral collections that are cleaned up with `--cleanup`.

## Polymorphic Loader (`src/ingestion/loader.py`)

The `DocumentLoader` abstraction normalizes both document types into a uniform `DocumentChunk` (text + flat metadata dict). The embedding pipeline never knows the source format.

```
DocumentLoader.for_path(path)  →  EpubLoader | PdfLoader
    .load(path)                →  List[DocumentChunk]
```

- `EpubLoader`: metadata from OPF (title, publisher, language, ISBN) — no sidecar files
- `PdfLoader`: text via pypdf, metadata from same-name `.json` sidecar under `downloads/`

## Two-Pass Named Vector Ingest

Per the unified-embedding-server spec, Qdrant supports upserting named vectors independently on the same point ID. The ingest pipeline exploits this:

1. **Pass 1 (dense)**: load files → chunk → `get_dense_vectors()` → upsert with `{"dense": vec}` + payload
2. **Pass 2 (sparse)**: scroll the same collection → `get_sparse_vectors()` (512-word chunking + max-aggregation) → upsert same point IDs with `{"sparse": SparseVector(...)}`

## Known Issues

1. **CLI `ingest` command is broken**: `src/cli/main.py` imports `src.embedding.dense_embedder.Embedder` which no longer exists (only a stale `.pyc` remains). The embedding layer was migrated to `servers/embedding_server/` but the CLI was not updated. Fix: update the CLI to use `servers.embedding_server.client.get_dense_vectors()` directly.

2. **Paper pipeline downloads to flat `downloads/`**: `embed_papers.py` writes PDFs and JSON metadata flat under `downloads/`. The `embed_papers_to_qdrant.py` script expects this layout. The `process_category()` function also creates category subdirectories but the actual PDFs land in the root `downloads/` dir.

## Environment

Required in `.env`:
```
QDRANT_URL=http://192.168.68.75:6333
EMBEDDING_SERVER_URL=http://192.168.68.75:8100
```

The embedding server must be running on the GPU box with both dense (embeddinggemma-300m) and sparse (MiniCOIL) models loaded. Check with `curl http://192.168.68.75:8100/health`.
