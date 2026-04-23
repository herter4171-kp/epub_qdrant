# Ingestion Pipeline Overview

## Architecture

```
test_books/*.epub  ──→  epub_parser  ──→  semantic_chunker  ──→  embedding server  ──→  Qdrant
downloads/*.pdf    ──→  paper_section_splitter ──→ semantic_chunker ──→ embedding server ──→ Qdrant
downloads/*.json   (metadata sidecar, read during paper ingestion)
```

### Components

| Component | Location | Role |
|-----------|----------|------|
| EPUB parser | `src/ingestion/epub_parser.py` | Extracts sections with real heading titles from raw HTML + OPF metadata |
| Paper section splitter | `src/ingestion/paper_section_splitter.py` | Splits PDF text into named academic sections, excludes References/Bibliography/Appendix |
| Semantic chunker | `src/ingestion/semantic_chunker.py` | Three-layer pipeline: structural → semantic boundary detection → recursive splitting via semchunk |
| Embedding server | `servers/embedding_server/` | FastAPI service on GPU box exposing `/embed_dense` (embeddinggemma-300m, 768d) and `/embed_sparse` (MiniCOIL) |
| Embedding client | `servers/embedding_server/client.py` | Thin HTTP client — `get_dense_vectors()`, `get_sparse_vectors()`, `health_check()` |
| Storage | `src/storage/` | Qdrant client wrapper — collection lifecycle, upsert, search, scroll |
| Config | `src/config.py` | Reads `.env` for `QDRANT_URL`, `EMBEDDING_SERVER_URL`, `CHUNK_SIZE`, `TOKENIZER_JSON`, etc. |
| Old chunker (compat) | `src/ingestion/chunker.py` | Legacy fixed-window chunker — no longer imported by loaders |
| Old paper loader (compat) | `src/ingestion/paper_loader.py` | Legacy paper chunker — no longer imported by loaders |

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

1. **Paper pipeline downloads to flat `downloads/`**: `embed_papers.py` writes PDFs and JSON metadata flat under `downloads/`. The `embed_papers_to_qdrant.py` script expects this layout.

## Environment

Required in `.env`:
```
QDRANT_URL=http://192.168.68.75:6333
EMBEDDING_SERVER_URL=http://192.168.68.75:8100
TOKENIZER_JSON=tokenizer.json
```

`TOKENIZER_JSON` points to the embeddinggemma-300m `tokenizer.json` file. Defaults to `./tokenizer.json` in the project root. Also accepted via `--tokenizer-json` CLI flag.

The embedding server must be running on the GPU box with both dense (embeddinggemma-300m) and sparse (MiniCOIL) models loaded. Check with `curl http://192.168.68.75:8100/health`.
