# EPUB-to-Qdrant Ingestion Pipeline

Standalone pipeline that reads EPUB files, generates embeddings via Ollama, and stores vectors in Qdrant for semantic search.

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
# Ingest all EPUBs from a directory
python -m src.main ingest ./my_books

# With progress output
python -m src.main ingest /path/to/epubs --limit 5

# Or via the entry point script
epub_qdrant ingest ./my_books
```

### Search

```bash
# Search a specific collection
python -m src.main search <collection-name> "your query here"

# Limit results
python -m src.main search python-basics "how to use decorators" --top-k 5
```

### List Collections

```bash
python -m src.main list-collections
```

### Delete a Collection

```bash
python -m src.main delete-collection <collection-name>
```

## Configuration

Environment variables (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://192.168.68.75:6333` | Qdrant server URL |
| `OLLAMA_URL` | `http://192.168.68.75:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `embeddinggemma:300m` | Ollama embedding model name |
| `CHUNK_SIZE` | `500` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Token overlap between chunks |
| `VECTOR_SIZE` | `768` | Embedding vector dimensions |
| `DISTANCE` | `Cosine` | Vector distance metric |

## Architecture

```
src/
  __init__.py     # Package marker
  config.py       # Settings from env vars
  epub_parser.py  # EPUB text extraction with heading detection
  chunker.py      # Paragraph-aware chunking with overlap
  embedder.py     # Ollama embedding API calls
  storage.py      # Qdrant collection management and search
  main.py         # CLI entry point (ingest/search)
```

### Flow

1. **Parse**: `epub_parser.py` reads EPUB structure, extracts text by chapter/section
2. **Chunk**: `chunker.py` splits text into ~500 token chunks with 100 token overlap, respecting paragraph boundaries
3. **Embed**: `embedder.py` calls Ollama `/api/embed` for each chunk
4. **Store**: `storage.py` upserts vectors + metadata into Qdrant

Each EPUB gets its own Qdrant collection named after the book filename.

## Dependencies

- `epub` - EPUB file reading
- `qdrant-client` - Qdrant Python client
- `requests` - HTTP for Ollama API
- `python-dotenv` - `.env` file loading
- `click` - CLI framework