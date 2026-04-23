# Proposal: Single-Book Sparse/Dense Weight Sweep

## Goal
Find optimal sparse weight for hybrid search on **one EPUB book** from `test_books/`, using that book's actual chunk content to drive query generation and relevance evaluation.

## Current State

- **20 EPUBs** in `test_books/`, all Apress publisher books, ~6,212 total chunks across `books` / `books-named` collections
- **Existing sweep** (`scripts/sweep_sparse_weight.py`): queries both collections → blind LLM scoring → RRF weight sweep
- **MiniCOIL sparse vectors** live in `-named` collections alongside dense vectors
- **No book-specific query corpus** — current sweep uses hardcoded generic AI queries

## Approach

### Step 1: Pick Book & Extract Content Profile
Pick one EPUB (e.g., `masteringretrieval-augmentedgeneration.epub` — 6K+ chunks, RAG topic = high retrieval signal).

Use `src/epub_parser.py` (or scroll Qdrant) to profile the book's content:
- Chapter/section headings (from EPUB structure)
- Chunk count, avg token count, topic distribution
- Metadata: publisher, isbn, language, source_file

### Step 2: Generate Query Corpus from Book Content
Extract representative snippets from the book's chunks. Feed them to LLM → generate 10-20 natural-language queries that a user would ask **about this specific book**.

Query types:
- **Exact keyword match** (sparse-favoring): "what does recursive character text splitting do", "explain HNSW indexing"
- **Semantic/conceptual** (dense-favoring): "how do you evaluate RAG systems in production", "what are best practices for prompt templates"
- **Metadata-targeted** (filter-seeking): "Apress book on enterprise AI", "RAG with LangChain and Python"
- **Mixed** (hybrid-optimal): "how does retrieval-augmented generation work with private data"

### Step 3: Single-Book Sweep
Modify sweep script to:
- Target **one** `-named` collection subset (use `FieldCondition(key="source_file", match="masteringretrieval-augmentedgeneration.epub")`)
- Fetch dense + sparse rank lists per query
- LLM-score each chunk for relevance to the generated query
- Sweep sparse weight 0.0–2.0 in 0.25 steps
- Report avg_relevance@5 per weight

### Step 4: Output
JSON results + summary table:
```
Weight   Avg Relevance@5    vs weight=1.0
0.00              2.10              -0.30
0.25              2.25              -0.15  ← recommended
0.50              2.35               -0.05
0.75              2.40               +0.00
1.00              2.40               +0.00  ← best
1.25              2.38              -0.02
...
```

## File Changes

| File | Change |
|------|--------|
| `scripts/sweep_single_book.py` | **NEW** — single-book sweep, query generation from chunks |
| `docs/single-book-sweep-proposal.md` | **NEW** — this doc |

## Why This Matters
Current sweep uses generic queries across all 20 books. Optimal weight differs by content domain. Book-level sweep reveals:
- Does sparse help for technical manual content? (yes — exact term matching)
- Does sparse hurt for narrative content? (maybe — dense captures context better)
- Per-book tuning vs global default trade-off