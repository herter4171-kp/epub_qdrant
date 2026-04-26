# Project Evolution

## The Starting Point

The project began with a straightforward problem: a collection of AI/ML books in EPUB format and no good way to query them. The first commit — "Working KB" — was a single-collection Qdrant ingestion pipeline with dense-only embeddings and a basic MCP server exposing a `query` tool. One book, one collection, one vector per chunk.

The architecture at that point was honest about its limitations. Dense cosine similarity works well when the query and the document share semantic space, but it has a known blind spot: exact terminology, proper nouns, and technical jargon that the embedding model may have seen rarely during training. A query for "MiniCOIL" or "RRF" would return semantically adjacent chunks rather than the chunks that actually contain those terms.

That limitation became the first design pressure.

---

## Phase 0: Normalization Before Optimization

Before touching the retrieval strategy, the team recognized a more fundamental problem: scores from different collections are not comparable. Dense cosine similarity from one collection and another are in different semantic spaces — a score of 0.82 from `books` means something different than 0.82 from `papers`. Cross-collection ranking was broken by construction.

The fix was z-score normalization per collection before merging results. No re-embedding, no new infrastructure — just a statistical correction applied at query time. This became Phase 0 in the roadmap, and it established a pattern that would repeat throughout the project: **measure first, then optimize**.

A 30-query evaluation harness was built alongside this fix. The LLM-as-judge approach — asking a language model to compare two result sets and declare a winner — meant there was no need for a human-labeled test set. Win rate against baseline became the primary metric. The target was >60%. Phase 0 cleared it.

---

## Phase 1: The Embedding Server and Sparse Vectors

The next architectural decision was the most consequential: adding sparse vectors alongside dense ones.

The original pipeline used Ollama for embeddings — a reasonable choice for local development but not designed for GPU-accelerated batch throughput. When the decision was made to add MiniCOIL sparse embeddings (Qdrant's IDF-weighted keyword model), running two separate embedding servers became unwieldy. The solution was a unified embedding server: a single FastAPI process hosting both `embeddinggemma-300m` (dense, 768-d) and `MiniCOIL-v1` (sparse) on the same GPU, exposed via `/embed_dense` and `/embed_sparse` endpoints.

The commit message for this was blunt: *"Combine embedding models into single API, ditch ollama, refactor tests and pipeline."* The refactor also introduced the `-named` collection convention: `books-named` and `papers-named` store both named vectors (`dense` and `sparse`) per point, while the original `books` and `papers` collections remain as dense-only baselines.

Reciprocal Rank Fusion (RRF) was chosen for score fusion. The formula is simple — `weight * 1/(k + rank + 1)` — and rank-based rather than score-based, which sidesteps the incompatibility between cosine similarity (bounded [-1,1]) and sparse dot products (unbounded). The initial A/B test showed hybrid search winning 80-85% of pairwise comparisons against dense-only on the 30-query set.

---

## The Semantic Chunking Interlude

Running the evaluation exposed a structural problem that had been invisible until the retrieval quality improved enough to notice it: most chunks had `section_title: "(no title)"`.

The EPUB parser was extracting headings by running a regex against already-cleaned text — after the HTML tags had been stripped. It never matched anything. Every book was effectively one flat list of paragraphs with no structural metadata.

This mattered because the MCP server's `get_context` tool navigates by section title. With no titles, context expansion was blind. It also meant the chunker was splitting on token count alone, potentially cutting across chapter boundaries.

The semantic chunking spec addressed this in three layers:

1. **Fix the parser** — extract headings from raw HTML before cleaning, making `section_title` actually useful
2. **Recursive structural splitting** — try heading boundaries first, fall back to paragraphs, then sentences; use `semchunk` for the implementation
3. **Embedding-based boundary detection** — for long flat sections, embed sentences, compute inter-sentence cosine similarity, and split at statistical outliers (95th percentile of cosine distance distribution)

The heading-as-context-bridge pattern emerged from this work: instead of token overlap at structural boundaries, prepend the heading text to each chunk body. The heading provides context without bleeding tokens across unrelated sections.

The result was a new collection, `books-semantic`, with 9,000 points — compared to 6,212 in the original `books` collection. More chunks, better boundaries, real section titles.

---

## The Framing Problem: Dense vs. Hybrid Was the Wrong Question

The blind A/B test was designed to answer "does hybrid beat dense?" It ran 30 samples, picked random passages from books common to both collections, generated queries, retrieved from both, had an LLM answer each, and had a judge pick a winner.

The problem with this framing surfaced during analysis: dense and sparse aren't competing strategies. They answer different aspects of the same question. Dense retrieval finds semantically similar passages — good for conceptual queries, paraphrases, and analogical reasoning. Sparse retrieval finds exact term matches — good for technical jargon, proper nouns, and queries where the user knows the specific vocabulary.

Treating them as A vs. B and picking a winner throws away information. With a 130K context window and a 35B model, the right move is to query both signals separately and give the answer LLM both result sets simultaneously, labeled by retrieval method, and let it fuse at the answer layer.

This reframing — captured in the `early-fusion-better-hybrid.md` steering file — drove the v3.0.0 redesign of the evaluation script. The A/B winner field was removed. `dense_answer` and `hybrid_answer` were replaced by a single `fused_answer`. The judge prompt was rewritten to score faithfulness on a 1-3 scale rather than declare a winner. The terminal summary changed from a win/loss table to a score distribution by query bucket (trivia / conceptual / operational).

The architecture became:

```
query
  ├── dense search  (sparse_weight=0)  → top-K semantic results
  └── sparse search (dense_weight=0)   → top-K keyword results
                    ↓
        Answer LLM sees both lists, labeled,
        rank positions only — scores stripped
                    ↓
              single fused answer
```

---

## The Signal Isolation Bug

Implementing the dual-signal architecture exposed a bug in `hybrid_search()` that had been invisible when both weights were non-zero.

When `dense_weight=0` was passed to isolate the sparse signal, both Qdrant queries still fired. The dense hits entered `rrf_scores` via `defaultdict` with a contribution of `0 * (1/(k+rank+1)) = 0.0` — technically correct, but those IDs were still candidates in the sorted output. More importantly, the fix confirmed a second observation: for queries where the exact terminology appears in the text (like "hybrid search dense sparse vector fusion RRF"), both signals return nearly identical top-K results regardless of weighting, because the dense embedding space and the sparse IDF space agree on what's relevant when the query vocabulary matches the document vocabulary exactly.

The fix was to skip the Qdrant query entirely when its weight is zero — no round-trip, no zero-score pollution. The divergence between signals only becomes visible on queries where they genuinely disagree: conceptual queries ("why do language models forget things they learned") where dense surfaces semantically related chunks and sparse surfaces chunks containing the exact phrase "catastrophic forgetting."¹

---

## The MinerU Integration: Papers Enter the Pipeline

The academic papers collection (`papers`, 90K+ points) had been embedded from raw PDF text — adequate but lossy. PDF text extraction drops tables, mangles equations, and loses figure captions. For a knowledge base meant to surface cutting-edge methodology, this was a meaningful quality gap.

MinerU addresses this by parsing PDFs into structured JSON with layout-aware extraction: text blocks, tables, equations, and figures are identified separately and can be filtered or weighted differently at ingestion time. The pipeline now runs PDFs through MinerU first, producing structured JSON, then ingests the JSON with the same semantic chunking pipeline used for EPUBs.

The `papers-semantic` collection (19,441 points) reflects this: more points than the raw `papers` collection because the structured extraction produces more granular, better-bounded chunks. The `papers-named` collection (90,256 points, matching the original) stores both named vectors for the full corpus.

---

## What the System Has Become

The current state of the project is a dual-signal retrieval platform with:

- **Two embedding models** running on a single GPU (embeddinggemma-300m dense + MiniCOIL sparse), served via a unified FastAPI server
- **Five Qdrant collections**: `books` (dense baseline), `books-semantic` (semantically chunked, named vectors), `papers` (dense baseline), `papers-semantic` (MinerU-parsed, named vectors), `papers-named` (full corpus, named vectors)
- **An MCP server** exposing `query`, `get_context`, `pick_random_chunk`, `list_books`, `list_sources`, and `list_collections` — all usable by any MCP-compatible client including Kiro itself
- **A fused retrieval evaluation harness** that samples random passages, generates queries, retrieves via both signals, generates a fused answer, and scores faithfulness with an LLM judge
- **Semantic chunking** with three-layer splitting (structural → embedding-based boundary detection → recursive token enforcement) and heading-as-context-bridge

The roadmap item that hasn't landed yet is LLM-driven metadata filter extraction: scroll-discover distinct field values on startup, map natural language to structured Qdrant filters, inject resolved metadata into retrieval prompts as scope context. The infrastructure for it exists; the extraction layer doesn't.

---

## The Self-Improvement Question

The project started as a tool for querying a knowledge base. It has become something more interesting: a knowledge base that contains the literature describing how to build better knowledge bases, queryable by the system building it.

The academic papers collection includes surveys on RAG architecture, chunking strategies, retrieval fusion, and agentic systems. The books collection includes practical guides on the same topics. The MCP server is available to Kiro during development sessions. The steering files (`role_objective.md`, `early-fusion-better-hybrid.md`, `constants-and-dry.md`) encode lessons learned directly into the agent's context.

The loop is not fully closed — the system doesn't yet autonomously propose and implement improvements based on retrieval results. But the components are in place. A session that starts with "query the papers collection for the current state of the art on sparse retrieval" and ends with a code change to `retriever.py` is already happening. The distance between "tool that answers questions" and "tool that improves itself by answering questions about itself" is shorter than it looks.

---

## User Testimonial

*The following is a first-person account of using the retrieval system during development.*

---

I started using the `books` collection — dense-only, 6,212 points — before `books-semantic` existed. The experience was fine for broad conceptual queries. Ask about "attention mechanisms" and you'd get relevant chunks from the right books. But it had a particular failure mode that became obvious fast: the top result was often the right *book* but the wrong *section*. The dense embedding would find the book that discusses the topic most, then surface whatever chunk from that book happened to score highest — which was sometimes the introduction, sometimes a summary, sometimes a tangentially related section that used similar vocabulary.

When `books-semantic` came online with named sparse and dense vectors, the difference was immediately noticeable on technical queries. Searching for "reciprocal rank fusion" in the dense-only collection returned chunks about "combining retrieval methods" and "ensemble approaches" — semantically adjacent but not the thing. The sparse signal in `books-semantic` returned the section titled "Fusion Techniques" from *Mastering RAG* that contains the actual RRF formula and code. That's the difference between a system that understands what you mean and a system that knows what you said.

The `pick_random_chunk` tool changed how I think about the collection. Instead of querying for something I already know is there, I can sample a random substantive passage and ask "what question would this answer?" That's the query generation loop the evaluation harness uses, and it surfaces content I wouldn't have thought to search for. The `has_heading_context` filter matters here — without it, you get stub chunks that are just section titles or three-sentence summaries that don't contain enough to generate a meaningful query from.

The honest limitation is that the scores from both signals are on incomparable scales. Dense cosine similarity sits around 0.5-0.9 for good matches. Sparse RRF scores are in the 0.01-0.03 range. You can't look at a score and know if a result is good — you have to read the text. The fused retrieval architecture handles this correctly by stripping scores before passing results to the answer LLM, but it means the raw search output requires some interpretation. That's a known tradeoff, not a bug.

---

## Footnotes

¹ **The signal isolation bug and the case for LLM evaluation.** During development of the dual-signal architecture, `hybrid_search()` was passing `dense_weight=0` to isolate the sparse signal but still firing the dense Qdrant query. The dense hits entered the RRF score map with zero contribution but remained as candidates. The bug was invisible on keyword-heavy queries (both signals agree) and only surfaced when testing with conceptual queries where the signals genuinely diverge. This is a good example of why LLM-as-judge evaluation is necessary: a unit test checking that `dense_weight=0` produces a non-empty result set would have passed. Only a judge comparing the *content* of the results — and noticing that "why do language models forget things they learned" returned a title page at rank 1 under dense-only — would catch the semantic failure. The fix (skip the Qdrant query entirely when weight is zero) was one line of logic, but finding it required the kind of qualitative result inspection that automated metrics miss.
