# Semantic Chunking: Problem Space and Path Forward

## Where We Are

Our current chunker (`src/ingestion/chunker.py`) uses fixed token-window splitting: 500 tokens per chunk, 100-token overlap, paragraph-aware but not meaning-aware. It splits on double newlines first, then falls back to sentence boundaries when a paragraph exceeds the window. This is the simplest viable approach — uniform chunks, predictable sizes, easy to reason about.

The EPUB parser (`src/ingestion/epub_parser.py`) extracts sections by matching `<h1>`–`<h3>` headings in the HTML. When no headings are found, the entire spine item becomes a single `(no title)` section. In practice, most of our Apress EPUBs produce sections titled `(no title)` because the heading structure doesn't match the regex pattern, or the content is flat XHTML without semantic headings.

This means our chunks carry almost no structural metadata. The `section_title` field is mostly useless, and `get_context` in the MCP server can't navigate by chapter because there are no chapters to navigate by.

## What the Books Say

Four books in our collection discuss chunking in depth. Here's what they converge on.

### The Chunking Taxonomy

The literature consistently identifies five strategies, ordered from simplest to most sophisticated:

1. **Fixed-length chunking** — split every N characters/tokens regardless of content. Simple, uniform, but breaks mid-sentence and loses semantic coherence. This is essentially what we do today.

2. **Sentence/paragraph-based chunking** — split at natural linguistic boundaries (periods, paragraph breaks). Preserves sentence integrity but doesn't consider whether adjacent sentences belong to the same topic.

3. **Recursive character splitting** — LangChain's `RecursiveCharacterTextSplitter`. Tries a hierarchy of separators in order: `\n## `, `\n### `, `\n\n`, `\n`, ` `, `""`. Falls through to finer separators only when coarser ones produce chunks that are too large. This is the de facto standard in LangChain-based RAG systems. Typical config: `chunk_size=1000, chunk_overlap=200`.

4. **Semantic chunking** — uses embeddings to detect topic boundaries. Embed consecutive sentences, compute cosine similarity between adjacent pairs, and split where similarity drops below a threshold. This is what LangChain's `SemanticChunker` does under the hood. The "Mastering RAG" book provides a `SemanticTextSplitter` class that wraps HuggingFace embeddings for this purpose.

5. **Document-structure-aware splitting** — split by the document's own structure (headings, sections, code blocks, tables). Requires parsing the format (HTML, Markdown, LaTeX) and using structural markers as primary split points. The books recommend this for academic papers (split by section/subsection), legal docs (by article/clause), and technical docs (by function/class).

### Key Insights from the Collection

**On chunk size**: "Mastering RAG" recommends 1000 characters with 200-character overlap as a starting point for `RecursiveCharacterTextSplitter`. "Mastering LangChain" notes that chunk size should balance context preservation against retrieval precision — too large and you get noise, too small and you fragment meaning. Our 500-token (~2000 char) window is actually in a reasonable range, but we're not respecting structural boundaries.

**On overlap**: Overlap exists to preserve context across chunk boundaries. "Mastering LangChain" describes the sliding window technique where the end of one chunk is repeated at the beginning of the next. Our 100-token overlap serves this purpose but is applied mechanically rather than at semantic boundaries.

**On semantic splitting**: "Mastering RAG" describes an embedding-based approach: embed sentences, measure cosine similarity between consecutive pairs, and split where the similarity drops. This identifies natural topic transitions. The book notes this "produces more meaningful chunks for your RAG system by keeping related concepts together, identifying natural topic boundaries, maintaining semantic coherence, and improving retrieval relevance."

**On document-specific strategies**: "Mastering RAG" explicitly recommends different splitting approaches per document type: "Academic papers might split by section, subsection, and paragraph. Code documentation might split by function, class, or module. Legal documents might split by article, clause, or provision." This is directly relevant — our EPUBs and PDFs have very different structures.

**On metadata preservation**: "Mastering RAG" emphasizes preserving metadata across chunks: "When splitting documents that contain important metadata (like author information, dates, or categories), you'll want to ensure this information is preserved across chunks." We do this for book-level metadata (publisher, ISBN) but lose section-level context because our section extraction is broken.

## The Gap

Our current approach has three concrete problems:

1. **No structural awareness**: We split on token count, not on document structure. A chunk can span the end of one chapter and the beginning of another. The recursive splitter approach (try headings first, fall back to paragraphs, then sentences) would be a direct improvement.

2. **Broken section extraction**: The EPUB parser's heading regex misses most section boundaries in our Apress books, producing `(no title)` for everything. This means we can't even do structure-aware splitting because we don't have the structure. Fixing the parser to extract headings from the raw HTML (before stripping tags) would give us the structural markers we need.

3. **No topic-boundary detection**: We don't use embeddings to find where the subject matter changes within a section. For long, flat sections (which is what we get when heading extraction fails), semantic chunking would identify natural breakpoints that our fixed-window approach misses.

## What We Could Build

A layered approach, ordered by dependency — each layer unlocks the next.

### Layer 1: Fix the EPUB parser (prerequisite for everything else)
Extract headings from raw HTML before cleaning. The current regex `<h([1-3])[^>]*>(.*?)</h\1>` runs against already-cleaned text where tags are stripped — it never matches. Fix: run heading extraction on the raw HTML bytes, then clean the content between headings. This alone makes `section_title` useful and gives the structural layer real boundaries to work with.

### Layer 2: Recursive splitting with structural separators
Replace the fixed-window chunker with a recursive approach. Separator hierarchy: `\n## ` → `\n### ` → `\n\n` → `\n` → `. ` → ` `. No new dependency needed — this is ~50 lines. Use `semchunk` (v4.0.0, MIT, pure Python, deps: `mpire[dill]` + `tqdm`) if we want the battle-tested version with overlap support and benchmarked 15% better RAG correctness than LangChain's recursive chunker.

### Layer 3: Embedding-based semantic splitting
For sections that are still too long after structural splitting, use our own embedding server to compute sentence-level embeddings and split at topic boundaries. Cost: one `/embed_dense` call per sentence at ingestion time — never at query time. Use percentile or IQR over the distribution of inter-sentence cosine similarities to find breakpoints (no fixed threshold). We already have the infrastructure; this is just a new function in `src/ingestion/`.

### Heading-as-context-bridge (concrete implementation decision)
At structural boundaries (headings), don't use token overlap. Instead:
1. Store the heading in `section_title` metadata on every chunk within that section
2. Prepend the heading text to the chunk body: `"## {heading}\n\n{chunk_text}"`

This gives the embedding model the section context without bleeding tokens across unrelated sections. Token overlap (20% ratio) is still used for within-section splits where there's no heading to serve as a bridge.

### For PDFs specifically
Academic papers have a known structure: abstract, introduction, related work, method, experiments, conclusion, references. A PDF-specific splitter could use regex patterns for section headers (common in arxiv papers) to split at those boundaries, then recursive splitting within each section. The abstract is typically one chunk; methodology and results need further splitting.

### Dependency decision
No LangChain. `semchunk` v4.0.0 for the structural/recursive layer, our embedding server for semantic boundary detection. This avoids the `langchain-core` dependency tree entirely while getting better benchmarked chunking quality.

## Research Answers

### Q1: Similarity threshold for topic boundary detection

The books do not give a specific cosine similarity cutoff for semantic chunking breakpoints. "Mastering RAG" describes the `SemanticTextSplitter` class that uses embeddings to "identify natural breakpoints where the semantic flow changes" but defers the threshold logic to the implementation. The closest guidance comes from the same book's evaluation chapter: "Scores above 0.8 indicate strong semantic alignment, while scores between 0.6 and 0.8 suggest moderate similarity. Scores below 0.4 typically indicate semantic divergence." These are for retrieval evaluation, not chunking, but they give a calibration range.

LangChain's `SemanticChunker` (in `langchain-experimental`) offers three threshold strategies: `percentile` (default), `standard_deviation`, and `interquartile`. Rather than a fixed cutoff, it computes the distribution of cosine distances between consecutive sentence groups and splits at statistical outliers. This is more robust than a hardcoded threshold.

Verdict: no magic number. Use a statistical method (percentile or IQR) over the distribution of inter-sentence similarities, not a fixed threshold.

### Q2: Handling long sections after structural splitting

The books unanimously recommend recursive fallback. "Generative AI Apps with LangChain and Python" describes `RecursiveCharacterTextSplitter`: "If any of the resulting chunks exceed the specified chunk_size, the splitter recursively applies the splitting process to those chunks. And this recursive splitting continues until all the chunks are within the desired chunk_size limit." "Mastering RAG" adds that for edge cases like "very long paragraphs or unusual formatting," you should implement "custom length functions for special content types" and "document-specific preprocessing steps." No book recommends truncation — they all treat it as information loss.

Verdict: recursive fallback with progressively finer separators. Never truncate.

### Q3: Chunking academic papers

"Mastering RAG" explicitly recommends treating paper sections as natural boundaries: "Academic papers might split by section, subsection, and paragraph." The book's code example shows a `create_smart_splitter` that handles a research paper with "Key Applications," "Methods and Results" as structural markers. The separator hierarchy for papers would be: section headers first (`\n## `, `\n### `), then paragraph breaks, then sentences.

However, the books don't recommend treating abstract/methodology/results as single monolithic chunks. They recommend splitting *at* those boundaries and then recursively splitting *within* each section if it exceeds the chunk size. The abstract is typically short enough to be one chunk; methodology and results sections usually need further splitting.

Verdict: use section headers as primary split points, then recursive splitting within each section. Don't try to keep an entire "Results" section as one chunk.

### Q4: Metadata preservation across chunks

"Mastering RAG" is the most explicit: the companion notebook shows how to "carry forward document-level metadata to all chunks" and "add chunk-specific metadata (like chunk numbers and positions)." The recommended metadata per chunk includes:
- Document-level: title, author, publisher, date, source_file, ISBN/arxiv_id
- Section-level: section_title, chapter_index, section_index
- Chunk-level: chunk_index, chunk_count, token_count, position within section

The book also recommends "handling documents with nested metadata structures" — meaning if you split a section into sub-chunks, each sub-chunk should carry the parent section's title.

Verdict: propagate all document-level and section-level metadata to every chunk. Add chunk-specific positional metadata. This is what we already do for document-level fields, but we're losing section-level context because of the `(no title)` problem.

### Q5: Overlap strategy at semantic/structural boundaries

"Mastering LangChain" describes overlapping chunks as a sliding window: "The sliding window technique creates chunks that overlap by a specified amount (e.g., 100 characters), ensuring that the end of one chunk is repeated at the beginning of the next chunk." "Mastering RAG" recommends "adjusting chunk overlap based on document structure" — meaning overlap should be configurable per document type, not a global constant.

The key insight from "Mastering RAG": "The 200-character overlap is our narrative bridge. It ensures that no critical information is lost between chunks." The standard recommendation is chunk_size=1000, chunk_overlap=200 (20% ratio). Our current 500-token chunks with 100-token overlap is also a 20% ratio, which aligns.

When splitting at structural boundaries (headings), overlap becomes less important because the heading itself serves as a context marker. The books don't explicitly say "skip overlap at heading boundaries," but the recursive splitter's behavior implies it: when you split at a `\n## ` separator, the heading text naturally provides context for the next chunk without needing token overlap.

Verdict: keep 20% overlap ratio for within-section splits. At structural boundaries (headings), overlap is less critical — the heading metadata serves as the context bridge instead.

### Q6: Dependencies and tooling from the internet

Three viable options, from heaviest to lightest:

**Option A: LangChain ecosystem** (heaviest, most featured)
- `langchain-text-splitters` v1.1.2 — `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`, `HTMLHeaderTextSplitter`. Requires `langchain-core`. MIT license.
- `langchain-experimental` v0.4.1 — `SemanticChunker` (embedding-based breakpoint detection with percentile/stddev/IQR thresholds). Requires `langchain-core` + `langchain-community`. MIT license.
- Pros: battle-tested, well-documented, the books all use it. Cons: pulls in the entire langchain-core dependency tree.

**Option B: semchunk** v4.0.0 (lightweight, production-proven)
- Pure Python, MIT license. Only deps: `mpire[dill]`, `tqdm`. Works with any tokenizer (tiktoken, transformers, custom callable).
- Hierarchical splitting algorithm: splits by newlines → tabs → whitespace → sentence terminators → clause separators → word joiners. Benchmarked at 15% better RAG correctness than LangChain's recursive chunker on Legal RAG QA.
- Supports overlap via `overlap` parameter (ratio or absolute tokens). Supports chunk offsets.
- Optional AI-powered chunking mode via Isaacus API (not needed for our use case).
- Cons: no built-in embedding-based semantic splitting (it's structural/heuristic only).

**Option C: semantic-text-splitter** v0.30.1 (Rust-backed, fastest)
- Python bindings over a Rust crate. MIT license. No required deps (tokenizer support optional via `tokenizers`).
- `TextSplitter` and `MarkdownSplitter` with Unicode-aware semantic levels (grapheme clusters → word boundaries → sentence boundaries → newline sequences → for Markdown: headings by level).
- Pros: fastest option, proper Unicode sentence boundary detection. Cons: no embedding-based splitting, no overlap support built-in.

**For our use case**, we already have our own embedding server and don't need LangChain's `SemanticChunker` to call OpenAI. The most practical path is:
- Use `semchunk` or `semantic-text-splitter` for the structural/recursive splitting layer
- Use our own embedding server for the optional semantic boundary detection layer (Layer 3 from the plan)
- Or: implement the recursive separator hierarchy ourselves (it's ~50 lines) and skip the dependency entirely

## References

All findings sourced from `books-fresh` collection:
- "Mastering Retrieval-Augmented Generation" (Apress, ISBN 979-8-8688-1808-0) — chapters on text splitting strategies, semantic splitting, recursive splitting, metadata preservation
- "Mastering LangChain" (Apress, ISBN 979-8-8688-1718-2) — chunking taxonomy, semantic vs fixed-length, overlap strategies
- "Building Applications with Large Language Models" (Apress, ISBN 979-8-8688-0569-1) — chunking strategy comparison, hybrid approaches
- "Generative AI Apps with LangChain and Python" (Apress, ISBN 979-8-8688-0882-1) — recursive splitting, Chunkviz visualization tool
- "The Practical Guide to Large Language Models" (Apress, ISBN 979-8-8688-2216-2) — cosine similarity for semantic comparison
- "Mastering Spring AI" (Apress, ISBN 979-8-8688-1001-5) — sliding window technique for context windows

Internet sources:
- [LangChain SemanticChunker API](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html) — breakpoint threshold types
- [semchunk on PyPI](https://pypi.org/project/semchunk/) v4.0.0 — hierarchical chunking benchmarks
- [semantic-text-splitter on PyPI](https://pypi.org/project/semantic-text-splitter/) v0.30.1 — Rust-backed Unicode-aware splitting
- [langchain-text-splitters on PyPI](https://pypi.org/project/langchain-text-splitters/) v1.1.2
