# First Self-Improve, Revision 2: Metadata-Aware Retrieval with MiniCOIL

> **Caveman mode on.** Big pic: our retriever dumb. Two collections (papers 90K, books 6K), different schemas. Metadata float in payload, never used. Vector = text only, no metadata signal. Retrieval = flat cosine, no hybrid, no cross-collection smarts. Fix: z-score first (immediate win), LLM-as-judge pairwise (no human labels needed), then MiniCOIL sparse vectors, hybrid search, per-collection chunking, LLM filter extraction with discovered taxonomy.

---

## 1. Executive Summary

We have two Qdrant collections with wildly different metadata schemas but zero coordination between what's stored and what's searched. Books have publisher, ISBN, language, chapter structure. Papers have arxiv_id, category, subcategory, publish_date. Neither set of metadata touches the embedding or the scoring.

**The fix, in phases of increasing impact and complexity:**

1. **Phase 0 — Z-score normalization** — fix cross-collection ranking immediately. No re-embedding needed. Scores from different collections live in different semantic spaces; z-score normalizes per-collection for fair cross-collection ranking.
2. **Phase 1 — LLM-as-judge pairwise pipeline** — set up evaluation using LLM judge to compare old vs new retrieval. No human-labeled test set needed. 30 random queries, pairwise comparison, win rate target >60%. Fast (~15 min of LLM calls).
3. **Phase 2 — MiniCOIL sparse re-embed + hybrid search** — store both dense (semantic) and sparse (keyword/metadata) vectors per point. ~96K points re-embed in minutes via ONNX/GPU. Fuse via Reciprocal Rank Fusion. Per-collection chunking (books = header-aware, papers = section-aware).
4. **Phase 3 — LLM-driven metadata filter extraction** — scroll-discover distinct values on startup, map natural language to structured Qdrant filters, inject resolved metadata into retrieval prompts as scope context.

**AWS Bedrock Knowledge Bases — trade-off analysis:**
Self-hosted Qdrant + MiniCOIL beats Bedrock on retrieval capability. Bedrock is simpler ops but locks out the key innovations this plan requires.

| | Self-hosted Qdrant | Bedrock KB + S3 |
|---|---|---|
| **Sparse vectors** | ✅ MiniCOIL hybrid search | ❌ No custom embeddings |
| **Cross-collection** | ✅ Full control, z-score normalization | ⚠️ Multi-collection possible, limited filters |
| **Chunking** | ✅ Custom per-collection | ❌ Bedrock-controlled |
| **Metadata filters** | ✅ Any FieldCondition | ⚠️ Exact-match only |
| **Cost** | GPU infra + maintenance | ~90% cheaper storage, pay-per-query |
| **Latency** | ~5ms GPU (local) | Higher (network hop to AWS) |
| **LLM** | Any (LiteLLM) | Bedrock models only |

**Bottom line**: Bedrock is a "move fast, manage nothing" choice. But for metadata-aware retrieval with sparse vectors, custom chunking, and cross-collection normalization — self-hosted Qdrant is the only way. Papers (90K) especially need the MiniCOIL sparse signal for arxiv_id and category matching.

---

## 2. Status Quo

### 2.1 Architecture

| Layer | Current State |
|---|---|
| **Embedding** | Ollama `embeddinggemma:300m` via `/api/embed` — plain text only |
| **Storage** | Qdrant, KEYWORD indexes on `source_file`, `book_title`, `section_title`, `publisher`, `language`, `isbn`, `arxiv_id`, `category`, `title` |
| **Retrieval** | Semantic cosine search → optional `FieldCondition` equality filter → group by section/book |
| **Cross-collection** | `search_collections()` concatenates results, sorts globally by score |
| **Filtering** | Manual `filter_by: Dict[str, str]` from caller — no LLM extraction |
| **Metrics** | None. Zero evaluation. |

### 2.2 Schema Disparity

| Field | Books (6,212 points) | Papers (90,256 points) |
|---|---|---|
| Identifier | `isbn`, `book_title` | `arxiv_id`, `title` |
| Authorship | — | `authors` (list) |
| Classification | `publisher`, `language` | `category`, `subcategory` |
| Temporal | — | `publish_date` |
| Structure | `chapter_index`, `section_index`, `chunk_index` | `chunk_index`, `chunk_count` |
| Content | `text`, `section_title` | `text`, `title` |
| Common | `doc_type` ("book"), `source_file`, `chunk_index`, `token_count` | `doc_type` ("paper"), `source_file`, `chunk_index`, `token_count` |

### 2.3 Core Problems

1. **Embedding ignores metadata** — text only gets embedded. Year, category, ISBN, chapter position float in payload but never influence the vector.
2. **Retrieval is flat cosine** — no hybrid (BM25/sparse), no field weighting, no cross-collection fusion.
3. **Filters are manual** — caller must pass `Dict[str, str]`. No LLM-driven extraction from natural language.
4. **Cross-collection merges blindly** — score 0.6 in books ≠ score 0.6 in papers. Different semantic spaces, no normalization.
5. **No metrics** — zero evaluation of any kind. Cannot measure improvement.

---

## 3. Phase 0: Z-Score Normalization for Cross-Collection Retrieval

**Immediate win, no re-embedding needed.** Cross-collection ranking is broken because scores from different collections live in different semantic spaces. A score of 0.55 in papers (90K points) ≠ 0.55 in books (6K points).

### 3.1 The Problem

Qdrant returns cosine similarity scores. With 90K points, average cosine tends to be lower than with 6K points. Global sort penalizes the larger collection unfairly.

### 3.2 Solution: Per-Collection Z-Score Normalization

```python
def search_collections(self, query: str, collections: List[str],
                       top_k: int = 30) -> List[ChunkResult]:
    """Search multiple collections with per-collection z-score normalization."""
    
    all_results = []
    
    for coll in collections:
        results = self.search(coll, query, k=top_k * 2)
        
        # Z-score normalization within collection
        scores = [r.score for r in results]
        mean_score = np.mean(scores)
        std_score = np.std(scores) + 1e-8  # avoid div-by-zero
        
        for r in results:
            normalized = (r.score - mean_score) / std_score
            all_results.append(ChunkResult(
                **r.__dict__,
                score=normalized,
                _collection=coll,
            ))
    
    # Metadata boost based on query terms
    q_lower = query.lower()
    for r in all_results:
        boost = self._compute_metadata_boost(r, q_lower)
        r.score += boost
    
    all_results.sort(key=lambda x: -x.score)
    return all_results[:top_k]

def _compute_metadata_boost(self, chunk: ChunkResult, q_lower: str) -> float:
    """Boost if query explicitly references chunk's metadata."""
    boost = 0.0
    if chunk.get("publisher") and chunk.publisher.lower() in q_lower:
        boost += 0.15
    if chunk.get("category") and chunk.category.lower() in q_lower:
        boost += 0.10
    if chunk.get("subcategory") and chunk.subcategory.lower() in q_lower:
        boost += 0.10
    if chunk.get("doc_type") and chunk.doc_type.lower() in q_lower:
        boost += 0.05
    return boost
```

**Result**: All scores on the same scale. Fair cross-collection ranking. Works with current embeddings immediately.

---

## 4. Phase 1: LLM-as-Judge Pairwise Evaluation Pipeline

No human-labeled test set. No Precision@K, Recall@K, or RAGAS. We use LLM-as-judge pairwise comparison — the most practical metric for a solo developer. Cheap, fast, no labels needed.

### 4.1 Method

1. Take 30 random queries from real usage
2. Run retrieval with **old method** (current flat cosine) and **new method** (after each phase change)
3. Feed both result sets to an LLM judge: "Given query Q and these two result lists (A and B), which is more relevant?"
4. Track win/loss/tie rate

**Why it works**: Doesn't need ground truth relevance labels. Captures the entire retrieval→generation pipeline. LLM judges correlate well with human judgment for pairwise comparison. Cost: ~15 minutes of LLM calls total.

```
Prompt template:
"Query: {query}
Result Set A (old method, top 5): {contexts_a}
Result Set B (new method, top 5): {contexts_b}

Which result set is more relevant to the query?
Respond: {"winner": "A"|"B"|"tie", "reason": "..."}"
```

**Metric**: Win rate for new method. Target: >60% win rate = meaningful improvement.

### 4.2 Evaluation Code

```python
import asyncio
from litellm import acompletion

async def judge_pairwise(llm_client, query: str, results_a: List, results_b: List) -> str:
    """Run pairwise comparison via LLM judge."""
    contexts_a = "\n".join(f"{i+1}. [{h.score:.3f}] {h.text[:200]}..." for i, h in enumerate(results_a[:5]))
    contexts_b = "\n".join(f"{i+1}. [{h.score:.3f}] {h.text[:200]}..." for i, h in enumerate(results_b[:5]))
    
    prompt = f"""Query: {query}

Result Set A (old method):
{contexts_a}

Result Set B (new method):
{contexts_b}

Which result set is more relevant to the query?
Respond as JSON: {{"winner": "A"|"B"|"tie", "reason": "..."}}"""

    resp = await llm_client.acomplete(prompt, model="ollama/nomic-embed-text")
    import json
    return json.loads(resp.choices[0].message.content)

async def evaluate_phases(query_set: List[str], phases: Dict[str, callable]) -> Dict:
    """Compare all phases pairwise against baseline.
    
    Args:
        query_set: 30 random queries
        phases: {"baseline": fn, "phase_0": fn, "phase_2": fn, ...}
    
    Returns:
        {"phase_0": {"wins": 18, "losses": 10, "ties": 2, "win_rate": 0.60}, ...}
    """
    results = {}
    baseline_key = "baseline"
    
    for phase_name, phase_fn in phases.items():
        if phase_name == baseline_key:
            continue
        
        wins = losses = ties = 0
        for query in query_set:
            baseline_hits = phases[baseline_key](query)
            phase_hits = phase_fn(query)
            judgment = await judge_pairwise(llm_client, query, baseline_hits, phase_hits)
            
            if judgment["winner"] == phase_name:
                wins += 1
            elif judgment["winner"] == baseline_key:
                losses += 1
            else:
                ties += 1
        
        total = wins + losses + ties
        results[phase_name] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / total if total > 0 else 0,
        }
    
    return results
```

### 4.3 Metric Tracking Summary

| Phase | Metric to Track |
|---|---|
| **Phase 0** (z-score) | Win rate vs. baseline (Phase 1) |
| **Phase 2** (hybrid) | Win rate vs. baseline |
| **Phase 3** (filter extraction) | Win rate vs. baseline |

---

## 5. Phase 2: MiniCOIL Sparse Embedding via ONNX/GPU

### 5.1 Why MiniCOIL + Dense (Approach B)

MiniCOIL is a lightweight sparse encoder (~30MB weights) that generates sparse vectors suitable for keyword/metadata matching. Combined with existing dense (semantic) vectors, this gives **hybrid search**: keyword precision + semantic recall.

| Aspect | MiniCOIL + Dense (Approach B) | Metadata-Prepended (Approach A) |
|---|---|---|
| Infra cost | Re-embed all points (ONNX/GPU) | Zero (but we're not using this) |
| Precision | High — exact keyword match via sparse vector | Low — semantic only |
| Metadata signal | Explicit in sparse vector space | Implicit in dense embedding |
| Qdrant support | Native (named vectors: dense + sparse) | No change needed |

### 5.2 ONNX Runtime on GPU — Fast Re-Embedding

**Current**: Ollama `/api/embed` — single/batch calls via HTTP. Took all night for ~96K points.

**New**: ONNX Runtime on GPU. Direct compute, no API overhead.

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class MiniCOILEmbedder:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load ONNX model on GPU
        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
    
    def embed_batch(self, texts: List[str], batch_size: int = 512) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                   max_length=512, return_tensors="np")
            # GPU inference: negligible latency per batch
            outputs = self.session.run(None, inputs)
            # outputs[0] = sparse attention weights (sparse vector)
            sparse_vec = outputs[0].squeeze(axis=0)  # (batch_size, seq_len)
            for vec in sparse_vec:
                # Convert to sparse format: {(token_id): weight, ...}
                sparse_indices = np.where(vec > 0)[0]
                sparse_values = vec[sparse_indices].tolist()
                results.append((sparse_indices.tolist(), sparse_values))
        return results
```

**Performance**:
- Batch size: 512-1024 on GPU
- Per-point overhead: <5ms (tokenize + GPU forward + sparse conversion)
- Total for 96K points: ~3-4 minutes
- vs. current Ollama: ~8-12 hours (all night)

### 5.3 Metadata-Informed Text for Dense Embedding

While re-embedding, also modify the dense embedding input to include metadata:

```python
def embed_with_metadata(chunk: Chunk) -> Tuple[List[float], List[float]]:
    """Return (dense_vector, sparse_vector) for a chunk."""
    
    # Dense: metadata-prepended text (for semantic signal)
    metadata_prefix = _build_metadata_prefix(chunk)
    dense_text = f"{metadata_prefix} {chunk.text}"
    dense_vector = ollama_embedder.embed_single(dense_text)  # or switch ONNX for dense too
    
    # Sparse: MiniCOIL on raw text (or metadata+text for keyword signal)
    sparse_input = f"{metadata_prefix} {chunk.text}"
    sparse_vector = mincoil_embedder.embed_batch([sparse_input])[0]
    
    return dense_vector, sparse_vector

def _build_metadata_prefix(chunk: Chunk) -> str:
    """Build structured metadata prefix for embedding."""
    parts = []
    if chunk.doc_type == "book":
        if chunk.publisher: parts.append(f"publisher:{chunk.publisher}")
        if chunk.language: parts.append(f"lang:{chunk.language}")
        if chunk.isbn: parts.append(f"isbn:{chunk.isbn}")
        if chunk.book_title: parts.append(f"title:{chunk.book_title}")
    elif chunk.doc_type == "paper":
        if chunk.category: parts.append(f"cat:{chunk.category}")
        if chunk.subcategory: parts.append(f"subcat:{chunk.subcategory}")
        if chunk.arxiv_id: parts.append(f"arxiv:{chunk.arxiv_id}")
        if chunk.publish_date: parts.append(f"date:{chunk.publish_date}")
        if chunk.authors: parts.append(f"authors:{','.join(chunk.authors)}")
    return " ".join(parts)
```

### 5.4 Qdrant Payload Update

After re-embedding, each point stores both vectors:

```python
point = PointStruct(
    id=chunk_id,
    vector={
        "dense": dense_vector,      # 768-d semantic
        "sparse": sparse_vector,    # sparse: (indices, values)
    },
    payload=chunk.metadata,  # all existing metadata intact
)
```

---

## 6. Hybrid Search with Dense + Sparse (RRF)

Qdrant supports named vectors. We store `dense` and `sparse` as named vectors per point.

### 6.1 Query-Time Fusion

```python
def hybrid_search(self, collection: str, query: str, 
                  query_dense: List[float], query_sparse: Dict,
                  k: int = 20, filter: Optional[Filter] = None) -> List[ChunkResult]:
    """Search with both dense and sparse vectors, fuse via RRF."""
    
    # Dense search
    dense_hits = self.client.search(
        collection_name=collection,
        query_vector=("dense", query_dense),
        limit=k * 2,
        query_filter=filter,
    )
    
    # Sparse search
    sparse_hits = self.client.search(
        collection_name=collection,
        query_vector=("sparse", query_sparse),
        limit=k * 2,
        query_filter=filter,
    )
    
    # Reciprocal Rank Fusion
    rrf_scores = defaultdict(float)
    k_rrf = 60  # standard RRF constant
    
    for rank, hit in enumerate(dense_hits):
        rrf_scores[hit.id] += 1.0 / (k_rrf + rank + 1)
    for rank, hit in enumerate(sparse_hits):
        rrf_scores[hit.id] += 1.0 / (k_rrf + rank + 1)
    
    # Merge and reattach payload
    results = []
    for point_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]:
        hit = next(h for h in dense_hits + sparse_hits if h.id == point_id)
        results.append(ChunkResult(
            text=hit.payload.get("text", ""),
            score=rrf_score,
            **hit.payload,
        ))
    return results
```

### 6.2 Why RRF Over Weighted Averaging

| Method | Problem |
|---|---|
| Weighted avg of scores | Dense and sparse live in different similarity spaces — scores are incomparable |
| RRF | Rank-based, no calibration needed. Works across different vector types |
| Bilingual LM reranking | Overkill for initial retrieval. Use as Phase 4+ refinement |

---

## 7. Chunking Strategy per Collection

Chunking strategy has measurable impact on retrieval quality. One-size-fits-all is a mistake.

### 7.1 Problem: Current Chunking Is Collection-Agnostic

Both collections likely use the same chunking logic, but books and papers have fundamentally different structures:

- **Books**: Hierarchical (chapter → section → subsection → paragraph), variable depth, prose-heavy
- **Papers**: Rigid (Abstract, Intro, Methods, Results, Discussion, Limitations, Conclusion), formula-heavy, dense

### 7.2 Books: Header-Aware Recursive Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

book_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,          # Target ~800 tokens per chunk
    chunk_overlap=150,       # 150-token overlap for context continuity
    separators=[
        "\n## ",             # Chapter headings
        "\n### ",            # Section headings
        "\n#### ",           # Subsection headings
        "\n\n",              # Paragraph breaks
        "\n",                # Line breaks
        " ",                 # Word boundaries
    ],
)
```

**Why**: Preserves chapter/section boundaries. A chunk won't straddle two topics. Overlap catches info that spans boundaries.

**Target**: 800-token chunks, 150-token overlap. Adjust based on LLM-as-judge win rate:
- Win rate drops → chunks degraded retrieval, adjust size up or down by 200 tokens |

### 7.3 Papers: Section-Aware Splitting

```python
# Pseudocode for paper chunking
PAPER_SECTIONS = [
    "abstract", "introduction", "introd.",
    "related work", "background", "literature",
    "methods", "methodology", "method",
    "experiments", "evaluation", "results",
    "discussion", "conclusion",
    "limitations", "future work", "references",
]

def chunk_papers(paper_text: str) -> List[Chunk]:
    sections = split_by_section_headings(paper_text, PAPER_SECTIONS)
    chunks = []
    for section_name, section_text in sections:
        # Split each section into 500-1000 token chunks
        # but respect sub-bounds (sentence/paragraph)
        section_chunks = semantic_chunk(section_text, size=800, overlap=100)
        for sc in section_chunks:
            chunks.append(Chunk(
                text=sc.text,
                section_name=section_name,   # metadata: which section
                arxiv_id=metadata.arxiv_id,
                # ... other metadata
            ))
    return chunks
```

**Why**: Papers have rigid structure. The Methods section of a paper is semantically different from Results. Chunking by section boundary preserves this signal. When combined with MiniCOIL sparse vectors, the section name becomes a keyword that helps exact-match queries.

### 7.4 Chunking Metrics to Watch

| Metric | Signal | Action |
|---|---|---|
| LLM-as-judge win rate drops | Chunks degraded retrieval quality | Re-evaluate chunk size/boundaries |

---

## 8. Phase 3: LLM-Driven Metadata Filter Extraction

**The full pipeline: discover taxonomy → parse query → build filter → inject into retrieval.**

### 8.1 Step 1: Discover Distinct Values via Qdrant Scroll

On startup, scroll each collection to discover the actual taxonomy of metadata values. No hardcoded schemas. No separate catalog to keep in sync.

```python
class MetadataCatalog:
    """Discovers distinct metadata values from Qdrant via scroll on startup."""
    
    def __init__(self, qdrant_client, collections: List[str]):
        self.catalog: Dict[str, Dict[str, Set[str]]] = {}
        self._discover(collections)
    
    def _discover(self, collections: List[str]):
        for collection in collections:
            self.catalog[collection] = {}
            seen_values: Dict[str, Set[str]] = {}
            
            # Scroll with pagination — 1000 points per collection is plenty
            offset = None
            for _ in range(10):  # max 10 pages = 10,000 points
                records, next_offset = self.client.scroll(
                    collection_name=collection,
                    limit=1000,
                    offset=offset,
                    with_payload=["*"],
                    with_vectors=False,
                )
                if not records:
                    break
                
                for record in records:
                    payload = record.payload
                    for field in self.KNOWN_METADATA_FIELDS:
                        value = payload.get(field)
                        if value:
                            if field not in seen_values:
                                seen_values[field] = set()
                            if isinstance(value, list):
                                seen_values[field].update(str(v) for v in value)
                            else:
                                seen_values[field].add(str(value))
                
                if next_offset is None:
                    break
                offset = next_offset
            
            self.catalog[collection] = seen_values
            logger.info(f"Discovered taxonomy for {collection}: {len(seen_values)} fields, "
                        f"{sum(len(v) for v in seen_values.values())} total values")

# Known fields per collection type
KNOWN_METADATA_FIELDS = {
    "papers": ["publisher", "category", "subcategory", "doc_type"],
    "books": ["publisher", "language", "doc_type"],
}
```

**Result**: In memory taxonomy like `{"publisher": {"Apress", "O'Reilly", ...}, "category": {"application-papers", "capability-papers", ...}}`. Zero sync overhead. Re-ingest → re-scan → taxonomy refreshes automatically.

### 8.2 Step 2: LLM-Mapped Structured Filter Extraction

```python
class MetadataFilterExtractor:
    """Extract structured metadata filters from natural language queries.
    
    Uses scroll-discovered taxonomy for value-matching to avoid LLM hallucination.
    """
    
    def __init__(self, llm_client: LLMClient, metadata_catalog: MetadataCatalog):
        self.llm = llm_client
        self.catalog = metadata_catalog
    
    def extract(self, query: str, collection: str) -> Optional[Filter]:
        """Parse query → structured Qdrant Filter using discovered taxonomy."""
        
        # Build taxonomy context for the LLM
        taxonomy = self.catalog.catalog.get(collection, {})
        fields_context = []
        for field, values in taxonomy.items():
            sample_values = sorted(values)[:10]  # limit to avoid context bloat
            fields_context.append(f"{field}: {sample_values}")
            if len(values) > 10:
                fields_context.append(f"  ... and {len(values) - 10} more")
        
        prompt = f"""Extract metadata filters from this query.

Known metadata values for this collection:
{chr(10).join(fields_context)}

Query: "{query}"

Return ONLY valid JSON:
{{
  "must_match": [{{"field": "...", "value": "..."}}],
  "must_not": [{{"field": "...", "value": "..."}}],
  "should_match": [{{"field": "...", "value": "..."}}]
}}

Rules:
- Match query terms to known values from the taxonomy above
- Use exact casing from the taxonomy (e.g., "Apress" not "apress")
- Use "must_not" for exclusions ("except", "not", "without")
- Use "should_match" for soft preferences ("recent", "preferably")
- Use "must" for explicit mentions ("from Apress", "Apress books")
- Only include fields mentioned or strongly implied
- If no match found for a field, omit it

Response:"""
        
        response = self.llm.complete(prompt)
        parsed = json.loads(response)
        return self._json_to_qdrant_filter(parsed, taxonomy)
    
    def _json_to_qdrant_filter(self, parsed: Dict, taxonomy: Dict) -> Filter:
        """Convert parsed JSON to Qdrant Filter."""
        conditions = []
        
        for condition_type in ["must_match", "must_not", "should_match"]:
            for item in parsed.get(condition_type, []):
                field = item["field"]
                value = item["value"]
                
                # Normalize: case-insensitive exact match
                values_set = taxonomy.get(field, set())
                # Find matching known value (case-insensitive)
                matched_value = None
                for known in values_set:
                    if known.lower() == value.lower():
                        matched_value = known
                        break
                
                # Use matched value if found, else use raw value
                actual_value = matched_value or value
                
                match_condition = MatchValue(value=actual_value) if isinstance(actual_value, str) else MatchAny(values=[actual_value])
                
                if condition_type == "must_match":
                    conditions.append(FieldCondition(key=field, match=match_condition))
                elif condition_type == "must_not":
                    conditions.append(FieldCondition(key=field, match=match_condition, negation=True))
                elif condition_type == "should_match":
                    conditions.append(FieldCondition(key=field, match=match_condition))
        
        if not conditions:
            return None
        
        return Filter(must=conditions)
```

### 8.3 Step 3: Inject Resolved Metadata into Retrieval Prompts

When the LLM generates the answer (via `agent-lookup`), inject the resolved metadata values as scope context. This completes the metadata loop: the LLM knows *what scope* was searched.

```python
def retrieve_with_context(self, query: str, collection: str) -> List[ChunkResult]:
    """Retrieve with metadata filter extraction and scope-injected retrieval."""
    
    # Step 1: Extract structured filter from query
    qdrant_filter = self.filter_extractor.extract(query, collection)
    
    # Step 2: Get the resolved values for context injection
    resolved = self._get_resolved_filter(query, collection)
    scope_context = self._build_scope_context(resolved)
    
    # Step 3: Build retrieval prompt with scope context
    enriched_query = f"{scope_context}: {query}"
    
    # Step 4: Perform vector search with filter
    results = self.search(collection, enriched_query, filter=qdrant_filter)
    
    # Step 5: Pass enriched results to agent-lookup
    return results

def _get_resolved_filter(self, query: str, collection: str) -> Dict:
    """Get the resolved filter values (for context injection)."""
    # Same logic as extract(), but returns the resolved values dict
    # rather than a Qdrant Filter object
    ...

def _build_scope_context(self, resolved: Dict) -> str:
    """Build scope context string for the LLM retrieval prompt.
    
    Example: "searching within Apress publications for agentic AI patterns"
    """
    if not resolved:
        return ""
    
    scopes = []
    if "publisher" in resolved:
        scopes.append(f"{resolved['publisher']} publications")
    if "category" in resolved:
        scopes.append(f"{resolved['category']} papers")
    if "subcategory" in resolved:
        scopes.append(f"{resolved['subcategory']}")
    
    if scopes:
        return f"searching within {' and '.join(scopes)}"
    return ""
```

**Example flow**:
1. Query: "Apress books on agentic AI"
2. Filter extracted: `must_match: [{field: "publisher", value: "Apress"}]`
3. Scope context: "searching within Apress publications"
4. Enriched query: "searching within Apress publications: Apress books on agentic AI"
5. agent-lookup receives enriched query → retrieves context from Apress books only
6. Answer respects the scope constraint

### 8.4 LLM Architecture (Single/Multi-LLM)

LiteLLM handles both:

```python
# Single instance (Ollama)
llm = LiteLLM(model="ollama/nomic-embed-text")

# Commercial (scaling not a problem)
llm = LiteLLM(model="gpt-4o")
llm = LiteLLM(model="anthropic/claude-3.5-sonnet")

# Multi-LLM consensus (broadcast to all, take majority vote)
results = llm.broadcast(prompt)  # {model_name: response}
```

---

## 9. Implementation Priority

### Phase 0: Z-Score Normalization (immediate, no re-embedding)

| Step | Action | Effort |
|---|---|---|
| 0.1 | Implement z-score normalization in `search_collections()` | 0.5 day |
| 0.2 | Add metadata boost for explicit query mentions | 0.25 day |

**Total**: ~0.75 days. Immediate cross-collection improvement.

### Phase 1: LLM-as-Judge Pairwise Evaluation (no human labels needed)

| Step | Action | Effort |
|---|---|---|
| 1.1 | Collect 30 random queries from real usage | 0.5 day |
| 1.2 | Write `judge_pairwise` and `evaluate_phases` functions | 1 day |
| 1.3 | Run baseline evaluation (old method vs. Phase 0 z-score) | 0.25 day |

### Phase 2: MiniCOIL + Hybrid Search + Chunking

| Step | Action | Effort |
|---|---|---|
| 2.1 | Install `onnxruntime-gpu`, `transformers`, `microsoft/miniCOIL` | 0.5 day |
| 2.2 | Write `MiniCOILEmbedder` class (ONNX/GPU, batch 512) | 1 day |
| 2.3 | Write metadata-informed text builder (`_build_metadata_prefix`) | 0.5 day |
| 2.4 | Rewrite chunking: books = header-aware, papers = section-aware | 1 day |
| 2.5 | Re-run embedding pipeline for both collections | 0.5 day (3-4 min re-embed) |
| 2.6 | Update Qdrant schema: add `sparse` named vector | 0.5 day |
| 2.7 | Write `hybrid_search()` with RRF fusion | 1 day |
| 2.8 | Run LLM-as-judge pairwise vs. baseline | 0.25 day |

**Total**: ~5.5 days of dev, ~3-4 min per full re-embed.

### Phase 3: LLM Filter Extraction

| Step | Action | Effort |
|---|---|---|
| 3.1 | Write `MetadataCatalog` (scroll-based taxonomy discovery) | 1 day |
| 3.2 | Write `MetadataFilterExtractor` with LiteLLM | 1 day |
| 3.3 | Add scope context injection into retrieval prompts | 0.5 day |
| 3.4 | Wire filter extraction into MCP server search | 0.5 day |
| 3.5 | Run LLM-as-judge pairwise vs. baseline | 0.25 day |

### Phase 4: Polish

| Step | Action | Effort |
|---|---|---|
| 4.1 | Fine-tune chunk sizes based on pairwise win rates | 0.5 day |

---

## 10. References

| Source | Key Concept |
|---|---|
| [1] Mastering Retrieval-Augmented Generation | Precision@K, Recall@K, NDCG@K, RAGAS metrics, Context Precision/Recall, chunking strategies (header-aware, semantic, sentence-based) |
| [2] Building Applications with Large Language Models | Chunk size trade-offs: too big = context overload, too small = hallucination risk. Per-document chunking strategies. |
| [3] Multi-Dimensional Knowledge Profiling (arXiv:2601.15170) | Two-level retrieval: metadata filter → weighted semantic search with LLM-assigned field weights |
| [4] Deep Research Survey (arXiv:2512.02038) | Hybrid search: keyword + dense + visual retrieval, RRF fusion |
| [5] Qdrant Hybrid Search Tutorial | Dense + sparse vectors, RRF fusion in Qdrant |
| [6] LangChain SelfQueryRetriever | LLM-driven filter extraction pattern |
| [7] MiniCOIL (microsoft/miniCOIL) | Lightweight sparse encoder, ONNX-compatible, ~30MB |
| [8] Microsoft: Boost RAG with Metadata Filters | Tag-based filtering alongside vector search |