# Hybrid Search vs Dense Search: Evaluation Plan

> **Goal**: Compare dense-only (`papers`/`books`) vs hybrid (`papers-named`/`books-named`) retrieval across 5 query types × 5 queries = 25 total. Stratified LLM-as-judge pairwise comparison. Both collections queried for every query. Single LLM instance, sequential execution.

---

## 1. Background

### 1.1 Methods Compared

| Method | Collections | Vectors | Fusion |
|--------|------------|---------|--------|
| **Dense** | `books`, `papers` | Semantic (embeddinggemma) | z-score normalization + metadata boost |
| **Hybrid** | `books-named`, `papers-named` | Semantic (dense) + keyword (sparse MiniCOIL) | Reciprocal Rank Fusion (RRF, k=60) |

### 1.2 Why Stratify by Query Type?

Literature consistently shows that **no single aggregate number captures retrieval quality**. Dense and sparse methods excel on different query types:

| Query Type | Dense | Sparse | Hybrid |
|-----------|-------|--------|--------|
| Exact match (paper title, method name) | ❌ | ✅✅ | ✅✅ |
| Conceptual (architecture, design pattern) | ✅✅ | ⚠️ | ✅ |
| Comparison (A vs B) | ✅ | ⚠️ | ✅ |
| Procedure (how to implement) | ✅ | ✅ | ✅✅ |
| Meta/scope (filter by category, publisher) | ⚠️ | ✅✅ | ✅✅ |

---

## 2. Query Set: 25 Stratified Queries

### 2.1 Query Type 1: Exact Method / Paper Name (5 queries)

Pure keyword/symbol matching. These query for specific method names, paper titles, or framework names that sparse vectors should catch perfectly. Dense embeddings treat specific acronyms as noise.

| ID | Query | Rationale |
|----|-------|-----------|
| E1-1 | "What is the ReAct framework for agent reasoning loops" | Specific method name "ReAct" — sparse hit |
| E1-2 | "Show me the LLaVA-CoT step-by-step reasoning approach" | "LLaVA-CoT" paper title — sparse hit |
| E1-3 | "What does tree of thought do for LLM reasoning" | "tree of thought" specific technique name |
| E1-4 | "How does reflexion work as a self-reflection agent pattern" | "reflexion" specific method name |
| E1-5 | "What is tool alignment in agentic AI systems" | "tool alignment" — specific term from corpus |

**Expected**: Hybrid wins on most of these. Dense may partially match on conceptual queries but will miss exact method names.

### 2.2 Query Type 2: Conceptual Architecture (5 queries)

Pure semantic queries. These ask about design principles, patterns, and system-level reasoning. Dense embeddings should excel here.

| ID | Query | Rationale |
|----|-------|-----------|
| C2-1 | "How should an enterprise agent handle permissions approvals and audit logs" | Broad architectural pattern |
| C2-2 | "What are the main failure modes of web agents in realistic browsing" | Conceptual analysis question |
| C2-3 | "How do self-evolving agents accumulate reusable skills without unstable behavior" | Self-evolution concept |
| C2-4 | "What safeguards are needed for agents that browse run code and call external APIs" | Safety architecture pattern |
| C2-5 | "How should sub-agent creation work in orchestration to prevent sprawl" | Orchestration design pattern |

**Expected**: Dense and hybrid tied or dense slightly favored. Sparse adds little value for pure conceptual questions.

### 2.3 Query Type 3: Comparison / Tradeoff (5 queries)

These ask for comparative analysis — A vs B, tradeoffs, when to use one over another. Both dense (semantic comparison) and sparse (specific terms) contribute.

| ID | Query | Rationale |
|----|-------|-----------|
| C3-1 | "What are the tradeoffs between single AI scientist agent and multi-agent discovery pipeline" | Explicit comparison, both terms in corpus |
| C3-2 | "Compare debate vs consensus patterns for multi-agent problem solving" | Two specific pattern names + comparison |
| C3-3 | "What is better for agent memory short-term working memory vs long-term user memory" | Comparison of two memory designs |
| C3-4 | "When should I use chain of thought vs tree of thought for agent reasoning" | Two specific methods compared |
| C3-5 | "Single agent vs multi-agent for scientific discovery what are the tradeoffs" | Multi-agent vs single agent comparison |

**Expected**: Hybrid may edge out dense. The specific method names (chain of thought, tree of thought) benefit sparse; the comparison structure benefits dense.

### 2.4 Query Type 4: Procedure / Implementation (5 queries)

"How to" questions about implementing specific agent capabilities. These sit in the overlap zone — dense captures the procedure intent, sparse catches the specific technique.

| ID | Query | Rationale |
|----|-------|-----------|
| P4-1 | "How to build an agent that decides when to use a tool versus answering directly" | ReAct-like decision loop implementation |
| P4-2 | "How to give an agent reasoning structure without brittle chain-of-thought templates" | Implementation with specific constraint |
| P4-3 | "How to make a coding agent ask clarifying questions only when needed" | Specific behavioral pattern implementation |
| P4-4 | "How to design memory that doesn't become a junk drawer of redundant facts" | Practical implementation problem |
| P4-5 | "How to build an API agent that discovers schema and recovers from malformed responses" | Specific architecture with error handling |

**Expected**: Hybrid slightly favored. The specific technique names (chain-of-thought) help sparse; the implementation intent helps dense.

### 2.5 Query Type 5: Meta / Scope Filtering (5 queries)

Queries that implicitly or explicitly target specific subsets of the corpus — by category, publisher, or doc type. Sparse vectors should excel at these because category names, publisher names, and doc types are exact tokens.

| ID | Query | Rationale |
|----|-------|-----------|
| M5-1 | "Show me papers about agentic AI simulation from application category" | Targets specific category |
| M5-2 | "What does Apress publish on enterprise AI applications" | Targets specific publisher |
| M5-3 | "Find research on deep research agents and their verification methods" | Specific paper cluster |
| M5-4 | "What capability papers discuss memory and self-evolution" | Targets capability-papers category |
| M5-5 | "Apress books about RAG evaluation and production monitoring" | Publisher + topic filter |

**Expected**: Hybrid strongly favored. Publisher names (Apress), category names (application-papers, capability-papers), and specific terms are exact sparse matches. Dense alone will spread results across unrelated topics.

---

## 3. Evaluation Methodology

### 3.1 LLM-as-Judge Pairwise (Per Literature)

**From**: DeepSynth-Eval, DeepResearchBench, AI Search Paradigm papers

For each query, run both methods and present both result sets to an LLM judge:

```
Result Set A (baseline — dense + z-score) vs Result Set B (hybrid — dense + sparse + RRF)

Judge decides: winner = "A" | "B" | "tie"
Metric: Normalized Win Rate = (#Win - #Lose) / (#Win + #Tie + #Lose)
```

### 3.2 Per-Stratum Reporting

Results are NOT aggregated into a single number. Instead, reported **per query type**:

| Query Type | Hybrid Wins | Dense Wins | Ties | Hybrid Win Rate |
|-----------|------------|-----------|------|-----------------|
| Exact Match (E1) | ? | ? | ? | ? |
| Conceptual (C2) | ? | ? | ? | ? |
| Comparison (C3) | ? | ? | ? | ? |
| Procedure (P4) | ? | ? | ? | ? |
| Meta/Scope (M5) | ? | ? | ? | ? |
| **Total** | **?** | **?** | **?** | **?** |

### 3.3 Additional Metrics Per Query

- **top5_avg_score**: Average score of top 5 results (normalized for dense, RRF-fused for hybrid)
- **cross_collection_ratio**: How balanced the result set is between books and papers
- **judge_reason**: LLM's explanation for the judgment

---

## 4. Implementation

### 4.1 New Evaluation Script

Create `scripts/evaluate_hybrid_vs_dense.py` — a focused evaluation that:

1. Uses the 25 queries from Section 2
2. Runs both methods sequentially (no concurrency)
3. Stores results in `results/hybrid_vs_dense.json`
4. Outputs per-stratum and aggregate tables

### 4.2 Execution Plan

| Step | Command | Time |
|------|---------|------|
| 1. Run dense-only retrieval (25 queries) | `python scripts/evaluate_hybrid_vs_dense.py --method dense` | ~30s |
| 2. Run hybrid retrieval (25 queries) | `python scripts/evaluate_hybrid_vs_dense.py --method hybrid` | ~45s |
| 3. Run LLM judge (50 comparisons) | `python scripts/evaluate_hybrid_vs_dense.py --judge` | ~15 min |
| 4. Generate report | automatic | <1s |

**Total**: ~16 minutes. Single LLM instance, sequential.

### 4.3 Output Format

```json
{
  "version": "2.0",
  "model": "openai/qwen36",
  "evaluated_at": "2026-04-23T...",
  "query_types": {
    "exact_match": {
      "hybrid_wins": 4,
      "dense_wins": 0,
      "ties": 1,
      "win_rate": 0.80,
      "queries": { ... }
    },
    "conceptual": { ... },
    "comparison": { ... },
    "procedure": { ... },
    "meta_scope": { ... }
  },
  "aggregate": {
    "hybrid_wins": 18,
    "dense_wins": 5,
    "ties": 2,
    "win_rate": 0.72
  }
}
```

---

## 5. What This Answers

| Question | How We Know |
|----------|------------|
| Does hybrid improve retrieval on agentic AI corpus? | Aggregate win rate |
| Where does hybrid help most? | Per-stratum win rates |
| Where does hybrid NOT help? | Per-stratum win rates (expect tied or dense-favored on conceptual) |
| Is the RRF fusion actually adding signal? | Comparison vs procedure types where both dense and sparse contribute |
| Does sparse help scope-filtering queries? | Meta/scope stratum win rate (expect highest gain) |

---

## 6. Single-GPU Constraint

- **No concurrent LLM calls** — all queries run sequentially
- **Single LLM instance** — one `LLMClient` reused across all judge calls
- **No concurrent retriever calls** — each query runs one at a time
- **No re-embedding** — uses existing `-named` collections on Qdrant
- **MiniCOIL server already running** on GPU box (192.168.68.75:9000)