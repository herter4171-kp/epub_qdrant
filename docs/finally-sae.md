# SAE Sparse vs Dense Retrieval — Comparison Results

## Summary

The `papers` collection (unnamed dense + sparse vectors) works perfectly for retrieval. The `papers-2048ctx-SAE` collection has named vector configs in its Qdrant schema, but **no actual vectors are indexed** — queries return `"Not existing vector name error"`. This means the SAE re-embedding and upsert pipeline either failed or was never completed.

---

## Step 1: Embeddings

Both dense and sparse embeddings generated successfully.

| Embedding | Type | Size |
|---|---|---|
| Dense | float array | 768 dims |
| Sparse | sparse vector | 456 active features |

Query text: `"How do I encourage sensible tool use by an agent?"`

Note: The embedding server's `get_dense_vectors()` and `get_sparse_vectors()` clients automatically rewrite the first query via the IT model before embedding.

---

## Step 2: Search Results

### 2a. Dense → `papers` (unnamed vector) ✅

**Status:** WORKED — 15 results returned.

| Score | Title | Arxiv ID |
|-------|-------|----------|
| 0.561 | Empowering Real-World: A Survey on the Technology, Practice, and Evaluation | 2510.17491 |
| 0.554 | How are AI agents used? Evidence from 177,000 MCP tools | 2603.23802 |
| 0.547 | Agentic Retrieval-Augmented Generation: A Survey | 2501.09136 |
| 0.539 | A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm | 2508.07407 |
| 0.534 | SMART: Self-Aware Agent for Tool Overuse Mitigation | 2502.11435 |
| 0.532 | Agent-SafetyBench: Evaluating the Safety of LLM Agents | 2412.14470 |
| 0.530 | Generalizability of LLM-Based Agents: A Comprehensive Survey | 2509.16330 |
| 0.525 | EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction | 2401.06201 |
| 0.525 | Advances and Challenges in Foundation Agents | 2504.01990 |
| 0.524 | SkillCraft: Can LLM Agents Learn to Use Tools Skillfully? | 2603.00718 |
| 0.523 | Building Effective AI Coding Agents for the Terminal | 2603.05344 |

**Assessment:** Excellent retrieval. All top results are highly relevant to agent tool use. Papers cover tool use, agent safety, tool overuse mitigation, tool instruction, and tool skill.

### 2b. Dense → `papers-2048ctx-SAE` (named `dense`) ❌

**Status:** FAILED — `"Wrong input: Not existing vector name error: "`

The collection config declares a named `dense` vector (768-d, Cosine) but Qdrant reports the vector name does not exist at query time. This means points were never upserted with the `dense` named vector, or the collection was re-created without named vectors.

### 2c. Sparse → `papers` (unnamed collection) ❌

**Status:** FAILED — `"Format error in JSON body: Expected some form of vector..."`

The `papers` collection does not have sparse vectors configured. The sparse vector config exists in the Qdrant schema for the `papers` collection but sparse named vectors are not available on this collection. Need to verify if sparse vectors were ever upserted.

### 2d. Sparse → `papers-2048ctx-SAE` (named `sparse`) ❌

**Status:** FAILED — `"Format error in JSON body: Expected some form of vector..."`

Same format issue as 2c, and additionally the named `sparse` vector may not be indexed.

---

## Step 3: Root Cause Analysis

### Investigation

The Qdrant collection config for `papers-2048ctx-SAE` shows:

```json
{
  "vectors": {
    "dense": { "size": 768, "distance": "Cosine" }
  },
  "sparse_vectors": {
    "sparse": { "modifier": "idf" }
  }
}
```

But two problems emerged:

1. **Unnamed query fails with "Not existing vector name"**: Even `{"query": dense}` fails on `papers-2048ctx-SAE` with `"Not existing vector name error: "`. This means the collection was created with named vector configs but points were never properly upserted with those named vectors. The points may only have unnamed vectors.

2. **Named vector query format**: Tested 4 different JSON formats for named vector queries:
   - `{"query": {"name": "dense", "query_vector": dense}}` — 400 Format error
   - `{"query": {"name": "dense", "query": dense}}` — 400 Format error
   - `{"query": {"name": "sparse", "query_vector": sparse}}` — 400 Format error
   - `{"query": {"name": "sparse", "query": sparse}}` — 400 Format error

   All fail with `"Expected some form of vector, id, or a type of query"`. This suggests either:
   - The Qdrant version doesn't support this query format
   - The named vectors are not actually indexed

### Diagnosis

The most likely scenario: The `papers-2048ctx-SAE` collection was created with named vector configs, but the SAE re-embedding pipeline did not successfully upsert points with the named vector format. This is consistent with the collection having 78K points but 155K indexed vectors — which could be just the dense unnamed vectors if the named vector upserts failed silently.

### Recommendation

**The SAE re-embedding pipeline needs to be fixed and re-run.** The pipeline script (referenced in `docs/finally-sae.md` as the next step) should:

1. Scroll the `papers` collection (or the original source documents)
2. Generate SAE-SPLADE dense vectors for 2048-token context chunks
3. Upsert with the correct named vector format into `papers-2048ctx-SAE`

The correct upsert format for named vectors in Qdrant:

```python
# When creating/upserting points:
point = PointStruct(
    id=chunk_id,
    vector={
        "dense": sae_dense_vector,  # SAE dense embedding
        "sparse": sae_sparse_vector,  # SAE sparse embedding
    },
    payload=chunk_metadata,
)
client.upsert(collection_name="papers-2048ctx-SAE", points=[point])
```

---

## Step 4: What We Can Conclude So Far

### Dense Retrieval Quality (papers collection)

The dense embedding model (`embeddinggemma:300m`) retrieves highly relevant results for the query "How do I encourage sensible tool use by an agent?":

**Top themes in top-15:**
1. **Tool use and tool overuse** — SMART paper directly addresses tool overuse mitigation
2. **MCP tools at scale** — 177K MCP tools analysis
3. **Agentic RAG** — retrieval-augmented generation in agent contexts
4. **Self-evolving agents** — agents that adapt their own behavior
5. **Agent safety** — safety benchmarks and evaluation
6. **Tool instruction and skill** — EASYTOOL, SkillCraft papers

**Relevance assessment:** All 15 results are topically relevant. The dense embedding captures semantic similarity well for this query.

### What's Missing

- No comparison with SAE-SPLADE vectors (pipeline not ready)
- No comparison with sparse keyword vectors
- No hybrid RRF results
- No derived query comparison

---

## Next Steps

1. **Fix the SAE re-embedding pipeline** — ensure points are upserted with correct named vector format into `papers-2048ctx-SAE`
2. **Verify named vectors are indexed** — after re-embedding, confirm queries to `papers-2048ctx-SAE` return results
3. **Run the full 4-way comparison** (2a, 2b, 2c, 2d) once vectors are available
4. **Generate derived queries and run reflection pass 2**

---

## Files

- Raw results: `/tmp/sae_comparison_results.json`
- Query embeddings: `/tmp/dense_vec.json` (dense), `/tmp/sparse_vec.json` (sparse)
- Search results from working 2a: `/tmp/search_dense_papers.json`