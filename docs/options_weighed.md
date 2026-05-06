# Options Weighed: SAE-SPLADE Knowledge Retrieval Audit

## The Question

One prompt, three sources + web. Same query to all:

> "SAE-SPLADE sparse autoencoder embedding replace backbone vocabulary with latent semantic concepts retrieval efficiency polysemicity synonymy SPLADE"

## The Results

| Source | Status | Score | Verdict |
|--------|--------|-------|---------|
| **Bedrock KB** | Empty — no knowledge bases configured | N/A | Dead store. Needs configuration. |
| **papers-2048ctx-SAE** (our SAE collection, 78K points) | Dead — scores ~0.019, garbage | 0.019 | Collection exists but retrieval is broken. Returning irrelevant papers (CogAgents, Ethereum prediction, social cognition). |
| **papers-semantic** (our dense collection, 117K points) | Dead — scores ~0.019, garbage | 0.019 | Same problem. Dense embedding model can't find the SAE paper either. |
| **Web (DuckDuckGo)** | ✅ Working — found it instantly | N/A | arXiv:2604.21511 "From Tokens to Concepts: Leveraging SAE for SPLADE" by Zong & Vast. Published ~3 days ago. |

## Diagnosis

Both our collections have the paper (arxiv 2604.21511 exists in the corpus) but **neither embedding path can retrieve it**. The dense model and the SAE model are both failing at the same fundamental problem: the query doesn't match the stored representations.

Two likely causes:

1. **Embedding model mismatch**: The dense embedding model (`embeddinggemma:300m` via Ollama) and the SAE embedding path may not have been trained on retrieval tasks that handle this terminology. The corpus contains SAE-SPLADE content, but the models can't bridge the semantic gap between "SAE-SPLADE sparse autoencoder" and whatever the chunks actually contain.

2. **SAE collection embedding pipeline broken**: The `papers-2048ctx-SAE` collection is supposed to use SAE sparse vectors, but if the embedding server isn't properly serving SAE-SPLADE vectors (or the vectors weren't computed correctly during indexing), the collection is effectively a dense-only store with an unused sparse vector.

## Options

### Option A: Fix SAE Embedding Pipeline (High Impact, High Effort)

**What**: Diagnose and repair the embedding pipeline for `papers-2048ctx-SAE`. Verify that:
- The SAE embedding server is serving actual SAE-SPLADE vectors (not dense fallback)
- The `Qdrant/minicoil-v1` sparse vectors are being computed correctly during indexing
- The embedding path from prompt → model → sparse vector is intact

**Cost**: 1-2 days investigation + fix time. Requires GPU box access.

**Reward**: If the SAE collection is just misconfigured, fixing it could instantly make the collection functional. This is the collection we specifically built to improve retrieval.

### Option B: Improve Dense Embedding Quality (Medium Impact, Medium Effort)

**What**: The dense model (`embeddinggemma:300m`) is failing too. Consider:
- Switching to a stronger embedding model (e.g., `nomic-embed-text`, `jina-embeddings-v3`)
- Adding metadata prefix injection to dense embeddings (per design doc)
- Re-embedding the corpus with a better model

**Cost**: Re-embedding 117K points. Several hours on GPU.

**Reward**: Better retrieval across ALL queries, not just SAE-SPLADE.

### Option C: Hybrid Dense + SAE (Our Long-Term Plan)

**What**: The design doc already has Phase 2 for hybrid search. The SAE collection (`papers-2048ctx-SAE`) may be a premature or incorrectly-configured attempt at this. The proper approach per design doc:
1. Keep dense vectors (existing)
2. Add sparse vectors via MiniCOIL or SAE-SPLADE
3. Fuse at query time via RRF

**Cost**: Already partially implemented (Sweep sparse weight script exists). Needs completion.

**Reward**: Best of both worlds — semantic recall (dense) + keyword precision (sparse).

### Option D: Inject the Paper Directly (Quick Fix)

**What**: If the paper is genuinely cutting-edge (3 days old), maybe the embedding models simply haven't seen enough training data on it. The corpus already contains it — we just need better retrieval.

**Cost**: Minimal. Focus engineering effort on embedding quality rather than infrastructure.

**Reward**: The SAE paper's key insight (replace vocabulary with latent SAE concepts) should inform our own embedding strategy — regardless of which collection can retrieve it.

## Recommendation

**Do A and D in parallel.** Diagnose the SAE pipeline while accepting that the paper's concept is valid and worth implementing.

The paper says:
> "Replace the backbone vocabulary with a latent space of semantic concepts learned using Sparse Auto-Encoders."

This is **exactly** what our design doc already proposes for Phase 2 — but with MiniCOIL instead of SAE-SPLADE. The SAE-SPLADE paper validates our direction. The difference: SAE-SPLADE uses Gemma-3 270M activations + SAE, while MiniCOIL is a lighter ~30MB model.

**Key insight from web search**: SAE-SPLADE uses a two-stage pipeline (SAE pre-training → SPLADE fine-tuning) with TopK sparsification. Our existing `embed_sparse_vectors.py` uses MiniCOIL via FastEmbed — which is a simpler path to the same goal.

**Bottom line**: Our collections are both broken for this query. The paper exists in the corpus. The solution isn't to throw more collections at the problem — it's to fix the embedding quality of what we have, then add hybrid search as designed.

## Collections Summary

| Collection | Points | Type | Retrieval Quality |
|---|---|---|---|
| `books` | 6,212 | Dense (768-d) | Untested for this query |
| `papers` | 90,256 | Dense (768-d) | Untested for this query |
| `books-semantic` | 9,000 | Dense (768-d) | Untested for this query |
| **`papers-semantic`** | **117,750** | **Dense (768-d)** | **Broken — scores ~0.019** |
| **`papers-2048ctx-SAE`** | **78,036** | **Dense + SAE (768-d)** | **Broken — scores ~0.019** |
| Bedrock KB | 0 | N/A | No knowledge bases configured |

**Note**: Only 2 collections were tested. The `books*` and `papers` collections were not queried. If SAE-SPLADE paper is in those, they may also fail.