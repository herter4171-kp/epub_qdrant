# Phase 2: MiniCOIL Sparse Re-embed + Hybrid Search — Implementation Plan

> **Status**: Phase 0 (z-score normalization) is already implemented in `retriever.py`.
> Phase 1 (LLM-as-judge evaluation harness) is complete — baseline run stored in `results.json`.
> This plan covers Phase 2: adding MiniCOIL sparse vectors and hybrid search.

---

Use the caveman skill.  You do not have access to the GPU box, so if you need something to happen there, ask the user.

## Goal

Add **sparse keyword/metadata-matching vectors** alongside existing dense (semantic) vectors, then fuse both at query time using **Reciprocal Rank Fusion (RRF)** for hybrid search.

**Key principle**: Do NOT re-read source files. Instead, scroll existing Qdrant collections, compute sparse vectors from stored payloads, and upsert the new named vectors into **new target collections** — keeping originals intact for safe comparison.

---

## Rationale

| Problem | Why source-file re-embedding is bad |
|---|---|
| **Slow** | Re-embedding from disk requires downloading PDFs → pypdf extraction → Ollama API calls. Takes hours for 96K points. |
| **Fragile** | Source files may be missing, corrupted, or out-of-sync with what's already in Qdrant. |
| **Redundant** | The text and metadata are already stored in Qdrant payloads. No need to re-extract. |
| **Wasteful** | We already have dense vectors. We only need to add sparse vectors — a lightweight ONNX/GPU operation. |

**Better approach**: Scroll Qdrant → compute sparse via FastEmbed GPU → upsert named vectors into new collections. ~3-6 minutes for all 96K points on GPU.

**Why new collections instead of in-place?**
- Qdrant cannot add named vector configs to an existing unnamed-vector collection — migration is required anyway.
- Keeping originals intact means you can run baseline vs. hybrid comparisons against live data without risk.
- New collections let you validate before cutting over.

---

## Architecture

```
Mac Mini (dev)                          Linux GPU Box (192.168.68.75)
┌─────────────────────────────┐         ┌──────────────────────────────┐
│ scripts/embed_sparse_        │  HTTP   │ mcp_servers/minicoil_server/ │
│   vectors.py                │────────▶│   server.py                  │
│                             │         │   fastembed-gpu + RTX 5090   │
│ mcp_servers/retrieval/      │  HTTP   │   :9000                      │
│   retriever.py              │────────▶│                              │
└─────────────────────────────┘         └──────────────────────────────┘
         │                                          │
         │ qdrant-client                            │ fastembed-gpu
         ▼                                          ▼
┌─────────────────┐                    ┌────────────────────────┐
│ Qdrant          │                    │ Qdrant/minicoil-v1     │
│  books          │                    │ ONNX model (~15MB)     │
│  papers         │                    │ auto-downloaded on     │
│  books-named    │                    │ first startup          │
│  papers-named   │                    └────────────────────────┘
└─────────────────┘
```

---

## MiniCOIL Overview

MiniCOIL is a **sparse neural embedding model** from Qdrant, served via the FastEmbed library.

| Property | Value |
|----------|-------|
| **Model** | `Qdrant/minicoil-v1` |
| **Inference library** | `fastembed-gpu` |
| **Model size** | ~15MB (auto-downloaded on first startup) |
| **GPU** | RTX 5090 via `fastembed-gpu` + ONNX Runtime CUDA |
| **Qdrant config** | Must use `Modifier.IDF` on sparse vector collection config |
| **Embedding paths** | `model.embed()` for documents, `model.query_embed()` for queries — **do not mix** |

> **Note**: The ONNX Runtime warning `GPU device discovery failed: Failed to open file: /sys/class/drm/card0/device/vendor` is harmless on Linux Mint. ONNX Runtime falls through to other discovery methods. Confirm CUDA is active with `python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"` — `CUDAExecutionProvider` should appear in the list.

---

## Implementation Steps

### Step 1: Install Dependencies

**On the GPU Linux box:**
```bash
pip install fastembed-gpu fastapi uvicorn
```

**On the Mac (dev machine):**
```bash
pip install qdrant-client requests
```

No PyTorch, no transformers, no raw ONNX wrangling.

---

### Step 2: MiniCOIL Embedding Server

Create `mcp_servers/minicoil_server/server.py` on the GPU box:

```python
"""MiniCOIL sparse embedding server.

Loads Qdrant/minicoil-v1 once on startup via fastembed-gpu,
then serves batched sparse embeddings over HTTP.

GPU box setup:
    pip install fastembed-gpu fastapi uvicorn
    uvicorn mcp_servers.minicoil_server.server:app --host 0.0.0.0 --port 9000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastembed import SparseTextEmbedding

app = FastAPI(title="MiniCOIL Embedding Server")
model = None


@app.on_event("startup")
async def startup():
    global model
    model = SparseTextEmbedding(model_name="Qdrant/minicoil-v1")


class EmbedRequest(BaseModel):
    texts: List[str]
    is_query: bool = False  # False for document indexing, True for query-time


class SparseVector(BaseModel):
    indices: List[int]
    values: List[float]


class EmbedResponse(BaseModel):
    vectors: List[SparseVector]


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if not request.texts:
        return EmbedResponse(vectors=[])
    if len(request.texts) > 1024:
        raise HTTPException(400, "Max 1024 texts per request")

    if request.is_query:
        embeddings = list(model.query_embed(request.texts))
    else:
        embeddings = list(model.embed(request.texts))

    return EmbedResponse(vectors=[
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in embeddings
    ])


@app.get("/health")
async def health():
    return {"status": "ok", "ready": model is not None}
```

**Start the server on the GPU box:**
```bash
uvicorn mcp_servers.minicoil_server.server:app --host 0.0.0.0 --port 9000
```

**Verify from the Mac:**
```bash
curl http://192.168.68.75:9000/health
# → {"status":"ok","ready":true}
```

---

### Step 3: MiniCOIL Client (shared utility)

Create `mcp_servers/minicoil_server/client.py` — used by both the indexing script and retriever:

```python
"""Thin client for the MiniCOIL embedding server."""

import requests
from typing import List, Dict

MINICOIL_URL = "http://192.168.68.75:9000/embed"


def get_sparse_vectors(texts: List[str], is_query: bool = False) -> List[Dict]:
    """Call the MiniCOIL server and return sparse vectors.

    Args:
        texts: List of strings to embed.
        is_query: True at search time, False during indexing.
                  MiniCOIL uses different embedding paths — do not mix.

    Returns:
        List of dicts with 'indices' and 'values' keys.
    """
    resp = requests.post(MINICOIL_URL, json={"texts": texts, "is_query": is_query})
    resp.raise_for_status()
    return resp.json()["vectors"]
```

---

### Step 4: Create New Collections with Named Vector Configs

At the top of `scripts/embed_sparse_vectors.py`, create the target collections before scrolling:

```python
from qdrant_client.models import VectorParams, SparseVectorParams, Distance, Modifier

COLLECTION_MAP = {
    "books": "books-named",
    "papers": "papers-named",
}

for src, dst in COLLECTION_MAP.items():
    client.create_collection(
        collection_name=dst,
        vectors_config={
            "dense": VectorParams(size=768, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF)
        }
    )
```

**Named vector schema:**

| Vector | Type | Dimensions | Distance |
|--------|------|-----------|----------|
| `dense` | `list[float]` | 768 | COSINE |
| `sparse` | `SparseVector(indices, values)` | variable | IDF (dot product) |

---

### Step 5: Scroll + Embed + Upsert Script

Create `scripts/embed_sparse_vectors.py`:

```python
"""Phase 2: Scroll existing collections, compute MiniCOIL sparse vectors,
upsert into new named-vector collections.

Run from Mac:
    python3 scripts/embed_sparse_vectors.py
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, SparseVector,
    VectorParams, SparseVectorParams,
    Distance, Modifier,
)
from mcp_servers.minicoil_server.client import get_sparse_vectors

QDRANT_URL = "http://localhost:6333"  # adjust if needed
BATCH_SIZE = 256
COLLECTION_MAP = {
    "books": "books-named",
    "papers": "papers-named",
}

client = QdrantClient(url=QDRANT_URL)

# Create target collections
for dst in COLLECTION_MAP.values():
    client.create_collection(
        collection_name=dst,
        vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    print(f"Created collection: {dst}")

# Scroll, embed, upsert
for src, dst in COLLECTION_MAP.items():
    print(f"\nProcessing {src} → {dst}")
    offset = None
    total = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=src,
            limit=BATCH_SIZE,
            offset=offset,
            with_vectors=True,   # p.vector is list[float] for unnamed collections
            with_payload=True,
        )
        if not points:
            break

        # Skip points with empty text
        valid = [p for p in points if p.payload.get("text", "").strip()]
        if not valid:
            if next_offset is None:
                break
            offset = next_offset
            continue

        texts = [p.payload["text"] for p in valid]
        sparse_vecs = get_sparse_vectors(texts, is_query=False)

        upsert_points = [
            PointStruct(
                id=p.id,
                vector={
                    "dense": p.vector,  # raw list[float] from unnamed collection
                    "sparse": SparseVector(
                        indices=sv["indices"],
                        values=sv["values"],
                    ),
                },
                payload=p.payload,
            )
            for p, sv in zip(valid, sparse_vecs)
        ]

        client.upsert(collection_name=dst, points=upsert_points)
        total += len(upsert_points)
        print(f"  Upserted {total} points...", end="\r")

        if next_offset is None:
            break
        offset = next_offset

    print(f"\n  Done: {total} points migrated to {dst}")
```

---

### Step 6: Implement Hybrid Search in `retriever.py`

Add `hybrid_search()` and update `search_collections()`:

```python
from collections import defaultdict
from mcp_servers.minicoil_server.client import get_sparse_vectors

def hybrid_search(
    self,
    collection: str,
    query_dense: List[float],
    query_sparse: Dict,
    k: int = 20,
    filter: Optional[Filter] = None,
) -> List[ChunkResult]:
    """Search with dense and sparse vectors, fuse via Reciprocal Rank Fusion."""

    dense_hits = self.client.search(
        collection_name=collection,
        query_vector=("dense", query_dense),
        limit=k * 2,
        query_filter=filter,
    )

    sparse_hits = self.client.search(
        collection_name=collection,
        query_vector=("sparse", query_sparse),
        limit=k * 2,
        query_filter=filter,
    )

    # RRF fusion
    rrf_scores = defaultdict(float)
    k_rrf = 60

    for rank, hit in enumerate(dense_hits):
        rrf_scores[hit.id] += 1.0 / (k_rrf + rank + 1)
    for rank, hit in enumerate(sparse_hits):
        rrf_scores[hit.id] += 1.0 / (k_rrf + rank + 1)

    # Merge, reattach payload
    all_hits = {h.id: h for h in dense_hits + sparse_hits}
    results = []
    for point_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]:
        hit = all_hits[point_id]
        results.append(ChunkResult(
            text=hit.payload.get("text", ""),
            score=rrf_score,
            **{k: v for k, v in hit.payload.items() if k != "text"},
        ))
    return results
```

**Update `search_collections()`** to:
1. Generate sparse query vector: `get_sparse_vectors([query], is_query=True)[0]`
2. Call `hybrid_search()` targeting `books-named` and `papers-named`
3. Apply existing z-score normalization across fused results

---

### Step 7: Run Phase 2 Evaluation

```bash
python3 scripts/evaluate.py phase_2_hybrid
```

Compares hybrid retrieval against baseline stored in `results.json`.

---

## File Changes Summary

| File | Change |
|------|--------|
| `mcp_servers/minicoil_server/server.py` | **NEW** — FastAPI server, runs on GPU box, serves sparse embeddings |
| `mcp_servers/minicoil_server/client.py` | **NEW** — thin HTTP client, shared by indexing script and retriever |
| `scripts/embed_sparse_vectors.py` | **NEW** — creates `-named` collections, scrolls, embeds, upserts |
| `mcp_servers/retrieval/mcp_server/retriever.py` | Add `hybrid_search()`, update `search_collections()` |
| `scripts/evaluate.py` | Add `phase_2_hybrid` method to RetrieverRunner |
| `pyproject.toml` | Add `fastembed-gpu` (GPU box), `requests` (Mac) |

---

## Performance Expectations

| Operation | Time |
|-----------|------|
| Server startup + model download (~15MB) | ~5s |
| Create 2 new collections | <1s |
| Scroll + embed 96K points (RTX 5090, batch=256) | ~3-6 minutes |
| Hybrid search per query | ~10-20ms (vs ~5ms dense-only) |
| Evaluation (30 queries) | ~3 minutes |

---

## Migration Strategy

```
Original collections (read-only after Phase 2)     New collections (active)
┌──────────────────────┐                          ┌──────────────────────────┐
│ books                │  ◄── scroll ──────────▶  │ books-named              │
│ papers               │  ◄── scroll ──────────▶  │ papers-named             │
└──────────────────────┘                          └──────────────────────────┘
  dense only, baseline                              dense + sparse, RRF hybrid
```

Once Phase 2 evaluation confirms improvement, point `retriever.py` permanently at the `-named` collections. Originals remain as rollback.

---

## Phase 3 Preview: LLM Filter Extraction

After Phase 2 completes, Phase 3 adds:

1. **MetadataCatalog** — scroll-discover taxonomy from Qdrant
2. **MetadataFilterExtractor** — LLM-driven filter extraction from natural language
3. **Scope context injection** — resolved metadata into retrieval prompts

See `.clinerules/first-self-improve-r2.md` for full Phase 3 details.
