# SPLADE Drop-In Implementation
## Instructions for Qwen 3.6

**Context:** We are replacing the current `SparseEmbedder` (Gemma 3 270M + JumpReLU SAE)
with a production-ready SPLADE model. This is a temporary baseline that will run
alongside the SAE-SPLADE training pipeline. The goal is working hybrid RAG today.

**All HuggingFace models must be downloaded to local paths and loaded from disk.**
Never load directly from HuggingFace at runtime — the embedding server has no
guarantee of internet access and cold-start latency is unacceptable.

---

## Step 1: Download Models to Local Paths (Human Executes)

Run these commands once. They download model weights to the local HuggingFace cache
and then copy to the project's model storage directory.

```bash
# Primary SPLADE model
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='naver/splade-cocondenser-ensemble-distil',
    local_dir='/tank/huggingface/splade-cocondenser-ensemble-distil',
    local_dir_use_symlinks=False
)
print(f'Downloaded to: {path}')
"

# Dense model is already local at /tank/huggingface/embeddinggemma-300m
# IT model is already local at /tank/huggingface/gemma-3-270m-it
# No additional downloads needed for dense or rewriter
```

**Verify download:**
```bash
ls /tank/huggingface/splade-cocondenser-ensemble-distil/
# Must contain: config.json, tokenizer.json, model.safetensors (or pytorch_model.bin)
```

---

## Step 2: Update `embedder.py`

### Constants to add at the top of `embedder.py`

```python
# SPLADE paths
SPLADE_LOCAL_PATH = "/tank/huggingface/splade-cocondenser-ensemble-distil"
SPLADE_VOCAB_SIZE = 30522      # BERT vocabulary size — sparse vector dimension
SPLADE_MAX_DOC_LENGTH = 256    # Truncate documents to this many tokens
SPLADE_MAX_QUERY_LENGTH = 32   # Truncate queries (after IT rewrite) to this
SPLADE_THRESHOLD = 0.0         # Keep all nonzero values — SPLADE is already sparse
```

### Remove or comment out these existing constants (no longer used)

```python
# SAE_LOCAL_PATH — no longer used
# BACKBONE_LOCAL_PATH — no longer used (was Gemma 3 270M PT)
# SAE_ID — no longer used
# SAE_HOOK_LAYER — no longer used
# SPLADE_THRESHOLD = 0.01 — replaced with 0.0 (SPLADE handles its own sparsity)
# INTERNAL_BATCH_SIZE — keep, reuse for SPLADE batching
```

### Replace the entire `JumpReLUSAE` class and `_load_jumprelu_sae` function

Delete both entirely. They are replaced by `AutoModelForMaskedLM` loaded directly.

### Replace the entire `SparseEmbedder` class

```python
class SparseEmbedder:
    """SPLADE sparse embedder using naver/splade-cocondenser-ensemble-distil.

    Produces sparse vectors over the BERT vocabulary (~30k dimensions).
    Loaded after DenseEmbedder so dense claims VRAM first.

    Drop-in replacement for the previous SAE-based SparseEmbedder.
    Same encode(texts, is_query) -> List[Dict] interface preserved.
    """

    def __init__(self, internal_batch_size: int = INTERNAL_BATCH_SIZE):
        self.internal_batch_size = internal_batch_size
        self._load_model()

    def _load_model(self):
        from transformers import AutoModelForMaskedLM

        logger.info("Loading SPLADE model from %s", SPLADE_LOCAL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(
            SPLADE_LOCAL_PATH,
            local_files_only=True,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            SPLADE_LOCAL_PATH,
            local_files_only=True,
            torch_dtype=torch.float32,   # float32 — model is small, bfloat16 not needed
            device_map="cuda",
        ).eval()

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            "SPLADE model loaded from %s | vocab_size=%d | "
            "VRAM allocated: %.2f GB, reserved: %.2f GB",
            SPLADE_LOCAL_PATH,
            SPLADE_VOCAB_SIZE,
            allocated,
            reserved,
        )
        assert next(self.model.parameters()).device.type == "cuda", \
            "FATAL: SPLADE model is on CPU. Check device_map."

    def _encode_batch(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """Run one micro-batch synchronously on GPU."""
        max_length = SPLADE_MAX_QUERY_LENGTH if is_query else SPLADE_MAX_DOC_LENGTH

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to("cuda")

        with torch.no_grad():
            output = self.model(**inputs)
            # output.logits shape: [batch, seq_len, vocab_size]

            # SPLADE aggregation:
            # 1. relu — zero out negative logits
            # 2. log1p — saturate large activations
            # 3. mask padding tokens
            # 4. max over sequence — one value per vocab term per document
            vecs = torch.log1p(torch.relu(output.logits))
            mask = inputs["attention_mask"].unsqueeze(-1).to(vecs.dtype)
            sparse_vecs = (vecs * mask).max(dim=1).values
            # sparse_vecs shape: [batch, vocab_size ~30522]

        results = []
        for vec in sparse_vecs:
            nonzero_mask = vec > SPLADE_THRESHOLD
            indices = nonzero_mask.nonzero(as_tuple=False).squeeze(-1)
            results.append({
                "indices": indices.cpu().tolist(),
                "values": vec[indices].cpu().float().tolist(),
            })

            # Diagnostic logging for first batch only
            if len(results) == 1:
                nnz = len(results[0]["indices"])
                logger.debug("SPLADE NNZ sample: %d active dims", nnz)

        return results

    def encode(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """Encode texts into SPLADE sparse vectors.

        is_query controls truncation length:
            is_query=True  → max 32 tokens  (rewritten queries)
            is_query=False → max 256 tokens (documents)

        Args:
            texts: List of strings to embed.
            is_query: Whether these are queries (True) or documents (False).

        Returns:
            List of dicts with 'indices' and 'values' keys.
        """
        all_results = []
        for i in range(0, len(texts), self.internal_batch_size):
            batch = texts[i : i + self.internal_batch_size]
            all_results.extend(self._encode_batch(batch, is_query=is_query))

        # Log NNZ statistics for monitoring
        nnz_counts = [len(r["indices"]) for r in all_results]
        if nnz_counts:
            logger.info(
                "SPLADE encode: %d texts | NNZ min=%d max=%d mean=%.1f | is_query=%s",
                len(texts),
                min(nnz_counts),
                max(nnz_counts),
                sum(nnz_counts) / len(nnz_counts),
                is_query,
            )
        return all_results

    async def encode_async(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """Async wrapper for FastAPI routes. Identical interface to before."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.encode(texts, is_query),
        )
```

---

## Step 3: Update Qdrant Collection Configuration

**Critical:** The sparse vector dimension changes from 65536 (SAE) to 30522 (BERT vocab).
If a Qdrant collection already exists with the wrong dimension, it must be deleted
and recreated — Qdrant does not support changing vector dimensions in-place.

### Collection creation code

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    SparseVectorParams, SparseIndexParams,
    CreateCollection,
)

client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "papers"  # adjust to your collection name

# Delete existing collection if it exists (dimension change requires recreation)
if client.collection_exists(COLLECTION_NAME):
    logger.warning(
        "Collection '%s' exists — deleting to recreate with correct dimensions. "
        "All indexed documents will be lost and must be re-indexed.",
        COLLECTION_NAME
    )
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        # Dense named vector — Snowflake embedding model output
        "dense": VectorParams(
            size=768,               # Snowflake arctic-embed output dimension
            distance=Distance.COSINE,
        ),
    },
    sparse_vectors_config={
        # Sparse named vector — SPLADE over BERT vocabulary
        "sparse": SparseVectorParams(
            index=SparseIndexParams(
                on_disk=False,      # Keep in RAM for fast retrieval
            ),
        ),
        # Note: sparse vectors do not specify a fixed size — Qdrant infers
        # from the maximum index value seen during insertion (will be ~30521)
    },
)
logger.info("Collection '%s' created with dense(768) + sparse(BERT vocab)", COLLECTION_NAME)
```

### Document ingestion format

```python
from qdrant_client.models import PointStruct, NamedVector, NamedSparseVector, SparseVector

def ingest_paper(client, point_id: int, text: str, metadata: dict,
                  dense_vec: List[float], sparse_result: dict):
    """
    Insert one paper into Qdrant with both named vectors.

    sparse_result: dict with 'indices' and 'values' keys
                   as returned by SparseEmbedder.encode()
    """
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(
                        indices=sparse_result["indices"],
                        values=sparse_result["values"],
                    ),
                },
                payload=metadata,  # title, abstract, authors, etc.
            )
        ],
    )
```

### Hybrid query format

```python
from qdrant_client.models import (
    Prefetch, Query, FusionQuery, Fusion,
    NamedSparseVector, SparseVector,
)
import numpy as np

def hybrid_search(client, query_text: str, dense_embedder, sparse_embedder,
                   top_k: int = 10, prefetch_limit: int = 100):
    """
    Hybrid search using Reciprocal Rank Fusion over dense + sparse results.

    prefetch_limit: how many candidates each sub-search retrieves before fusion.
                    Higher = better recall, slower. 100 is a good starting point
                    for a 1700-paper corpus.
    """
    # Encode query with both models
    dense_vec = dense_embedder.encode([query_text])[0]
    sparse_result = sparse_embedder.encode([query_text], is_query=True)[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            # Dense sub-search
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=prefetch_limit,
            ),
            # Sparse sub-search
            Prefetch(
                query=SparseVector(
                    indices=sparse_result["indices"],
                    values=sparse_result["values"],
                ),
                using="sparse",
                limit=prefetch_limit,
            ),
        ],
        # RRF fusion — smoothing constant 60 is standard
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return results.points
```

---

## Step 4: Update `server.py`

No changes needed to the FastAPI routes — the `SparseEmbedder` interface is
identical. The `/embed_sparse` endpoint continues to work as-is.

Update the `/models` endpoint to reflect the new model:

```python
@app.get("/models", response_model=ModelsResponse)
def models():
    return ModelsResponse(
        dense=DENSE_MODEL,
        sparse=SPLADE_LOCAL_PATH,   # was BACKBONE_LOCAL_PATH
    )
```

---

## Step 5: Smoke Test After Deployment

Run this after starting the server to confirm SPLADE is working correctly:

```python
import requests

# Test sparse encoding
resp = requests.post("http://localhost:8100/embed_sparse", json={
    "texts": [
        "sparse autoencoder feature learning interpretability",
        "dense retrieval dual encoder BERT",
        "hybrid search reciprocal rank fusion"
    ],
    "is_query": False
})
vecs = resp.json()["vectors"]

for i, vec in enumerate(vecs):
    nnz = len(vec["indices"])
    max_val = max(vec["values"]) if vec["values"] else 0
    print(f"Vec {i}: NNZ={nnz}, max_val={max_val:.4f}")

# Expected output (approximate):
# Vec 0: NNZ=~80-150, max_val=~2.0-4.0
# Vec 1: NNZ=~60-120, max_val=~2.0-4.0
# Vec 2: NNZ=~70-130, max_val=~2.0-4.0

# Red flags:
# NNZ=0 → SPLADE not activating, check model loaded correctly
# NNZ>500 → threshold too low or wrong model
# max_val<0.1 → relu/log1p not applied correctly
# All vecs have identical indices → aggregation bug

# Test query encoding (shorter truncation)
resp = requests.post("http://localhost:8100/embed_sparse", json={
    "texts": ["what papers discuss sparse retrieval"],
    "is_query": True
})
q_vec = resp.json()["vectors"][0]
print(f"Query NNZ: {len(q_vec['indices'])}")
# Expected: NNZ=~30-80 (shorter due to 32 token limit)
```

---

## What Does NOT Change

- `DenseEmbedder` — unchanged, still uses Snowflake model
- `QueryRewriter` — unchanged, still uses Gemma 3 270M IT
- All FastAPI routes — unchanged
- `_executor` thread pool — unchanged
- VRAM cap logic — unchanged
- The IT model query rewrite → sparse encode flow — unchanged

---

## Expected NNZ Ranges for SPLADE

For reference when monitoring logs:

| Text type | Expected NNZ | Notes |
|-----------|-------------|-------|
| Short query (32 tok) | 30–80 | After IT rewrite |
| Abstract (256 tok) | 80–200 | Typical paper abstract |
| Full paper chunk | 100–300 | Longer passages |
| Empty string | 0 | Should not occur — add guard |

If NNZ consistently exceeds 400, add a threshold:
```python
SPLADE_THRESHOLD = 0.1  # Prune low-confidence activations
```

---

## Known Issues and Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `local_files_only=True` raises error | Model not downloaded yet | Run Step 1 download first |
| `AutoModelForMaskedLM` not imported | Missing import in embedder.py | Add to transformers imports at top |
| NNZ=0 for all vectors | `relu` applied after `log1p` not before | Order must be `log1p(relu(logits))` |
| VRAM OOM after loading | SPLADE + dense + IT all on same GPU | SPLADE is ~268MB — should fit; check other processes |
| Qdrant upsert fails on sparse | `indices` contains duplicates | SPLADE should not produce duplicates; add `assert len(set(indices)) == len(indices)` |
| Query NNZ same as doc NNZ | `is_query` not passed through to `_encode_batch` | Confirm `is_query` flows from `encode()` to `_encode_batch()` |
| Collection exists error on startup | Old collection with wrong dimension | Delete and recreate — data must be re-indexed |
