# Engineering Spec: Replace MiniCOIL/fastembed with SAE-SPLADE (gemma-scope-2-270m-pt)

## Why We Are Doing This

The current sparse embedder uses MiniCOIL via `fastembed`. MiniCOIL is a SPLADE-family model: it produces
sparse vectors where each active dimension corresponds to a token in the model's vocabulary. This means
"corrosion inhibitor" and "anti-rust compound" score as different concepts because they are different tokens,
even if the underlying meaning is identical. This is the polysemicity/synonymy problem that plagues all
vocabulary-bound sparse retrievers.

SAE-SPLADE replaces the vocabulary projection with a Sparse Autoencoder (SAE). The SAE has learned a latent
space of semantic concepts from the model's internal activations. Active dimensions in the output vector
correspond to those learned concepts, not surface tokens. For domain-specific document corpora with
standardized templates and shared terminology, this produces more semantically consistent retrieval —
especially when query phrasing diverges from document phrasing, which is almost always true in practice.

We are using `google/gemma-scope-2-270m-pt` (a pretrained Gemma 3 270M base model) as the backbone and one
of its associated Gemma Scope 2 SAE checkpoints for the concept projection. The goal is a surgical drop-in
replacement for the `SparseEmbedder` class with no changes to the server API or the dense embedder path.

---

## Scope

**Only `embedder.py` changes.** Specifically, only the `SparseEmbedder` class.

`server.py`, `client.py`, and the `/embed_sparse` endpoint contract are untouched. The replacement must
preserve the existing return type: `List[Dict]` where each dict has `"indices": List[int]` and
`"values": List[float]`.

---

## Prerequisites & Dependencies

### CUDA requirement
This server runs on an RTX 5090 (Blackwell, sm_120, compute capability 12.0). CUDA 12.8 is required.
PyTorch must be installed from the cu128 index:

```bash
pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

Do NOT install flash-attn. It is not required and Blackwell (sm_120) support in flash-attn is not
confirmed stable. PyTorch's built-in SDPA will be used instead.

### New dependency: sae-lens
```bash
pip install sae-lens==6.37.6
```

`sae-lens` is the standard library for loading Gemma Scope checkpoints. It handles the checkpoint format,
config parsing, and weight loading. Do NOT attempt to load the SAE weights manually with `torch.load` or
`nn.Linear` — the checkpoint format is not a plain state dict.

### Remove: fastembed-gpu
`fastembed-gpu` (and its `fastembed` parent) can be removed from requirements once the new embedder is
confirmed working. Do not remove it during this change — keep it until the new path is verified.

---

## Local Model Paths

Both models must already be cloned locally. Do not download from HuggingFace during startup.

| Model | Local path |
|---|---|
| Backbone (Gemma 3 270M PT) | `/tank/huggingface/gemma-3-270m-pt` |
| SAE weights (Gemma Scope 2) | `/tank/huggingface/gemma-scope-2-270m-pt` |

The SAE checkpoint directory structure under the local clone is:
```
gemma-scope-2-270m-pt/
  resid_post/
    layer_12_width_65k_l0_medium/    ← use this one
    layer_12_width_16k_l0_small/
    layer_12_width_262k_l0_medium/
    ...
```

Use `resid_post/layer_12_width_65k_l0_medium`. Rationale: 65k width gives enough concept granularity for
domain retrieval without excessive Qdrant index bloat; `medium` L0 (target 30-60 active features per token)
is the recommended default. If this checkpoint is missing from the local clone, fall back to
`layer_12_width_16k_l0_small`.

---

## Implementation

### SparseEmbedder class (replaces the existing one in embedder.py)

```python
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModel
from sae_lens import SAE

logger = logging.getLogger(__name__)

# Thread pool for non-blocking GPU inference (one worker — GPU is single-threaded)
_executor = ThreadPoolExecutor(max_workers=1)

SAE_LOCAL_PATH = "/tank/huggingface/gemma-scope-2-270m-pt"
BACKBONE_LOCAL_PATH = "/tank/huggingface/gemma-3-270m-pt"
SAE_ID = "layer_12_width_65k_l0_medium"
SAE_HOOK_LAYER = 12          # resid_post after layer 12
SPLADE_THRESHOLD = 0.01      # prune near-zero activations before returning
INTERNAL_BATCH_SIZE = 32     # micro-batch size for GPU inference


class SparseEmbedder:
    """SAE-SPLADE sparse embedder using gemma-scope-2-270m-pt.

    Replaces the fastembed/MiniCOIL SparseEmbedder.
    Drop-in: same encode(texts, is_query) -> List[Dict] interface.
    """

    def __init__(self, internal_batch_size: int = INTERNAL_BATCH_SIZE):
        self.internal_batch_size = internal_batch_size
        self._load_models()
        self._validate_dimensions()

    def _load_models(self):
        logger.info("Loading Gemma 3 270M PT backbone from %s", BACKBONE_LOCAL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(BACKBONE_LOCAL_PATH)
        self.backbone = AutoModel.from_pretrained(
            BACKBONE_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            output_hidden_states=True,
            # No attn_implementation override — SDPA is fine for sm_120
        ).eval()

        logger.info("Loading Gemma Scope 2 SAE: %s / %s", SAE_LOCAL_PATH, SAE_ID)
        # SAE.from_pretrained accepts a local directory path as `release`
        self.sae, self.sae_cfg, _ = SAE.from_pretrained(
            release=SAE_LOCAL_PATH,
            sae_id=SAE_ID,
        )
        self.sae = self.sae.to("cuda").to(torch.bfloat16)

        logger.info("Both models loaded. Running device assertions.")
        assert next(self.backbone.parameters()).device.type == "cuda", \
            "FATAL: Backbone is on CPU. Check device_map."
        # SAE internal weights
        assert next(self.sae.parameters()).device.type == "cuda", \
            "FATAL: SAE is on CPU. .to('cuda') failed."

    def _validate_dimensions(self):
        """Confirm backbone hidden dim matches SAE expected input dim.

        Gemma 3 270M PT has hidden_size=1152. The SAE cfg.d_in must match.
        If these differ, something is wrong with the checkpoint or the backbone path.
        """
        backbone_hidden = self.backbone.config.hidden_size
        sae_input_dim = self.sae_cfg["d_in"]  # exposed by sae-lens cfg dict
        assert backbone_hidden == sae_input_dim, (
            f"Dimension mismatch: backbone hidden_size={backbone_hidden} "
            f"but SAE d_in={sae_input_dim}. "
            f"Check that backbone and SAE checkpoint correspond to the same model."
        )
        logger.info(
            "Dimension check passed: backbone hidden_size=%d, SAE d_in=%d, SAE width=%d",
            backbone_hidden,
            sae_input_dim,
            self.sae_cfg["d_sae"],
        )

    def _encode_batch(self, texts: List[str]) -> List[Dict]:
        """Run one micro-batch synchronously on GPU. Called from thread pool."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to("cuda")

        with torch.no_grad():
            outputs = self.backbone(**inputs)
            # hidden_states is a tuple of (n_layers + 1) tensors, each [batch, seq, hidden]
            # Index SAE_HOOK_LAYER + 1 because index 0 is the embedding layer
            hidden = outputs.hidden_states[SAE_HOOK_LAYER + 1]

            # SAE encode: hidden [batch, seq, hidden] -> activations [batch, seq, sae_width]
            # sae-lens SAE.encode() expects [batch * seq, hidden]; reshape accordingly
            batch_size, seq_len, hidden_dim = hidden.shape
            flat_hidden = hidden.reshape(-1, hidden_dim)
            flat_acts = self.sae.encode(flat_hidden)            # [batch*seq, sae_width]
            acts = flat_acts.reshape(batch_size, seq_len, -1)  # [batch, seq, sae_width]

            # SPLADE max-pooling with log saturation, masked to real tokens
            logged = torch.log1p(acts)                          # log(1 + x)
            mask = inputs["attention_mask"].unsqueeze(-1).to(logged.dtype)
            sparse_vecs = (logged * mask).max(dim=1).values     # [batch, sae_width]

        results = []
        for vec in sparse_vecs:
            nonzero_mask = vec > SPLADE_THRESHOLD
            indices = nonzero_mask.nonzero(as_tuple=False).squeeze(-1)
            results.append({
                "indices": indices.cpu().tolist(),
                "values": vec[indices].cpu().float().tolist(),
            })
        return results

    def encode(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """Encode texts into SAE-SPLADE sparse vectors.

        `is_query` is accepted for API compatibility but has no effect here —
        the same SAE is used for both documents and queries, consistent with
        how SPLADE symmetric models work. Asymmetric query handling can be
        added later via the IT model at the client/agent layer.

        Args:
            texts: List of strings to embed.
            is_query: Ignored (kept for interface compatibility).

        Returns:
            List of dicts with 'indices' and 'values' keys.
        """
        all_results = []
        for i in range(0, len(texts), self.internal_batch_size):
            batch = texts[i : i + self.internal_batch_size]
            all_results.extend(self._encode_batch(batch))
        return all_results

    async def encode_async(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """Async wrapper for use in FastAPI async routes.

        Pushes GPU work to the thread pool so the event loop is not blocked.
        Call this from async route handlers instead of encode() directly.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.encode(texts, is_query),
        )
```

---

## FastAPI Route Update (server.py)

To avoid blocking the event loop during GPU inference, update the `/embed_sparse` route to use the async
wrapper. This is the fix for the throughput problem — the dense and sparse routes can interleave instead of
serializing.

Change the route signature and call in `server.py`:

```python
# Before:
@app.post("/embed_sparse", response_model=SparseEmbedResponse)
def embed_sparse(req: SparseEmbedRequest):
    ...
    raw = _sparse.encode(req.texts, is_query=req.is_query)

# After:
@app.post("/embed_sparse", response_model=SparseEmbedResponse)
async def embed_sparse(req: SparseEmbedRequest):
    ...
    raw = await _sparse.encode_async(req.texts, is_query=req.is_query)
```

Apply the same async pattern to `/embed_dense` if the dense embedder also supports it. Both routes
sharing a single thread pool executor (one worker) is correct — the GPU is not multi-threaded; this just
frees the event loop to accept new connections while one request is on the GPU.

---

## Startup Validation Sequence

The `_load_models()` and `_validate_dimensions()` calls in `__init__` will:

1. Load backbone and SAE from local disk
2. Assert both are on CUDA (fail loudly if not)
3. Assert `backbone.config.hidden_size == sae_cfg["d_in"]` (fail loudly if not)
4. Log the SAE width (`d_sae`) so you can confirm which checkpoint is active

If startup fails with a dimension mismatch, check:
- That `BACKBONE_LOCAL_PATH` points to Gemma 3 270M PT (not a different size)
- That `SAE_ID` matches a checkpoint actually present in `SAE_LOCAL_PATH/resid_post/`
- That `sae_cfg["d_in"]` equals 1152 (the expected hidden size for this model)

Do not suppress these assertions. Silent dimension mismatches produce vectors that look valid but are garbage.

---

## Quick Smoke Test

Run this after deploying to confirm the pipeline is working end-to-end:

```python
import torch

embedder = SparseEmbedder()

# Basic forward pass
results = embedder.encode(["salt loop leak repair procedures"])
assert len(results) == 1
assert len(results[0]["indices"]) > 0, "No active features — check threshold or model load"
assert len(results[0]["indices"]) < 5000, (
    f"Too many active features ({len(results[0]['indices'])}). "
    f"Raise SPLADE_THRESHOLD or check SAE L0."
)

# Confirm GPU residency
assert next(embedder.backbone.parameters()).device.type == "cuda"
assert next(embedder.sae.parameters()).device.type == "cuda"

# Confirm values are float (not bfloat16) — Qdrant expects float32
assert isinstance(results[0]["values"][0], float)

# Confirm no NaN or Inf
vals = torch.tensor(results[0]["values"])
assert not vals.isnan().any(), "NaN in output vectors"
assert not vals.isinf().any(), "Inf in output vectors"

print(f"OK — {len(results[0]['indices'])} active features")
print(f"Sample indices: {results[0]['indices'][:5]}")
print(f"Sample values:  {results[0]['values'][:5]}")
```

A healthy output will show 20–150 active features for a typical sentence. If you see 0 features, lower
`SPLADE_THRESHOLD`. If you see >500 features consistently, raise it or switch to a smaller-L0 checkpoint
(`l0_small`).

---

## Using the IT Model Later (gemma-scope-2-270m-it)

Do not plug `gemma-scope-2-270m-it` into the embedding server. It does not belong there.

The right place is at the agent/client layer, upstream of the embedding call. The pattern is:

1. User asks: *"how do we handle salt loop leaks?"*
2. Agent sends that to the IT model with a system prompt instructing it to reformulate the query into
   precise technical language matching the document corpus vocabulary.
3. IT model returns: *"Procedures for molten salt reactor coolant loop leak detection, sealant
   application, and structural repair protocols."*
4. Agent sends the reformulated string to `/embed_sparse` and `/embed_dense` as normal.

This keeps the server stateless and simple. The IT model is query expansion/rewriting middleware, not
part of the indexing pipeline. The PT model handles all indexing (document embedding); the IT model
handles query understanding. They never need to be in the same process.

When ready to wire this up, the IT model can be served as a separate lightweight inference endpoint or
called directly in the agent's query pipeline before hitting the embedding server.

---

## Notes for the Agentic IDE

- `sae-lens==6.37.6` is the current PyPI release. Pin this version.
- `SAE.from_pretrained(release=<local_path>, sae_id=<subfolder_name>)` is the correct local-load API.
  The `release` argument accepts a local directory path, not just HuggingFace release names.
- `sae_cfg` returned by `from_pretrained` is a dict. Access `sae_cfg["d_in"]` and `sae_cfg["d_sae"]`
  for dimension info. Do not assume these values; read them from the checkpoint.
- The backbone is loaded with `AutoModel`, not `AutoModelForCausalLM`, because we only need hidden states,
  not logits.
- `output_hidden_states=True` is set at load time via the constructor. Alternatively it can be passed at
  inference time in the `forward()` call — either works.
- `hidden_states[SAE_HOOK_LAYER + 1]` uses +1 because index 0 is the token embedding layer, not a
  transformer block output.
- The `values` list in the output dict is converted to Python `float` (`.float().tolist()`) before
  returning. Do not return `bfloat16` values — Qdrant and most downstream consumers expect float32.
- If `SAE.from_pretrained` raises a path error, verify the local clone contains the
  `resid_post/layer_12_width_65k_l0_medium/` subdirectory and a `config.json` inside it.
