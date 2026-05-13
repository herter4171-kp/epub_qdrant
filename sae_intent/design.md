# SAE on SPLADE — Design Document (Revised)

## Overview

This system augments an existing SPLADE sparse retrieval pipeline with a Sparse Autoencoder (SAE) that re-expresses SPLADE's token-level activations into a larger, more semantically disentangled latent space. The expanded sparse vectors are added to a **new Qdrant collection** (not the existing one). Fusion is handled by the LLM in-context, not by a ranking algorithm.

The corpus is whatever the existing Qdrant collection contains — N vectors, determined at scroll time. The activation matrix is approximately 215MB. The entire pipeline runs on the RTX 5090. The DGX Spark is not required.

The SAE is trained offline using SAELens v6 (6.43.0) with a `DataProvider` (a plain `Iterator[torch.Tensor]` generator) + `mixing_buffer()` for shuffling. At serving time only the encoder half runs.

---

## System Architecture

### Data Flow — Indexing

```
Existing Qdrant collection (scroll with_payload=True, with_vectors=False)
  → SPLADE forward pass (naver/splade-cocondenser-ensembledistil)
  → Pre-threshold activation capture (30,522-dim float32)
  → NumPy memmap (~215MB)
  → SAELens v6 training (SAETrainer + DataProvider wrapping memmap)
  → Trained SAE encoder exported as TorchScript
  → Corpus re-encoded: SPLADE → SAE encoder → Top-K gate
  → Sparse vectors (indices + values, K nonzeros each)
  → Qdrant: create new collection, upsert sae_sparse vectors
```

### Data Flow — Query Time

```
Query text
  → SPLADE forward pass (pre-threshold activation)
  → TorchScript SAE encoder → ReLU → Top-K gate
  → Qdrant sae_sparse search → top-N sparse candidates + scores

Query text (parallel)
  → Dense embedding (768-dim)
  → Qdrant dense search on source collection → top-N dense candidates + scores

Both candidate sets + chunk text + scores → LLM (fuses in-context)
```

### What the LLM Receives

Both result sets are passed with chunk text and retrieval scores. No RRF, no unified ranking step. The LLM performs fusion as part of its reasoning — consistent with the existing retrieval architecture.

---

## Hardware

The corpus is small enough that the RTX 5090 handles everything. The DGX Spark is not in scope.

| Task | Hardware |
|---|---|
| Activation capture | RTX 5090 |
| SAE training (via SAELens) | RTX 5090 |
| Qdrant indexing | RTX 5090 |
| Query-time serving | RTX 5090 |

---

## Component Design

### 1. Activation Capture

**Purpose:** Extract pre-threshold SPLADE activations from the existing Qdrant collection and persist them as a memory-mappable float32 matrix of shape `(N, 30522)`.

Activations must be captured before SPLADE's sparsification threshold. Post-threshold vectors have already discarded small-but-meaningful activations that the SAE needs in order to learn good features.

The pooling operation is `log1p(relu(logits)).max(dim=sequence)` — SPLADE's native pooling, reproduced exactly.

The corpus mean vector `(30522,)` is computed over all documents and saved separately. It initializes the SAE pre-encoder bias and is architecturally load-bearing for training stability.

**Corpus ingestion:** text is scrolled directly from the existing Qdrant collection payload (`with_payload=True, with_vectors=False`). This is correct by construction — the text is exactly what was indexed, the IDs are the existing point IDs, and the chunking is whatever is already in the collection. No JSON parsing, no re-chunking, no alignment risk.

**SPLADE truncation:** 256 tokens for documents, 32 tokens for queries. SAE input is intrinsically derived from the same SPLADE pass that produces the existing sparse vectors.

---

### 2. SAELens v6 Integration

SAELens v6 (6.43.0) provides Top-K SAE architecture, dead feature tracking, auxiliary dead-feature loss, decoder column normalization, and WandB logging as native capabilities. The only custom code required is a single data pipeline shim.

**DataProvider:** SAELens v6 uses `SAETrainer` + `DataProvider` (a plain `Iterator[torch.Tensor]` generator) + `mixing_buffer()` for shuffling. No `ActivationsStore` subclass exists in v6. The DataProvider wraps the `(N, 30522)` memmap tensor and yields shuffled batches of shape `(batch_size, 30522)`.

**SAELens training configuration:**

| Parameter | Value |
|---|---|
| `architecture` | `"topk"` |
| `activation_fn_kwargs` | `{"k": K}` (K measured from corpus) |
| `d_in` | `30522` |
| `d_sae` | `30522 × expansion_factor` (default 4× = 122,088) |
| `dtype` | `"float32"` |
| `normalize_activations` | `"none"` (pre-bias handles this) |
| WandB logging | enabled |

---

### 3. SAE Architecture (via SAELens)

**Pre-encoder bias** — `(30522,)` parameter initialized to corpus mean. Subtracted before encoding, added back after decoding. Prevents the encoder from wasting capacity on the mean signal shared across all documents.

**Encoder** — linear `30522 → d_sae` with bias, followed by ReLU. Non-negativity is architecturally required: SAE output values become Qdrant sparse vector weights. Negative weights would subtract relevance during dot-product scoring, which is semantically broken.

**Top-K gate** — keeps exactly K values per vector, zeroes the rest. Hard constant — not a loss term or threshold. Sparsity is exact and predictable per document, making Qdrant index size precisely estimable.

**Decoder** — linear `d_sae → 30522`, no bias. Column vectors constrained to unit L2 norm at all times via renormalization after every optimizer step. Prevents scale collapse where the model hides information in decoder column magnitude rather than direction.

**Serving artifact:** encoder only (pre-bias + encoder weight + encoder bias). Exported as TorchScript after training. The decoder is not deployed.

---

### 4. Qdrant Integration

**New collection:** SAE sparse vectors go into a **new collection** specified at ingest time. Payload is copied from source (includes `dense_chunk_ids` for cross-referencing). No modification to existing collections.

**Point IDs:** Integer type in existing collections (sequential). SAE collection gets its own independent integer IDs. No ID alignment needed between collections.

**Sparse vector format:** integer indices of K nonzero SAE features + float32 activation values at those positions.

**Index configuration:** `on_disk=false`. At typical corpus sizes the sparse index is negligible in size.

---

### 5. Serving — /embed_sae Endpoint

The SAE encoder lives in the **embedding server** as a new `/embed_sae` endpoint. It runs the full pipeline internally: text → SPLADE → SAE encoder → sparse output. Must be testable independently.

**Endpoint design:**
- `/embed_sparse` — unchanged (backward compatibility, returns raw SPLADE)
- `/embed_sae` — new default endpoint (returns SAE sparse vectors)

**Retrieval strategy:** SAE **replaces** raw SPLADE as the default sparse signal. Retrieval is dense + SAE sparse. Raw SPLADE remains available for backward compatibility but is no longer the default retrieval path.

---

### 6. Evaluation

The existing framework produces LLM-rated relevance and satisfaction scores for dense and raw SPLADE conditions, matching the structure of the contour data from the research artifacts. The SAE evaluation adds a third retrieval condition — SPLADE→SAE sparse — to the same framework using the same queries, same LLM judge, and same scoring rubric. Results slot directly into the existing data as new columns, enabling comparison on the same satisfaction/relevance axes already established.

---

## Key Parameters

| Parameter | Value | Source |
|---|---|---|
| `input_dim` | 30,522 | BERT/SPLADE vocabulary — fixed |
| `expansion_factor` | 4 (configurable) | Default 4× expansion |
| `d_sae` | 122,088 | 30,522 × 4 |
| `K` | Measure from corpus | Mean nonzero count in pre-threshold SPLADE output |
| `batch_size` | 256 | Appropriate for corpus size |
| `lr` | 2e-4 | Standard SAE learning rate |
| `aux_weight` | 1/32 | Dead feature revival without dominating primary loss |
| `epochs` | 20 | Will converge fast on small corpus |
| Dense vector dim | 768 | Matches existing Qdrant collection |
| SPLADE doc truncation | 256 tokens | SPLADE default |
| SPLADE query truncation | 32 tokens | Comparability with past results |

---

## Failure Modes

| Failure | Symptom | Mitigation |
|---|---|---|
| High dead features (>10%) | Dead % stable above threshold after epoch 5 | Increase aux_weight or reduce latent_dim |
| Poor reconstruction | Top tokens mismatched in spot-check | Verify pre-threshold capture; check pre-bias init |
| Negative sparse values | Qdrant scoring breaks | ReLU prevents this; verify in post-training sparsity check |

---

## Explicit Exclusions

- RRF or any rank fusion — LLM fuses in-context
- L1 regularization — Top-K gives exact sparsity control
- DGX Spark — corpus is small, RTX 5090 is sufficient
- Decoder at serving time — encoder-only TorchScript export
- Post-threshold SPLADE activations as SAE input
- Custom PyTorch training loop — SAELens v6 handles this natively
- Quantization — revisit only if storage becomes a concern
- Rebuilding the existing Qdrant collection
