# SAE on SPLADE — Requirements (Revised)

## Functional Requirements

### FR-1: Corpus Ingestion and Activation Capture

**FR-1.1** The system shall scroll the entire existing Qdrant collection using `with_payload=True, with_vectors=False` and extract the point ID and chunk text from each point's payload. The resulting `(point_id, text)` pairs are the canonical input to the activation extractor. Point IDs shall be persisted to `chunk_ids.npy` in scroll order.

**FR-1.2** Each document shall be tokenized using the `naver/splade-cocondenser-ensembledistil` tokenizer with truncation at 256 tokens. Queries shall be truncated at 32 tokens.

**FR-1.3** The system shall run each tokenized document through the SPLADE model and apply the native pooling operation: `log1p(relu(logits)).max(dim=sequence)`. No sparsification threshold shall be applied at this stage.

**FR-1.4** The system shall produce a single `(N, 30522)` float32 matrix persisted to disk as a NumPy memmap file, where N is the number of points in the collection.

**FR-1.5** The system shall compute and persist the corpus mean activation vector `(30522,)` as a separate `.npy` file. This is required for SAE pre-encoder bias initialization.

**FR-1.6** The system shall log the distribution of per-document nonzero activation counts. The mean of this distribution is used to determine K.

**FR-1.7** The count of scrolled Qdrant points shall match the expected corpus size exactly before activation capture begins.

---

### FR-2: SAELens Configuration and DataProvider

**FR-2.1** The SAE shall be trained using SAELens v6 (6.43.0) with `architecture="topk"`.

**FR-2.2** The system shall implement a `DataProvider` (a plain `Iterator[torch.Tensor]` generator) that wraps the pre-captured memmap tensor and yields shuffled batches. This is the sole integration point between the activation capture pipeline and SAELens.

**FR-2.3** The DataProvider shall shuffle document order across epochs using `mixing_buffer()`.

**FR-2.4** The SAELens configuration shall set `d_in=30522`, `d_sae=30522 × expansion_factor` (default 4× = 122,088), `activation_fn_kwargs={"k": K}`, `dtype="float32"`, and `normalize_activations="none"`.

**FR-2.5** WandB logging shall be enabled. Training shall emit reconstruction loss, auxiliary loss, and dead feature percentage.

**FR-2.6** The SAE pre-encoder bias shall be initialized to the corpus mean vector from FR-1.5 before training begins, outside the SAELens training loop.

---

### FR-3: SAE Architectural Constraints

**FR-3.1** All encoder output values shall be non-negative, enforced by ReLU before the Top-K gate.

**FR-3.2** The Top-K gate shall produce exactly K nonzero values per vector. K is a fixed integer, not a threshold or regularization coefficient.

**FR-3.3** Decoder column vectors shall be constrained to unit L2 norm at all times via renormalization after every optimizer step.

**FR-3.4** The decoder shall have no bias term.

---

### FR-4: Training

**FR-4.1** Training shall run on the RTX 5090 for 20 epochs.

**FR-4.2** SAELens's native dead-feature auxiliary loss shall be enabled and weighted at 1/32 of the primary reconstruction loss.

**FR-4.3** Gradient clipping shall be applied at max norm 1.0.

**FR-4.4** The optimizer shall be Adam with lr=2e-4, β1=0.9, β2=0.999, with cosine decay schedule over the full training run.

**FR-4.5** A model checkpoint shall be saved at the end of every epoch, including model weights, optimizer state, scheduler state, epoch number, and the config used.

---

### FR-5: Validation

**FR-5.1** Before indexing, reconstruction fidelity shall be verified on 20 randomly sampled documents. The top-20 tokens by weight in the reconstructed vector must overlap with the top-20 in the original activation by at least 60% on average.

**FR-5.2** A sparsity exactness check shall be run across all documents. Every encoded vector must have exactly K nonzero values and no negative values. Any violation is a critical error blocking indexing.

**FR-5.3** A dead feature audit shall be run across the full corpus after training. More than 10% dead features blocks indexing and triggers a retraining recommendation.

**FR-5.4** A feature report shall be generated listing the top 20 most frequently firing features, with the top 20 documents by activation value for each. Saved as a Markdown file for human review before indexing proceeds.

---

### FR-6: Serving Artifact Export

**FR-6.1** The encoder half of the SAE (pre-encoder bias + encoder weight + encoder bias) shall be exported as a TorchScript module.

**FR-6.2** The exported TorchScript module shall accept a batch of pre-threshold SPLADE activation tensors of shape `(B, 30522)` and return sparse vector tuples `(indices, values)` with exactly K nonzero entries per vector.

**FR-6.3** The decoder shall not be included in the serving artifact.

---

### FR-7: Qdrant Integration

**FR-7.1** SAE sparse vectors shall be added to a **new Qdrant collection** specified at ingest time. The existing collection shall not be modified.

**FR-7.2** The SAE collection payload shall copy source payload fields including `dense_chunk_ids` for cross-referencing back to the dense collection.

**FR-7.3** The SAE sparse vector shall be stored in Qdrant sparse format: integer indices + float32 values.

**FR-7.4** The sparse index shall be configured with `on_disk=false`.

**FR-7.5** Point IDs in the SAE collection shall be independent integers. No ID alignment with the existing collection is required.

---

### FR-8: Query Pipeline

**FR-8.1** At query time, the query text shall be run through SPLADE (pre-threshold activation) then through the TorchScript SAE encoder to produce a sparse query vector.

**FR-8.2** Dense and sparse Qdrant queries shall be issued in parallel.

**FR-8.3** Both result sets shall be passed to the LLM with chunk text and retrieval scores. No ranking fusion shall be applied before the LLM.

---

### FR-9: Serving Endpoint

**FR-9.1** The embedding server shall expose a new `/embed_sae` endpoint that runs the full pipeline internally: text → SPLADE → SAE encoder → sparse output.

**FR-9.2** The `/embed_sparse` endpoint shall remain unchanged for backward compatibility.

**FR-9.3** SAE sparse retrieval shall **replace** raw SPLADE as the default sparse signal. Retrieval is dense + SAE sparse.

---

### FR-10: Evaluation

**FR-10.1** The SAE sparse retrieval shall be evaluated as a third condition alongside the existing dense and raw SPLADE conditions, using the same query set, LLM judge, and scoring rubric already in use.

**FR-10.2** Evaluation outputs shall include LLM-rated relevance and satisfaction scores for the SAE sparse condition, in the same format as existing dense and SPLADE results.

**FR-10.3** Results shall be comparable against the existing dense and sparse baseline data without re-running those conditions.

---

## Non-Functional Requirements

### NFR-1: Hardware

**NFR-1.1** The entire pipeline (capture, training, indexing, serving) shall run on the RTX 5090. The DGX Spark is not required.

**NFR-1.2** The activation matrix (~215MB) shall fit comfortably in RTX 5090 VRAM. If VRAM is constrained during training due to SAELens overhead, the matrix may be held in system RAM and paged as needed.

**NFR-1.3** The TorchScript SAE encoder forward pass shall add no more than 5ms of latency per query on the RTX 5090.

---

### NFR-2: Reproducibility

**NFR-2.1** A fixed random seed shall be set before training.

**NFR-2.2** The corpus mean vector, K value, expansion_factor, and all hyperparameters shall be written to a `config.yaml` alongside model checkpoints.

**NFR-2.3** Training loss curves and dead feature percentage curves shall be persisted as artifacts alongside the final checkpoint (via WandB and as local files).

---

### NFR-3: Correctness Invariants

**NFR-3.1** No negative values shall appear in any SAE output vector at any point — indexing or query time.

**NFR-3.2** K used during indexing and K used during query encoding shall be identical (same exported TorchScript model used for both).

**NFR-3.3** Pre-threshold SPLADE activations shall be used as SAE input at both indexing and query time. Post-threshold activations shall never be fed to the SAE.

**NFR-3.4** The SAE collection shall be independent of the existing collection — no shared IDs, no shared schema modifications.

---

## Dependencies

| Dependency | Version Constraint | Purpose |
|---|---|---|
| PyTorch | ≥ 2.3 | SAE operations and TorchScript export |
| SAELens | 6.43.0 | SAE training framework |
| Transformers (HuggingFace) | ≥ 4.40 | SPLADE model loading |
| NumPy | ≥ 1.26 | Activation matrix memmap |
| Qdrant Client | ≥ 1.9 | Sparse and dense vector operations |
| CUDA | ≥ 12.4 | RTX 5090 (Blackwell) support |
| WandB | latest | Training observability |

---

## Out of Scope

- RRF or any rank fusion algorithm
- L1 regularization as a sparsity mechanism
- DGX Spark (corpus size does not require it)
- Post-threshold SPLADE activations as SAE training data
- Decoder at serving time
- Custom PyTorch training loop (SAELens v6 provides this)
- Quantization of sparse vector weights
- Rebuilding the existing Qdrant collection
- ID alignment between SAE and dense collections
