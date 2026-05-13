# SAE on SPLADE — Tasks (Revised)

## How to Read This Document

Tasks are ordered by dependency. A task cannot begin until all tasks it depends on are complete. All tasks run on the RTX 5090. Estimated effort is in engineer-hours for a single ML engineer — the corpus is whatever the collection has so runtimes are short throughout.

Status values: `- [ ]` not started · `- [~]` in progress · `- [x]` complete · `- [!]` blocked

---

## Phase 0 — Pre-Work

### T0.1 — Measure corpus SPLADE nonzero statistics
**Effort:** 1–2 hours  
**Depends on:** nothing  
**Blocks:** T0.2, T1.1

Run a sample of 200–300 papers through SPLADE and record the distribution of nonzero activation counts in the pre-threshold output vector. Compute mean, median, p10, p90. The mean (or median if the distribution is skewed) becomes K. Record in `config.yaml`.

**Done when:** Distribution plot saved. K value chosen and documented.

---

### T0.2 — Confirm expansion_factor and storage budget
**Effort:** 30 minutes  
**Depends on:** T0.1  
**Blocks:** T2.1

Compute expected sparse index size: `N × K × 8 bytes`. Confirm `expansion_factor=4` (giving `d_sae = 30522 × 4 = 122,088`) as the default. Record in `config.yaml`.

**Done when:** `config.yaml` contains `expansion_factor`, `K`, `d_in=30522`, `d_sae=122088`, `dense_dim=768`.

---

### T0.3 — Verify environment
**Effort:** 1 hour  
**Depends on:** nothing  
**Blocks:** T1.1, T2.1

Confirm on RTX 5090: CUDA ≥ 12.4, PyTorch ≥ 2.3, SAELens 6.43.0, HuggingFace Transformers ≥ 4.40, Qdrant client ≥ 1.9, WandB. Load `naver/splade-cocondenser-ensembledistil` and run a single forward pass without error. Confirm TorchScript export works.

**Done when:** All imports succeed. SPLADE forward pass produces output of shape `(1, 30522)`.

---

## Phase 1 — Activation Capture

### T1.1 — Scroll Qdrant collection and extract payload
**Effort:** 1–2 hours  
**Depends on:** T0.3  
**Blocks:** T1.2

Scroll the entire existing Qdrant collection using the client's scroll API with `with_payload=True` and `with_vectors=False`. For each point, extract the point ID and the chunk text from the payload. Write `(point_id, text)` pairs to an ordered list that becomes the canonical input to the activation extractor. Persist the point ID array to `chunk_ids.npy` in scroll order — this is the authoritative ID mapping for all subsequent upserts.

This approach is correct by construction: the text is exactly what was indexed, the IDs are exactly the Qdrant point IDs, and the chunking is whatever chunking already exists in the collection. No alignment verification step is needed because there is no separate source to misalign with.

**Done when:** All points scrolled. `chunk_ids.npy` saved. Count of extracted chunks matches Qdrant collection point count exactly.

---

### T1.2 — Implement SPLADE activation extractor
**Effort:** 2 hours  
**Depends on:** T1.1  
**Blocks:** T1.3

Write the module that: tokenizes input text at 256 tokens (32 for queries); runs the SPLADE forward pass; applies `log1p(relu(logits)).max(dim=1)` to produce the pre-threshold activation vector; returns float32 tensors of shape `(batch_size, 30522)`. Batch size 64–128 is safe on the RTX 5090 for this input length.

Unit tests: output shape is `(batch_size, 30522)`; all values ≥ 0; at least 500 nonzero values per document (guards against accidental thresholding).

**Done when:** Unit tests pass.

---

### T1.3 — Run full activation capture
**Effort:** 30 minutes runtime  
**Depends on:** T1.1, T1.2  
**Blocks:** T1.4

Run all documents through the extractor. Write results to a NumPy memmap file `activations.npy` of shape `(N, 30522)`. Persist chunk IDs in matching order to `chunk_ids.npy`. Log progress.

**Done when:** `activations.npy` exists with correct shape (~215MB). Chunk ID array length matches row count.

---

### T1.4 — Compute corpus statistics
**Effort:** 30 minutes  
**Depends on:** T1.3  
**Blocks:** T2.1

Compute corpus mean vector `(30522,)` over all rows and save as `corpus_mean.npy`. Compute per-document nonzero counts and compare against T0.1 sample — should match within expected variance. Spot-check 10 random activation vectors: top tokens should be semantically plausible given paper content.

**Done when:** `corpus_mean.npy` saved. Spot-check passed. Statistics documented.

---

## Phase 2 — SAELens Setup

### T2.1 — Implement DataProvider for SAELens
**Effort:** 2–3 hours  
**Depends on:** T0.2, T1.4  
**Blocks:** T2.2

Implement a `DataProvider` (a plain `Iterator[torch.Tensor]` generator) that wraps the `activations.npy` memmap and yields shuffled batches of shape `(batch_size, 30522)` on the correct device. Use `mixing_buffer()` for shuffling. This is the only custom code interfacing with SAELens internals.

Unit tests: `next_batch()` returns correct shape; all values ≥ 0; shuffling produces different order across calls.

**Done when:** Unit tests pass. DataProvider confirmed compatible with SAELens trainer's expected interface by instantiating the trainer in a dry-run with 10 steps.

---

### T2.2 — Configure SAELens training run
**Effort:** 1 hour  
**Depends on:** T2.1  
**Blocks:** T3.1

Write the SAELens training config: `architecture="topk"`, `d_in=30522`, `d_sae=122088` (or `30522 × expansion_factor`), `activation_fn_kwargs={"k": K}`, `dtype="float32"`, `normalize_activations="none"`, WandB enabled. Wire in the DataProvider. Set the pre-encoder bias to `corpus_mean.npy` before training starts.

**Done when:** Config instantiates without error. WandB run initializes. Pre-encoder bias verified to match corpus mean values.

---

## Phase 3 — Training

### T3.1 — Smoke test (5 epochs, full corpus)
**Effort:** 30 minutes  
**Depends on:** T2.2  
**Blocks:** T3.2

Run 5 epochs. Verify: reconstruction loss decreases in epoch 1; dead feature percentage is trending downward; no NaN or OOM; decoder columns remain unit-norm; WandB receives data. At typical corpus sizes this should complete in a few minutes.

**Done when:** 5 epochs complete cleanly. Loss curve and dead feature curve visible in WandB. No anomalies.

---

### T3.2 — Full 20-epoch training run
**Effort:** 20–30 minutes runtime  
**Depends on:** T3.1  
**Blocks:** T4.1

Run full 20 epochs. Monitor dead feature percentage — if it plateaus above 10% after epoch 8, increase aux_weight and resume from last checkpoint. Save final checkpoint with all artifacts.

**Done when:** 20 epochs complete. Dead feature % < 10% and stable. Final checkpoint saved. Loss curves archived locally and in WandB.

---

## Phase 4 — Validation

### T4.1 — Reconstruction fidelity check
**Effort:** 1 hour  
**Depends on:** T3.2  
**Blocks:** T4.4 (must pass)

Sample 20 documents. For each, run through SAE encode + decode. Compare top-20 tokens by weight in reconstruction vs. original. Compute mean overlap. Target: ≥ 60%.

If < 60%: document the failure, return to T3.2 with adjusted hyperparameters.

**Done when:** Mean overlap computed. Pass/fail recorded.

---

### T4.2 — Sparsity exactness check
**Effort:** 30 minutes  
**Depends on:** T3.2  
**Blocks:** T4.4 (must pass)

Encode all documents using the trained SAE encoder. Assert every output vector has exactly K nonzero values and no negative values. Any violation is a critical error.

**Done when:** Zero violations. Confirmed across full corpus.

---

### T4.3 — Dead feature audit
**Effort:** 30 minutes  
**Depends on:** T3.2  
**Blocks:** T4.4 (must pass)

Encode full corpus. Count features that fired zero times. If > 10% dead, block indexing and increase aux_weight for retraining.

**Done when:** Dead feature percentage documented. Pass/fail recorded.

---

### T4.4 — Feature interpretability report
**Effort:** 2 hours  
**Depends on:** T4.1, T4.2, T4.3 (all must pass)  
**Blocks:** T5.1

Identify top 20 most frequently firing features. For each, retrieve top 20 documents by activation value. Save as `feature_report.md`: one section per feature, with paper titles, activation values, and first 200 characters of text. Human reviewer reads and signs off that at least 15 of 20 features have a discernible semantic theme.

**Done when:** Report saved. Reviewer sign-off recorded.

---

## Phase 5 — Export and Indexing

### T5.1 — Export TorchScript serving artifact
**Effort:** 1 hour  
**Depends on:** T4.4  
**Blocks:** T5.2

Export the encoder half (pre-bias + encoder weight + encoder bias) as TorchScript. Verify the export round-trip is lossless by re-running T4.2 using the TorchScript module. Confirm output is identical to the PyTorch version.

**Done when:** TorchScript file saved. Sparsity check passes on TorchScript output.

---

### T5.2 — Create new Qdrant collection for SAE vectors
**Effort:** 1 hour  
**Depends on:** T5.1  
**Blocks:** T5.3

Create a new Qdrant collection for SAE sparse vectors. Configure sparse vector with `on_disk=false`. Verify the schema with a single test upsert and retrieval. Confirm the collection is independent of the existing collection.

**Done when:** New collection created. Test point upserted and sparse vector retrievable.

---

### T5.3 — Implement and run indexing pipeline
**Effort:** 2 hours  
**Depends on:** T5.2  
**Blocks:** T5.4

Implement the indexing loop: for each document, run through SPLADE → TorchScript SAE encoder; format output as Qdrant sparse vector; upsert to new collection with independent integer IDs. Payload includes `dense_chunk_ids` for cross-referencing. Process in batches. Log progress.

At typical corpus sizes this completes in minutes.

**Done when:** All points upserted. Spot-check 10 random points: sparse vector present, correct number of nonzeros (K), payload intact.

---

### T5.4 — Verify indexed collection
**Effort:** 30 minutes  
**Depends on:** T5.3  
**Blocks:** T6.1

Run a test sparse query against the new collection. Confirm results are returned with scores. Confirm that the same query via the dense named vector on the source collection also returns results. Both retrieval legs operational.

**Done when:** Both sparse and dense queries return results on test queries.

---

## Phase 6 — Query Pipeline and Evaluation

### T6.1 — Implement /embed_sae endpoint
**Effort:** 2 hours  
**Depends on:** T5.4  
**Blocks:** T6.2

Implement the `/embed_sae` endpoint on the existing embedding server. Accepts query text; runs SPLADE pre-threshold activation; runs TorchScript SAE encoder; returns sparse vector. Testable independently.

Measure end-to-end latency on 10 test queries. Target: < 200ms total.

**Done when:** Endpoint runs end-to-end. Latency target met.

---

### T6.2 — Offline evaluation vs. baseline
**Effort:** 3–4 hours  
**Depends on:** T6.1  
**Blocks:** T6.3

Run the full existing query set through the SAE sparse retrieval condition. Pass results to the LLM judge using the same prompt and scoring rubric used for the existing dense and raw SPLADE conditions. Record LLM-rated relevance and satisfaction scores.

Compare against existing baseline data. Flag any condition where SAE sparse recall is more than 5% below raw SPLADE — if so, investigate K or latent_dim before proceeding.

**Done when:** Evaluation scores recorded. Comparison table saved alongside existing baseline data.

---

### T6.3 — Full pipeline evaluation with LLM fusion
**Effort:** 3–4 hours  
**Depends on:** T6.2  
**Blocks:** nothing (final task)

Run the complete pipeline — both dense and SAE sparse candidates passed to the LLM — across the full query set. Score with the same LLM judge. Compare satisfaction and relevance against dense-only and SPLADE-only baselines. Reference the sparse fraction contour from the research artifacts to inform how many candidates from each leg to pass.

**Done when:** Full evaluation results documented. System declared ready for production use or returned to T3.2 with specific guidance.

---

## Full Dependency Order

```
T0.1 ──→ T0.2 ──→ T2.1
T0.3 ──→ T1.1 ──→ T1.2 ──→ T1.3 ──→ T1.4 ──→ T2.1
                                               ↓
                                T2.1 ──→ T2.2 ──→ T3.1 ──→ T3.2
                                                             ↓
                                          T4.1, T4.2, T4.3 ──→ T4.4
                                                                  ↓
                                               T5.1 ──→ T5.2 ──→ T5.3 ──→ T5.4
                                                                             ↓
                                                             T6.1 ──→ T6.2 ──→ T6.3
```

## Estimated Total Timeline

| Phase | Effort |
|---|---|
| Phase 0 | 2 hours |
| Phase 1 | 4 hours + 30 min runtime |
| Phase 2 | 4 hours |
| Phase 3 | 1 hour + 30 min runtime |
| Phase 4 | 4 hours |
| Phase 5 | 4 hours |
| Phase 6 | 8 hours |
| **Total** | **~27 engineer-hours** |

At this corpus size, no task is blocked by compute time. The bottleneck is engineering and validation time.
