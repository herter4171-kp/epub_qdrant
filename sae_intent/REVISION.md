Here's the complete list of decisions made during our review that differ from the original `./sae_intent` documents:

## Decisions / Changes from Original Spec

| # | Original (`sae_intent`) | Decision |
|---|---|---|
| 1 | Hardcoded "1,766 documents", "215MB" | Size is N vectors (determined at scroll time). Don't hardcode corpus size — it's whatever the collection has. |
| 2 | SPLADE truncation at 512 tokens (FR-1.2) | Use whatever SPLADE uses — currently 256 for docs, 32 for queries. SAE input is intrinsically derived from the same SPLADE pass that produces the existing sparse vectors. Keep 32-token query limit for comparability with past results. |
| 3 | Add `sae_sparse` named vector to existing collection (FR-7.1) | SAE sparse vectors go into a **new collection** specified at ingest time. Payload is copied from source (includes `dense_chunk_ids` for cross-referencing). No modification to existing collections. |
| 4 | Custom `ActivationsStore` subclass with `next_batch()` (FR-2.2) | SAELens v6 uses `SAETrainer` + `DataProvider` (a plain `Iterator[torch.Tensor]` generator) + `mixing_buffer()` for shuffling. No `ActivationsStore` subclass exists in v6. |
| 5 | No serving location specified | SAE encoder lives in the **embedding server** as a new `/embed_sae` endpoint. Runs the full pipeline internally: text → SPLADE → SAE encoder → sparse output. Must be testable independently. |
| 6 | SAE as a "third condition" alongside dense and raw SPLADE (FR-9.1) | SAE **replaces** raw SPLADE as the default sparse signal. Retrieval is dense + SAE sparse. Raw SPLADE (`/embed_sparse`) remains available for backward compatibility but is no longer the default retrieval path. |
| 7 | Scroll order concerns / ID alignment with existing collection | Doesn't matter. SAE collection is new with its own IDs. Payload carries `dense_chunk_ids` for cross-referencing. No ID alignment needed between collections. |
| 8 | `d_sae = 49152` (1.6× expansion) | Express as `expansion_factor` config parameter, **default 4×**. So `d_sae = 30522 × 4 = 122,088`. |
| 9 | Task format uses `[ ]` / `[~]` / `[x]` / `[!]` | Convert to Kiro `- [ ]` checkbox format. |
| 10 | No rollback plan documented | Rollback is version control. No explicit cleanup task needed. |

## Additional Clarifications Established

- **Ingestion flow**: Text → SPLADE (same as existing `/embed_sparse`) → SAE encoder → SAE sparse vector → new Qdrant collection
- **Query flow**: Query → `/embed_sae` (SPLADE + SAE encoder internally) → search SAE collection; parallel dense search on source collection → RRF → LLM
- **SAE collection payload**: Copies source payload fields including `dense_chunk_ids` array mapping back to the dense collection's point IDs
- **Endpoint design**: `/embed_sparse` unchanged (backward compat); `/embed_sae` is the new default endpoint callers should use
- **Point IDs**: Integer type in existing collections (sequential). SAE collection gets its own independent integer IDs.
- **SAELens version**: 6.43.0 (latest). Uses `StandardTrainingSAEConfig`, `SAETrainerConfig`, `SAETrainer.fit()`, `mixing_buffer()`.

