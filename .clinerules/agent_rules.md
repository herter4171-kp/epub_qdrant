# Rules for Agentic Coding

These rules are non-negotiable. Violations cause real data loss, hours of wasted compute, and corrupted indexes. Each rule includes a concrete example of what failure looks like.

---

## Fault Tolerance
Assume GPU jobs die. Assume network drops mid-transfer. Assume disks fill. Design so restart costs only the current unit of work, not everything before it.

*A two-phase ingest pipeline embedded dense vectors for 78,009 chunks over 27 minutes, then attempted to write all of them in a single upsert call. The call failed with a 1.4GB payload against a 32MB limit. Zero points were written. All embedding work was lost. Upserting in batches of 500 as embedding completes limits loss to one batch and makes partial progress visible immediately.*

---

## Observability
Log what happened and whether it was correct. Status codes are not proof of correctness. Log inputs, outputs, and quality metrics at every stage boundary. Silent success is indistinguishable from silent failure.

*The embedder logged that `/embed_sparse` returned 200 OK. It did not log that it returned 37 vectors with NNZ min=105 max=408 mean=200. The difference between those two log lines is the difference between knowing the model is working and knowing the server is running. The bot removed the useful logs and kept the useless ones, calling it a cleanup.*

---

## Scope
Do not expand scope without explicit instruction. Fixing a bug does not authorize refactoring. Adding a feature does not authorize restructuring. Do only what was asked.

*Asked to add a `--no-rewrite` flag to the embedding server, the bot restructured the startup sequence, collapsed model loading logic, and removed per-model exception handling. None of that was asked for. All of it introduced new failure modes.*

---

## Code Integrity
Do not replace working code. Extend it. Simpler code that loses fault tolerance, observability, or correctness under edge cases is regression, not improvement.

*The original embedder logged rewrite input and output with character counts, showed NNZ statistics per encode call, and warned on empty rewriter output with a fallback to the original query. The bot replaced it with a version that logged nothing meaningful. It called this a cleanup. The result was a system that appeared to work while silently producing garbage.*

---

## Memory
Treat memory as a finite resource. Release it as soon as it is no longer needed. Process in bounded batches. Never accumulate a collection whose size depends on input scale. A crash should lose one batch, not everything.

*The ingest script accumulated all chunk texts, metadata dicts, and 768-dimensional dense vectors for all 78,009 chunks simultaneously before writing anything. Each list grew proportionally to corpus size with no upper bound. The single upsert call then attempted to send a 1.4GB JSON payload to Qdrant, which has a 32MB limit. The call failed. Nothing was written. Upserting in batches of 500 caps memory usage, limits crash loss to one batch, and lets you see Qdrant point count climb in real time.*

---

## Pipelines
Phases are ordered for reasons. Do not collapse, reorder, or migrate work between phases without understanding why the boundary exists.

*The ingest pipeline calls dense embedding in Phase 1 and sparse embedding in Phase 2 because Phase 2 can resume independently — it scrolls existing points from Qdrant and fills in missing sparse vectors. The bot wired the IT query rewriter into Phase 1 dense embedding. 78,036 document chunks were rewritten by an instruction-tuned model that returned `"```python"` and `"Okay I understand"` as search queries. These garbage strings were embedded and stored as dense vectors in Qdrant.*

---

## Phase Idempotency
Every phase must be resumable. A phase that can be restarted without repeating completed work is idempotent. Phases that are not idempotent are fragile by design.

*Phase 2 sparse embedding scrolls the entire collection and re-embeds sparse vectors for every point, including points that already have them. A crash mid-Phase 2 means restarting from the beginning and re-embedding everything. The fix is one check: fetch sparse vectors during scroll and skip points where sparse is already populated.*

---

## Separation of Concerns
Query-time components run at query time. Index-time components run at index time. Do not cross these boundaries.

*The IT query rewriter exists to reformulate user queries before retrieval. It is a query-time component. The bot called it on every document chunk during indexing. The instruction-tuned model responded to paper abstracts as if they were user instructions, producing responses like `"This is a good overview of the key areas of research"` which were silently embedded as dense vectors and stored in Qdrant. The index was corrupted. The logs showed only HTTP 200 responses.*

---

## Dependencies
Load from local paths. Do not introduce network calls where local resources exist. Validate paths explicitly at startup.

*Models live in `/tank/huggingface/`. Always pass `local_files_only=True`. A missing hyphen in a path constant — `splade-cocondenser-ensembledistil` instead of `splade-cocondenser-ensemble-distil` — causes a silent fallback to a network download at startup. The correct behavior is an immediate assertion failure with a clear error message, not a silent network call.*

---

## Introspection
Before acting, re-read the original instructions. Before completing, verify the result matches the original intent — not just the last message.

*The original design called for the IT model to rewrite user queries at query time. Three conversations later the bot had wired it into document indexing at index time, removed the logging that would have shown this was happening, and reported success based on HTTP 200 responses. It had drifted so far from the original intent that it was actively corrupting the index while appearing to work correctly. Re-reading the original architecture description would have caught this immediately.*
