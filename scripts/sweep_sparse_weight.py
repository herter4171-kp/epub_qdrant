#!/usr/bin/env python3
"""
Sparse weight sweep: find the optimal sparse/dense RRF balance.

For a small set of low-Jaccard queries (where the ratio actually matters),
this script:
  1. Fetches dense and sparse rank lists once from Qdrant (books-named / papers-named)
  2. Scores each unique chunk blindly via LLM (1-3 relevance, no collection label)
  3. Sweeps sparse_weight 0.0 -> 2.0 in 0.25 steps
  4. Simulates RRF at each weight using cached ranks + cached relevance scores
  5. Reports avg_relevance@5 per weight -> tells you where to set the multiplier

Qdrant 1.17.1 client syntax: uses NamedVector / NamedSparseVector model objects.

Usage:
    python scripts/sweep_sparse_weight.py
    python scripts/sweep_sparse_weight.py --top-k 10 --output sweep_results.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_mcp_server_dir = _project_root / "mcp_servers" / "retrieval"
if str(_mcp_server_dir) not in sys.path:
    sys.path.insert(0, str(_mcp_server_dir))

from mcp_server.retriever import Retriever
from mcp_server.config import settings
from mcp_server.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sweep")

# Low-Jaccard queries: where sparse weight actually changes results
# Selected from compare_collections.py output (Jaccard < 0.20)
QUERIES = [
    "I want to experiment with multi-agent problem solving where agents debate, specialize, and then converge. What patterns are worth trying first?",
    "What scaffolding do strong terminal-based coding agents use around the model itself?",
    "What are the main failure modes of web agents in realistic browsing tasks, and how should I evaluate them?",
    "How would you build a GUI agent that can operate a clunky enterprise web app with inconsistent layouts?",
    "What should I measure to catch post-merge quality problems in agent-generated pull requests?",
]

SPARSE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
K_RRF = 60
HYBRID_COLLECTIONS = ["books-named", "papers-named"]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RankedChunk:
    """A chunk with its dense rank, sparse rank, and text for LLM scoring."""
    point_id: str          # Qdrant point ID (used as key)
    doc_id: str            # arxiv_id or book title (human-readable)
    text: str
    title: str
    dense_rank: Optional[int] = None   # None = not in dense results
    sparse_rank: Optional[int] = None  # None = not in sparse results
    relevance_score: Optional[int] = None  # 1-3, filled by LLM


@dataclass
class QueryData:
    query: str
    chunks: Dict[str, RankedChunk] = field(default_factory=dict)  # point_id -> chunk


# ── Retrieval: fetch dense and sparse rank lists separately ──────────────────

def fetch_rank_lists(retriever: Retriever, query: str, top_k: int) -> QueryData:
    """
    Fetch dense-only and sparse-only rank lists from the named collections.
    Uses Qdrant 1.17.1 query_points() API with using= parameter.

    Pattern from test_hybrid_search.py:
      client.query_points(collection_name=..., query=vec, using="dense", limit=...)
      client.query_points(collection_name=..., query=SparseVector(...), using="sparse", limit=...)
    """
    from qdrant_client.models import SparseVector

    qdata = QueryData(query=query)
    client = retriever._client

    # Get the query embeddings via Retriever helpers
    dense_vector = retriever._embed(query)
    sparse_query = retriever._embed_sparse(query)  # {"indices": [...], "values": [...]}

    for collection in HYBRID_COLLECTIONS:
        # --- Dense-only search ---
        try:
            dense_hits = client.query_points(
                collection_name=collection,
                query=dense_vector,
                using="dense",
                limit=top_k,
            )
            for rank, hit in enumerate(dense_hits.points):
                pid = str(hit.id)
                payload = hit.payload or {}
                doc_id = payload.get("arxiv_id") or payload.get("book_title", "")[:80].lower()
                text = payload.get("text", "")
                title = payload.get("title", doc_id)
                if pid not in qdata.chunks:
                    qdata.chunks[pid] = RankedChunk(
                        point_id=pid, doc_id=doc_id, text=text, title=title
                    )
                if qdata.chunks[pid].dense_rank is None:
                    qdata.chunks[pid].dense_rank = rank
        except Exception as e:
            logger.warning(f"Dense search failed on {collection}: {e}")

        # --- Sparse-only search ---
        # sparse_query is a dict {"indices": [...], "values": [...]} from MiniCOIL client
        try:
            sparse_hits = client.query_points(
                collection_name=collection,
                query=SparseVector(
                    indices=sparse_query["indices"],
                    values=sparse_query["values"],
                ),
                using="sparse",
                limit=top_k,
            )
            for rank, hit in enumerate(sparse_hits.points):
                pid = str(hit.id)
                payload = hit.payload or {}
                doc_id = payload.get("arxiv_id") or payload.get("book_title", "")[:80].lower()
                text = payload.get("text", "")
                title = payload.get("title", doc_id)
                if pid not in qdata.chunks:
                    qdata.chunks[pid] = RankedChunk(
                        point_id=pid, doc_id=doc_id, text=text, title=title
                    )
                if qdata.chunks[pid].sparse_rank is None:
                    qdata.chunks[pid].sparse_rank = rank
        except Exception as e:
            logger.warning(f"Sparse search failed on {collection}: {e}")

    logger.info(
        f"  Fetched {len(qdata.chunks)} unique chunks "
        f"({sum(1 for c in qdata.chunks.values() if c.dense_rank is not None)} dense, "
        f"{sum(1 for c in qdata.chunks.values() if c.sparse_rank is not None)} sparse)"
    )
    return qdata


# ── RRF simulation ────────────────────────────────────────────────────────────

def simulate_rrf(qdata: QueryData, sparse_weight: float, top_k: int) -> List[RankedChunk]:
    """
    Re-run RRF fusion with a given sparse weight multiplier.
    Returns the top_k chunks sorted by fused score.
    """
    scores: Dict[str, float] = defaultdict(float)

    for pid, chunk in qdata.chunks.items():
        if chunk.dense_rank is not None:
            scores[pid] += 1.0 / (K_RRF + chunk.dense_rank + 1)
        if chunk.sparse_rank is not None:
            scores[pid] += sparse_weight * (1.0 / (K_RRF + chunk.sparse_rank + 1))

    ranked = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
    return [qdata.chunks[pid] for pid in ranked[:top_k]]


# ── Blind LLM relevance scoring ───────────────────────────────────────────────

async def score_chunk(llm: LLMClient, query: str, chunk: RankedChunk) -> int:
    """
    Ask LLM to rate a single chunk's relevance to the query on a 1-3 scale.
    No collection label, no score -- completely blind.
    Returns 1, 2, or 3.
    """
    prompt = f"""Rate how relevant this text is to the query. Reply with ONLY a single digit: 1, 2, or 3.

1 = Not relevant (off-topic, tangential, or too generic)
2 = Somewhat relevant (related topic but doesn't directly address the query)
3 = Directly relevant (directly addresses what the query is asking)

Query: {query}

Text: {chunk.text[:600]}

Reply with only 1, 2, or 3."""

    try:
        response = await llm.answer(query=query, context=prompt)
        # Extract first digit found
        for char in response.strip():
            if char in ("1", "2", "3"):
                return int(char)
        logger.warning(f"Could not parse relevance score from: {response[:50]}")
        return 1
    except Exception as e:
        logger.error(f"LLM scoring failed: {e}")
        return 1


async def score_all_chunks(llm: LLMClient, all_query_data: List[QueryData]) -> None:
    """Score all unique chunks across all queries. Modifies in place."""
    tasks = []
    chunk_refs = []  # (qdata_index, point_id)

    for qi, qdata in enumerate(all_query_data):
        for pid, chunk in qdata.chunks.items():
            tasks.append(score_chunk(llm, qdata.query, chunk))
            chunk_refs.append((qi, pid))

    total = len(tasks)
    logger.info(f"Scoring {total} chunks with LLM (blind, 1-3 scale)...")

    # Run in batches of 10 to avoid hammering the API
    batch_size = 10
    for i in range(0, total, batch_size):
        batch = tasks[i:i + batch_size]
        refs = chunk_refs[i:i + batch_size]
        results = await asyncio.gather(*batch)
        for (qi, pid), score in zip(refs, results):
            all_query_data[qi].chunks[pid].relevance_score = score
        logger.info(f"  Scored {min(i + batch_size, total)}/{total} chunks")


# ── Main sweep ────────────────────────────────────────────────────────────────

def compute_avg_relevance_at_k(qdata: QueryData, sparse_weight: float, top_k: int) -> float:
    top_chunks = simulate_rrf(qdata, sparse_weight, top_k)
    scores = [c.relevance_score or 1 for c in top_chunks]
    return sum(scores) / len(scores) if scores else 0.0


async def run_sweep(top_k: int = 5, output_path: Optional[str] = None) -> None:
    retriever = Retriever()

    # Build LLM client (same pattern as evaluate.py)
    openai_base = os.getenv("OPENAI_API_BASE")
    api_url = openai_base.rstrip("/") if openai_base else settings.LITELLM_API_URL
    api_key = os.getenv("LITELLM_API_KEY") or settings.LITELLM_API_KEY
    model = "openai/qwen36"
    llm = LLMClient(api_url=api_url, api_key=api_key, model=model)

    # Step 1: fetch rank lists for all queries
    logger.info("Fetching dense + sparse rank lists...")
    all_query_data = []
    for query in QUERIES:
        logger.info(f"  Query: {query[:70]}...")
        qdata = fetch_rank_lists(retriever, query, top_k=top_k * 3)  # fetch more than top_k so sweep has room
        all_query_data.append(qdata)

    # Step 2: blind LLM scoring of all unique chunks
    await score_all_chunks(llm, all_query_data)

    # Step 3: sweep sparse weights
    logger.info("\nSweeping sparse weights...")
    weight_results: Dict[float, List[float]] = {w: [] for w in SPARSE_WEIGHTS}

    for qdata in all_query_data:
        for w in SPARSE_WEIGHTS:
            avg_rel = compute_avg_relevance_at_k(qdata, w, top_k)
            weight_results[w].append(avg_rel)

    # Step 4: print summary table
    print(f"\n{'Weight':>8}  {'Avg Relevance@' + str(top_k):>18}  {'vs weight=1.0':>14}")
    print("-" * 46)

    baseline_avg = sum(weight_results[1.0]) / len(weight_results[1.0])
    best_weight = max(SPARSE_WEIGHTS, key=lambda w: sum(weight_results[w]))

    for w in SPARSE_WEIGHTS:
        avg = sum(weight_results[w]) / len(weight_results[w])
        delta = avg - baseline_avg
        marker = "  <- best" if w == best_weight else ""
        print(f"{w:>8.2f}  {avg:>18.4f}  {delta:>+14.4f}{marker}")

    print(f"\nBest sparse weight: {best_weight}")
    print(f"Queries evaluated:  {len(QUERIES)}")
    print(f"Top-K:              {top_k}")

    # Per-query breakdown
    print(f"\nPer-query avg relevance@{top_k} at best weight ({best_weight}):")
    for qdata in all_query_data:
        avg = compute_avg_relevance_at_k(qdata, best_weight, top_k)
        print(f"  {qdata.query[:65]:<65}  {avg:.3f}")

    # Save results
    if output_path:
        out = {
            "top_k": top_k,
            "queries": QUERIES,
            "sparse_weights_tested": SPARSE_WEIGHTS,
            "best_weight": best_weight,
            "weight_summary": {
                str(w): {
                    "avg_relevance_at_k": round(sum(weight_results[w]) / len(weight_results[w]), 4),
                    "per_query": [round(v, 4) for v in weight_results[w]],
                }
                for w in SPARSE_WEIGHTS
            },
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(run_sweep(top_k=args.top_k, output_path=args.output))