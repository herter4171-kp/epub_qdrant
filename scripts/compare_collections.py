#!/usr/bin/env python3
"""
Dead-simple dense vs hybrid collection comparison.

Runs the same queries against:
  - books + papers          (dense-only baseline)
  - books-named + papers-named  (hybrid: dense + sparse + RRF)

Reports per-query doc overlap and a summary table.
No LLM judge. No score normalization. Just doc IDs.

Usage:
    python compare_collections.py
    python compare_collections.py --top-k 10
    python compare_collections.py --output overlap_results.json
"""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# ── same path setup as evaluate.py ──────────────────────────────────────────
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_mcp_server_dir = _project_root / "mcp_servers" / "retrieval"
if str(_mcp_server_dir) not in sys.path:
    sys.path.insert(0, str(_mcp_server_dir))

from mcp_server.retriever import Retriever, ChunkResult

# ── queries (same 30 as evaluate.py) ────────────────────────────────────────
QUERIES = [
    "Help me design a planning module for an LLM agent that can break a long task into executable subtasks without overplanning.",
    "What is the current best practice for giving an agent reasoning structure without forcing brittle chain-of-thought templates?",
    "Show me how to build an agent that decides when to use a tool versus when to answer directly.",
    "How should I architect multi-tool orchestration so an agent can chain search, code execution, and document drafting safely?",
    "What scaffolding do strong terminal-based coding agents use around the model itself?",
    "What is a good memory design for an agent that needs short-term working memory and long-term user memory?",
    "How do I stop an agent's memory from turning into a junk drawer full of redundant or low-value facts?",
    "I want an agent that can localize bugs in a large repository. What retrieval and hypothesis-testing loop should I start with?",
    "How would you benchmark an agent that performs refactors rather than one-off bug fixes?",
    "I want an agent that can notice its own mistakes and retry with a different strategy. What self-correction loop should I use?",
    "How do I make a coding agent ask clarifying questions only when needed instead of either guessing wildly or stopping constantly?",
    "What should I measure to catch post-merge quality problems in agent-generated pull requests?",
    "How would you build a GUI agent that can operate a clunky enterprise web app with inconsistent layouts?",
    "What are the main failure modes of web agents in realistic browsing tasks, and how should I evaluate them?",
    "I want a mobile agent that can carry out a multi-step task across apps. What architecture would make that robust?",
    "How should an enterprise agent handle permissions, approvals, and audit logs when acting on behalf of employees?",
    "What is the cleanest way to build an API agent that can discover schema details and recover from malformed tool responses?",
    "I need a data agent that can query a warehouse, validate the SQL, and then generate charts with commentary. How would you structure it?",
    "What would a serious research agent pipeline look like for literature review, source ranking, note synthesis, and citation tracking?",
    "How do deep research agents verify intermediate conclusions instead of just producing polished nonsense?",
    "I'm interested in agentic scientific simulation. How would an agent iteratively propose, run, and revise model configurations?",
    "What are the tradeoffs between a single AI scientist agent and a multi-agent scientific discovery pipeline?",
    "How should sub-agent creation work in a larger orchestration system so I do not end up with agent sprawl?",
    "I want to experiment with multi-agent problem solving where agents debate, specialize, and then converge. What patterns are worth trying first?",
    "How do self-evolving agents accumulate reusable skills without drifting into unstable behavior?",
    "What safeguards are needed when agents can modify their own prompts, memories, or tool policies over time?",
    "Help me think through safety for agents that can browse the web, run code, and call external APIs.",
    "How does agent tuning look like in practice for a smaller open model that needs to behave more like a reliable operator?",
    "How should I evaluate agents in production beyond task success rate—latency, cost, recovery rate, human override rate, what else?",
    "Can you sketch a benchmark for comparing a data-analysis agent against human analysts on realistic business tasks?",
]


def get_doc_ids(chunks: List[ChunkResult]) -> List[str]:
    """
    Extract a stable identifier per chunk.
    Prefers arxiv_id or book_title+chapter as a doc-level key
    so we're measuring document overlap, not chunk overlap.
    """
    ids = []
    for c in chunks:
        if c.arxiv_id:
            ids.append(c.arxiv_id)
        elif c.book_title:
            # book_title alone as doc identifier (ignore chunk offset)
            ids.append(c.book_title.strip().lower()[:80])
        else:
            # fallback: first 60 chars of text (not great, but visible)
            ids.append(c.text[:60].strip())
    return ids


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union else 0.0


def run_query(retriever: Retriever, query: str, collections: List[str], top_k: int) -> List[ChunkResult]:
    bundle = retriever.search_collections(
        query=query,
        top_k=top_k,
        collections=collections,
    )
    chunks = []
    for g in bundle.groups:
        chunks.extend(g.results)
    chunks.sort(key=lambda x: x.score, reverse=True)
    return chunks[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5, help="Results per query to compare (default: 5)")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    top_k = args.top_k
    retriever = Retriever()

    dense_collections  = ["books", "papers"]
    hybrid_collections = ["books-named", "papers-named"]

    results = []
    overlap_scores = []

    print(f"\n{'Query':<60} {'Overlap':>8}  {'Dense-only unique':>18}  {'Hybrid-only unique':>18}")
    print("-" * 110)

    for query in QUERIES:
        dense_chunks  = run_query(retriever, query, dense_collections,  top_k)
        hybrid_chunks = run_query(retriever, query, hybrid_collections, top_k)

        dense_ids  = get_doc_ids(dense_chunks)
        hybrid_ids = get_doc_ids(hybrid_chunks)

        dense_set  = set(dense_ids)
        hybrid_set = set(hybrid_ids)

        overlap    = jaccard(dense_ids, hybrid_ids)
        only_dense  = sorted(dense_set  - hybrid_set)
        only_hybrid = sorted(hybrid_set - dense_set)

        overlap_scores.append(overlap)

        short_q = query[:57] + "..." if len(query) > 60 else query
        print(f"{short_q:<60} {overlap:>7.1%}  {len(only_dense):>18}  {len(only_hybrid):>18}")

        results.append({
            "query": query,
            "jaccard_overlap": round(overlap, 4),
            "dense_doc_ids":   dense_ids,
            "hybrid_doc_ids":  hybrid_ids,
            "only_in_dense":   only_dense,
            "only_in_hybrid":  only_hybrid,
        })

    avg_overlap = sum(overlap_scores) / len(overlap_scores)
    min_overlap = min(overlap_scores)
    max_overlap = max(overlap_scores)

    high_overlap  = sum(1 for s in overlap_scores if s >= 0.8)
    mid_overlap   = sum(1 for s in overlap_scores if 0.4 <= s < 0.8)
    low_overlap   = sum(1 for s in overlap_scores if s < 0.4)

    print("\n" + "=" * 110)
    print(f"  Queries:        {len(QUERIES)}")
    print(f"  Top-K:          {top_k}")
    print(f"  Avg Jaccard:    {avg_overlap:.1%}   (min {min_overlap:.1%}, max {max_overlap:.1%})")
    print(f"  High overlap (>=80%):  {high_overlap} queries  ← sparse doing nothing here")
    print(f"  Mid  overlap (40-79%): {mid_overlap} queries  ← sparse adding something")
    print(f"  Low  overlap (<40%):   {low_overlap} queries  ← sparse dominating / very different results")
    print("=" * 110)

    if avg_overlap >= 0.8:
        print("\n  ⚠  High average overlap — sparse vectors are not meaningfully changing results.")
        print("     Check: are sparse vectors actually indexed in books-named / papers-named?")
        print("     Check: are sparse query vectors being sent, or is Qdrant falling back to dense?")
    elif avg_overlap <= 0.4:
        print("\n  ✓  Low overlap — sparse is pulling in different documents.")
        print("     Now you need a relevance signal to know if they're *better* documents.")
    else:
        print("\n  ~  Mixed overlap — sparse is contributing on some queries but not others.")
        print("     Inspect the low-overlap queries manually to see what sparse is surfacing.")

    if args.output:
        out = {
            "top_k": top_k,
            "avg_jaccard": round(avg_overlap, 4),
            "min_jaccard": round(min_overlap, 4),
            "max_jaccard": round(max_overlap, 4),
            "high_overlap_count": high_overlap,
            "mid_overlap_count":  mid_overlap,
            "low_overlap_count":  low_overlap,
            "per_query": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results written to {args.output}")


if __name__ == "__main__":
    main()
