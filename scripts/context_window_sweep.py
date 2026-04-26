#!/usr/bin/env python3
"""Context-window sweep: does wider retrieval context help the LLM answer better?

Outer loop: user-supplied prompts (or defaults).
Inner loop: num_extra_chunks in [0, 1, 2, 4, 8] — chunks added around the
            top-1 hit via get_context(radius=N).

For each (prompt, radius) cell:
  - Retrieve top-1 hit via hybrid search
  - Expand with get_context(radius=N) to get surrounding chunks
  - Ask LLM to answer using that context
  - Ask LLM to rate its own answer: 1=bullshit/vague, 2=partial, 3=grounded/specific
  - Note chunk sizes (token counts) to flag suspiciously small chunks

Output: table + JSON.

Usage:
    python scripts/context_window_sweep.py
    python scripts/context_window_sweep.py --prompts "What is RRF?" "How does DPR work?"
    python scripts/context_window_sweep.py --collection papers-semantic --radii 0 2 4 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from servers.mcp_server.retriever import Retriever
from servers.mcp_server.llm_client import LLMClient
from servers.mcp_server.config import settings

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("context_sweep")

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_COLLECTION = "papers-semantic"
DEFAULT_RADII = [0, 1, 2, 4, 8]
DEFAULT_PROMPTS = [
    "How does Reciprocal Rank Fusion combine dense and sparse retrieval signals?",
    "What are the failure modes of multi-hop retrieval in agentic RAG systems?",
    "How do self-evolving agents accumulate reusable skills without drifting?",
]

SMALL_CHUNK_THRESHOLD = 50  # tokens — flag chunks below this

JUDGE_SYSTEM = """\
You are a strict evaluator of answer quality. You will be given a question and an answer
produced from retrieved passages. Score the answer on a 1–3 scale:

1 = Vague or unsupported — answer is generic, hallucinates, or fails to engage with specifics
2 = Partial — answer has some grounded claims but also vague or unsupported sections
3 = Grounded and specific — every claim is traceable to the retrieved passages; no padding

Respond with JSON only: {"score": 1|2|3, "reason": "one sentence"}"""

ANSWER_SYSTEM = """\
You are a precise technical assistant. Answer the question using only the provided passages.
Be specific — name mechanisms, frameworks, and tradeoffs from the passages.
If the passages lack enough information, say so explicitly. No padding."""


# ── Core logic ───────────────────────────────────────────────────────────────

async def run_cell(
    prompt: str,
    radius: int,
    retriever: Retriever,
    llm: LLMClient,
    collection: str,
) -> Dict:
    """Run one (prompt, radius) cell. Returns a result dict."""

    # Step 1: top-1 hit
    bundle = retriever.search(
        query=prompt,
        top_k=1,
        collection=collection,
    )
    if not bundle.groups or not bundle.groups[0].results:
        return {
            "radius": radius,
            "error": "no results from search",
            "chunks": [],
            "chunk_token_counts": [],
            "small_chunks": [],
            "answer": "",
            "score": 0,
            "reason": "no results",
        }

    top_hit = bundle.groups[0].results[0]
    source_file = top_hit.source_file
    section_title = top_hit.section_title or top_hit.section

    # Step 2: expand with context
    if radius == 0:
        # No expansion — just the top hit
        chunks = [top_hit]
    else:
        ctx_bundle = retriever.get_context(
            source_file=source_file,
            section_title=section_title if section_title else None,
            query=prompt if not section_title else None,
            radius=radius,
            collection=collection,
        )
        if ctx_bundle.groups and ctx_bundle.groups[0].results:
            chunks = ctx_bundle.groups[0].results
        else:
            chunks = [top_hit]

    # Step 3: note chunk sizes
    token_counts = []
    for c in chunks:
        # token_count may be on the raw payload; ChunkResult may not carry it
        tc = getattr(c, "token_count", None) or len(c.text.split())
        token_counts.append(tc)

    small_chunks = [
        {"chunk_index": getattr(c, "chunk_index", i), "tokens": tc}
        for i, (c, tc) in enumerate(zip(chunks, token_counts))
        if tc < SMALL_CHUNK_THRESHOLD
    ]

    # Step 4: build context string
    context_parts = []
    for i, c in enumerate(chunks, 1):
        src = c.source_file or ""
        sec = c.section_title or c.section or ""
        context_parts.append(f"[Passage {i}] {src} — {sec}\n{c.text}")
    context_str = "\n\n".join(context_parts)

    # Step 5: get answer
    answer = await llm.answer(
        query=prompt,
        context=context_str,
        system_prompt=ANSWER_SYSTEM,
    )

    # Step 6: judge
    judge_prompt = f"QUESTION:\n{prompt}\n\nANSWER:\n{answer}"
    judge_raw = await llm.answer(
        query=judge_prompt,
        context="",
        system_prompt=JUDGE_SYSTEM,
    )

    score = 2
    reason = "parse error"
    import re
    m = re.search(r'\{[^{}]*"score"[^{}]*\}', judge_raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            score = int(parsed.get("score", 2))
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "radius": radius,
        "source_file": source_file,
        "section_title": section_title,
        "num_chunks": len(chunks),
        "chunk_token_counts": token_counts,
        "avg_tokens_per_chunk": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
        "small_chunks": small_chunks,
        "answer_excerpt": answer[:300],
        "score": score,
        "reason": reason,
    }


async def sweep_prompt(
    prompt: str,
    radii: List[int],
    retriever: Retriever,
    llm: LLMClient,
    collection: str,
) -> Dict:
    """Run all radii for one prompt."""
    print(f"\n{'─'*70}")
    print(f"PROMPT: {prompt}")
    print(f"{'─'*70}")
    print(f"{'radius':>8}  {'chunks':>7}  {'avg_tok':>8}  {'score':>6}  {'small?':>7}  reason")
    print(f"{'─'*8}  {'─'*7}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*40}")

    cells = []
    for radius in radii:
        cell = await run_cell(prompt, radius, retriever, llm, collection)
        cells.append(cell)

        small_flag = f"{len(cell.get('small_chunks', []))} tiny" if cell.get("small_chunks") else "ok"
        print(
            f"{cell['radius']:>8}  "
            f"{cell.get('num_chunks', 0):>7}  "
            f"{cell.get('avg_tokens_per_chunk', 0):>8.0f}  "
            f"{cell.get('score', 0):>6}  "
            f"{small_flag:>7}  "
            f"{cell.get('reason', '')[:60]}"
        )

    return {"prompt": prompt, "cells": cells}


async def main_async(args: argparse.Namespace) -> None:
    retriever = Retriever()
    llm = LLMClient()
    collection = args.collection

    prompts = args.prompts or DEFAULT_PROMPTS
    radii = args.radii or DEFAULT_RADII

    print(f"\nContext-window sweep")
    print(f"Collection : {collection}")
    print(f"Radii      : {radii}")
    print(f"Prompts    : {len(prompts)}")
    print(f"Small chunk threshold: {SMALL_CHUNK_THRESHOLD} tokens")

    results = []
    for prompt in prompts:
        row = await sweep_prompt(prompt, radii, retriever, llm, collection)
        results.append(row)

    # Summary table across all prompts
    print(f"\n{'═'*70}")
    print("SUMMARY — avg score by radius across all prompts")
    print(f"{'radius':>8}  {'avg_score':>10}  {'avg_chunks':>11}  {'total_tiny':>11}")
    print(f"{'─'*8}  {'─'*10}  {'─'*11}  {'─'*11}")

    for radius in radii:
        scores, chunk_counts, tiny_counts = [], [], []
        for row in results:
            for cell in row["cells"]:
                if cell["radius"] == radius:
                    if cell.get("score"):
                        scores.append(cell["score"])
                    chunk_counts.append(cell.get("num_chunks", 0))
                    tiny_counts.append(len(cell.get("small_chunks", [])))
        avg_s = sum(scores) / len(scores) if scores else 0
        avg_c = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0
        total_t = sum(tiny_counts)
        print(f"{radius:>8}  {avg_s:>10.2f}  {avg_c:>11.1f}  {total_t:>11}")

    # Save JSON
    output = {
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "radii": radii,
        "small_chunk_threshold": SMALL_CHUNK_THRESHOLD,
        "results": results,
    }
    out_path = _root / "results" / "context_window_sweep.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Context-window sweep")
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION,
        help=f"Qdrant collection (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--radii", nargs="+", type=int, default=None,
        help=f"Radius values to sweep (default: {DEFAULT_RADII})",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=None,
        help="Prompts to test (default: built-in set of 3)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
