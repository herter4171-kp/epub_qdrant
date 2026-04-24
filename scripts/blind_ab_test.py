#!/usr/bin/env python3
"""Blind A/B test: dense-only vs hybrid retrieval.

For each book common to both collections: pick random passage, generate query,
retrieve from both, Answer_LLM × 2, Judge_LLM scores faithfulness.
MCP server does retrieval only. Script does answer generation + judging.

No classes, no async, no framework.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

import requests

# Load .env before importing settings
from dotenv import load_dotenv
load_dotenv()

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from servers.mcp_server.config import settings

# ─── Constants ───────────────────────────────────────────────────────

SCRIPT_VERSION = "1.0.0"
MCP_URL = f"http://localhost:{settings.MCP_PORT}/mcp"
LITELLM_URL = settings.LITELLM_API_URL
LITELLM_KEY = settings.LITELLM_API_KEY

# Per-role model/temperature
QUERY_MODEL = settings.LITELLM_MODEL
QUERY_TEMPERATURE = 0.3
ANSWER_MODEL = settings.LITELLM_MODEL
ANSWER_TEMPERATURE = 0.3
JUDGE_MODEL = settings.LITELLM_MODEL
JUDGE_TEMPERATURE = 0.0  # deterministic judging

SOURCE_EXCERPT_LEN = 500
ANSWER_EXCERPT_LEN = 200
MCP_TIMEOUT = 120
LLM_TIMEOUT = 120


# ─── ANSI Color Helpers ──────────────────────────────────────────────


def green(s):
    return f"\033[32m{s}\033[0m"


def red(s):
    return f"\033[31m{s}\033[0m"


def yellow(s):
    return f"\033[33m{s}\033[0m"


def bold(s):
    return f"\033[1m{s}\033[0m"


def dim(s):
    return f"\033[2m{s}\033[0m"


# ─── MCP Call Helper ─────────────────────────────────────────────────


def mcp_call(tool, args):
    """JSON-RPC 2.0 tools/call to MCP server. Robust response parsing."""
    resp = requests.post(
        MCP_URL,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        },
        timeout=MCP_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise RuntimeError(body["error"]["message"])
    result = body.get("result", body)
    # 1. Direct JSON result (no content wrapper)
    if isinstance(result, dict) and "content" not in result:
        return result
    # 2. result.content[0].text (standard MCP)
    content = result.get("content", [])
    if content and isinstance(content, list):
        text = content[0].get("text", "")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"raw_text": text}
    # 3. Fallback
    return result


# ─── LLM Call Helper ─────────────────────────────────────────────────


def llm_call(system, user, model=None, temperature=None):
    """OpenAI chat completions format to LiteLLM endpoint."""
    resp = requests.post(
        f"{LITELLM_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LITELLM_KEY}"},
        json={
            "model": model or ANSWER_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else ANSWER_TEMPERATURE,
        },
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ─── CLI ─────────────────────────────────────────────────────────────


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Blind A/B test: dense-only vs hybrid retrieval."
    )
    parser.add_argument("dense_collection", help="Qdrant collection for dense-only retrieval")
    parser.add_argument("hybrid_collection", help="Qdrant collection for hybrid retrieval")
    parser.add_argument(
        "--positions", type=int, default=1, help="Random samples per book (default: 1)"
    )
    parser.add_argument(
        "--output",
        default="results/blind_ab_test.json",
        help="Output JSON path (default: results/blind_ab_test.json)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--top-k", type=int, default=None, help="Top-k for retrieval (default: MCP default)"
    )
    return parser.parse_args(argv)


# ─── System Prompts (verbatim from requirements) ────────────────────

QUERY_SYSTEM_PROMPT = (
    "You are given a passage from a technical document. Write one specific "
    "retrieval question that this passage answers. Output only the question, "
    "nothing else."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a precise technical research assistant. You are given a question "
    "and a set of retrieved passages from a knowledge base of AI and machine "
    "learning literature. Answer the question by synthesizing only what the "
    "passages contain. Be specific — name mechanisms, frameworks, and tradeoffs "
    "that appear in the passages rather than speaking in generalities. If the "
    "passages do not contain enough information to answer the question, say so "
    "explicitly rather than filling gaps with your own knowledge. Do not pad "
    "your answer with caveats, introductions, or summaries. Write for an "
    "engineer who wants the answer, not an explanation of how you found it."
)

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator of retrieval quality. You will be given a "
    "source passage, a question generated from that passage, two retrieved "
    "passage bundles, and two answers — Answer A and Answer B — each produced "
    "from a different retrieval system.\n\n"
    "Your job is to determine which answer is more faithful to the source "
    "passage and better supported by its retrieved passages.\n\n"
    "Faithful means: the answer contains specific facts, mechanisms, or claims "
    "that are present in the source passage or clearly supported by the "
    "retrieved passages, and does not introduce facts that contradict or are "
    "absent from the provided material.\n\n"
    "Do not reward confidence or fluency. An answer that sounds authoritative "
    "but draws on knowledge outside the provided source/retrieval material is "
    "worse than a shorter answer that accurately reflects what the material "
    "actually says.\n\n"
    'Respond with a JSON object only, no explanation outside it: '
    '{"winner": "A" | "B" | "tie", "reason": "one sentence"}\n\n'
    "A tie is appropriate when both answers are equally faithful or equally "
    "unfaithful."
)

EMPTY_RETRIEVAL_MSG = (
    "No passages were retrieved for this query. "
    "State that no relevant information was found."
)


# ─── Core Functions ──────────────────────────────────────────────────


def discover_books(dense, hybrid):
    """Discover books common to both collections. Returns list of book dicts."""
    # Verify collections exist
    collections_resp = mcp_call("list_collections", {})
    collections_list = collections_resp.get("collections", [])
    existing = {c["name"] if isinstance(c, dict) else c for c in collections_list}
    for coll in (dense, hybrid):
        if coll not in existing:
            print(f"ERROR: Collection '{coll}' not found. Available: {existing}", file=sys.stderr)
            sys.exit(1)

    # Get books per collection
    dense_books = mcp_call("list_books", {"collection": dense}).get("books", [])
    hybrid_books = mcp_call("list_books", {"collection": hybrid}).get("books", [])

    dense_by_sf = {b["source_file"]: b for b in dense_books}
    hybrid_by_sf = {b["source_file"]: b for b in hybrid_books}

    # Intersect by source_file
    common_sfs = set(dense_by_sf.keys()) & set(hybrid_by_sf.keys())

    # Warn about skipped books
    for sf in set(dense_by_sf.keys()) - common_sfs:
        print(f"  WARN: '{sf}' in dense only, skipping", file=sys.stderr)
    for sf in set(hybrid_by_sf.keys()) - common_sfs:
        print(f"  WARN: '{sf}' in hybrid only, skipping", file=sys.stderr)

    return [dense_by_sf[sf] for sf in sorted(common_sfs)]


def intersect_books(dense_books, hybrid_books):
    """Pure intersection logic — returns books present in both lists by source_file."""
    dense_sfs = {b["source_file"] for b in dense_books}
    hybrid_sfs = {b["source_file"] for b in hybrid_books}
    common = dense_sfs & hybrid_sfs
    dense_by_sf = {b["source_file"]: b for b in dense_books}
    return [dense_by_sf[sf] for sf in sorted(common)]


def pick_passage(collection, source_file):
    """Pick random passage from a specific book via MCP. Returns chunk dict or None."""
    try:
        result = mcp_call("pick_random_chunk", {
            "collection": collection,
            "source_file": source_file,
        })
        if "error" in result:
            print(f"  pick_passage error: {result['error']}", file=sys.stderr)
            return None
        return result
    except Exception as e:
        print(f"  pick_passage exception: {e}", file=sys.stderr)
        return None


def generate_query(passage_text):
    """Generate retrieval query from passage. Returns (query, raw) or (None, None)."""
    try:
        raw = llm_call(QUERY_SYSTEM_PROMPT, passage_text,
                       model=QUERY_MODEL, temperature=QUERY_TEMPERATURE)
        query = raw.strip().strip('"')
        return query, raw
    except Exception as e:
        print(f"  generate_query exception: {e}", file=sys.stderr)
        return None, None


def retrieve(collection, query, sparse_weight=None, top_k=None):
    """Query MCP in search mode. Returns result dict or None."""
    try:
        call_args = {"mode": "search", "collection": collection, "query": query}
        if sparse_weight is not None:
            call_args["sparse_weight"] = sparse_weight
        if top_k is not None:
            call_args["top_k"] = top_k
        return mcp_call("query", call_args)
    except Exception as e:
        print(f"  retrieve exception: {e}", file=sys.stderr)
        return None


def _extract_passages(results):
    """Extract text passages from MCP search results."""
    texts = []
    for group in results.get("groups", []):
        for chunk in group.get("chunks", []):
            t = chunk.get("text", "")
            if t:
                texts.append(t)
    return texts


def _has_chunks(results):
    """Check if results contain any chunks."""
    for group in results.get("groups", []):
        if group.get("chunks"):
            return True
    return False


def generate_answer(query, retrieved_results):
    """Generate answer from retrieved passages. Returns (answer, zero_chunk) or (None, False)."""
    zero_chunk = not _has_chunks(retrieved_results) if retrieved_results else True
    passages = _extract_passages(retrieved_results) if retrieved_results else []

    if zero_chunk or not passages:
        user_msg = f"Question: {query}\n\n{EMPTY_RETRIEVAL_MSG}"
    else:
        passages_text = "\n\n---\n\n".join(passages)
        user_msg = (
            f"Question: {query}\n\n"
            f"Retrieved passages:\n\n{passages_text}\n\n"
            "Answer only from these passages."
        )

    try:
        answer = llm_call(ANSWER_SYSTEM_PROMPT, user_msg,
                          model=ANSWER_MODEL, temperature=ANSWER_TEMPERATURE)
        return answer, zero_chunk
    except Exception as e:
        print(f"  generate_answer exception: {e}", file=sys.stderr)
        return None, zero_chunk


def parse_judge_response(raw):
    """Parse judge JSON response. Returns dict with winner, reason, judge_raw."""
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        parsed = json.loads(text)
        winner = parsed.get("winner", "tie")
        if winner not in ("A", "B", "tie"):
            winner = "tie"
        return {
            "winner": winner,
            "reason": parsed.get("reason", ""),
            "judge_raw": raw,
        }
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {"winner": "tie", "reason": "judge_error", "judge_raw": raw}


def judge(source_passage, query, results_a, results_b, answer_a, answer_b):
    """Judge two answers for faithfulness. Returns dict with winner, reason, judge_raw."""
    passages_a = "\n\n".join(_extract_passages(results_a)) if results_a else "(no passages)"
    passages_b = "\n\n".join(_extract_passages(results_b)) if results_b else "(no passages)"

    user_msg = (
        f"Source passage:\n{source_passage}\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved passages for A:\n{passages_a}\n\n"
        f"Retrieved passages for B:\n{passages_b}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}"
    )

    try:
        raw = llm_call(JUDGE_SYSTEM_PROMPT, user_msg,
                       model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE)
        return parse_judge_response(raw)
    except Exception as e:
        print(f"  judge exception: {e}", file=sys.stderr)
        return {"winner": "tie", "reason": "judge_error", "judge_raw": str(e)}


def compute_hit_rank(source_chunk, results):
    """Compute whether source chunk appears in retrieved results.

    Tries 3 strategies in priority order, stops at first match:
    1. source_file + chunk_index range overlap (structural)
    2. normalized text overlap >50% (fuzzy fallback)
    Returns {rank: int|None, match_method: str}.
    """
    chunks = []
    for group in results.get("groups", []):
        chunks.extend(group.get("chunks", []))
    chunks.sort(key=lambda c: c.get("score", 0), reverse=True)

    source_file = source_chunk.get("source_file", "")
    seed_idx = source_chunk.get("seed_chunk_index")
    chunk_range = source_chunk.get("chunk_range", [seed_idx, seed_idx] if seed_idx is not None else None)
    source_text = source_chunk.get("text", "")[:200].lower()

    # Strategy 1: source_file + chunk_index range overlap
    if chunk_range and chunk_range[0] is not None:
        for i, c in enumerate(chunks):
            if (c.get("source_file") == source_file
                    and chunk_range[0] <= c.get("chunk_index", -1) <= chunk_range[1]):
                return {"rank": i + 1, "match_method": "chunk_index"}

    # Strategy 2: text overlap fallback
    if source_text:
        source_words = set(source_text.split())
        for i, c in enumerate(chunks):
            chunk_text = c.get("text", "")[:200].lower()
            if chunk_text:
                overlap = len(source_words & set(chunk_text.split()))
                if overlap / max(len(source_words), 1) > 0.5:
                    return {"rank": i + 1, "match_method": "text_overlap"}

    return {"rank": None, "match_method": "none"}


def map_verdict(winner_ab, a_src, b_src):
    """Map judge A/B/tie verdict to dense/hybrid/tie."""
    if winner_ab == "A":
        return a_src
    elif winner_ab == "B":
        return b_src
    return "tie"



# ─── Print + Output ──────────────────────────────────────────────────


def print_sample(sample):
    """Colorized per-sample terminal output."""
    winner = sample.get("winner", "tie")
    print(f"\n  {bold(sample.get('book_title', '?'))}")
    print(f"  {dim(sample['source_passage_excerpt'][:200])}")
    print(f"  Query: {sample['query']}")
    print(f"  Dense answer:  {dim(sample['dense_answer'][:ANSWER_EXCERPT_LEN])}")
    print(f"  Hybrid answer: {dim(sample['hybrid_answer'][:ANSWER_EXCERPT_LEN])}")

    if winner == "dense":
        label = f"{green('DENSE wins')} — {sample.get('reason', '')}"
    elif winner == "hybrid":
        label = f"{green('HYBRID wins')} — {sample.get('reason', '')}"
    else:
        label = f"{yellow('TIE')} — {sample.get('reason', '')}"
    print(f"  Verdict: {label}")

    if sample.get("zero_chunk_retrieval"):
        print(f"  {yellow('(zero-chunk retrieval)')}")


def compute_aggregates(samples):
    """Compute hit counts/rates from samples list."""
    total = len(samples)
    dense_hits = sum(1 for s in samples if s.get("dense_source_hit_rank") is not None)
    hybrid_hits = sum(1 for s in samples if s.get("hybrid_source_hit_rank") is not None)
    return {
        "dense_hit_count": dense_hits,
        "hybrid_hit_count": hybrid_hits,
        "dense_hit_rate": dense_hits / total if total else 0.0,
        "hybrid_hit_rate": hybrid_hits / total if total else 0.0,
    }


def print_summary(metadata):
    """Print final summary line."""
    print(f"\n{'=' * 60}")
    print(f"  Total: {metadata['total_samples']}  "
          f"Dense: {green(metadata['dense_wins'])}  "
          f"Hybrid: {green(metadata['hybrid_wins'])}  "
          f"Ties: {yellow(metadata['ties'])}")
    print(f"  Dense hit rate: {metadata['dense_hit_rate']:.1%}  "
          f"Hybrid hit rate: {metadata['hybrid_hit_rate']:.1%}")
    if metadata.get("judge_error_count"):
        print(f"  Judge errors: {red(metadata['judge_error_count'])}")
    if metadata.get("zero_chunk_count"):
        print(f"  Zero-chunk samples: {yellow(metadata['zero_chunk_count'])}")
    print(f"{'=' * 60}")



# ─── Main ────────────────────────────────────────────────────────────


def main(argv=None):
    args = parse_args(argv)
    print(f"blind_ab_test v{SCRIPT_VERSION}")
    print(f"Dense: {args.dense_collection}  Hybrid: {args.hybrid_collection}")
    print(f"Positions/book: {args.positions}  Seed: {args.seed}  Top-k: {args.top_k}")

    if args.seed is not None:
        random.seed(args.seed)

    top_k = args.top_k

    # Discover common books
    common_books = discover_books(args.dense_collection, args.hybrid_collection)
    if not common_books:
        print("ERROR: No common books between collections.", file=sys.stderr)
        sys.exit(1)
    print(f"\nFound {len(common_books)} common books.")

    samples = []
    dense_wins = 0
    hybrid_wins = 0
    ties = 0
    judge_error_count = 0
    zero_chunk_count = 0

    for book in common_books:
        sf = book["source_file"]
        title = book.get("book_title", sf)
        print(f"\n{'─' * 60}")
        print(f"Book: {bold(title)} ({sf})")

        for pos in range(args.positions):
            try:
                # 1. Pick random passage (book-scoped)
                chunk = pick_passage(args.dense_collection, sf)
                if not chunk:
                    continue

                # 2. Generate query
                query, query_raw = generate_query(chunk["text"])
                if not query:
                    continue

                # 3. Dense retrieval (sparse_weight=0)
                dense_results = retrieve(args.dense_collection, query,
                                         sparse_weight=0, top_k=top_k)
                if dense_results is None:
                    continue

                # 4. Hybrid retrieval (default sparse_weight)
                hybrid_results = retrieve(args.hybrid_collection, query,
                                          sparse_weight=None, top_k=top_k)
                if hybrid_results is None:
                    continue

                # 5–7. Generate answers
                dense_answer, dense_zero = generate_answer(query, dense_results)
                if dense_answer is None:
                    continue
                hybrid_answer, hybrid_zero = generate_answer(query, hybrid_results)
                if hybrid_answer is None:
                    continue

                zero_chunk = dense_zero or hybrid_zero
                if zero_chunk:
                    zero_chunk_count += 1

                # 8. Random A/B assignment
                if random.random() < 0.5:
                    a_src, b_src = "dense", "hybrid"
                else:
                    a_src, b_src = "hybrid", "dense"

                answer_a = dense_answer if a_src == "dense" else hybrid_answer
                answer_b = dense_answer if b_src == "dense" else hybrid_answer
                results_a = dense_results if a_src == "dense" else hybrid_results
                results_b = dense_results if b_src == "dense" else hybrid_results

                # 9–10. Judge
                verdict = judge(chunk["text"], query,
                                results_a, results_b, answer_a, answer_b)

                if verdict["reason"] == "judge_error":
                    judge_error_count += 1

                # 11. Map A/B → dense/hybrid/tie
                winner = map_verdict(verdict["winner"], a_src, b_src)

                # Track wins
                if winner == "dense":
                    dense_wins += 1
                elif winner == "hybrid":
                    hybrid_wins += 1
                else:
                    ties += 1

                # Hit rank
                dense_hit = compute_hit_rank(chunk, dense_results)
                hybrid_hit = compute_hit_rank(chunk, hybrid_results)

                # Build sample dict
                sample = {
                    "book_source_file": sf,
                    "book_title": title,
                    "position_index": pos,
                    "source_chunk_id": chunk.get("point_id"),
                    "source_metadata": {
                        k: chunk.get(k) for k in (
                            "point_id", "source_file", "title", "section_title",
                            "seed_chunk_index", "chunk_range", "chunks_used",
                            "total_tokens", "token_count", "arxiv_id", "authors",
                        )
                    },
                    "source_passage": chunk.get("text", ""),
                    "source_passage_excerpt": chunk.get("text", "")[:SOURCE_EXCERPT_LEN],
                    "query": query,
                    "query_generation_raw": query_raw,
                    "dense_raw_results": dense_results,
                    "hybrid_raw_results": hybrid_results,
                    "dense_retrieved_passages": _extract_passages(dense_results),
                    "hybrid_retrieved_passages": _extract_passages(hybrid_results),
                    "dense_answer": dense_answer,
                    "hybrid_answer": hybrid_answer,
                    "answer_a_source": a_src,
                    "answer_b_source": b_src,
                    "answer_a": answer_a,
                    "answer_b": answer_b,
                    "judge_winner": verdict["winner"],
                    "winner": winner,
                    "reason": verdict["reason"],
                    "judge_raw": verdict["judge_raw"],
                    "dense_source_hit_rank": dense_hit["rank"],
                    "dense_source_hit_method": dense_hit["match_method"],
                    "hybrid_source_hit_rank": hybrid_hit["rank"],
                    "hybrid_source_hit_method": hybrid_hit["match_method"],
                    "zero_chunk_retrieval": zero_chunk,
                }

                # 12. Print
                print_sample(sample)

                # 13. Append
                samples.append(sample)

            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                continue

    # Aggregates
    agg = compute_aggregates(samples)

    metadata = {
        "dense_collection": args.dense_collection,
        "hybrid_collection": args.hybrid_collection,
        "positions_per_book": args.positions,
        "total_samples": len(samples),
        "dense_wins": dense_wins,
        "hybrid_wins": hybrid_wins,
        "ties": ties,
        "judge_error_count": judge_error_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "query_model": QUERY_MODEL,
        "mcp_url": MCP_URL,
        "top_k": top_k,
        "dense_hit_count": agg["dense_hit_count"],
        "hybrid_hit_count": agg["hybrid_hit_count"],
        "dense_hit_rate": agg["dense_hit_rate"],
        "hybrid_hit_rate": agg["hybrid_hit_rate"],
        "zero_chunk_count": zero_chunk_count,
        "script_version": SCRIPT_VERSION,
    }

    # Write JSON
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metadata": metadata, "samples": samples}, f, indent=2)
    print(f"\nResults written to {output_path}")

    print_summary(metadata)


if __name__ == "__main__":
    main()
