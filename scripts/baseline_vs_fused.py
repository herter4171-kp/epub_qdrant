#!/usr/bin/env python3
"""v4 A/B comparison: baseline dense-only RAG vs fused hybrid retrieval.

For each book common to both collections: pick random passage, generate query,
retrieve from both pipelines, generate answers, judge head-to-head.

Tracks retrieval overlap: what's unique to baseline, unique to fused, common to both.

No classes, no async, no framework.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

import requests

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from servers.mcp_server.config import settings

# ─── Constants ───────────────────────────────────────────────────────

SCRIPT_VERSION = "4.0.0"
MCP_URL = f"http://localhost:{settings.MCP_PORT}/mcp"
LITELLM_URL = settings.LITELLM_API_URL
LITELLM_KEY = settings.LITELLM_API_KEY or os.environ.get("LITELLM_API_KEY", "")

QUERY_MODEL = settings.LITELLM_MODEL
QUERY_TEMPERATURE = 0.3
ANSWER_MODEL = settings.LITELLM_MODEL
ANSWER_TEMPERATURE = 0.3
JUDGE_MODEL = settings.LITELLM_MODEL
JUDGE_TEMPERATURE = 0.0

SOURCE_EXCERPT_LEN = 500
ANSWER_EXCERPT_LEN = 200
MCP_TIMEOUT = 120
LLM_TIMEOUT = 120

DEFAULT_TOP_K = 5
DEFAULT_DENSE_K = 5
DEFAULT_SPARSE_K = 5


# ─── ANSI ────────────────────────────────────────────────────────────

def green(s):  return f"\033[32m{s}\033[0m"
def red(s):    return f"\033[31m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"
def dim(s):    return f"\033[2m{s}\033[0m"


# ─── MCP / LLM Helpers ──────────────────────────────────────────────

def mcp_call(tool, args):
    """JSON-RPC 2.0 tools/call to MCP server."""
    resp = requests.post(
        MCP_URL,
        json={
            "jsonrpc": "2.0", "id": 1,
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
    if isinstance(result, dict) and "content" not in result:
        return result
    content = result.get("content", [])
    if content and isinstance(content, list):
        text = content[0].get("text", "")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"raw_text": text}
    return result


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
        description="v4 A/B: baseline dense-only vs fused hybrid retrieval."
    )
    parser.add_argument("baseline_collection",
                        help="Dense-only collection (e.g. books)")
    parser.add_argument("fused_collection",
                        help="Hybrid collection with dense+sparse vectors (e.g. books-semantic)")
    parser.add_argument("--baseline-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Top-k for baseline dense retrieval (default: {DEFAULT_TOP_K})")
    parser.add_argument("--dense-k", type=int, default=DEFAULT_DENSE_K,
                        help=f"Top-k for fused dense signal (default: {DEFAULT_DENSE_K})")
    parser.add_argument("--sparse-k", type=int, default=DEFAULT_SPARSE_K,
                        help=f"Top-k for fused sparse signal (default: {DEFAULT_SPARSE_K})")
    parser.add_argument("--positions", type=int, default=1,
                        help="Random samples per book (default: 1)")
    parser.add_argument("--output", default="results/baseline_vs_fused.json",
                        help="Output JSON path")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--books", type=int, default=None,
                        help="Limit to first N common books (default: all)")
    return parser.parse_args(argv)


# ─── Query Generation ────────────────────────────────────────────────

QUERY_TYPE_BUCKETS = {
    "numeric_fact_lookup": "trivia",
    "named_entity_lookup": "trivia",
    "conceptual_explanation": "conceptual",
    "comparison_tradeoff": "conceptual",
    "paper_or_method_discovery": "conceptual",
    "implementation_guidance": "operational",
    "architecture_design": "operational",
    "failure_mode_debugging": "operational",
    "evaluation_method": "operational",
    "security_or_governance": "operational",
    "operational_monitoring": "operational",
}

QUERY_SYSTEM_PROMPT = (
    "You are generating realistic retrieval queries for a working professional "
    "using a private knowledge base about AI, machine learning, LLMs, RAG, and "
    "agentic AI.\n\n"
    "Given a source passage, write one natural question that an engineer, "
    "researcher, architect, security lead, product owner, or technical "
    "decision-maker might actually ask because they need to make a design, "
    "implementation, evaluation, purchasing, security, research, or operational "
    "decision.\n\n"
    "Do not write quiz questions.\n"
    "Do not say \"according to the passage.\"\n"
    "Do not mention \"the passage\", \"the text\", \"the excerpt\", or \"the source.\"\n"
    "Do not ask arbitrary trivia questions.\n"
    "Do not ask for a percentage, list, definition, named component, or fact "
    "unless that fact would matter for a real work task.\n\n"
    "Prefer questions about how to build, choose, debug, evaluate, compare, "
    "secure, deploy, monitor, govern, or operationalize something.\n\n"
    "The question must be answerable from the source passage, but it should "
    "sound like it came from someone trying to do real work.\n\n"
    "Return JSON only in this format:\n"
    "{\n"
    '  "query": "one realistic work question",\n'
    '  "query_type": "one of the allowed query types",\n'
    '  "why_this_is_realistic": "one short sentence explaining the work context"\n'
    "}\n\n"
    "Allowed query types:\n"
    "- conceptual_explanation\n"
    "- implementation_guidance\n"
    "- architecture_design\n"
    "- comparison_tradeoff\n"
    "- failure_mode_debugging\n"
    "- evaluation_method\n"
    "- security_or_governance\n"
    "- operational_monitoring\n"
    "- named_entity_lookup\n"
    "- numeric_fact_lookup\n"
    "- paper_or_method_discovery\n\n"
    "Use named_entity_lookup or numeric_fact_lookup only when the name, exact "
    "value, API, model, framework, benchmark, or statistic would realistically "
    "matter to a worker's decision."
)


# ─── System Prompts ──────────────────────────────────────────────────

BASELINE_ANSWER_SYSTEM_PROMPT = (
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

FUSED_ANSWER_SYSTEM_PROMPT = BASELINE_ANSWER_SYSTEM_PROMPT  # same prompt, different retrieval

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator of retrieval-augmented answers. You will be "
    "given a question and two answer bundles — Answer A (baseline) and Answer B "
    "(fused) — each produced from a different retrieval pipeline. Each answer is "
    "shown alongside the passages that were retrieved for it.\n\n"
    "Your job is to determine which answer is more useful, specific, and "
    "grounded in its own retrieved passages.\n\n"
    "Evaluate each answer on three criteria:\n"
    "1. Grounding: Does the answer make claims supported by its retrieved passages? "
    "An answer that invents facts not present in its passages is worse than one "
    "that stays within what the passages actually say.\n"
    "2. Specificity: Does the answer name concrete mechanisms, frameworks, "
    "tradeoffs, or examples from its passages, or does it speak in vague "
    "generalities?\n"
    "3. Completeness: Does the answer address the question thoroughly given "
    "what its passages contain?\n\n"
    "Do not penalize an answer for containing information absent from the OTHER "
    "answer's passages. Each answer should be judged against its own retrieval "
    "context. Do not reward confidence or fluency over substance.\n\n"
    'Respond with JSON only: {"winner": "A" | "B" | "tie", "reason": "one sentence"}\n\n'
    "A tie is appropriate when both answers are equally well-grounded and useful."
)

EMPTY_RETRIEVAL_MSG = (
    "No passages were retrieved for this query. "
    "State that no relevant information was found."
)


# ─── Core Functions ──────────────────────────────────────────────────

def discover_books(collection):
    """Discover books in a collection. Returns list of book dicts."""
    collections_resp = mcp_call("list_collections", {})
    collections_list = collections_resp.get("collections", [])
    existing = {c["name"] if isinstance(c, dict) else c for c in collections_list}
    if collection not in existing:
        print(f"ERROR: Collection '{collection}' not found. Available: {existing}", file=sys.stderr)
        sys.exit(1)
    return mcp_call("list_books", {"collection": collection}).get("books", [])


def find_common_books(baseline_books, fused_books):
    """Find books present in both collections by source_file."""
    baseline_sfs = {b["source_file"] for b in baseline_books}
    fused_sfs = {b["source_file"] for b in fused_books}
    common = baseline_sfs & fused_sfs
    baseline_by_sf = {b["source_file"]: b for b in baseline_books}
    return [baseline_by_sf[sf] for sf in sorted(common)]


def pick_passage(collection, source_file, seed=None):
    """Pick random passage from a specific book via MCP."""
    try:
        call_args = {"collection": collection, "source_file": source_file}
        if seed is not None:
            call_args["seed"] = seed
        result = mcp_call("pick_random_chunk", call_args)
        if "error" in result:
            print(f"  pick_passage error: {result['error']}", file=sys.stderr)
            return None
        return result
    except Exception as e:
        print(f"  pick_passage exception: {e}", file=sys.stderr)
        return None


def generate_query(passage_text):
    """Generate retrieval query from passage."""
    try:
        user_msg = (
            f"SOURCE PASSAGE:\n{passage_text}\n\n"
            "Generate one realistic work query from this passage.\n\nReturn JSON only."
        )
        raw = llm_call(QUERY_SYSTEM_PROMPT, user_msg,
                       model=QUERY_MODEL, temperature=QUERY_TEMPERATURE)
        text = _strip_markdown_fences(raw)
        try:
            parsed = json.loads(text)
            query = parsed.get("query", "").strip()
            query_type = parsed.get("query_type", "unknown")
            why = parsed.get("why_this_is_realistic", "")
        except (json.JSONDecodeError, TypeError):
            query = text.split("\n")[0].strip().strip('"')
            query_type = "unknown"
            why = "query_generation_parse_error"
        if not query:
            return None
        bucket = QUERY_TYPE_BUCKETS.get(query_type, "operational")
        return {
            "query": query, "query_type": query_type,
            "query_bucket": bucket, "why_this_is_realistic": why,
            "query_generation_raw": raw,
        }
    except Exception as e:
        print(f"  generate_query exception: {e}", file=sys.stderr)
        return None


def retrieve(collection, query, sparse_weight=None, dense_weight=None, top_k=None):
    """Query MCP in search mode. Returns result dict or None."""
    try:
        call_args = {"mode": "search", "collection": collection, "query": query}
        if sparse_weight is not None:
            call_args["sparse_weight"] = sparse_weight
        if dense_weight is not None:
            call_args["dense_weight"] = dense_weight
        if top_k is not None:
            call_args["top_k"] = top_k
        return mcp_call("query", call_args)
    except Exception as e:
        print(f"  retrieve exception: {e}", file=sys.stderr)
        return None


# ─── Passage Extraction & Overlap ────────────────────────────────────

def _strip_markdown_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


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
    for group in results.get("groups", []):
        if group.get("chunks"):
            return True
    return False


def _flatten_chunks(results):
    """Flatten MCP grouped results into list of chunk dicts with point_id."""
    chunks = []
    for group in results.get("groups", []):
        for chunk in group.get("chunks", []):
            if chunk.get("text"):
                chunks.append(dict(chunk))
    chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
    # Deduplicate by point_id
    seen = set()
    deduped = []
    for c in chunks:
        pid = c.get("point_id")
        if pid and pid not in seen:
            seen.add(pid)
            deduped.append(c)
        elif not pid:
            deduped.append(c)
    return deduped


def compute_overlap(baseline_chunks, fused_chunks):
    """Compute set overlap between baseline and fused retrieval by text fingerprint.

    Uses first 200 chars of text as fingerprint since point_ids differ across collections.
    Returns dict with counts and lists.
    """
    def fingerprint(text):
        return text[:200].strip().lower()

    baseline_fps = {}
    for c in baseline_chunks:
        fp = fingerprint(c.get("text", ""))
        if fp:
            baseline_fps[fp] = c

    fused_fps = {}
    for c in fused_chunks:
        fp = fingerprint(c.get("text", ""))
        if fp:
            fused_fps[fp] = c

    common_fps = set(baseline_fps.keys()) & set(fused_fps.keys())
    baseline_only_fps = set(baseline_fps.keys()) - common_fps
    fused_only_fps = set(fused_fps.keys()) - common_fps

    return {
        "common_count": len(common_fps),
        "baseline_only_count": len(baseline_only_fps),
        "fused_only_count": len(fused_only_fps),
        "baseline_total": len(baseline_fps),
        "fused_total": len(fused_fps),
        "jaccard": (len(common_fps) / len(baseline_fps | fused_fps)
                    if (baseline_fps or fused_fps) else 0.0),
    }


# ─── Fused Pipeline Helpers (from v3) ────────────────────────────────

RERANKER_MODEL = settings.LITELLM_MODEL
RERANKER_TEMPERATURE = 0.0

RERANKER_SYSTEM_PROMPT = (
    "You are a relevance reranker for a technical knowledge base. You will receive\n"
    "a query and a set of candidate passages, each labeled with how it was found\n"
    "(semantic search, keyword search, or both).\n\n"
    "Your task is to reorder the passages by relevance to the query based on their\n"
    "content. Read each passage carefully and reason about which ones best answer\n"
    "the query. Consider:\n"
    "- Passages found by BOTH signals are likely highly relevant\n"
    "- For factual/entity queries, keyword-matched passages may be more precise\n"
    "- For conceptual queries, semantic-matched passages may capture better context\n"
    "- Judge relevance by passage content, not by position in this list\n\n"
    'Return a JSON array of passage indices in relevance order, most relevant first:\n'
    '{"ranked_indices": [3, 1, 5, 2, 4, ...]}'
)

SIGNAL_LABELS = {
    "dense": "semantic",
    "sparse": "keyword",
    "both": "both (semantic + keyword)",
}


def _flatten_with_rank(results):
    """Flatten MCP grouped results into ranked chunk list with point_id dedup."""
    chunks = []
    for group in results.get("groups", []):
        for chunk in group.get("chunks", []):
            if chunk.get("text") and chunk.get("point_id"):
                chunks.append(dict(chunk))
    chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
    seen = set()
    deduped = []
    for c in chunks:
        pid = c["point_id"]
        if pid not in seen:
            seen.add(pid)
            deduped.append(c)
    for i, c in enumerate(deduped):
        c["rank"] = i + 1
    return deduped


def dedup_and_normalize(dense_results, sparse_results):
    """Merge dense and sparse results into deduplicated, normalized candidate set."""
    dense_chunks = _flatten_with_rank(dense_results)
    sparse_chunks = _flatten_with_rank(sparse_results)

    dense_scores = [c["score"] for c in dense_chunks]
    sparse_scores = [c["score"] for c in sparse_chunks]
    d_min, d_max = (min(dense_scores), max(dense_scores)) if dense_scores else (0, 0)
    s_min, s_max = (min(sparse_scores), max(sparse_scores)) if sparse_scores else (0, 0)

    def norm(val, lo, hi):
        return 1.0 if hi == lo else (val - lo) / (hi - lo)

    candidates = {}
    for c in dense_chunks:
        pid = c.get("point_id")
        if not pid:
            continue
        candidates[pid] = {
            "point_id": pid, "text": c["text"],
            "source_file": c.get("source_file", ""),
            "title": c.get("title", ""),
            "section_title": c.get("section_title", ""),
            "chunk_index": c.get("chunk_index"),
            "signal": "dense",
            "dense_score": c["score"], "sparse_score": None,
            "dense_rank": c["rank"], "sparse_rank": None,
            "dense_score_norm": norm(c["score"], d_min, d_max),
            "sparse_score_norm": None,
        }
    for c in sparse_chunks:
        pid = c.get("point_id")
        if not pid:
            continue
        if pid in candidates:
            candidates[pid]["signal"] = "both"
            candidates[pid]["sparse_score"] = c["score"]
            candidates[pid]["sparse_rank"] = c["rank"]
            candidates[pid]["sparse_score_norm"] = norm(c["score"], s_min, s_max)
        else:
            candidates[pid] = {
                "point_id": pid, "text": c["text"],
                "source_file": c.get("source_file", ""),
                "title": c.get("title", ""),
                "section_title": c.get("section_title", ""),
                "chunk_index": c.get("chunk_index"),
                "signal": "sparse",
                "dense_score": None, "sparse_score": c["score"],
                "dense_rank": None, "sparse_rank": c["rank"],
                "dense_score_norm": None,
                "sparse_score_norm": norm(c["score"], s_min, s_max),
            }
    return list(candidates.values())


def _u_shape_order(passages):
    n = len(passages)
    if n <= 2:
        return passages
    mid = (n + 1) // 2
    return passages[:mid] + passages[mid:][::-1]


def _parse_reranker_response(raw, n):
    text = _strip_markdown_fences(raw)
    try:
        parsed = json.loads(text)
        indices = parsed.get("ranked_indices", [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        return list(range(n))
    seen = set()
    deduped = []
    for i in indices:
        if isinstance(i, int) and 0 <= i < n and i not in seen:
            deduped.append(i)
            seen.add(i)
    for i in range(n):
        if i not in seen:
            deduped.append(i)
    return deduped


def rerank(query, candidates):
    """LLM-based listwise reranking. Returns (reordered_candidates, raw_response)."""
    if not candidates:
        return [], ""
    passage_lines = []
    for i, c in enumerate(candidates):
        label = SIGNAL_LABELS.get(c["signal"], c["signal"])
        passage_lines.append(f"[{i}] Signal: {label}\n    {c['text'][:500]}")
    user_msg = (
        f"QUERY: {query}\n\nCANDIDATE PASSAGES:\n\n"
        + "\n\n".join(passage_lines)
        + "\n\nReturn the passage indices in relevance order as JSON."
    )
    try:
        raw = llm_call(RERANKER_SYSTEM_PROMPT, user_msg,
                       model=RERANKER_MODEL, temperature=RERANKER_TEMPERATURE)
        ranked_indices = _parse_reranker_response(raw, len(candidates))
        return [candidates[i] for i in ranked_indices], raw
    except Exception as e:
        print(f"  rerank exception: {e}", file=sys.stderr)
        return candidates, str(e)


# ─── Answer Generation ───────────────────────────────────────────────

def generate_baseline_answer(query, results):
    """Generate answer from baseline dense-only retrieval."""
    zero_chunk = not _has_chunks(results) if results else True
    passages = _extract_passages(results) if results else []

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
        return llm_call(BASELINE_ANSWER_SYSTEM_PROMPT, user_msg,
                        model=ANSWER_MODEL, temperature=ANSWER_TEMPERATURE)
    except Exception as e:
        print(f"  generate_baseline_answer exception: {e}", file=sys.stderr)
        return None


def generate_fused_answer(query, reranked_passages):
    """Generate answer from reranked fused passages."""
    if not reranked_passages:
        user_msg = f"Question: {query}\n\n{EMPTY_RETRIEVAL_MSG}"
    else:
        ordered = _u_shape_order(reranked_passages)
        passage_lines = []
        for i, c in enumerate(ordered):
            source = c.get("title") or c.get("source_file", "unknown")
            passage_lines.append(f"[{i + 1}] {source}: {c['text']}")
        user_msg = (
            f"QUESTION:\n{query}\n\n"
            f"RETRIEVED PASSAGES:\n" + "\n\n".join(passage_lines) + "\n\n"
            "Answer the question using these passages."
        )
    try:
        return llm_call(FUSED_ANSWER_SYSTEM_PROMPT, user_msg,
                        model=ANSWER_MODEL, temperature=ANSWER_TEMPERATURE)
    except Exception as e:
        print(f"  generate_fused_answer exception: {e}", file=sys.stderr)
        return None


# ─── Judge ───────────────────────────────────────────────────────────

def judge(query, baseline_passages, fused_passages, answer_a, answer_b):
    """Head-to-head judge. Each answer evaluated against its own retrieved passages.

    Returns {winner, reason, judge_raw}.
    Applies verbosity normalization: truncate both to min(len) * 1.2.
    """
    max_chars = int(min(len(answer_a), len(answer_b)) * 1.2)
    a_trimmed = answer_a[:max_chars]
    b_trimmed = answer_b[:max_chars]

    baseline_text = "\n\n---\n\n".join(baseline_passages) if baseline_passages else "(no passages)"
    fused_text = "\n\n---\n\n".join(fused_passages) if fused_passages else "(no passages)"

    user_msg = (
        f"Question:\n{query}\n\n"
        f"── Answer A (baseline) ──\n"
        f"Retrieved passages for A:\n{baseline_text}\n\n"
        f"Answer A:\n{a_trimmed}\n\n"
        f"── Answer B (fused) ──\n"
        f"Retrieved passages for B:\n{fused_text}\n\n"
        f"Answer B:\n{b_trimmed}"
    )
    try:
        raw = llm_call(JUDGE_SYSTEM_PROMPT, user_msg,
                       model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE)
        return _parse_judge_response(raw)
    except Exception as e:
        print(f"  judge exception: {e}", file=sys.stderr)
        return {"winner": "tie", "reason": "judge_error", "judge_raw": str(e)}


def _parse_judge_response(raw):
    text = _strip_markdown_fences(raw)
    try:
        parsed = json.loads(text)
        winner = parsed.get("winner", "tie")
        if winner not in ("A", "B", "tie"):
            winner = "tie"
        return {"winner": winner, "reason": parsed.get("reason", ""), "judge_raw": raw}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {"winner": "tie", "reason": "judge_error", "judge_raw": raw}


# ─── Hit Rank ────────────────────────────────────────────────────────

def compute_hit_rank(source_chunk, results):
    """Check if source chunk appears in retrieved results."""
    chunks = []
    for group in results.get("groups", []):
        chunks.extend(group.get("chunks", []))
    chunks.sort(key=lambda c: c.get("score", 0), reverse=True)

    source_file = source_chunk.get("source_file", "")
    seed_idx = source_chunk.get("seed_chunk_index")
    chunk_range = source_chunk.get("chunk_range",
                                   [seed_idx, seed_idx] if seed_idx is not None else None)
    source_text = source_chunk.get("text", "")[:200].lower()

    if chunk_range and chunk_range[0] is not None:
        for i, c in enumerate(chunks):
            if (c.get("source_file") == source_file
                    and chunk_range[0] <= c.get("chunk_index", -1) <= chunk_range[1]):
                return {"rank": i + 1, "match_method": "chunk_index"}

    if source_text:
        source_words = set(source_text.split())
        for i, c in enumerate(chunks):
            chunk_text = c.get("text", "")[:200].lower()
            if chunk_text:
                overlap = len(source_words & set(chunk_text.split()))
                if overlap / max(len(source_words), 1) > 0.5:
                    return {"rank": i + 1, "match_method": "text_overlap"}

    return {"rank": None, "match_method": "none"}


# ─── Print + Output ──────────────────────────────────────────────────

def print_sample(sample):
    """Colorized per-sample terminal output."""
    winner = sample.get("judge_winner", "tie")
    winner_color = green if winner == "B" else (red if winner == "A" else yellow)
    overlap = sample.get("retrieval_overlap", {})

    print(f"\n  {bold(sample.get('book', '?'))}")
    print(f"  {dim(sample.get('source_passage_excerpt', '')[:200])}")
    print(f"  Query: {sample['query']}  [{sample.get('query_bucket', '?')}]")
    print(f"  Baseline answer: {dim(str(sample.get('baseline_answer', ''))[:ANSWER_EXCERPT_LEN])}")
    print(f"  Fused answer:    {dim(str(sample.get('fused_answer', ''))[:ANSWER_EXCERPT_LEN])}")
    print(f"  Overlap: common={overlap.get('common_count', 0)} "
          f"baseline-only={overlap.get('baseline_only_count', 0)} "
          f"fused-only={overlap.get('fused_only_count', 0)} "
          f"jaccard={overlap.get('jaccard', 0):.2f}")
    print(f"  Judge: {winner_color(f'winner={winner}')} — {sample.get('judge_reason', '')}")


def compute_aggregates(samples):
    """Compute v4 aggregates."""
    total = len(samples)
    if not total:
        return {
            "baseline_wins": 0, "fused_wins": 0, "ties": 0,
            "baseline_win_rate": 0.0, "fused_win_rate": 0.0, "tie_rate": 0.0,
            "avg_jaccard": 0.0, "avg_common": 0.0,
            "avg_baseline_only": 0.0, "avg_fused_only": 0.0,
            "baseline_hit_count": 0, "fused_hit_count": 0,
            "baseline_hit_rate": 0.0, "fused_hit_rate": 0.0,
            "bucket_summary": {},
        }

    baseline_wins = sum(1 for s in samples if s["judge_winner"] == "A")
    fused_wins = sum(1 for s in samples if s["judge_winner"] == "B")
    ties = sum(1 for s in samples if s["judge_winner"] == "tie")

    overlaps = [s.get("retrieval_overlap", {}) for s in samples]
    avg_jaccard = sum(o.get("jaccard", 0) for o in overlaps) / total
    avg_common = sum(o.get("common_count", 0) for o in overlaps) / total
    avg_baseline_only = sum(o.get("baseline_only_count", 0) for o in overlaps) / total
    avg_fused_only = sum(o.get("fused_only_count", 0) for o in overlaps) / total

    baseline_hits = sum(1 for s in samples if s.get("baseline_hit_rank") is not None)
    fused_hits = sum(1 for s in samples if s.get("fused_hit_rank") is not None)

    bucket_summary = {}
    for bucket_name in ("trivia", "conceptual", "operational"):
        bs = [s for s in samples if s.get("query_bucket") == bucket_name]
        bw = sum(1 for s in bs if s["judge_winner"] == "A")
        fw = sum(1 for s in bs if s["judge_winner"] == "B")
        ti = sum(1 for s in bs if s["judge_winner"] == "tie")
        bucket_summary[bucket_name] = {
            "samples": len(bs),
            "baseline_wins": bw, "fused_wins": fw, "ties": ti,
        }

    return {
        "baseline_wins": baseline_wins,
        "fused_wins": fused_wins,
        "ties": ties,
        "baseline_win_rate": round(baseline_wins / total, 3),
        "fused_win_rate": round(fused_wins / total, 3),
        "tie_rate": round(ties / total, 3),
        "avg_jaccard": round(avg_jaccard, 3),
        "avg_common": round(avg_common, 1),
        "avg_baseline_only": round(avg_baseline_only, 1),
        "avg_fused_only": round(avg_fused_only, 1),
        "baseline_hit_count": baseline_hits,
        "fused_hit_count": fused_hits,
        "baseline_hit_rate": round(baseline_hits / total, 3),
        "fused_hit_rate": round(fused_hits / total, 3),
        "bucket_summary": bucket_summary,
    }


def print_summary(metadata):
    """Print v4 comparison summary table."""
    agg = metadata

    print(f"\n{'=' * 70}")
    print(f"  BASELINE vs FUSED — Head-to-Head Results")
    print(f"  {'─' * 55}")
    print(f"  Baseline wins:  {red(agg['baseline_wins'])}")
    print(f"  Fused wins:     {green(agg['fused_wins'])}")
    print(f"  Ties:           {yellow(agg['ties'])}")
    print(f"  {'─' * 55}")

    bucket_summary = agg.get("bucket_summary", {})
    print(f"\n  {'':20s} {'baseline':>10s}  {'fused':>10s}  {'tie':>10s}  {'samples':>7s}")
    print(f"  {'─' * 60}")
    for bucket_name in ("trivia", "conceptual", "operational"):
        b = bucket_summary.get(bucket_name, {})
        bw, fw, ti = b.get("baseline_wins", 0), b.get("fused_wins", 0), b.get("ties", 0)
        n = b.get("samples", 0)
        label = bucket_name.capitalize()
        print(f"  {label:20s} {red(bw):>19s}  {green(fw):>19s}  {yellow(ti):>19s}  {n:>7d}")

    print(f"\n  {'─' * 55}")
    print(f"  Retrieval Overlap:")
    print(f"    Avg Jaccard:       {agg['avg_jaccard']:.3f}")
    print(f"    Avg common:        {agg['avg_common']:.1f}")
    print(f"    Avg baseline-only: {agg['avg_baseline_only']:.1f}")
    print(f"    Avg fused-only:    {agg['avg_fused_only']:.1f}")
    print(f"  Hit Rates:")
    print(f"    Baseline: {agg['baseline_hit_rate']:.1%}  Fused: {agg['fused_hit_rate']:.1%}")
    print(f"{'=' * 70}")


# ─── Main ────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    baseline_col = args.baseline_collection
    fused_col = args.fused_collection

    print(f"baseline_vs_fused v{SCRIPT_VERSION}")
    print(f"Baseline: {baseline_col} (k={args.baseline_k})")
    print(f"Fused:    {fused_col} (dense-k={args.dense_k}, sparse-k={args.sparse_k})")
    print(f"Positions/book: {args.positions}  Seed: {args.seed}")

    if args.seed is not None:
        random.seed(args.seed)

    # Discover books in both collections
    baseline_books = discover_books(baseline_col)
    fused_books = discover_books(fused_col)
    common_books = find_common_books(baseline_books, fused_books)

    if not common_books:
        print("ERROR: No books common to both collections.", file=sys.stderr)
        sys.exit(1)

    if args.books:
        common_books = common_books[:args.books]

    print(f"\nFound {len(common_books)} books to process.")

    samples = []

    for book in common_books:
        sf = book["source_file"]
        title = book.get("book_title", sf)
        print(f"\n{'─' * 60}")
        print(f"Book: {bold(title)} ({sf})")

        for pos in range(args.positions):
            try:
                # 1. Pick passage from baseline collection (deterministic if seeded)
                book_seed = None
                if args.seed is not None:
                    book_seed = hash((args.seed, sf, pos)) & 0xFFFFFFFF
                chunk = pick_passage(baseline_col, sf, seed=book_seed)
                if not chunk:
                    continue

                # 2. Generate query (same for both pipelines)
                query_result = generate_query(chunk["text"])
                if not query_result:
                    continue
                query = query_result["query"]

                # 3a. Baseline retrieval: dense-only on baseline collection
                baseline_results = retrieve(baseline_col, query, top_k=args.baseline_k)
                if baseline_results is None:
                    continue

                # 3b. Fused retrieval: dense + sparse signals on hybrid collection
                dense_results = retrieve(fused_col, query,
                                         sparse_weight=0, dense_weight=1,
                                         top_k=args.dense_k)
                sparse_results = retrieve(fused_col, query,
                                          sparse_weight=1, dense_weight=0,
                                          top_k=args.sparse_k)
                if dense_results is None or sparse_results is None:
                    continue

                # 4. Fused pipeline: dedup → rerank
                candidates = dedup_and_normalize(dense_results, sparse_results)
                reranked, reranker_raw = rerank(query, candidates)

                # 5. Compute retrieval overlap (baseline vs fused combined set)
                baseline_chunks = _flatten_chunks(baseline_results)
                fused_chunks_flat = _flatten_chunks(dense_results)
                # Add sparse-only chunks to fused set
                sparse_flat = _flatten_chunks(sparse_results)
                fused_pids = {c.get("point_id") for c in fused_chunks_flat if c.get("point_id")}
                for c in sparse_flat:
                    if c.get("point_id") not in fused_pids:
                        fused_chunks_flat.append(c)
                        fused_pids.add(c.get("point_id"))

                overlap = compute_overlap(baseline_chunks, fused_chunks_flat)

                # 6. Generate answers
                baseline_answer = generate_baseline_answer(query, baseline_results)
                fused_answer = generate_fused_answer(query, reranked)
                if baseline_answer is None or fused_answer is None:
                    continue

                # 7. Judge head-to-head
                baseline_passages = _extract_passages(baseline_results)
                fused_passages_text = [c.get("text", "") for c in reranked if c.get("text")]
                verdict = judge(query, baseline_passages, fused_passages_text,
                                baseline_answer, fused_answer)

                # 8. Hit ranks
                baseline_hit = compute_hit_rank(chunk, baseline_results)
                fused_hit = compute_hit_rank(chunk, dense_results)

                # 9. Build sample
                sample = {
                    "book": title,
                    "position_index": pos,
                    "source_chunk_id": chunk.get("point_id"),
                    "source_metadata": {
                        k: chunk.get(k) for k in (
                            "point_id", "source_file", "title", "section_title",
                            "seed_chunk_index", "chunk_range", "chunks_used",
                            "total_tokens", "token_count",
                        )
                    },
                    "source_passage": chunk.get("text", ""),
                    "source_passage_excerpt": chunk.get("text", "")[:SOURCE_EXCERPT_LEN],
                    "query": query,
                    "query_type": query_result["query_type"],
                    "query_bucket": query_result["query_bucket"],
                    "why_this_is_realistic": query_result["why_this_is_realistic"],
                    "query_generation_raw": query_result["query_generation_raw"],
                    "baseline_raw_results": baseline_results,
                    "baseline_retrieved_passages": _extract_passages(baseline_results),
                    "fused_dense_raw_results": dense_results,
                    "fused_sparse_raw_results": sparse_results,
                    "fused_candidates": candidates,
                    "fused_reranked": reranked,
                    "reranker_raw": reranker_raw,
                    "retrieval_overlap": overlap,
                    "baseline_answer": baseline_answer,
                    "fused_answer": fused_answer,
                    "judge_winner": verdict["winner"],
                    "judge_reason": verdict["reason"],
                    "judge_raw": verdict["judge_raw"],
                    "baseline_hit_rank": baseline_hit["rank"],
                    "baseline_hit_method": baseline_hit["match_method"],
                    "fused_hit_rank": fused_hit["rank"],
                    "fused_hit_method": fused_hit["match_method"],
                }

                print_sample(sample)
                samples.append(sample)

            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                continue

    # Aggregates
    agg = compute_aggregates(samples)

    metadata = {
        "version": SCRIPT_VERSION,
        "baseline_collection": baseline_col,
        "fused_collection": fused_col,
        "baseline_k": args.baseline_k,
        "dense_k": args.dense_k,
        "sparse_k": args.sparse_k,
        "positions_per_book": args.positions,
        "total_samples": len(samples),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "query_model": QUERY_MODEL,
        "reranker_model": RERANKER_MODEL,
        "mcp_url": MCP_URL,
        "script_version": SCRIPT_VERSION,
        **agg,
    }

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metadata": metadata, "samples": samples}, f, indent=2)
    print(f"\nResults written to {output_path}")

    print_summary(metadata)


if __name__ == "__main__":
    main()
