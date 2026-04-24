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

SCRIPT_VERSION = "3.0.0"
MCP_URL = f"http://localhost:{settings.MCP_PORT}/mcp"
LITELLM_URL = settings.LITELLM_API_URL
LITELLM_KEY = settings.LITELLM_API_KEY or os.environ.get("LITELLM_API_KEY", "")

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

# v3 CLI defaults
DEFAULT_DENSE_K = 5
DEFAULT_SPARSE_K = 5


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
        description="Fused retrieval with LLM-based reranking (v3.0.0)."
    )
    parser.add_argument("collection", help="Qdrant hybrid collection (has both dense + sparse vectors)")
    parser.add_argument(
        "--dense-k", type=int, default=DEFAULT_DENSE_K,
        help=f"Top-k for dense signal retrieval (default: {DEFAULT_DENSE_K})"
    )
    parser.add_argument(
        "--sparse-k", type=int, default=DEFAULT_SPARSE_K,
        help=f"Top-k for sparse signal retrieval (default: {DEFAULT_SPARSE_K})"
    )
    parser.add_argument(
        "--positions", type=int, default=1,
        help="Random samples per book (default: 1)"
    )
    parser.add_argument(
        "--output", default="results/blind_ab_test.json",
        help="Output JSON path (default: results/blind_ab_test.json)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args(argv)


# ─── Query Type Bucketing ────────────────────────────────────────────

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


# ─── System Prompts ─────────────────────────────────────────────────

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


def discover_books(collection):
    """Discover books in a single collection. Returns list of book dicts."""
    collections_resp = mcp_call("list_collections", {})
    collections_list = collections_resp.get("collections", [])
    existing = {c["name"] if isinstance(c, dict) else c for c in collections_list}
    if collection not in existing:
        print(f"ERROR: Collection '{collection}' not found. Available: {existing}", file=sys.stderr)
        sys.exit(1)

    books = mcp_call("list_books", {"collection": collection}).get("books", [])
    return books


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
    """Generate retrieval query from passage. Returns dict with query, query_type, etc."""
    try:
        user_msg = f"SOURCE PASSAGE:\n{passage_text}\n\nGenerate one realistic work query from this passage.\n\nReturn JSON only."
        raw = llm_call(QUERY_SYSTEM_PROMPT, user_msg,
                       model=QUERY_MODEL, temperature=QUERY_TEMPERATURE)
        # Try JSON parse
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            parsed = json.loads(text)
            query = parsed.get("query", "").strip()
            query_type = parsed.get("query_type", "unknown")
            why = parsed.get("why_this_is_realistic", "")
        except (json.JSONDecodeError, TypeError):
            # Fallback: first non-empty line as query
            query = text.split("\n")[0].strip().strip('"')
            query_type = "unknown"
            why = "query_generation_parse_error"
        if not query:
            return None
        bucket = QUERY_TYPE_BUCKETS.get(query_type, "operational")
        return {
            "query": query,
            "query_type": query_type,
            "query_bucket": bucket,
            "why_this_is_realistic": why,
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


# ─── v3 Pure Functions ───────────────────────────────────────────────


def _flatten_with_rank(results):
    """Flatten MCP grouped results into ranked chunk list.

    Sorts by score descending, assigns 1-based ranks, excludes empty-text chunks.
    Deduplicates by point_id (keeps highest score).
    """
    chunks = []
    for group in results.get("groups", []):
        for chunk in group.get("chunks", []):
            if chunk.get("text") and chunk.get("point_id"):
                chunks.append(dict(chunk))  # shallow copy to avoid mutating input
    chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
    # Deduplicate by point_id — first occurrence wins (highest score)
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
    """Merge dense and sparse results into deduplicated, normalized candidate set.

    1. Flatten chunks from both result sets
    2. Set-union by point_id — duplicates marked as "both" signal
    3. Min-max normalize scores within each signal to [0,1]
    4. Return unified candidate list with signal labels and normalized scores
    """
    dense_chunks = _flatten_with_rank(dense_results)
    sparse_chunks = _flatten_with_rank(sparse_results)

    # Min-max normalize scores within each signal
    dense_scores = [c["score"] for c in dense_chunks]
    sparse_scores = [c["score"] for c in sparse_chunks]
    d_min, d_max = (min(dense_scores), max(dense_scores)) if dense_scores else (0, 0)
    s_min, s_max = (min(sparse_scores), max(sparse_scores)) if sparse_scores else (0, 0)

    def norm(val, lo, hi):
        if hi == lo:
            return 1.0
        return (val - lo) / (hi - lo)

    # Build candidate map by point_id (set-union)
    candidates = {}

    for c in dense_chunks:
        pid = c.get("point_id")
        if not pid:
            continue
        candidates[pid] = {
            "point_id": pid,
            "text": c["text"],
            "source_file": c.get("source_file", ""),
            "title": c.get("title", ""),
            "section_title": c.get("section_title", ""),
            "chunk_index": c.get("chunk_index"),
            "signal": "dense",
            "dense_score": c["score"],
            "sparse_score": None,
            "dense_rank": c["rank"],
            "sparse_rank": None,
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
                "point_id": pid,
                "text": c["text"],
                "source_file": c.get("source_file", ""),
                "title": c.get("title", ""),
                "section_title": c.get("section_title", ""),
                "chunk_index": c.get("chunk_index"),
                "signal": "sparse",
                "dense_score": None,
                "sparse_score": c["score"],
                "dense_rank": None,
                "sparse_rank": c["rank"],
                "dense_score_norm": None,
                "sparse_score_norm": norm(c["score"], s_min, s_max),
            }

    return list(candidates.values())


def _u_shape_order(passages):
    """Reorder passages in U-shape: top half first, bottom half reversed.

    Mitigates lost-in-the-middle by placing most relevant passages at beginning
    and end of context where LLM attention is strongest.

    Input:  [1, 2, 3, 4, 5, 6, 7, 8]  (ranked by relevance)
    Output: [1, 2, 3, 4, 8, 7, 6, 5]  (U-shape)
    """
    n = len(passages)
    if n <= 2:
        return passages
    mid = (n + 1) // 2  # ceil(n/2)
    return passages[:mid] + passages[mid:][::-1]


def _strip_markdown_fences(text):
    """Strip markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def _parse_reranker_response(raw, n):
    """Parse reranker JSON response. Returns list of valid indices.

    Graceful degradation: if parsing fails or indices are invalid,
    returns range(n) (original order).
    """
    text = _strip_markdown_fences(raw)

    try:
        parsed = json.loads(text)
        indices = parsed.get("ranked_indices", [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        return list(range(n))

    # Validate: keep only ints in [0, n), deduplicate
    seen = set()
    deduped = []
    for i in indices:
        if isinstance(i, int) and 0 <= i < n and i not in seen:
            deduped.append(i)
            seen.add(i)

    # Append missing indices at end
    for i in range(n):
        if i not in seen:
            deduped.append(i)

    return deduped


def _parse_judge_response_v3(raw):
    """Parse judge JSON. Returns {score, reason, judge_raw}.

    Graceful degradation: score=2, reason="judge_error" on parse failure.
    """
    text = _strip_markdown_fences(raw)

    try:
        parsed = json.loads(text)
        score = parsed.get("score", 2)
        if score not in (1, 2, 3):
            score = 2
        return {
            "score": score,
            "reason": parsed.get("reason", ""),
            "judge_raw": raw,
        }
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {"score": 2, "reason": "judge_error", "judge_raw": raw}


# ─── v3 LLM-Calling Functions ───────────────────────────────────────

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

ANSWER_SYSTEM_PROMPT_V3 = (
    "You are a precise technical research assistant. You are given a question\n"
    "and a set of retrieved passages from a knowledge base of AI and machine\n"
    "learning literature.\n\n"
    "Answer the question by synthesizing from the passages. Be specific — name\n"
    "mechanisms, frameworks, and tradeoffs that appear in the passages rather\n"
    "than speaking in generalities. If the passages do not contain enough\n"
    "information to answer the question, say so explicitly rather than filling\n"
    "gaps with your own knowledge. Do not pad your answer with caveats,\n"
    "introductions, or summaries. Write for an engineer who wants the answer,\n"
    "not an explanation of how you found it."
)

JUDGE_SYSTEM_PROMPT_V3 = (
    "You are an impartial evaluator of retrieval quality. You will be given a\n"
    "source passage, a question generated from that passage, the retrieved passages\n"
    "used to answer it, and an answer.\n\n"
    "Your job is to score the answer for faithfulness to the source passage and\n"
    "the retrieved material on a scale of 1-3:\n\n"
    "1 = Unfaithful: introduces facts not in the source or retrieved passages,\n"
    "    or contradicts them\n"
    "2 = Partially faithful: mostly grounded but includes at least one unsupported claim\n"
    "3 = Faithful: every claim is supported by the source passage or retrieved passages\n\n"
    "Respond with JSON only:\n"
    '{"score": 1|2|3, "reason": "one sentence"}'
)


def rerank(query, candidates):
    """LLM-based listwise reranking of unified candidate set.

    Returns (reordered_candidates, raw_llm_response).
    """
    if not candidates:
        return [], ""

    passage_lines = []
    for i, c in enumerate(candidates):
        label = SIGNAL_LABELS.get(c["signal"], c["signal"])
        passage_lines.append(
            f"[{i}] Signal: {label}\n"
            f"    {c['text'][:500]}"
        )

    user_msg = (
        f"QUERY: {query}\n\n"
        f"CANDIDATE PASSAGES:\n\n"
        + "\n\n".join(passage_lines)
        + "\n\nReturn the passage indices in relevance order as JSON."
    )

    try:
        raw = llm_call(RERANKER_SYSTEM_PROMPT, user_msg,
                       model=RERANKER_MODEL, temperature=RERANKER_TEMPERATURE)
        ranked_indices = _parse_reranker_response(raw, len(candidates))
        reranked = [candidates[i] for i in ranked_indices]
        return reranked, raw
    except Exception as e:
        print(f"  rerank exception: {e}", file=sys.stderr)
        return candidates, str(e)


def generate_fused_answer(query, reranked_passages):
    """Generate answer from reranked passages. Returns answer string or None."""
    if not reranked_passages:
        user_msg = f"Question: {query}\n\n{EMPTY_RETRIEVAL_MSG}"
    else:
        ordered = _u_shape_order(reranked_passages)
        passage_lines = []
        for i, c in enumerate(ordered):
            source = c.get("title") or c.get("source_file", "unknown")
            passage_lines.append(f"[{i + 1}] {source}: {c['text']}")
        passages_block = "\n\n".join(passage_lines)
        user_msg = (
            f"QUESTION:\n{query}\n\n"
            f"RETRIEVED PASSAGES:\n{passages_block}\n\n"
            "Answer the question using these passages."
        )

    try:
        return llm_call(ANSWER_SYSTEM_PROMPT_V3, user_msg,
                        model=ANSWER_MODEL, temperature=ANSWER_TEMPERATURE)
    except Exception as e:
        print(f"  generate_fused_answer exception: {e}", file=sys.stderr)
        return None


def judge_faithfulness(source_passage, query, retrieved_passages, answer):
    """Score answer faithfulness on 1-3 scale.

    Returns {score, reason, judge_raw}.
    """
    passages_text = "\n\n---\n\n".join(
        c.get("text", "") for c in retrieved_passages
    ) if retrieved_passages else "(no passages retrieved)"

    user_msg = (
        f"Source passage:\n{source_passage}\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved passages:\n{passages_text}\n\n"
        f"Answer:\n{answer}"
    )

    try:
        raw = llm_call(JUDGE_SYSTEM_PROMPT_V3, user_msg,
                       model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE)
        return _parse_judge_response_v3(raw)
    except Exception as e:
        print(f"  judge_faithfulness exception: {e}", file=sys.stderr)
        return {"score": 2, "reason": "judge_error", "judge_raw": str(e)}


# ─── v2 Functions (kept for backward compat until Phase 5) ──────────


def discover_books_v2(dense, hybrid):
    """v2: Discover books common to both collections."""
    pass  # kept for reference only


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
    """Judge two answers for faithfulness. Returns dict with winner, reason, judge_raw.

    Applies verbosity normalization: both answers are truncated to
    min(len(a), len(b)) * 1.2 characters before judging, so the judge
    cannot reward length differences.
    """
    # Verbosity bias mitigation: truncate both answers to min length * 1.2
    # so the judge cannot reward length differences.
    #
    # Basis:
    #   - Commey, "When 'Better' Prompts Hurt", Texas A&M University
    #     (arXiv:2601.22025) — documents 10-20% verbosity bias magnitude
    #     and recommends truncating to shorter answer + 20%.
    #   - Shi et al., "Deep Research: A Systematic Survey", Shandong University
    #     (arXiv:2512.02038) — confirms LLM judges "prefer longer responses,
    #     be affected by answer ordering, reward particular writing styles."
    #   - Anwar et al., "Foundational Challenges in Assuring Alignment and
    #     Safety of Large Language Models", University of Cambridge
    #     (arXiv:2404.09932) — documents "preference for verbose and longer
    #     answers" as a systematic cognitive bias in LLM-based evaluation.
    max_chars = int(min(len(answer_a), len(answer_b)) * 1.2)
    answer_a_trimmed = answer_a[:max_chars]
    answer_b_trimmed = answer_b[:max_chars]

    passages_a = "\n\n".join(_extract_passages(results_a)) if results_a else "(no passages)"
    passages_b = "\n\n".join(_extract_passages(results_b)) if results_b else "(no passages)"

    user_msg = (
        f"Source passage:\n{source_passage}\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved passages for A:\n{passages_a}\n\n"
        f"Retrieved passages for B:\n{passages_b}\n\n"
        f"Answer A:\n{answer_a_trimmed}\n\n"
        f"Answer B:\n{answer_b_trimmed}"
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
    """Colorized per-sample terminal output for v3."""
    score = sample.get("judge_score", 0)
    score_color = green if score == 3 else (yellow if score == 2 else red)

    print(f"\n  {bold(sample.get('book', '?'))}")
    print(f"  {dim(sample.get('source_passage_excerpt', '')[:200])}")
    print(f"  Query: {sample['query']}  [{sample.get('query_bucket', '?')}]")
    print(f"  Fused answer: {dim(str(sample.get('fused_answer', ''))[:ANSWER_EXCERPT_LEN])}")
    print(f"  Judge: {score_color(f'score={score}')} — {sample.get('judge_reason', '')}")


def compute_aggregates(samples):
    """Compute score-based aggregates for v3 fused retrieval."""
    total = len(samples)
    scores = [s["judge_score"] for s in samples]
    avg_score = sum(scores) / total if total else 0.0

    dense_hits = sum(1 for s in samples if s.get("dense_source_hit_rank") is not None)
    sparse_hits = sum(1 for s in samples if s.get("sparse_source_hit_rank") is not None)

    dedup_sizes = [s.get("dedup_set_size", 0) for s in samples]
    both_counts = [s.get("both_signal_count", 0) for s in samples]

    bucket_summary = {}
    for bucket_name in ("trivia", "conceptual", "operational"):
        bucket_samples = [s for s in samples if s.get("query_bucket") == bucket_name]
        bucket_scores = [s["judge_score"] for s in bucket_samples]
        bucket_summary[bucket_name] = {
            "avg_score": sum(bucket_scores) / len(bucket_scores) if bucket_scores else 0.0,
            "samples": len(bucket_samples),
            "score_distribution": {
                1: sum(1 for sc in bucket_scores if sc == 1),
                2: sum(1 for sc in bucket_scores if sc == 2),
                3: sum(1 for sc in bucket_scores if sc == 3),
            },
        }

    return {
        "avg_judge_score": round(avg_score, 2),
        "dense_hit_count": dense_hits,
        "sparse_hit_count": sparse_hits,
        "dense_hit_rate": round(dense_hits / total, 3) if total else 0.0,
        "sparse_hit_rate": round(sparse_hits / total, 3) if total else 0.0,
        "avg_dedup_set_size": round(sum(dedup_sizes) / total, 1) if total else 0.0,
        "avg_both_signal_count": round(sum(both_counts) / total, 1) if total else 0.0,
        "bucket_summary": bucket_summary,
    }


def print_summary(metadata):
    """Print v3 score distribution table."""
    bucket_summary = metadata.get("bucket_summary", {})

    print(f"\n{'=' * 70}")
    print(f"  {'':20s} {'score=1':>8s}  {'score=2':>8s}  {'score=3':>8s}  {'avg':>6s}  {'samples':>7s}")
    print(f"  {'─' * 60}")
    for bucket_name in ("trivia", "conceptual", "operational"):
        b = bucket_summary.get(bucket_name, {})
        dist = b.get("score_distribution", {})
        s1, s2, s3 = dist.get(1, 0), dist.get(2, 0), dist.get(3, 0)
        avg = b.get("avg_score", 0.0)
        n = b.get("samples", 0)
        label = bucket_name.capitalize()
        print(f"  {label:20s} {red(s1):>17s}  {yellow(s2):>17s}  {green(s3):>17s}  {avg:>6.2f}  {n:>7d}")
    print(f"  {'─' * 60}")

    # Total row
    all_scores = []
    total_dist = {1: 0, 2: 0, 3: 0}
    total_n = 0
    for b in bucket_summary.values():
        dist = b.get("score_distribution", {})
        for k in (1, 2, 3):
            total_dist[k] += dist.get(k, 0)
        total_n += b.get("samples", 0)
    avg_total = metadata.get("avg_judge_score", 0.0)
    print(f"  {'Total':20s} {red(total_dist[1]):>17s}  {yellow(total_dist[2]):>17s}  "
          f"{green(total_dist[3]):>17s}  {avg_total:>6.2f}  {total_n:>7d}")
    print(f"{'=' * 70}")



# ─── Main ────────────────────────────────────────────────────────────


def main(argv=None):
    args = parse_args(argv)
    collection = args.collection
    dense_k = args.dense_k
    sparse_k = args.sparse_k

    print(f"blind_ab_test v{SCRIPT_VERSION}")
    print(f"Collection: {collection}  Dense-k: {dense_k}  Sparse-k: {sparse_k}")
    print(f"Positions/book: {args.positions}  Seed: {args.seed}")

    if args.seed is not None:
        random.seed(args.seed)

    # Discover books
    books = discover_books(collection)
    if not books:
        print("ERROR: No books in collection.", file=sys.stderr)
        sys.exit(1)
    print(f"\nFound {len(books)} books.")

    samples = []

    for book in books:
        sf = book["source_file"]
        title = book.get("book_title", sf)
        print(f"\n{'─' * 60}")
        print(f"Book: {bold(title)} ({sf})")

        for pos in range(args.positions):
            try:
                # 1. Pick random passage (book-scoped)
                chunk = pick_passage(collection, sf)
                if not chunk:
                    continue

                # 2. Generate query
                query_result = generate_query(chunk["text"])
                if not query_result:
                    continue
                query = query_result["query"]
                query_type = query_result["query_type"]
                query_bucket = query_result["query_bucket"]

                # 3. Dense signal retrieval (sparse_weight=0, dense_weight=1)
                dense_results = retrieve(collection, query,
                                         sparse_weight=0, dense_weight=1,
                                         top_k=dense_k)
                if dense_results is None:
                    continue

                # 4. Sparse signal retrieval (sparse_weight=1, dense_weight=0)
                sparse_results = retrieve(collection, query,
                                          sparse_weight=1, dense_weight=0,
                                          top_k=sparse_k)
                if sparse_results is None:
                    continue

                # 5. Dedup and normalize
                candidates = dedup_and_normalize(dense_results, sparse_results)
                dedup_set_size = len(candidates)
                both_signal_count = sum(1 for c in candidates if c["signal"] == "both")

                # 6. Rerank
                reranked, reranker_raw = rerank(query, candidates)

                # 7. Generate fused answer
                fused_answer = generate_fused_answer(query, reranked)
                if fused_answer is None:
                    continue

                # 8. Judge faithfulness
                verdict = judge_faithfulness(chunk["text"], query, reranked, fused_answer)

                # 9. Hit rank for both signals
                dense_hit = compute_hit_rank(chunk, dense_results)
                sparse_hit = compute_hit_rank(chunk, sparse_results)

                # 10. Build v3 sample dict
                sample = {
                    "book": title,
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
                    "query_type": query_type,
                    "query_bucket": query_bucket,
                    "why_this_is_realistic": query_result["why_this_is_realistic"],
                    "query_generation_raw": query_result["query_generation_raw"],
                    "dense_raw_results": dense_results,
                    "sparse_raw_results": sparse_results,
                    "dense_retrieved_passages": _extract_passages(dense_results),
                    "sparse_retrieved_passages": _extract_passages(sparse_results),
                    "dedup_set_size": dedup_set_size,
                    "both_signal_count": both_signal_count,
                    "reranked_passages": reranked,
                    "reranker_raw": reranker_raw,
                    "fused_answer": fused_answer,
                    "judge_score": verdict["score"],
                    "judge_reason": verdict["reason"],
                    "judge_raw": verdict["judge_raw"],
                    "dense_source_hit_rank": dense_hit["rank"],
                    "dense_source_hit_method": dense_hit["match_method"],
                    "sparse_source_hit_rank": sparse_hit["rank"],
                    "sparse_source_hit_method": sparse_hit["match_method"],
                }

                # 11. Print
                print_sample(sample)

                # 12. Append
                samples.append(sample)

            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                continue

    # Aggregates
    agg = compute_aggregates(samples)

    metadata = {
        "version": "3.0.0",
        "collection": collection,
        "dense_k": dense_k,
        "sparse_k": sparse_k,
        "positions_per_book": args.positions,
        "total_samples": len(samples),
        "avg_judge_score": agg["avg_judge_score"],
        "bucket_summary": agg["bucket_summary"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "query_model": QUERY_MODEL,
        "reranker_model": RERANKER_MODEL,
        "mcp_url": MCP_URL,
        "dense_hit_count": agg["dense_hit_count"],
        "sparse_hit_count": agg["sparse_hit_count"],
        "dense_hit_rate": agg["dense_hit_rate"],
        "sparse_hit_rate": agg["sparse_hit_rate"],
        "avg_dedup_set_size": agg["avg_dedup_set_size"],
        "avg_both_signal_count": agg["avg_both_signal_count"],
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
