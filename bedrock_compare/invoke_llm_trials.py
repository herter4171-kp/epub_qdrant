#!/usr/bin/env python3
"""Unified LLM pipeline: harvest responses then judge them in a single run.

Reads query result files from query_results/, calls the LLM API for each source
(harvest phase), then calls the LLM judge to score each response (judge phase),
and writes both query_responses/ and response_assessment/ outputs.

Usage:
    python3 invoke_llm_trials.py --limit 2
    python3 invoke_llm_trials.py --limit 10 --overwrite
    python3 invoke_llm_trials.py --prompt-dense prompt_dense.txt --prompt-semantic prompt_semantic.txt --prompt-judge prompt_judge.txt

Supersedes:
    harvest_responses.py + judge_responses_for_runs.py (run separately)
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Import shared logging helpers.
try:
    from bedrock_compare.logging_utils import (
        log_key, log_info, log_dim, log_green, log_red, log_yellow,
        log_blue, log_cyan,
        truncate_line, seen_prompt,
    )
except ImportError:
    from logging_utils import (
        log_key, log_info, log_dim, log_green, log_red, log_yellow,
        log_blue, log_cyan,
        truncate_line, seen_prompt,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("invoke_llm_trials")


# ── Helpers ───────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_source_payload(payload: Any) -> str:
    """Serialize a source payload for the assistant message."""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def select_input_files(results_dir: Path, limit: Optional[int]) -> List[Path]:
    """Select input files sorted by prompt ID (numeric), then fallback to filename.

    Filenames are like ``1_spatial_orientation_1_2.json`` — the leading number
    is the prompt ID.  Lexicographic sorting puts ``10_`` before ``2_``, so we
    sort by the extracted integer first, then by the full filename for ties.
    """
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0")
    files = list(results_dir.glob("*.json"))

    def _sort_key(path: Path):
        stem = path.stem  # e.g. "1_spatial_orientation_1_2"
        try:
            prompt_id = int(stem.split("_")[0])
        except ValueError:
            prompt_id = 0
        return (prompt_id, stem)

    files.sort(key=_sort_key)
    if limit is None:
        return files
    return files[:limit]


# ── Prompt loading ────────────────────────────────────────────────────────

def _load_system_prompt(path: str) -> str:
    """Load system prompt from file, or raise."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"System prompt file not found: {path}")
    return p.read_text(encoding="utf-8")


# Map source names to prompt keys
SOURCE_PROMPT_MAP = {
    "papers": "dense",
    "bedrock": "dense",
    "papers_semantic": "semantic",
}
DEFAULT_PROMPT_KEY = "dense"


def resolve_source_prompt(source_name: str, prompts: Dict[str, str]) -> str:
    """Return the system prompt text for a given source name."""
    prompt_key = SOURCE_PROMPT_MAP.get(source_name, DEFAULT_PROMPT_KEY)
    return prompts[prompt_key]


# ── Phase 1: Harvest ────────────────────────────────────────────────────

def call_llm(client: OpenAI, system_prompt: str, source_payload: Any,
             prompt: str, model: str) -> Dict[str, Any]:
    """Call OpenAI-compatible chat completions endpoint (harvest phase)."""
    serialized = serialize_source_payload(source_payload)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Results: " + serialized},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
    )
    return {
        "choices": [{"message": {"content": response.choices[0].message.content}}],
        "model": response.model,
    }


def extract_response_text(response: Dict[str, Any]) -> str:
    """Extract text content from response dict."""
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def model_name_from_response(response: Dict[str, Any]) -> Optional[str]:
    """Try to extract model name from response."""
    try:
        return response.get("model") or None
    except Exception:
        return None


# ── Phase 2: Judge ──────────────────────────────────────────────────────

def build_judge_prompt(query: str, responses: Dict[str, Any], sources: Optional[Dict[str, Any]] = None) -> str:
    """Build the user message for the LLM judge.

    Order per judge prompt spec:
      1. The original user prompt
      2. The retrieval results each source had access to
      3. Each source's generated response
    """
    parts = [f"Prompt: {query}\n\n"]

    # ── Retrieval results ──────────────────────────────────────────────
    if sources:
        parts.append("Retrieval Results:\n")
        for source_name, source_payload in sources.items():
            content = source_payload.get("content", [])
            if not content:
                continue
            parts.append(f"--- Source: {source_name} ---\n")
            for item in content:
                text = item.get("text", "") or ""
                text_trunc = text[:500]
                parts.append(text_trunc)
                parts.append("\n")
            parts.append("---\n")

    # ── Responses ──────────────────────────────────────────────────────
    parts.append("Responses:\n")
    for source_name, resp_data in responses.items():
        text = resp_data.get("response_text", "") or ""
        model = resp_data.get("model", "unknown")
        error = resp_data.get("error")
        if error:
            parts.append(f"--- Source: {source_name} (model: {model}, ERROR) ---\n{error['message']}\n")
        else:
            parts.append(f"--- Source: {source_name} (model: {model}) ---\n{text}\n")

    return "\n".join(parts)


def call_judge_llm(
    client: OpenAI,
    system_prompt: str,
    user_message: str,
    model: str,
) -> Dict[str, Any]:
    """Call the LLM judge and return the parsed response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=8192,
        temperature=0.0,
    )
    choice = response.choices[0]
    content = choice.message.content if hasattr(choice.message, 'content') else None
    reasoning_content = choice.message.reasoning_content if hasattr(choice.message, 'reasoning_content') else None
    raw_finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else None
    raw_usage = response.usage if hasattr(response, 'usage') else None

    if raw_finish_reason == 'length':
        logger.warning("  LLM judge hit token limit (finish_reason=length). Output may be truncated. usage=%s", raw_usage)

    if not content:
        logger.debug("  LLM judge returned empty content. finish_reason=%s usage=%s",
                      raw_finish_reason, raw_usage)
        if hasattr(choice, 'delta') and choice.delta and hasattr(choice.delta, 'content'):
            content = choice.delta.content
            logger.debug("  Content found in choice.delta instead")
        if not content:
            logger.debug("  Raw choice.message: %s", choice.message)
            logger.debug("  Raw choice: %s", {k: v for k, v in vars(choice).items() if not k.startswith('_')})

    result = {
        "choices": [{"message": {"content": content}}],
        "model": response.model,
    }
    if reasoning_content:
        result["reasoning_content"] = reasoning_content

    return result


def _validate_score_block(val: Any, valid_keys: set) -> Optional[Dict[str, Any]]:
    """Validate a source's score block. Accepts either:

    Old schema: {"rating": int, "basis": str}
    New schema: {"retrieval_score": int, "retrieval_basis": str,
                 "response_score": int, "response_basis": str}

    Returns normalized dict or None.
    """
    if not isinstance(val, dict):
        return None

    # Old schema: rating + basis
    if "rating" in val and "basis" in val:
        rating = val["rating"]
        if isinstance(rating, int) and 1 <= rating <= 10:
            return {
                "rating": rating,
                "basis": str(val["basis"]),
            }
        logger.warning("  Invalid rating for source %s: %s (must be int 1-10)", valid_keys, rating)
        return None

    # New schema: retrieval_score + retrieval_basis + response_score + response_basis
    if "retrieval_score" in val and "retrieval_basis" in val and \
       "response_score" in val and "response_basis" in val:
        rs = val["retrieval_score"]
        ps = val["response_score"]
        ok = True
        if not (isinstance(rs, int) and 1 <= rs <= 10):
            logger.warning("  Invalid retrieval_score for source %s: %s (must be int 1-10)", valid_keys, rs)
            ok = False
        if not (isinstance(ps, int) and 1 <= ps <= 10):
            logger.warning("  Invalid response_score for source %s: %s (must be int 1-10)", valid_keys, ps)
            ok = False
        if ok:
            return {
                "retrieval_score": rs,
                "retrieval_basis": str(val["retrieval_basis"]),
                "response_score": ps,
                "response_basis": str(val["response_basis"]),
            }
        return None

    logger.warning("  Invalid response structure for source %s: %s", valid_keys, val)
    return None


def parse_judge_response(
    response_text: str, source_names: List[str]
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Parse the LLM's JSON scoring response.

    Returns a dict mapping source_name -> score object.
    Supports both old schema (rating/basis) and new schema
    (retrieval_score/retrieval_basis/response_score/response_basis).
    Returns None if parsing fails.
    """
    text = response_text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug("  JSON parse failed: %s\n  Raw response (first 500 chars):\n%s",
                      e, text[:500])
        return None

    # Validate: keys should be source names or "source_verdict", values should have score blocks
    result = {}
    valid_keys = set(source_names) | {"source_verdict"}
    for key, val in parsed.items():
        if key not in valid_keys:
            logger.warning("  Unexpected source key in judge response: %s", key)
            continue
        block = _validate_score_block(val, key)
        if block is not None:
            result[key] = block

    return result if result else None


# ── Core pipeline ────────────────────────────────────────────────────────

def process_file(
    fpath: Path,
    responses_out_path: Path,
    assessment_out_path: Path,
    model: str,
    client: OpenAI,
    prompts: Dict[str, str],
    overwrite: bool,
) -> bool:
    """Process one input file: harvest responses, then judge them.

    Returns True on success.
    """
    log_cyan(f"▶ {fpath.name}")

    with open(fpath, "r", encoding="utf-8") as f:
        result = json.load(f)

    prompt = result["prompt"]
    sources = result.get("sources", {})

    # Print the prompt if we haven't seen it before
    prompt_trunc = truncate_line(prompt, shutil.get_terminal_size().columns - 10)
    if not seen_prompt(prompt):
        log_key(f"  [NEW PROMPT] {prompt_trunc}")
    else:
        log_dim(f"  [KNOWN PROMPT] {prompt_trunc}")

    # ── Harvest Phase ──────────────────────────────────────────────────
    harvested_responses: Dict[str, Any] = {}
    harvest_success = False

    for source_name, source_payload in sources.items():
        started_epoch = time.time()

        log_blue(f"  LLM calling: {source_name}...")

        try:
            system_prompt = resolve_source_prompt(source_name, prompts)
            response = call_llm(client, system_prompt, source_payload, prompt, model)
            elapsed = round(time.time() - started_epoch, 3)

            resp_text = extract_response_text(response)
            log_green(f"  ✓ {source_name}: {truncate_line(resp_text)} ({elapsed}s)")

            harvested_responses[source_name] = {
                "response_text": resp_text,
                "model": model_name_from_response(response) or model,
                "started_at": datetime.fromtimestamp(started_epoch, tz=timezone.utc).isoformat(),
                "completed_at": datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "error": None,
            }
            harvest_success = True
        except Exception as exc:
            elapsed = 0.0
            log_red(f"  ✗ {source_name}: {type(exc).__name__}: {truncate_line(str(exc))}")
            logger.warning("  source %s FAILED: %s", source_name, exc)

            harvested_responses[source_name] = {
                "response_text": "",
                "model": model,
                "started_at": datetime.fromtimestamp(started_epoch, tz=timezone.utc).isoformat(),
                "completed_at": datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

    # Write query_responses output
    if harvest_success:
        response_output: Dict[str, Any] = {
            "input_file": fpath.name,
            "input_id": result.get("id"),
            "category": result.get("category"),
            "proficiency": result.get("proficiency"),
            "topk": result.get("topk"),
            "prompt": prompt,
            "timestamp": result.get("timestamp"),
            "responses": harvested_responses,
        }

        if not responses_out_path.exists() or overwrite:
            responses_out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(responses_out_path, "w", encoding="utf-8") as f:
                json.dump(response_output, f, ensure_ascii=False, sort_keys=True, indent=2)
            log_dim(f"  → {responses_out_path.name} (responses)")
        else:
            log_dim(f"  ⏭ {responses_out_path.name} (already exists)")
    else:
        log_yellow("  No successful harvests, skipping judge phase")
        return False

    # ── Judge Phase ────────────────────────────────────────────────────
    source_names = list(harvested_responses.keys())
    log_info(f"  Sources: {', '.join(source_names)}")
    log_dim(f"  Query: {truncate_line(prompt, shutil.get_terminal_size().columns - 10)}")

    user_message = build_judge_prompt(prompt, harvested_responses, sources)

    started_epoch = time.time()
    log_blue("  LLM judge: calling completion endpoint...")
    try:
        llm_response = call_judge_llm(client, prompts["judge"], user_message, model)
        elapsed = round(time.time() - started_epoch, 3)
        response_text = llm_response["choices"][0]["message"]["content"] or ""
        log_dim(f"  LLM judge returned in {elapsed:.1f}s")
    except Exception as exc:
        log_red(f"  LLM judge call FAILED: {exc}")
        elapsed = 0.0
        llm_response = {"model": model, "choices": [{"message": {"content": ""}}]}
        response_text = ""

    # Parse judge scores
    scores = None
    if response_text:
        scores = parse_judge_response(response_text, source_names)

    if scores is None:
        log_yellow("  Failed to parse judge response, assigning 0 for all sources")
        scores = {
            src: {"rating": 0, "basis": "Failed to parse judge response"}
            for src in source_names
        }

    # Print scores
    score_parts = []
    for src in source_names:
        block = scores.get(src, {})
        if isinstance(block, dict):
            rating = block.get("rating", block.get("response_score", "?"))
            score_parts.append(f"{src}: {rating}/10")
        else:
            score_parts.append(f"{src}: ?/10")
    log_green(f"  {' | '.join(score_parts)}")

    # Write response_assessment output
    assessment_output: Dict[str, Any] = {
        "input_file": fpath.name,
        "input_id": result.get("input_id"),
        "category": result.get("category"),
        "proficiency": result.get("proficiency"),
        "topk": result.get("topk"),
        "prompt": prompt,
        "timestamp": result.get("timestamp"),
        "responses": harvested_responses,
        "scores": scores,
        "judge_model": llm_response.get("model", model),
        "judge_elapsed_seconds": elapsed,
        "judged_at": now_iso(),
        "raw_judge_response": response_text[:2000] if response_text else "",
    }

    if not assessment_out_path.exists() or overwrite:
        assessment_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(assessment_out_path, "w", encoding="utf-8") as f:
            json.dump(assessment_output, f, ensure_ascii=False, sort_keys=True, indent=2)
        log_dim(f"  → {assessment_out_path.name} (assessment)")
    else:
        log_dim(f"  ⏭ {assessment_out_path.name} (already exists)")

    return True


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified LLM pipeline: harvest responses then judge them."
    )
    parser.add_argument(
        "--query-results-dir",
        default="query_results",
        help="Directory containing query result JSON files (default: query_results)",
    )
    parser.add_argument(
        "--query-responses-dir",
        default="query_responses",
        help="Directory to write response JSON files (default: query_responses)",
    )
    parser.add_argument(
        "--response-assessment-dir",
        default="response_assessment",
        help="Directory to write scored assessment JSON files (default: response_assessment)",
    )
    parser.add_argument(
        "--prompt-dense",
        required=True,
        help="Path to file containing the system prompt for dense sources (papers, bedrock)",
    )
    parser.add_argument(
        "--prompt-semantic",
        required=True,
        help="Path to file containing the system prompt for semantic source (papers-semantic)",
    )
    parser.add_argument(
        "--prompt-judge",
        required=True,
        help="Path to file containing the system prompt for the judge",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N input files (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--model",
        default="gemma4",
        help="Model name (default: gemma4)",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("LITELLM_API_URL", "https://litellm.twr.church/v1"),
        help="OpenAI-compatible API base URL (default: $LITELLM_API_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LITELLM_API_KEY", ""),
        help="API key (default: $LITELLM_API_KEY)",
    )

    args = parser.parse_args()

    results_dir = Path(args.query_results_dir)
    responses_dir = Path(args.query_responses_dir)
    assessment_dir = Path(args.response_assessment_dir)

    # Validate input directory
    if not results_dir.exists():
        logger.error("Query results directory not found: %s", results_dir)
        sys.exit(1)
    if not results_dir.is_dir():
        logger.error("Query results path is not a directory: %s", results_dir)
        sys.exit(1)

    # Load all 3 prompts
    prompts = {
        "dense": _load_system_prompt(args.prompt_dense),
        "semantic": _load_system_prompt(args.prompt_semantic),
        "judge": _load_system_prompt(args.prompt_judge),
    }

    # Create OpenAI client
    client = OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
    )

    # Health check
    try:
        models = client.models.list()
        log_dim(f"Connected to {args.api_base} (model: {args.model})")
    except Exception as e:
        log_red(f"Failed to connect to {args.api_base}: {e}")
        sys.exit(1)

    # Select files
    files = select_input_files(results_dir, args.limit)
    if not files and args.limit not in (0, None):
        log_red(f"No JSON files found in {results_dir}")
        sys.exit(1)

    if args.limit is not None:
        log_key(f"Selected {len(files)} input files from {results_dir}")
    else:
        log_key(f"Selected all input files from {results_dir}")

    if args.limit == 0:
        log_key("Limit is 0, exiting.")
        return

    responses_dir.mkdir(parents=True, exist_ok=True)
    assessment_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    success_count = 0
    fail_count = 0

    for fpath in files:
        responses_out = responses_dir / fpath.name
        assessment_out = assessment_dir / fpath.name
        try:
            ok = process_file(
                fpath, responses_out, assessment_out,
                args.model, client, prompts, args.overwrite,
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            log_red(f"Failed to process {fpath.name}: {e}")
            fail_count += 1

    log_key(f"\nDone. {success_count} succeeded, {fail_count} failed out of {len(files)} files.")
    log_dim(f"Responses in: {responses_dir}")
    log_dim(f"Assessments in: {assessment_dir}")


if __name__ == "__main__":
    main()