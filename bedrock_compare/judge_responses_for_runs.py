#!/usr/bin/env python3
"""Judge LLM responses from query result JSON files.

Reads harvested response files from query_responses/, calls an OpenAI-compatible
LLM API to score each response (1-10) based on the judge prompt, writes scored
assessments to response_assessment/ with matching filenames.

Usage:
    python3 judge_responses_for_runs.py --limit 10
    python3 judge_responses_for_runs.py --limit 10 --overwrite
    python3 judge_responses_for_runs.py --prompt-judge system_prompts/prompt_judge.txt
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

_logging_level = os.getenv("JUDGE_LOG_LEVEL", "INFO").upper()
# Import shared logging helpers.
try:
    from bedrock_compare.logging_utils import (
        log_key, log_info, log_dim, log_green, log_red, log_yellow,
        log_blue, log_cyan,
        truncate_line,
    )
except ImportError:
    from logging_utils import (
        log_key, log_info, log_dim, log_green, log_red, log_yellow,
        log_blue, log_cyan,
        truncate_line,
    )

logging.basicConfig(
    level=logging.DEBUG if _logging_level == "DEBUG" else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("judge")


# ── Helpers ─────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_input_files(
    responses_dir: Path, limit: Optional[int]
) -> List[Path]:
    """Select response files sorted by filename (prompt ID order)."""
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0")
    files = list(responses_dir.glob("*.json"))

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


# ── Core pipeline ───────────────────────────────────────────────────────────

def build_judge_prompt(query: str, responses: Dict[str, Any]) -> str:
    """Build the user message for the LLM judge.

    Includes the query and all response texts, formatted for comparative
    quality assessment.
    """
    parts = [f"Prompt: {query}\n\nResponses:\n"]
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
        logger.warning("  LLM hit token limit (finish_reason=length). Output may be truncated. usage=%s", raw_usage)

    if not content:
        logger.debug("  LLM returned empty content. finish_reason=%s usage=%s",
                     raw_finish_reason, raw_usage)
        # Try delta fallback (some backends return via delta)
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
        # Remove first and last fence lines
        lines = lines[1:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug("  JSON parse failed: %s\n  Raw response (first 500 chars):\n%s",
                     e, text[:500])
        return None

    # Validate: keys should be source names, values should have score blocks
    result = {}
    valid_keys = set(source_names)
    for key, val in parsed.items():
        if key not in valid_keys:
            logger.warning("  Unexpected source key in judge response: %s", key)
            continue
        block = _validate_score_block(val, key)
        if block is not None:
            result[key] = block

    return result if result else None


def process_file(
    fpath: Path,
    out_path: Path,
    system_prompt: str,
    model: str,
    client: OpenAI,
    overwrite: bool,
) -> bool:
    """Process one response file. Returns True on success."""
    log_cyan(f"▶ {fpath.name}")

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data.get("prompt", "")
    responses = data.get("responses", {})

    if not responses:
        log_yellow("  No responses found, skipping")
        return False

    source_names = list(responses.keys())
    log_info(f"  Sources: {', '.join(source_names)}")
    log_dim(f"  Query: {truncate_line(prompt, shutil.get_terminal_size().columns - 10)}")

    # Build judge prompt
    user_message = build_judge_prompt(prompt, responses)

    # Call LLM judge — blue announcement
    started_epoch = time.time()
    log_blue("  LLM judge: calling completion endpoint...")
    try:
        llm_response = call_judge_llm(client, system_prompt, user_message, model)
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

    # Print scores — always show them
    score_parts = []
    for src in source_names:
        block = scores.get(src, {})
        if isinstance(block, dict):
            rating = block.get("rating", block.get("response_score", "?"))
            score_parts.append(f"{src}: {rating}/10")
        else:
            score_parts.append(f"{src}: ?/10")
    log_green(f"  {' | '.join(score_parts)}")

    # Build output
    output: Dict[str, Any] = {
        "input_file": fpath.name,
        "input_id": data.get("input_id"),
        "category": data.get("category"),
        "proficiency": data.get("proficiency"),
        "topk": data.get("topk"),
        "prompt": prompt,
        "timestamp": data.get("timestamp"),
        "responses": responses,
        "scores": scores,
        "judge_model": llm_response.get("model", model),
        "judge_elapsed_seconds": elapsed,
        "judged_at": now_iso(),
        "raw_judge_response": response_text[:2000] if response_text else "",
    }

    # Write output
    if not out_path.exists() or overwrite:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, sort_keys=True, indent=2)
        log_dim(f"  → {out_path.name}")
    else:
        log_dim(f"  ⏭ {out_path.name} (already exists)")

    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Judge LLM responses from query result files."
    )
    parser.add_argument(
        "--query-responses-dir",
        default="query_responses",
        help="Directory containing harvested response JSON files (default: query_responses)",
    )
    parser.add_argument(
        "--response-assessment-dir",
        default="response_assessment",
        help="Directory to write scored assessment JSON files (default: response_assessment)",
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
        help="Overwrite existing assessment files",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("judge").setLevel(logging.DEBUG)

    responses_dir = Path(args.query_responses_dir)
    assessment_dir = Path(args.response_assessment_dir)

    # Validate input directory
    if not responses_dir.exists():
        log_red(f"Query responses directory not found: {responses_dir}")
        sys.exit(1)
    if not responses_dir.is_dir():
        log_red(f"Query responses path is not a directory: {responses_dir}")
        sys.exit(1)

    # Load judge system prompt
    judge_prompt_path = Path(args.prompt_judge)
    if not judge_prompt_path.exists():
        log_red(f"Judge prompt file not found: {args.prompt_judge}")
        sys.exit(1)
    system_prompt = judge_prompt_path.read_text(encoding="utf-8")

    # Create OpenAI client
    client = OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
    )

    # Health check — dimmed
    try:
        models = client.models.list()
        log_dim(f"Connected to {args.api_base} (model: {args.model})")
    except Exception as e:
        log_red(f"Failed to connect to {args.api_base}: {e}")
        sys.exit(1)

    # Select files
    files = select_input_files(responses_dir, args.limit)
    if not files and args.limit not in (0, None):
        log_red(f"No JSON files found in {responses_dir}")
        sys.exit(1)

    if args.limit is not None:
        log_key(f"Selected {len(files)} input files from {responses_dir}")
    else:
        log_key(f"Selected all input files from {responses_dir}")

    if args.limit == 0:
        log_key("Limit is 0, exiting.")
        return

    assessment_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    success_count = 0
    fail_count = 0

    for fpath in files:
        out_path = assessment_dir / fpath.name
        try:
            ok = process_file(fpath, out_path, system_prompt, args.model, client, args.overwrite)
            if ok:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            log_red(f"Failed to process {fpath.name}: {e}")
            fail_count += 1

    log_key(f"\nDone. {success_count} succeeded, {fail_count} failed out of {len(files)} files.")
    log_dim(f"Assessments in: {assessment_dir}")


if __name__ == "__main__":
    main()