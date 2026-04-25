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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

logging.basicConfig(
    level=logging.DEBUG,
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


def parse_judge_response(
    response_text: str, source_names: List[str]
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Parse the LLM's JSON scoring response.

    Returns a dict mapping source_name -> {"rating": int, "basis": str}.
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

    # Validate: keys should be source names, values should have rating and basis
    result = {}
    valid_keys = set(source_names)
    for key, val in parsed.items():
        if key not in valid_keys:
            logger.warning("  Unexpected source key in judge response: %s", key)
            continue
        if isinstance(val, dict) and "rating" in val and "basis" in val:
            rating = val["rating"]
            if isinstance(rating, int) and 1 <= rating <= 10:
                result[key] = {
                    "rating": rating,
                    "basis": str(val["basis"]),
                }
            else:
                logger.warning(
                    "  Invalid rating for source %s: %s (must be int 1-10)",
                    key, rating,
                )
        else:
            logger.warning("  Invalid response structure for source %s: %s", key, val)

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
    logger.info("Processing %s", fpath.name)

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data.get("prompt", "")
    responses = data.get("responses", {})

    if not responses:
        logger.warning("  No responses found in %s, skipping", fpath.name)
        return False

    source_names = list(responses.keys())
    logger.info("  Sources: %s", ", ".join(source_names))

    # Build judge prompt
    user_message = build_judge_prompt(prompt, responses)

    # Call LLM judge
    started_epoch = time.time()
    try:
        logger.info("  Calling LLM judge...")
        llm_response = call_judge_llm(client, system_prompt, user_message, model)
        elapsed = round(time.time() - started_epoch, 3)
        response_text = llm_response["choices"][0]["message"]["content"] or ""
        logger.info("  LLM judge returned in %.1fs", elapsed)
    except Exception as exc:
        logger.error("  LLM judge call FAILED: %s", exc)
        elapsed = 0.0
        llm_response = {"model": model, "choices": [{"message": {"content": ""}}]}
        response_text = ""

    # Parse judge scores
    scores = None
    if response_text:
        scores = parse_judge_response(response_text, source_names)

    if scores is None:
        # Failed to parse — assign 0 for all sources
        logger.warning("  Failed to parse judge response, assigning 0 for all sources")
        scores = {
            src: {"rating": 0, "basis": "Failed to parse judge response"}
            for src in source_names
        }

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
        logger.info("Wrote %s", out_path.name)
    else:
        logger.info("Skipping %s (already exists, use --overwrite)", out_path.name)

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

    args = parser.parse_args()

    responses_dir = Path(args.query_responses_dir)
    assessment_dir = Path(args.response_assessment_dir)

    # Validate input directory
    if not responses_dir.exists():
        logger.error("Query responses directory not found: %s", responses_dir)
        sys.exit(1)
    if not responses_dir.is_dir():
        logger.error("Query responses path is not a directory: %s", responses_dir)
        sys.exit(1)

    # Load judge system prompt
    judge_prompt_path = Path(args.prompt_judge)
    if not judge_prompt_path.exists():
        logger.error("Judge prompt file not found: %s", args.prompt_judge)
        sys.exit(1)
    system_prompt = judge_prompt_path.read_text(encoding="utf-8")

    # Create OpenAI client
    client = OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
    )

    # Health check
    try:
        models = client.models.list()
        logger.info("Connected to %s (model: %s)", args.api_base, args.model)
    except Exception as e:
        logger.error("Failed to connect to %s: %s", args.api_base, e)
        sys.exit(1)

    # Select files
    files = select_input_files(responses_dir, args.limit)
    if not files and args.limit not in (0, None):
        logger.error("No JSON files found in %s", responses_dir)
        sys.exit(1)

    if args.limit is not None:
        logger.info("Selected %d input files from %s", len(files), responses_dir)
    else:
        logger.info("Selected all input files from %s", responses_dir)

    if args.limit == 0:
        logger.info("Limit is 0, exiting.")
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
            logger.error("Failed to process %s: %s", fpath.name, e)
            fail_count += 1

    logger.info("\nDone. %d succeeded, %d failed out of %d files.",
                success_count, fail_count, len(files))
    logger.info("Assessments in: %s", assessment_dir)


if __name__ == "__main__":
    main()