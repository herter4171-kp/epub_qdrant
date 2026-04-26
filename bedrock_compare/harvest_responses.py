#!/usr/bin/env python3
"""Harvest LLM responses from query result JSON files.

Reads query result files from query_results/, calls OpenAI-compatible LLM API
for each source, writes harvested responses to query_responses/ with matching
filenames.

Usage:
    python harvest_responses.py --limit 1
    python harvest_responses.py --limit 10 --overwrite
    python harvest_responses.py --prompt-dense prompt_dense.txt --prompt-semantic prompt_semantic.txt
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

# Import shared logging helpers (still keep logger for backward compat).
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
logger = logging.getLogger("harvest")


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


# ── Core pipeline ────────────────────────────────────────────────────────

# Map source names to system prompt filenames
SOURCE_PROMPT_MAP = {
    "papers": "prompt_dense.txt",
    "bedrock": "prompt_dense.txt",
    "papers_semantic": "prompt_semantic.txt",
}

# Fallback: if a source name is not in the map, use dense prompt
DEFAULT_PROMPT_FILE = "prompt_dense.txt"

# Module-level prompt texts (set by main() at startup)
prompt_dense: str = ""
prompt_semantic: str = ""


def _load_system_prompt(path: str) -> str:
    """Load system prompt from file, or raise."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"System prompt file not found: {path}")
    return p.read_text(encoding="utf-8")


def resolve_source_prompt(source_name: str) -> str:
    """Return the system prompt text for a given source name.

    Uses SOURCE_PROMPT_MAP to select which pre-loaded prompt to use.
    Falls back to the dense prompt if the source is not in the map.
    """
    prompt_file = SOURCE_PROMPT_MAP.get(source_name, DEFAULT_PROMPT_FILE)
    if prompt_file == "prompt_dense.txt":
        return prompt_dense
    elif prompt_file == "prompt_semantic.txt":
        return prompt_semantic
    return prompt_dense


def call_llm(client: OpenAI, system_prompt: str, source_payload: Any,
             prompt: str, model: str) -> Dict[str, Any]:
    """Call OpenAI-compatible chat completions endpoint."""
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


def process_file(
    fpath: Path,
    out_path: Path,
    model: str,
    client: OpenAI,
    overwrite: bool,
) -> bool:
    """Process one input file. Returns True on success."""
    log_cyan(f"▶ {fpath.name}")

    with open(fpath, "r", encoding="utf-8") as f:
        result = json.load(f)

    prompt = result["prompt"]
    sources = result.get("sources", {})

    # Print the prompt if we haven't seen it before
    prompt_trunc = truncate_line(prompt, shutil.get_terminal_size().columns - 10)
    if not seen_prompt(prompt):
        log_key(f"  [NEW] {prompt_trunc}")
    else:
        log_dim(f"  [SEEN] {prompt_trunc}")

    output: Dict[str, Any] = {
        "input_file": fpath.name,
        "input_id": result.get("id"),
        "category": result.get("category"),
        "proficiency": result.get("proficiency"),
        "topk": result.get("topk"),
        "prompt": prompt,
        "timestamp": result.get("timestamp"),
        "responses": {},
    }

    success = False
    for source_name, source_payload in sources.items():
        started_epoch = time.time()

        # Blue LLM announcement before each source call
        log_blue(f"  LLM calling: {source_name}...")

        try:
            system_prompt = resolve_source_prompt(source_name)
            response = call_llm(client, system_prompt, source_payload, prompt, model)
            elapsed = round(time.time() - started_epoch, 3)

            resp_text = extract_response_text(response)
            log_green(f"  ✓ {source_name}: {truncate_line(resp_text)} ({elapsed}s)")

            output["responses"][source_name] = {
                "response_text": resp_text,
                "model": model_name_from_response(response) or model,
                "started_at": datetime.fromtimestamp(started_epoch, tz=timezone.utc).isoformat(),
                "completed_at": datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "error": None,
            }
            success = True
        except Exception as exc:
            elapsed = 0.0
            log_red(f"  ✗ {source_name}: {type(exc).__name__}: {truncate_line(str(exc))}")
            logger.warning("  source %s FAILED: %s", source_name, exc)

            output["responses"][source_name] = {
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

    # Write output
    if not out_path.exists() or overwrite:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, sort_keys=True, indent=2)
        log_dim(f"  → {out_path.name}")
    else:
        log_dim(f"  ⏭ {out_path.name} (already exists)")

    return success


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Harvest LLM responses from query result JSON files."
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
        "--limit",
        type=int,
        default=None,
        help="Process only the first N input files (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing response files",
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

    # Validate input directory
    if not results_dir.exists():
        logger.error("Query results directory not found: %s", results_dir)
        sys.exit(1)
    if not results_dir.is_dir():
        logger.error("Query results path is not a directory: %s", results_dir)
        sys.exit(1)

    global prompt_dense, prompt_semantic
    prompt_dense = _load_system_prompt(args.prompt_dense)
    prompt_semantic = _load_system_prompt(args.prompt_semantic)

    # Create OpenAI client (OpenAI-compatible endpoint)
    client = OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
    )

    # Health check — dimmed, background
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

    # Process files
    success_count = 0
    fail_count = 0
    seen_count = 0

    for fpath in files:
        out_path = responses_dir / fpath.name
        try:
            ok = process_file(fpath, out_path, args.model, client, args.overwrite)
            if ok:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            log_red(f"Failed to process {fpath.name}: {e}")
            fail_count += 1

    log_key(f"\nDone. {success_count} succeeded, {fail_count} failed out of {len(files)} files.")
    log_dim(f"Responses in: {responses_dir}")


if __name__ == "__main__":
    main()