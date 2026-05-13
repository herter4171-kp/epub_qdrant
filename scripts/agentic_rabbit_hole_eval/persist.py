"""Persistence for the agentic rabbit-hole eval harness.

Output structure:
    {run_dir}/
        config.json           Standard config snapshot
        failures.jsonl        One line per failed case
        cases/
            prompt_001.json   One CaseResult per prompt
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    """Write JSON atomically (tmp + rename). Pretty-printed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.rename(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def ensure_run_dir(run_dir: str) -> None:
    """Create run directory structure."""
    os.makedirs(os.path.join(run_dir, "cases"), exist_ok=True)


def write_config(snapshot: Dict[str, Any], run_dir: str) -> str:
    """Write config.json, return path."""
    path = os.path.join(run_dir, "config.json")
    _write_json_atomic(path, snapshot)
    return path


def write_case(case_result: Any, run_dir: str, prompt_index: int,
               sparse_k: int = 0, tag: str = "") -> str:
    """Write case JSON, return path."""
    tag_part = f"_{tag}" if tag else ""
    fname = f"prompt_{prompt_index:03d}_sk_{sparse_k}{tag_part}.json"
    path = os.path.join(run_dir, "cases", fname)
    _write_json_atomic(path, case_result.to_dict())
    return path


def append_failure(run_dir: str, failure: Dict[str, Any]) -> None:
    """Append one line to failures.jsonl."""
    path = os.path.join(run_dir, "failures.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(failure) + "\n")


def init_failures_file(run_dir: str) -> None:
    """Create empty failures.jsonl."""
    path = os.path.join(run_dir, "failures.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        pass  # empty