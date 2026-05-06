"""Persist retrieval/critique JSON + read helpers."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Iterator, Optional

from .schemas import Critique, RetrievalSet, ConfigSnapshot

_RETRIEVALS_DIR = "retrievals"
_CRITIQUES_DIR = "critiques"


def _slug_prompt(index: int) -> str:
    return f"prompt_{index:03d}"


def _slug_case(index: int, sparse_k: int) -> str:
    return f"{_slug_prompt(index)}_sk_{sparse_k}"


def _retrieval_path(prompt_index: int, sparse_k: int, run_dir: str) -> str:
    slug = _slug_case(prompt_index, sparse_k)
    return os.path.join(run_dir, _RETRIEVALS_DIR, f"{slug}.json")


def _critique_path(prompt_index: int, sparse_k: int, run_dir: str) -> str:
    slug = _slug_case(prompt_index, sparse_k)
    return os.path.join(run_dir, _CRITIQUES_DIR, f"{slug}.json")


def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    """Write JSON atomically (tmp + rename). Pretty-printed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.rename(tmp, path)
    except:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def ensure_run_dir(run_dir: str) -> None:
    os.makedirs(os.path.join(run_dir, _RETRIEVALS_DIR), exist_ok=True)
    os.makedirs(os.path.join(run_dir, _CRITIQUES_DIR), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "report_assets"), exist_ok=True)


def write_retrieval(retrieval: RetrievalSet, run_dir: str) -> str:
    """Write retrieval JSON, return path."""
    path = _retrieval_path(retrieval.prompt_index, retrieval.sparse_k, run_dir)
    write_json_atomic(path, retrieval.to_dict())
    return path


def write_critique(critique: Critique, run_dir: str) -> str:
    """Write critique JSON, return path. One file per (prompt_index, sparse_k);
    multi-sample runs hold all judge_outputs in the file's list."""
    path = _critique_path(critique.prompt_index, critique.sparse_k, run_dir)
    write_json_atomic(path, critique.to_dict())
    return path


def write_config(snapshot: ConfigSnapshot, run_dir: str) -> str:
    """Write config.json, return path."""
    path = os.path.join(run_dir, "config.json")
    write_json_atomic(path, snapshot.to_dict())
    return path


def read_critique(
    run_dir: str, prompt_index: int, sparse_k: int,
) -> Optional[Dict[str, Any]]:
    """Read a critique JSON by prompt_index + sparse_k."""
    path = _critique_path(prompt_index, sparse_k, run_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_retrieval(run_dir: str, prompt_index: int, sparse_k: int) -> Optional[Dict[str, Any]]:
    """Read a retrieval JSON by prompt_index + sparse_k."""
    path = _retrieval_path(prompt_index, sparse_k, run_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_critiques(run_dir: str) -> Iterator[Dict[str, Any]]:
    """Yield all critique dicts in critiques/."""
    critiques_dir = os.path.join(run_dir, _CRITIQUES_DIR)
    if not os.path.isdir(critiques_dir):
        return
    for fname in sorted(os.listdir(critiques_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(critiques_dir, fname), "r", encoding="utf-8") as f:
                yield json.load(f)


def iter_retrievals(run_dir: str) -> Iterator[Dict[str, Any]]:
    """Yield all retrieval dicts in retrievals/."""
    retrievals_dir = os.path.join(run_dir, _RETRIEVALS_DIR)
    if not os.path.isdir(retrievals_dir):
        return
    for fname in sorted(os.listdir(retrievals_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(retrievals_dir, fname), "r", encoding="utf-8") as f:
                yield json.load(f)


def read_config(run_dir: str) -> Optional[Dict[str, Any]]:
    """Read config.json from run_dir."""
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
