"""Dataclasses for the agentic rabbit-hole eval harness."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallTurn:
    """One turn where the model called search_corpus."""
    depth: int  # 1-based call index
    query: str  # what the model passed to search_corpus
    chunks: List[dict]  # raw results from execute_search


@dataclass
class FinalAnswerTurn:
    """The model's final text response."""
    reply: str


@dataclass
class CaseResult:
    """Result of running one agentic case (one seed prompt)."""
    schema_version: int = 2
    prompt_index: int = 0
    prompt_text: str = ""
    sparse_collection: str = ""
    dense_collection: str = ""
    dense_k: int = 0
    sparse_k: int = 0
    model: str = ""
    model_base_url: str = ""
    max_query_depth: int = 0
    sparse_fraction: str = ""
    tool_calls_made: int = 0
    completed: bool = False
    timestamp_utc: str = ""
    _tag: str = ""
    turns: List[dict] = field(default_factory=list)  # serialized ToolCallTurn | FinalAnswerTurn

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "prompt_index": self.prompt_index,
            "prompt_text": self.prompt_text,
            "sparse_collection": self.sparse_collection,
            "dense_collection": self.dense_collection,
            "dense_k": self.dense_k,
            "sparse_k": self.sparse_k,
            "model": self.model,
            "model_base_url": self.model_base_url,
            "max_query_depth": self.max_query_depth,
            "sparse_fraction": self.sparse_fraction,
            "tool_calls_made": self.tool_calls_made,
            "completed": self.completed,
            "timestamp_utc": self.timestamp_utc,
            "_tag": self._tag,
            "turns": self.turns,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseResult":
        """Deserialize from dict."""
        obj = cls()
        obj.schema_version = data.get("schema_version", 2)
        obj.prompt_index = data.get("prompt_index", 0)
        obj.prompt_text = data.get("prompt_text", "")
        obj.sparse_collection = data.get("sparse_collection", "")
        obj.dense_collection = data.get("dense_collection", "")
        obj.dense_k = data.get("dense_k", 0)
        obj.sparse_k = data.get("sparse_k", 0)
        obj.model = data.get("model", "")
        obj.model_base_url = data.get("model_base_url", "")
        obj.max_query_depth = data.get("max_query_depth", 0)
        obj.tool_calls_made = data.get("tool_calls_made", 0)
        obj.completed = data.get("completed", False)
        obj.timestamp_utc = data.get("timestamp_utc", "")
        obj._tag = data.get("_tag", "")
        obj.turns = data.get("turns", [])
        return obj

    def write(self, run_dir: str, prompt_index: int, tag: str = "") -> str:
        """Write case JSON to run_dir/cases/prompt_XXX[.json] or prompt_XXX_tag.json."""
        os.makedirs(os.path.join(run_dir, "cases"), exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(run_dir, "cases", f"prompt_{prompt_index:03d}{suffix}.json")
        _write_json_atomic(path, self.to_dict())
        return path


def _write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    """Write JSON atomically (tmp + rename). Pretty-printed."""
    import tempfile
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