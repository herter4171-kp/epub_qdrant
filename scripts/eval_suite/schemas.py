"""Typed dataclasses for all artifacts."""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


@dataclass
class Prompt:
    index: int  # 1-based
    text: str


@dataclass
class MergedChunk:
    rank: int
    id: Any  # qdrant point id — int/uuid
    source: str  # "dense" or "sparse"
    token_count: int
    text: str
    title: str = ""        # publication title from payload
    judge_id: str = ""     # ephemeral random hash shown to judge


@dataclass
class RetrievalSet:
    prompt_index: int
    prompt_text: str
    topk: int
    sparse_k: int
    dense_k: int
    sparse_fraction: str  # e.g. "0.33"
    collection: str
    timestamp_utc: str
    dense_raw: List[Dict[str, Any]]
    sparse_raw: List[Dict[str, Any]]
    merged: List[MergedChunk]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # MergedChunk is already serializable via asdict
        return d


@dataclass
class CritiqueChunk:
    rank: int
    id: Any                 # qdrant point id
    source: str
    token_count: int
    text: str
    title: str = ""
    judge_id: str = ""      # ephemeral hash shown to judge


@dataclass
class CritiqueParsedChunk:
    rank: int
    id: str                 # judge-side hash
    relevance: int  # 1-10
    reason: str


@dataclass
class JudgeOutput:
    raw: str
    parsed: Optional[Dict[str, Any]]
    parse_ok: bool
    retried: bool
    error: Optional[str]


@dataclass
class Critique:
    schema_version: int = 1
    run_name: str = ""
    prompt_index: int = 0
    prompt_text: str = ""
    system_prompt_text: str = ""
    topk: int = 0
    sparse_k: int = 0
    dense_k: int = 0
    sparse_fraction: str = ""
    collection: str = ""
    embed_model_endpoint: str = ""
    judge_model: str = ""
    judge_base_url: str = ""
    timestamp_utc: str = ""
    chunks: List[CritiqueChunk] = field(default_factory=list)
    judge_output: Optional[JudgeOutput] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "prompt_index": self.prompt_index,
            "prompt_text": self.prompt_text,
            "system_prompt_text": self.system_prompt_text,
            "topk": self.topk,
            "sparse_k": self.sparse_k,
            "dense_k": self.dense_k,
            "sparse_fraction": self.sparse_fraction,
            "collection": self.collection,
            "embed_model_endpoint": self.embed_model_endpoint,
            "judge_model": self.judge_model,
            "judge_base_url": self.judge_base_url,
            "timestamp_utc": self.timestamp_utc,
        }
        d["chunks"] = [asdict(c) for c in self.chunks]
        if self.judge_output:
            d["judge_output"] = {
                "raw": self.judge_output.raw,
                "parsed": self.judge_output.parsed,
                "parse_ok": self.judge_output.parse_ok,
                "retried": self.judge_output.retried,
                "error": self.judge_output.error,
            }
        return d


@dataclass
class ConfigSnapshot:
    topk: int
    sparse_step: int
    prompts_file: str
    system_prompt_file: str
    num_prompts: Optional[int]
    collection: str
    qdrant_url: str
    embed_url: str
    judge_base_url: str
    judge_model: str
    judge_api_key: Optional[str]
    output_root: str
    prompts: List[str] = field(default_factory=list)
    system_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Redact non-empty API keys
        if d.get("judge_api_key") and d["judge_api_key"] != "":
            d["judge_api_key"] = "***"
        d["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return d