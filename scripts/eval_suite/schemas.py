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
    id: Any  # dense point id, or "+"-joined composite for combined sparse_resolved
    source: str  # "dense" or "sparse_resolved" (first retriever; see sources for all)
    token_count: int
    text: str
    title: str = ""
    sources: List[str] = field(default_factory=list)  # all retrievers that found this chunk                            # publication title from payload
    docket_id: str = ""                        # opaque per-result token shown to judge
    constituent_ids: List[Any] = field(default_factory=list)  # dense ids combined into this chunk
    originating_sparse_id: Any = None          # sparse point id that resolved here (None for direct dense)


@dataclass
class RetrievalSet:
    prompt_index: int
    prompt_text: str
    topk: int
    sparse_k: int
    dense_k: int
    sparse_fraction: str  # e.g. "0.33"
    dense_collection: str
    sparse_collection: str
    dense_vector_name: str  # "" means unnamed single-vector dense collection
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
    docket_id: str = ""     # opaque per-result token shown to judge


@dataclass
class CritiqueParsedChunk:
    rank: int
    id: str                 # docket_id echoed back by judge
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
    dense_collection: str = ""
    sparse_collection: str = ""
    dense_vector_name: str = ""
    embed_model_endpoint: str = ""
    judge_model: str = ""
    judge_base_url: str = ""
    timestamp_utc: str = ""
    chunks: List[CritiqueChunk] = field(default_factory=list)
    judge_outputs: List[JudgeOutput] = field(default_factory=list)

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
            "dense_collection": self.dense_collection,
            "sparse_collection": self.sparse_collection,
            "dense_vector_name": self.dense_vector_name,
            "embed_model_endpoint": self.embed_model_endpoint,
            "judge_model": self.judge_model,
            "judge_base_url": self.judge_base_url,
            "timestamp_utc": self.timestamp_utc,
        }
        d["chunks"] = [asdict(c) for c in self.chunks]
        d["judge_outputs"] = [
            {
                "raw": jo.raw,
                "parsed": jo.parsed,
                "parse_ok": jo.parse_ok,
                "retried": jo.retried,
                "error": jo.error,
            }
            for jo in self.judge_outputs
        ]
        return d


@dataclass
class ConfigSnapshot:
    topk: int
    sparse_step: int
    prompts_file: str
    system_prompt_file: str
    num_prompts: Optional[int]
    dense_collection: str
    sparse_collection: str
    dense_vector_name: str  # "" = unnamed single-vector dense collection
    qdrant_url: str
    embed_url: str
    judge_base_url: str
    judge_model: str
    judge_api_key: Optional[str]
    output_root: str
    judge_timeout_seconds: float = 180.0
    judge_per_chunk_timeout_seconds: float = 30.0
    judge_max_tokens: int = 2048
    judge_attempts: int = 3
    judges_per_case: int = 1
    case_timeout_seconds: float = 600.0
    turbo_submit: int = 0  # batch size for parallel judge submissions (0 = serial)
    sparse_only: bool = True  # use /embed_sae (True) vs /embed_sparse (False)
    prompts: List[str] = field(default_factory=list)
    system_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Redact non-empty API keys
        if d.get("judge_api_key") and d["judge_api_key"] != "":
            d["judge_api_key"] = "***"
        d["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return d