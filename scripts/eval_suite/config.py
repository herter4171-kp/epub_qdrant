"""Config dataclass + env+CLI merge."""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from .schemas import ConfigSnapshot

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", ".env")
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH)


@dataclass(frozen=True)
class ResolvedConfig:
    topk: int
    sparse_step: int
    prompts_file: str
    system_prompt_file: str
    num_prompts: Optional[int]
    dense_collection: str
    sparse_collection: str
    dense_vector_name: str  # "" means unnamed single-vector dense collection
    qdrant_url: str
    embed_url: str
    judge_base_url: str
    judge_model: str
    judge_api_key: str
    output_root: str
    judge_timeout_seconds: float
    judge_per_chunk_timeout_seconds: float
    judge_max_tokens: int
    judge_attempts: int
    judges_per_case: int
    case_timeout_seconds: float
    turbo_submit: int  # batch size for parallel judge submissions (0 = serial)

    def to_snapshot(self) -> ConfigSnapshot:
        return ConfigSnapshot(
            topk=self.topk,
            sparse_step=self.sparse_step,
            prompts_file=self.prompts_file,
            system_prompt_file=self.system_prompt_file,
            num_prompts=self.num_prompts,
            dense_collection=self.dense_collection,
            sparse_collection=self.sparse_collection,
            dense_vector_name=self.dense_vector_name,
            qdrant_url=self.qdrant_url,
            embed_url=self.embed_url,
            judge_base_url=self.judge_base_url,
            judge_model=self.judge_model,
            judge_api_key=self.judge_api_key,
            output_root=self.output_root,
            judge_timeout_seconds=self.judge_timeout_seconds,
            judge_per_chunk_timeout_seconds=self.judge_per_chunk_timeout_seconds,
            judge_max_tokens=self.judge_max_tokens,
            judge_attempts=self.judge_attempts,
            judges_per_case=self.judges_per_case,
            case_timeout_seconds=self.case_timeout_seconds,
            turbo_submit=self.turbo_submit,
        )


def _resolve_env(key: str, env_vars: list[str], default: Optional[str] = None) -> Optional[str]:
    for ek in env_vars:
        v = os.environ.get(ek)
        if v is not None:
            return v
    return default


def resolve_config(argv: list, env: dict = None) -> ResolvedConfig:
    """Parse CLI args + env vars, return ResolvedConfig."""
    parser = argparse.ArgumentParser(description="Retrieval eval suite")
    parser.add_argument("--topk", type=int, required=False)
    parser.add_argument("--sparse-step", type=int, required=False)
    parser.add_argument("--prompts-file", type=str, required=False)
    parser.add_argument("--system-prompt-file", type=str, required=False)
    parser.add_argument("--num-prompts", type=int, required=False, default=None)
    parser.add_argument("--dense-collection", type=str, required=False,
                        help="Qdrant collection holding dense vectors.")
    parser.add_argument("--sparse-collection", type=str, required=False,
                        help="Qdrant collection holding sparse vectors with dense_chunk_ids payload.")
    parser.add_argument("--dense-vector-name", type=str, required=False, default=None,
                        help='Named-vector key for dense (default "dense"). Pass "" for unnamed single-vector collection.')
    parser.add_argument("--qdrant-url", type=str, required=False)
    parser.add_argument("--embed-url", type=str, required=False)
    parser.add_argument("--judge-base-url", type=str, required=False)
    parser.add_argument("--judge-model", type=str, required=False)
    parser.add_argument("--judge-api-key", type=str, required=False)
    parser.add_argument("--output-root", type=str, required=False)
    parser.add_argument("--judge-timeout-seconds", type=float, required=False, default=None,
                        help="Total wall-clock cap per judge response (default 180).")
    parser.add_argument("--judge-per-chunk-timeout-seconds", type=float, required=False, default=None,
                        help="Inactivity cap between SSE chunks before abort (default 30).")
    parser.add_argument("--judge-max-tokens", type=int, required=False, default=None,
                        help="Server-side cap on tokens emitted per judge response (default 2048).")
    parser.add_argument("--judge-attempts", type=int, required=False, default=None,
                        help="Total attempts per judgement (single flat retry loop covering "
                             "timeout/loop/transport/parse failures). Default 3.")
    parser.add_argument("--judges-per-case", type=int, required=False, default=None,
                        help="Number of independent judge runs per (prompt, sparse_k) case for noise characterization. Default 1.")
    parser.add_argument("--case-timeout-seconds", type=float, required=False, default=None,
                        help="Hard wall-clock cap per case across all judges + retries. "
                             "When exceeded, remaining judge slots are filled with "
                             "case_wallclock_exceeded errors. Default 600.")
    parser.add_argument("--turbo-submit", type=int, required=False, default=0,
                        help="Batch size for parallel judge LLM submissions. When > 0, "
                             "judges are submitted in batches of this size using asyncio. "
                             "Default 0 (serial).")
    args = parser.parse_args(argv)

    topk = args.topk or int(os.environ.get("EVAL_TOPK", 0))
    if not topk:
        parser.error("--topk or EVAL_TOPK is required")
    if topk < 2 or topk % 2 != 0:
        parser.error(f"--topk must be even and >= 2, got {topk}")

    sparse_step = args.sparse_step or int(os.environ.get("EVAL_SPARSE_STEP", "2"))
    if sparse_step < 1 or sparse_step > topk:
        parser.error(f"--sparse-step must be 1..topk, got {sparse_step}")

    prompts_file = args.prompts_file or os.environ.get("EVAL_PROMPTS_FILE", "")
    if not prompts_file:
        parser.error("--prompts-file or EVAL_PROMPTS_FILE is required")

    system_prompt_file = args.system_prompt_file or os.environ.get("EVAL_SYSTEM_PROMPT_FILE", "")
    if not system_prompt_file:
        parser.error("--system-prompt-file or EVAL_SYSTEM_PROMPT_FILE is required")

    num_prompts = args.num_prompts
    if num_prompts is None:
        num_prompts_env = os.environ.get("EVAL_NUM_PROMPTS")
        if num_prompts_env:
            num_prompts = int(num_prompts_env)

    dense_collection = args.dense_collection or os.environ.get("DENSE_COLLECTION", "")
    if not dense_collection:
        parser.error("--dense-collection or DENSE_COLLECTION is required")

    sparse_collection = args.sparse_collection or os.environ.get("SPARSE_COLLECTION", "")
    if not sparse_collection:
        parser.error("--sparse-collection or SPARSE_COLLECTION is required")

    if args.dense_vector_name is not None:
        dense_vector_name = args.dense_vector_name
    else:
        dense_vector_name = os.environ.get("DENSE_VECTOR_NAME", "dense")

    qdrant_url = args.qdrant_url or os.environ.get("QDRANT_URL", "http://192.168.68.75:6333")
    embed_url = args.embed_url or os.environ.get("EMBED_URL") or os.environ.get("EMBEDDING_SERVER_URL", "http://192.168.68.75:8100")

    judge_base_url = args.judge_base_url
    if not judge_base_url:
        judge_base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if not judge_base_url:
        parser.error("--judge-base-url or LLM_BASE_URL or OPENAI_API_BASE is required")

    judge_model = args.judge_model or os.environ.get("LLM_MODEL") or os.environ.get("LITELLM_MODEL", "qwen36")

    judge_api_key = args.judge_api_key or os.environ.get("LLM_API_KEY") or os.environ.get("LITELLM_API_KEY", "")

    output_root = args.output_root or os.environ.get("EVAL_OUTPUT_ROOT", "./eval_results")

    def _resolve_float(arg, env_key, default):
        if arg is not None:
            return float(arg)
        env_val = os.environ.get(env_key)
        return float(env_val) if env_val else default

    def _resolve_int(arg, env_key, default):
        if arg is not None:
            return int(arg)
        env_val = os.environ.get(env_key)
        return int(env_val) if env_val else default

    judge_timeout_seconds = _resolve_float(
        args.judge_timeout_seconds, "JUDGE_TIMEOUT_SECONDS", 180.0,
    )
    judge_per_chunk_timeout_seconds = _resolve_float(
        args.judge_per_chunk_timeout_seconds, "JUDGE_PER_CHUNK_TIMEOUT_SECONDS", 30.0,
    )
    judge_max_tokens = _resolve_int(
        args.judge_max_tokens, "JUDGE_MAX_TOKENS", 2048,
    )
    judge_attempts = _resolve_int(
        args.judge_attempts, "JUDGE_ATTEMPTS", 3,
    )
    judges_per_case = _resolve_int(
        args.judges_per_case, "JUDGES_PER_CASE", 1,
    )
    case_timeout_seconds = _resolve_float(
        args.case_timeout_seconds, "CASE_TIMEOUT_SECONDS", 600.0,
    )

    turbo_submit = args.turbo_submit if args.turbo_submit is not None else 0
    if turbo_submit < 0:
        parser.error(f"--turbo-submit must be >= 0, got {turbo_submit}")

    if judge_timeout_seconds <= 0:
        parser.error(f"--judge-timeout-seconds must be > 0, got {judge_timeout_seconds}")
    if judge_per_chunk_timeout_seconds <= 0:
        parser.error(f"--judge-per-chunk-timeout-seconds must be > 0, got {judge_per_chunk_timeout_seconds}")
    if judge_max_tokens <= 0:
        parser.error(f"--judge-max-tokens must be > 0, got {judge_max_tokens}")
    if judge_attempts < 1:
        parser.error(f"--judge-attempts must be >= 1, got {judge_attempts}")
    if judges_per_case < 1:
        parser.error(f"--judges-per-case must be >= 1, got {judges_per_case}")
    if case_timeout_seconds <= 0:
        parser.error(f"--case-timeout-seconds must be > 0, got {case_timeout_seconds}")

    if num_prompts is not None and num_prompts < 1:
        parser.error(f"--num-prompts must be >= 1, got {num_prompts}")

    return ResolvedConfig(
        topk=topk,
        sparse_step=sparse_step,
        prompts_file=prompts_file,
        system_prompt_file=system_prompt_file,
        num_prompts=num_prompts,
        dense_collection=dense_collection,
        sparse_collection=sparse_collection,
        dense_vector_name=dense_vector_name,
        qdrant_url=qdrant_url,
        embed_url=embed_url,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        judge_api_key=judge_api_key,
        output_root=output_root,
        judge_timeout_seconds=judge_timeout_seconds,
        judge_per_chunk_timeout_seconds=judge_per_chunk_timeout_seconds,
        judge_max_tokens=judge_max_tokens,
        judge_attempts=judge_attempts,
        judges_per_case=judges_per_case,
        case_timeout_seconds=case_timeout_seconds,
        turbo_submit=turbo_submit,
    )
