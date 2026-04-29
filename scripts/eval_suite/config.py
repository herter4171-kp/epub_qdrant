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
    collection: str
    qdrant_url: str
    embed_url: str
    judge_base_url: str
    judge_model: str
    judge_api_key: str
    output_root: str

    def to_snapshot(self) -> ConfigSnapshot:
        return ConfigSnapshot(
            topk=self.topk,
            sparse_step=self.sparse_step,
            prompts_file=self.prompts_file,
            system_prompt_file=self.system_prompt_file,
            num_prompts=self.num_prompts,
            collection=self.collection,
            qdrant_url=self.qdrant_url,
            embed_url=self.embed_url,
            judge_base_url=self.judge_base_url,
            judge_model=self.judge_model,
            judge_api_key=self.judge_api_key,
            output_root=self.output_root,
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
    parser.add_argument("--collection", type=str, required=False)
    parser.add_argument("--qdrant-url", type=str, required=False)
    parser.add_argument("--embed-url", type=str, required=False)
    parser.add_argument("--judge-base-url", type=str, required=False)
    parser.add_argument("--judge-model", type=str, required=False)
    parser.add_argument("--judge-api-key", type=str, required=False)
    parser.add_argument("--output-root", type=str, required=False)
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

    collection = args.collection or os.environ.get("COLLECTION", "fuck-qwen36")

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

    if num_prompts is not None and num_prompts < 1:
        parser.error(f"--num-prompts must be >= 1, got {num_prompts}")

    return ResolvedConfig(
        topk=topk,
        sparse_step=sparse_step,
        prompts_file=prompts_file,
        system_prompt_file=system_prompt_file,
        num_prompts=num_prompts,
        collection=collection,
        qdrant_url=qdrant_url,
        embed_url=embed_url,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        judge_api_key=judge_api_key,
        output_root=output_root,
    )