#!/usr/bin/env python3
"""CLI entry point for the agentic rabbit-hole eval harness.

Usage:
    python -m scripts.agentic_rabbit_hole_eval.run [args]
"""

import json
import logging
import os
import re as _re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_scripts_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from .loop import run_case
    from .persist import ensure_run_dir, write_config, write_case, append_failure, init_failures_file
    from .search import execute_search
except ImportError:
    from agentic_rabbit_hole_eval.loop import run_case
    from agentic_rabbit_hole_eval.persist import ensure_run_dir, write_config, write_case, append_failure, init_failures_file
    from agentic_rabbit_hole_eval.search import execute_search

# ── ANSI color helpers ───────────────────────────────────────────────────────

_ANSI_RESET  = "\033[0m"
_ANSI_CYAN   = "\033[36m"
_ANSI_YELLOW = "\033[33m"
_ANSI_GREEN  = "\033[32m"
_ANSI_RED    = "\033[31m"
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{code}{text}{_ANSI_RESET}"


_HTTP_REQ_RE = _re.compile(r"HTTP Request:\s+(GET|POST)\b")
_STATUS_OK_RE = _re.compile(r'"HTTP/[\d.]+ 2\d\d')


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not _USE_COLOR:
            return msg
        if record.levelno >= logging.ERROR:
            return f"{_ANSI_RED}{msg}{_ANSI_RESET}"
        raw = record.getMessage()
        m = _HTTP_REQ_RE.search(raw)
        if m:
            method = m.group(1)
            ok = bool(_STATUS_OK_RE.search(raw))
            if method == "GET":
                return f"{_ANSI_YELLOW}{msg}{_ANSI_RESET}"
            return f"{(_ANSI_GREEN if ok else _ANSI_RED)}{msg}{_ANSI_RESET}"
        return msg


_root_handler = logging.StreamHandler(sys.stdout)
_root_handler.setFormatter(_ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))
_root_logger = logging.getLogger()
_root_logger.handlers.clear()
_root_logger.addHandler(_root_handler)
_root_logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_prompts(prompts_file: str, num_prompts: int = None) -> list:
    with open(prompts_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            prompts = [str(p) for p in data]
        elif isinstance(data, dict) and "prompts" in data:
            prompts = [str(p) for p in data["prompts"]]
        else:
            prompts = [content]
    except json.JSONDecodeError:
        prompts = [line for line in content.splitlines() if line.strip()]
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    return prompts


def _sweep_grid(topk: int, sparse_step: int) -> list:
    """sparse_k values: 0, sparse_step, 2*sparse_step, ..., topk."""
    return list(range(0, topk + 1, sparse_step))


def _parse_args(argv: list) -> dict:
    import argparse
    parser = argparse.ArgumentParser(description="Agentic rabbit-hole eval harness")
    parser.add_argument("--qdrant-url",               type=str, required=True)
    parser.add_argument("--dense-collection",         type=str, required=True)
    parser.add_argument("--sparse-collection",        type=str, required=True)
    parser.add_argument("--sparse-collection-control",type=str, default=None)
    parser.add_argument("--dense-vector-name",        type=str, default="")
    parser.add_argument("--embed-url",                type=str, required=True)
    parser.add_argument("--model",                    type=str, required=True)
    parser.add_argument("--model-base-url",           type=str, required=True)
    parser.add_argument("--model-api-key",            type=str, default="not-set")
    parser.add_argument("--prompts-file",             type=str, required=True)
    parser.add_argument("--num-prompts",              type=int, default=None)
    parser.add_argument("--output-root",              type=str, required=True)
    parser.add_argument("--max-query-depth",          type=int, default=0,
                        help="Max tool calls per case (default 0 = no tool offered)")
    parser.add_argument("--topk",                     type=int, default=6)
    parser.add_argument("--sparse-step",              type=int, default=2)
    parser.add_argument("--temperature",              type=float, default=0.1)
    parser.add_argument("--timeout",                  type=float, default=180.0)
    parser.add_argument("--case-timeout",             type=float, default=600.0)
    parser.add_argument("--case-batch-size",          type=int, default=0)
    parser.add_argument("--system-prompt-file",       type=str, default=None)
    return vars(parser.parse_args(argv))


# ── Single case runner ────────────────────────────────────────────────────────

def _run_single_case(
    prompt_index: int,
    prompt_text: str,
    sparse_k: int,
    done: int,
    total_cases: int,
    config: dict,
    system_prompt: str,
    case_deadline: float,
    tag: str = "",
    sparse_collection: str = "",
    sparse_only: bool = False,
) -> dict:
    import openai

    topk    = config["topk"]
    dense_k = topk - sparse_k
    frac    = f"{sparse_k / topk:.2f}"
    tag_label = f" [{tag}]" if tag else ""
    prefix = _c(
        f"[{done}/{total_cases} sk={sparse_k}/{topk} frac={frac}{tag_label}]",
        _ANSI_CYAN,
    )

    # Print the query so you know what's running.
    q_preview = prompt_text[:100] + ("…" if len(prompt_text) > 100 else "")
    print(f"{prefix} query: {q_preview}")

    client = openai.OpenAI(
        api_key=config["model_api_key"],
        base_url=config["model_base_url"],
    )

    def search_fn(q, excluded_dense_ids=None, excluded_sparse_ids=None):
        return execute_search(
            q,
            embed_url=config["embed_url"],
            qdrant_url=config["qdrant_url"],
            dense_collection=config["dense_collection"],
            sparse_collection=sparse_collection,
            dense_vector_name=config.get("dense_vector_name", ""),
            dense_k=dense_k,
            sparse_k=sparse_k,
            sparse_only=sparse_only,
            excluded_dense_ids=excluded_dense_ids,
            excluded_sparse_ids=excluded_sparse_ids,
        )

    try:
        case_result = run_case(
            seed_prompt=prompt_text,
            prompt_index=prompt_index,
            tag=tag,
            client=client,
            model=config["model"],
            system_prompt=system_prompt,
            max_query_depth=config["max_query_depth"],
            execute_search_fn=search_fn,
            temperature=config["temperature"],
            timeout_seconds=config["timeout"],
            case_deadline=case_deadline,
        )

        case_result.prompt_index    = prompt_index
        case_result.prompt_text     = prompt_text
        case_result.sparse_collection = sparse_collection
        case_result.dense_collection  = config["dense_collection"]
        case_result.dense_k         = dense_k
        case_result.sparse_k        = sparse_k
        case_result.sparse_fraction = frac
        case_result.model           = config["model"]
        case_result.model_base_url  = config["model_base_url"]
        case_result.max_query_depth = config["max_query_depth"]
        case_result._tag            = tag

        path = write_case(case_result, config["_run_dir"], prompt_index, sparse_k=sparse_k, tag=tag)
        calls = case_result.tool_calls_made
        depth = config["max_query_depth"]
        ok_indicator = _c("ok", _ANSI_GREEN) if case_result.completed else _c("incomplete", _ANSI_RED)
        print(f"{prefix} {ok_indicator} tool_calls={calls}/{depth} | wrote {os.path.basename(path)}")

        return {"prompt_index": prompt_index, "sparse_k": sparse_k, "tag": tag,
                "ok": case_result.completed, "path": path, "error": ""}

    except Exception as e:
        logger.error("Case p=%d sk=%d [%s] failed: %s", prompt_index, sparse_k, tag, e)
        append_failure(config["_run_dir"], {
            "prompt_index": prompt_index, "sparse_k": sparse_k,
            "tag": tag, "error": str(e),
        })
        print(f"{prefix} {_c('FAIL', _ANSI_RED)}: {e}")
        return {"prompt_index": prompt_index, "sparse_k": sparse_k, "tag": tag,
                "ok": False, "path": "", "error": str(e)}


# ── Main ─────────────────────────────────────────────────────────────────────

def run(argv: list = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    config = _parse_args(argv)

    prompts = _load_prompts(config["prompts_file"], config["num_prompts"])
    if not prompts:
        logger.error("No prompts loaded from %s", config["prompts_file"])
        sys.exit(1)

    sys_prompt_file = config.get("system_prompt_file")
    if not sys_prompt_file:
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        default = os.path.join(pkg_dir, "system_prompt.txt")
        if os.path.exists(default):
            sys_prompt_file = default
        else:
            logger.error("No system prompt file. Use --system-prompt-file or create system_prompt.txt")
            sys.exit(1)
    with open(sys_prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config["output_root"], timestamp)
    ensure_run_dir(run_dir)
    config["_run_dir"] = run_dir

    sparse_k_values = _sweep_grid(config["topk"], config["sparse_step"])

    use_tags = config.get("sparse_collection_control") is not None
    collections = [("variable" if use_tags else "", config["sparse_collection"], True)]
    if config.get("sparse_collection_control"):
        collections.append(("control", config["sparse_collection_control"], False))

    write_config({
        **{k: v for k, v in config.items() if not k.startswith("_")},
        "sparse_k_values": sparse_k_values,
        "collection_count": len(collections),
        "prompt_count": len(prompts),
        "system_prompt": system_prompt,
    }, run_dir)
    init_failures_file(run_dir)

    cases = []
    for prompt_index, prompt_text in enumerate(prompts):
        for sparse_k in sparse_k_values:
            for tag, sparse_col, sparse_only in collections:
                cases.append((prompt_index, prompt_text, sparse_k, tag, sparse_col, sparse_only))

    total_cases = len(cases)
    case_batch_size = config.get("case_batch_size", 0)
    all_results = []

    def _submit(idx, pidx, ptxt, sk, tag, sp_col, sp_only):
        return _run_single_case(
            prompt_index=pidx,
            prompt_text=ptxt,
            sparse_k=sk,
            done=idx + 1,
            total_cases=total_cases,
            config=config,
            system_prompt=system_prompt,
            case_deadline=time.monotonic() + config["case_timeout"],
            tag=tag,
            sparse_collection=sp_col,
            sparse_only=sp_only,
        )

    if case_batch_size > 0 and total_cases > 1:
        with ThreadPoolExecutor(max_workers=min(case_batch_size, total_cases)) as ex:
            futures = [ex.submit(_submit, i, *c) for i, c in enumerate(cases)]
            for f in futures:
                all_results.append(f.result())
    else:
        for i, c in enumerate(cases):
            all_results.append(_submit(i, *c))

    ok = sum(1 for r in all_results if r["ok"])
    fail = total_cases - ok
    print(
        f"\nRun complete. {len(prompts)} prompts × {len(sparse_k_values)} sparse fractions "
        f"× {len(collections)} collection(s) = {total_cases} cases. "
        f"{ok} completed, {fail} failed."
    )


if __name__ == "__main__":
    run()
