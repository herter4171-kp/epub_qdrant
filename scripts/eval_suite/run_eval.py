#!/usr/bin/env python3
"""Top-level orchestrator: CLI, sweep, per-config logging.

Usage:
    python scripts/eval_suite/run_eval.py --topk 6 --sparse-step 2 ...
"""

import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Ensure parent dir is importable so relative imports work when run
# as a module: python -m scripts.eval_suite.run_eval
# Also handle direct invocation: python scripts/eval_suite/run_eval.py
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_scripts_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from .config import resolve_config
    from .prompts import load_prompts
    from .embed import dense_embed, sparse_embed, sparse_embed_sae
    from .retrieve import retrieve
    from .critique import critique as critique_fn
    from .persist import (
        ensure_run_dir,
        write_retrieval,
        write_critique,
        write_config,
    )
    from .report import build_report, render_pdf
    from .schemas import Prompt
except ImportError:
    from eval_suite.config import resolve_config
    from eval_suite.prompts import load_prompts
    from eval_suite.embed import dense_embed, sparse_embed, sparse_embed_sae
    from eval_suite.retrieve import retrieve
    from eval_suite.critique import critique as critique_fn
    from eval_suite.persist import (
        ensure_run_dir,
        write_retrieval,
        write_critique,
        write_config,
    )
    from eval_suite.report import build_report, render_pdf
    from eval_suite.schemas import Prompt

# ── ANSI color helpers ───────────────────────────────────────────────────────

_ANSI_RESET = "\033[0m"
_ANSI_CYAN = "\033[36m"
_ANSI_YELLOW = "\033[33m"
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{code}{text}{_ANSI_RESET}"


import re as _re

_HTTP_REQ_RE = _re.compile(r"HTTP Request:\s+(GET|POST)\b")
_STATUS_OK_RE = _re.compile(r'"HTTP/[\d.]+ 2\d\d')


class _ColorFormatter(logging.Formatter):
    """Color HTTP traffic and error-level records on stdout."""

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
            # POST
            return f"{_ANSI_GREEN if ok else _ANSI_RED}{msg}{_ANSI_RESET}"
        return msg


# Logging — single stdout handler with color formatter.
_root_handler = logging.StreamHandler(sys.stdout)
_root_handler.setFormatter(_ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))
_root_logger = logging.getLogger()
_root_logger.handlers.clear()
_root_logger.addHandler(_root_handler)
_root_logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class _CaseResult:
    """Result of running one (prompt, sparse_k) case."""
    prompt_index: int
    prompt_text: str
    sparse_k: int
    sparse_frac: str
    done: int
    total_configs: int
    merged: int
    ok_n: int
    total_judges: int
    retrieval_path: str
    critique_path: str
    error: str = ""


def _run_single_case(
    prompt,
    sparse_k,
    sparse_frac,
    dense_k,
    embeddings,
    config,
    sparse_collection,
    tag,
    system_prompt,
    judges_per_case,
    done,
    total_configs,
    failures_path,
) -> _CaseResult:
    """Run one (prompt, sparse_k, collection) case: retrieve + critique."""
    tag_label = f" [{tag}]" if tag else ""
    prefix = _c(
        f"[{done}/{total_configs} sk={sparse_k}/{config.topk} frac={sparse_frac}{tag_label}]",
        _ANSI_CYAN,
    )
    _RETRIEVAL_ATTEMPTS = 2
    _RETRIEVAL_RETRY_DELAY = 1.0  # seconds

    status_parts: list = []
    retrieval_path = ""

    last_retrieval_exc: Exception | None = None
    for _attempt in range(_RETRIEVAL_ATTEMPTS):
        try:
            retrieval = retrieve(
                prompt=prompt,
                embeddings=embeddings,
                dense_k=dense_k,
                sparse_k=sparse_k,
                qdrant_url=config.qdrant_url,
                dense_collection=config.dense_collection,
                sparse_collection=sparse_collection,
                dense_vector_name=config.dense_vector_name,
                topk=config.topk,
            )
            last_retrieval_exc = None
            break
        except Exception as e:
            last_retrieval_exc = e
            if _attempt < _RETRIEVAL_ATTEMPTS - 1:
                logger.warning(
                    "Retrieval attempt %d/%d failed for prompt %d sk=%d: %s — retrying in %.1fs",
                    _attempt + 1, _RETRIEVAL_ATTEMPTS, prompt.index, sparse_k, e, _RETRIEVAL_RETRY_DELAY,
                )
                time.sleep(_RETRIEVAL_RETRY_DELAY)

    if last_retrieval_exc is not None:
        e = last_retrieval_exc
        logger.error("Retrieval failed for prompt %d sk=%d: %s", prompt.index, sparse_k, e)
        failures = []
        failures.append({
            "prompt_index": prompt.index,
            "prompt_text": prompt.text,
            "sparse_k": sparse_k,
            "sparse_fraction": sparse_frac,
            "error": str(e),
        })
        with open(failures_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt_index": prompt.index, "sparse_k": sparse_k,
                                "error": str(e)}) + "\n")
        status_parts.append(_c("RETRIEVAL FAIL", _ANSI_RED))
        print(f"{prefix} " + " | ".join(status_parts))
        return _CaseResult(
            prompt_index=prompt.index,
            prompt_text=prompt.text,
            sparse_k=sparse_k,
            sparse_frac=sparse_frac,
            done=done,
            total_configs=total_configs,
            merged=0,
            ok_n=0,
            total_judges=0,
            retrieval_path=retrieval_path,
            critique_path="",
            error=str(e),
        )

    # One critique call runs judges_per_case judgements internally.
    try:
        c = critique_fn(
            retrieval_set=retrieval,
            system_prompt=system_prompt,
            judge_base_url=config.judge_base_url,
            judge_model=config.judge_model,
            judge_api_key=config.judge_api_key,
            judge_temperature=config.judge_temperature,
            judge_timeout_seconds=config.judge_timeout_seconds,
            judge_per_chunk_timeout_seconds=config.judge_per_chunk_timeout_seconds,
            judge_max_tokens=config.judge_max_tokens,
            judge_attempts=config.judge_attempts,
            judges_per_case=judges_per_case,
            case_timeout_seconds=config.case_timeout_seconds,
            turbo_submit=config.turbo_submit,
        )
        critique_path = write_critique(c, config._run_dir, tag=tag)

        outs = c.judge_outputs or []
        ok_n = sum(1 for jo in outs if jo and jo.parse_ok)
        bad_n = len(outs) - ok_n
        if ok_n == 0:
            first_err = next((jo.error for jo in outs if jo and jo.error), "no parse")
            status_parts.append(_c(f"judges 0/{len(outs)} (last: {first_err})", _ANSI_RED))
        elif bad_n > 0:
            status_parts.append(_c(f"judges {ok_n}/{len(outs)}", _ANSI_YELLOW))
        else:
            status_parts.append(_c(f"judges {ok_n}/{len(outs)} ok", _ANSI_GREEN))

        status_parts.append(f"wrote {os.path.basename(critique_path)}")
        print(f"{prefix} " + " | ".join(status_parts))

        return _CaseResult(
            prompt_index=prompt.index,
            prompt_text=prompt.text,
            sparse_k=sparse_k,
            sparse_frac=sparse_frac,
            done=done,
            total_configs=total_configs,
            merged=len(retrieval.merged),
            ok_n=ok_n,
            total_judges=len(outs),
            retrieval_path=retrieval_path,
            critique_path=critique_path,
        )

    except Exception as e:
        logger.error("Critique failed for prompt %d sk=%d: %s",
                     prompt.index, sparse_k, e)
        failures = []
        failures.append({
            "prompt_index": prompt.index,
            "prompt_text": prompt.text,
            "sparse_k": sparse_k,
            "sparse_fraction": sparse_frac,
            "error": str(e),
        })
        with open(failures_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt_index": prompt.index, "sparse_k": sparse_k,
                                "error": str(e)}) + "\n")
        status_parts.append(_c(f"CRITIQUE FAIL: {e}", _ANSI_RED))
        print(f"{prefix} " + " | ".join(status_parts))
        return _CaseResult(
            prompt_index=prompt.index,
            prompt_text=prompt.text,
            sparse_k=sparse_k,
            sparse_frac=sparse_frac,
            done=done,
            total_configs=total_configs,
            merged=len(retrieval.merged) if retrieval else 0,
            ok_n=0,
            total_judges=0,
            retrieval_path="",
            critique_path="",
            error=str(e),
        )


def sweep_grid(topk: int, sparse_step: int) -> list:
    """Generate sparse_k values: 0, sparse_step, 2*sparse_step, ..., up to topk."""
    return list(range(0, topk + 1, sparse_step))


def run_eval(argv: list = None) -> None:
    """Main entry point."""
    if argv is None:
        argv = sys.argv[1:]

    # Resolve config
    config = resolve_config(argv)
    snapshot = config.to_snapshot()

    # Load prompts
    prompts = load_prompts(config.prompts_file, config.num_prompts)
    if not prompts:
        logger.error("No prompts loaded from %s", config.prompts_file)
        sys.exit(1)

    # Load system prompt
    with open(config.system_prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Create run directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.output_root, timestamp)
    ensure_run_dir(run_dir)

    # Write config snapshot with prompts and system prompt
    snapshot.prompts = [p.text for p in prompts]
    snapshot.system_prompt = system_prompt
    write_config(snapshot, run_dir)

    # Write failures file (empty initially)
    failures_path = os.path.join(run_dir, "failures.jsonl")
    with open(failures_path, "w", encoding="utf-8") as f:
        pass  # empty

    # Sweep grid
    sparse_k_values = sweep_grid(config.topk, config.sparse_step)
    judges_per_case = config.judges_per_case

    # Store run_dir on config for _run_single_case to use
    object.__setattr__(config, '_run_dir', run_dir)

    # Collections: (tag, sparse_collection, sparse_only_flag)
    # Variable always uses SAE embeds; control (if configured) always uses SPLADE embeds.
    use_tags = config.sparse_collection_control is not None
    collections = [("variable" if use_tags else "", config.sparse_collection, True)]
    if config.sparse_collection_control:
        collections.append(("control", config.sparse_collection_control, False))

    # Embedding cache: (prompt.text, sparse_only) -> {"dense": [...], "sparse": {...}}
    # Dense vectors are shared across collections; sparse vectors depend on sparse_only.
    embed_cache: dict = {}
    sparse_only_flags = {col_sparse_only for _, _, col_sparse_only in collections}
    for prompt in prompts:
        texts = [prompt.text]
        dense_vecs = dense_embed(config.embed_url, texts)
        dense_vec = dense_vecs[0] if dense_vecs else []
        for flag in sparse_only_flags:
            key = (prompt.text, flag)
            if key not in embed_cache:
                if flag:
                    sparse_vecs = sparse_embed_sae(config.embed_url, texts)
                else:
                    sparse_vecs = sparse_embed(config.embed_url, texts)
                embed_cache[key] = {
                    "dense": dense_vec,
                    "sparse": sparse_vecs[0] if sparse_vecs else {"indices": [], "values": []},
                }

    # Build flat list of all cases: (prompt, sparse_k, sparse_frac, dense_k, sparse_col, tag, embed_key)
    cases = []
    for tag, sparse_col, col_sparse_only in collections:
        for prompt in prompts:
            for sparse_k in sparse_k_values:
                dense_k = config.topk - sparse_k
                sparse_frac = f"{sparse_k / config.topk:.2f}"
                embed_key = (prompt.text, col_sparse_only)
                cases.append((prompt, sparse_k, sparse_frac, dense_k, sparse_col, tag, embed_key))

    total_configs = len(cases)
    case_batch_size = config.case_batch_size
    all_results: list[_CaseResult] = []
    done = 0
    missing = 0

    if case_batch_size > 0 and len(cases) > 1:
        # Batch mode: submit cases in parallel
        max_workers = min(case_batch_size, len(cases))
        if max_workers < 1:
            max_workers = 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, (prompt, sparse_k, sparse_frac, dense_k, sparse_col, tag, embed_key) in enumerate(cases):
                embeddings = embed_cache[embed_key]
                future = executor.submit(
                    _run_single_case,
                    prompt=prompt,
                    sparse_k=sparse_k,
                    sparse_frac=sparse_frac,
                    dense_k=dense_k,
                    embeddings=embeddings,
                    config=config,
                    sparse_collection=sparse_col,
                    tag=tag,
                    system_prompt=system_prompt,
                    judges_per_case=judges_per_case,
                    done=idx + 1,
                    total_configs=total_configs,
                    failures_path=failures_path,
                )
                futures.append(future)

            for future in futures:
                result = future.result()
                all_results.append(result)
                if result.ok_n == 0:
                    missing += 1
    else:
        # Serial mode
        for idx, (prompt, sparse_k, sparse_frac, dense_k, sparse_col, tag, embed_key) in enumerate(cases):
            done += 1
            embeddings = embed_cache[embed_key]
            result = _run_single_case(
                prompt=prompt,
                sparse_k=sparse_k,
                sparse_frac=sparse_frac,
                dense_k=dense_k,
                embeddings=embeddings,
                config=config,
                sparse_collection=sparse_col,
                tag=tag,
                system_prompt=system_prompt,
                judges_per_case=judges_per_case,
                done=done,
                total_configs=total_configs,
                failures_path=failures_path,
            )
            all_results.append(result)
            if result.ok_n == 0:
                missing += 1

    n_collections = len(collections)
    # Report
    print(
        f"\nRun complete. {n_collections} collection(s) x {len(prompts)} prompts x "
        f"{len(sparse_k_values)} sparse-fractions x {judges_per_case} judges/case = "
        f"{total_configs} configs. {missing} missing."
    )
    print("Building report...", end=" ")
    report_path = build_report(run_dir)
    print("done.")
    print(f"Report: {report_path}")
    pdf_path = render_pdf(run_dir)
    if pdf_path:
        print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    run_eval()