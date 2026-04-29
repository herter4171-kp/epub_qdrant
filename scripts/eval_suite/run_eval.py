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
    from .embed import dense_embed, sparse_embed
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
    from eval_suite.embed import dense_embed, sparse_embed
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

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def sweep_grid(topk: int, step: int) -> list:
    """Generate sparse_k sweep values."""
    grid = list(range(0, topk + 1, step))
    if grid[-1] != topk:
        grid.append(topk)
    return grid


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
    total_configs = len(prompts) * len(sparse_k_values)
    done = 0
    missing = 0
    failures = []

    # Embedding cache: (text, embed_url) -> {"dense": [...], "sparse": {...}}
    embed_cache = {}

    for prompt in prompts:
        # Embed once per prompt (cached for all sparse_k values)
        cache_key = (prompt.text, config.embed_url)
        if cache_key not in embed_cache:
            texts = [prompt.text]
            dense_vecs = dense_embed(config.embed_url, texts)
            sparse_vecs = sparse_embed(config.embed_url, texts)
            embed_cache[cache_key] = {
                "dense": dense_vecs[0] if dense_vecs else [],
                "sparse": sparse_vecs[0] if sparse_vecs else {"indices": [], "values": []},
            }
        embeddings = embed_cache[cache_key]

        for sparse_k in sparse_k_values:
            done += 1
            dense_k = config.topk - sparse_k
            sparse_frac = f"{sparse_k / config.topk:.2f}"

            status_parts = []

            # Retrieve
            try:
                retrieval = retrieve(
                    prompt=prompt,
                    embeddings=embeddings,
                    dense_k=dense_k,
                    sparse_k=sparse_k,
                    qdrant_url=config.qdrant_url,
                    collection=config.collection,
                    topk=config.topk,
                )
                retrieval_path = write_retrieval(retrieval, run_dir)
                status_parts.append(f"merged={len(retrieval.merged)} (d={dense_k} s={sparse_k})")
                status_parts.append(f"wrote {os.path.basename(retrieval_path)}")
            except Exception as e:
                logger.error("Retrieval failed for prompt %d sk=%d: %s", prompt.index, sparse_k, e)
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
                missing += 1
                status_parts.append("FAIL")
                print(f"[{done}/{total_configs} sk={sparse_k}/{config.topk} frac={sparse_frac}] " + " | ".join(status_parts))
                continue

            # Critique
            try:
                c = critique_fn(
                    retrieval_set=retrieval,
                    system_prompt=system_prompt,
                    judge_base_url=config.judge_base_url,
                    judge_model=config.judge_model,
                    judge_api_key=config.judge_api_key,
                )
                critique_path = write_critique(c, run_dir)

                jout = c.judge_output
                if jout and jout.error == "empty_response_after_retry":
                    status_parts.append("empty, retrying... empty")
                    status_parts.append("flagged")
                    missing += 1
                elif jout and not jout.parse_ok:
                    status_parts.append(f"parse error: {jout.error}")
                    status_parts.append("flagged")
                    missing += 1
                else:
                    status_parts.append("ok")

                status_parts.append(f"wrote {os.path.basename(critique_path)}")
                print(f"[{done}/{total_configs} sk={sparse_k}/{config.topk} frac={sparse_frac}] " + " | ".join(status_parts))

            except Exception as e:
                logger.error("Critique failed for prompt %d sk=%d: %s", prompt.index, sparse_k, e)
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
                missing += 1
                status_parts.append(f"CRITIQUE FAIL: {e}")
                print(f"[{done}/{total_configs} sk={sparse_k}/{config.topk} frac={sparse_frac}] " + " | ".join(status_parts))

    # Report
    print(f"\nRun complete. {len(prompts)} prompts x {len(sparse_k_values)} sparse-fractions = {total_configs} configs. {missing} missing.")
    print("Building report...", end=" ")
    report_path = build_report(run_dir)
    print("done.")
    print(f"Report: {report_path}")
    pdf_path = render_pdf(run_dir)
    if pdf_path:
        print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    run_eval()