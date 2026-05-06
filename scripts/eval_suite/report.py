"""Build report.md + plots from persisted critiques."""

import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

from .persist import iter_critiques, read_config
from .plotting import (
    per_prompt_bar,
    per_prompt_table_csv,
    aggregate_contour,
    aggregate_table_csv,
    _get_satisfaction,
    _get_avg_relevance,
    _get_avg_relevance_by_source,
    _norm_frac,
)


def _sparse_fractions_set(critiques: List[Dict]) -> List[str]:
    fracs: Set[str] = set()
    for c in critiques:
        fracs.add(_norm_frac(c.get("sparse_fraction", "")))
    return sorted(fracs, key=lambda x: float(x))


def _get_judge_reply(critique: Dict) -> str:
    """Reply from the first parsed judgement (legacy or new). Returns ""
    when nothing parsed."""
    outs = critique.get("judge_outputs")
    if isinstance(outs, list):
        for jo in outs:
            if isinstance(jo, dict) and jo.get("parse_ok"):
                parsed = jo.get("parsed", {})
                if isinstance(parsed, dict):
                    return parsed.get("reply", "") or ""
        return ""
    jout = critique.get("judge_output")
    if not jout or not jout.get("parse_ok"):
        return ""
    parsed = jout.get("parsed", {})
    if not isinstance(parsed, dict):
        return ""
    return parsed.get("reply", "") or ""


def build_report(run_dir: str) -> str:
    """Build report.md. Returns path."""
    report_path = os.path.join(run_dir, "report.md")

    all_critiques = list(iter_critiques(run_dir))
    if not all_critiques:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Eval Report\n\nNo critiques found.\n")
        return report_path

    all_sparse_fractions = _sparse_fractions_set(all_critiques)
    topk = max((c.get("topk", 6) for c in all_critiques), default=6)

    prompt_critiques: Dict[int, List[Dict]] = {}
    for c in all_critiques:
        pi = c.get("prompt_index", 0)
        prompt_critiques.setdefault(pi, []).append(c)

    sorted_indices = sorted(prompt_critiques.keys())
    n_prompts = len(sorted_indices)

    failures_path = os.path.join(run_dir, "failures.jsonl")
    failures_count = 0
    if os.path.exists(failures_path):
        with open(failures_path, "r", encoding="utf-8") as f:
            failures_count = sum(1 for _ in f)

    os.makedirs(os.path.join(run_dir, "report_assets"), exist_ok=True)

    lines: List[str] = []
    lines.append("# Eval Report\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
    lines.append(f"Total prompts: {n_prompts}\n")
    lines.append(f"Total configurations: {n_prompts * len(all_sparse_fractions)}\n")
    lines.append(f"Missing configurations: {failures_count}\n")

    for pi in sorted_indices:
        pc = prompt_critiques.get(pi, [])
        lines.append(f"\n## Prompt {pi:03d}\n")
        prompt_text = pc[0].get("prompt_text", "") if pc else ""
        lines.append(f"**Query**: \"{prompt_text}\"\n")

        per_prompt_bar(run_dir, pi, all_critiques, all_sparse_fractions, topk)
        per_prompt_table_csv(run_dir, pi, all_critiques, all_sparse_fractions, topk)
        lines.append("\n")
        lines.append(f"![Prompt {pi:03d} bar chart](report_assets/prompt_{pi:03d}_bars.png){{ width=80% }}\n")
        lines.append("\n")

        lines.append("| sparse_frac | reply |\n")
        lines.append("|------|------|\n")
        for frac in all_sparse_fractions:
            reply = ""
            for c in pc:
                if _norm_frac(c.get("sparse_fraction", "")) != frac:
                    continue
                reply = _get_judge_reply(c)
                break
            reply_display = reply.replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {frac} | {reply_display} |\n")
        lines.append("\n")

    aggregate_table_csv(run_dir, all_critiques, all_sparse_fractions)
    aggregate_contour(run_dir, all_critiques, all_sparse_fractions, n_prompts)

    lines.append("\n## Aggregate\n\n")
    lines.append("![Aggregate contour](report_assets/aggregate_contour.png){ width=80% }\n\n")
    lines.append("| sparse_frac | satisfaction | avg_relevance | dense_rel | sparse_rel |\n")
    lines.append("|------|------|------|------|------|\n")
    for frac in all_sparse_fractions:
        import numpy as np
        sat_vals: List[float] = []
        rel_vals: List[float] = []
        d_vals: List[float] = []
        sp_vals: List[float] = []
        for c in all_critiques:
            if _norm_frac(c.get("sparse_fraction", "")) != frac:
                continue
            s = _get_satisfaction(c)
            if s >= 1:
                sat_vals.append(float(s))
            r = _get_avg_relevance(c)
            if not np.isnan(r):
                rel_vals.append(r)
            d, sp = _get_avg_relevance_by_source(c)
            if not np.isnan(d):
                d_vals.append(d)
            if not np.isnan(sp):
                sp_vals.append(sp)
        sat_s = f"{np.mean(sat_vals):.2f}" if sat_vals else ""
        rel_s = f"{np.mean(rel_vals):.2f}" if rel_vals else ""
        d_s = f"{np.mean(d_vals):.2f}" if d_vals else ""
        sp_s = f"{np.mean(sp_vals):.2f}" if sp_vals else ""
        lines.append(f"| {frac} | {sat_s} | {rel_s} | {d_s} | {sp_s} |\n")

    lines.append("\n## Missing Configurations\n")
    total_configs = n_prompts * len(all_sparse_fractions)
    present_configs = len({(c.get("prompt_index", 0), _norm_frac(c.get("sparse_fraction", "")))
                           for c in all_critiques})
    lines.append(f"Total expected: {total_configs}\n")
    lines.append(f"Present: {present_configs}\n")
    lines.append(f"Missing: {total_configs - present_configs}\n")

    config_data = read_config(run_dir)
    if config_data:
        lines.append("\n## Configuration\n")
        lines.append(f"Dense collection: {config_data.get('dense_collection', 'unknown')}\n")
        lines.append(f"Sparse collection: {config_data.get('sparse_collection', 'unknown')}\n")
        lines.append(f"Dense vector name: {config_data.get('dense_vector_name', '') or '(unnamed)'}\n")
        lines.append(f"Topk: {config_data.get('topk', '?')}\n")
        lines.append(f"Sparse step: {config_data.get('sparse_step', '?')}\n")
        lines.append(f"Judge model: {config_data.get('judge_model', '?')}\n")
        lines.append(f"Judge base URL: {config_data.get('judge_base_url', '?')}\n")
        lines.append(f"Embed URL: {config_data.get('embed_url', '?')}\n")
        lines.append(f"Timestamp: {config_data.get('timestamp_utc', '?')}\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return report_path


_PDF_ENGINE_CANDIDATES = ["typst", "weasyprint", "wkhtmltopdf", "pdflatex"]


def _pick_pdf_engine() -> Optional[str]:
    """First available engine on PATH, in non-LaTeX-first order."""
    for e in _PDF_ENGINE_CANDIDATES:
        if shutil.which(e):
            return e
    return None


def render_pdf(
    run_dir: str,
    md_filename: str = "report.md",
    engine: Optional[str] = None,
) -> Optional[str]:
    """Render report.md -> report.pdf via pandoc. Returns PDF path or None.

    Auto-picks engine when not specified. Skips with a warning if pandoc
    or no engine available.
    """
    if not shutil.which("pandoc"):
        logger.warning("pandoc not on PATH; skipping PDF render")
        return None

    chosen = engine or _pick_pdf_engine()
    if not chosen:
        logger.warning(
            "No PDF engine found (typst/weasyprint/wkhtmltopdf/pdflatex). "
            "Install one (e.g. `brew install typst`) to enable PDF output."
        )
        return None

    md_path = os.path.join(run_dir, md_filename)
    if not os.path.exists(md_path):
        logger.warning("md not found: %s", md_path)
        return None

    pdf_path = os.path.join(run_dir, os.path.splitext(md_filename)[0] + ".pdf")

    cmd = [
        "pandoc", md_path, "-o", pdf_path,
        f"--pdf-engine={chosen}",
        "--resource-path", run_dir,
        "--from", "markdown",
        "-V", "geometry:margin=0.75in",
    ]
    if chosen in ("pdflatex", "xelatex", "lualatex"):
        # Float placement: keep figures where written, prevent drift.
        cmd += ["-V", "figure-pos=H"]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("pandoc invocation failed: %s", e)
        return None

    if proc.returncode != 0:
        logger.warning(
            "pandoc render failed (engine=%s, rc=%d):\n%s",
            chosen, proc.returncode, proc.stderr.strip()[:2000],
        )
        return None

    return pdf_path
