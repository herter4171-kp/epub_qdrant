#!/usr/bin/env python3
"""Meta-study: pairwise LLM judgment of control vs variable retrieval.

For a study dir produced by run_eval.py with a control collection configured,
compare control vs variable replies per (prompt, sparse_fraction) using a
continuous-scale LLM judge. Outputs:
  {study_dir}/meta/verdicts.json      — cached per-cell verdicts (float 0-1 / null)
  {study_dir}/meta/meta_table.csv     — prompt × sparse_fraction table
  {study_dir}/meta/meta_histogram.png — mean-score histogram

Verdict encoding: 0.0 = full control preference, 1.0 = full variable preference,
0.5 = neutral. Null = missing data or judge failure; excluded from aggregates.

Usage:
    python scripts/eval_study.py SAE_VS_SPLADE/20260509_223012 [options]
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

try:
    from eval_suite.critique import _stream_judge, _strip_wrappers, StreamAbort
    from eval_suite.persist import read_config
except ImportError:
    sys.path.insert(0, os.path.dirname(_here))
    from eval_suite.critique import _stream_judge, _strip_wrappers, StreamAbort
    from eval_suite.persist import read_config

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_META_SYS_PROMPT = """\
* Overview
    * You are part of an agentic RAG workflow meant to down-select the most suited candidate answer to a particular prompt.
    * Upstream of you, LLM calls have generated responses to this prompt using one of two RAG embed and retrieve methods.
    * Your task is reporting the merit of these replies on a continuous scale between zero and one.
* Input
    * The prompt our candidate replies target.
    * A reply with ID 0.
    * A reply with ID 1.
* Evaluation
    * Given your understanding of language, research, and the subject-matter at hand, weigh the two replies against each other.
    * At one end of your scale, you have ID 0, and at the other ID 1.
    * The two digit floating point number you reply with is an expression of your preference between ID 0 and ID 1.
* Output
    * We are not after analysis or any verbiage from you at all!
    * Your reply shall be the requested two digit floating point number!\
"""

_CTRL_FILE_RE = re.compile(r"^prompt_(\d+)_sk_(\d+)_control\.json$")
_VAR_FILE_RE  = re.compile(r"^prompt_(\d+)_sk_(\d+)_variable\.json$")
_FLOAT_RE     = re.compile(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b")

_VERDICTS_FNAME = "verdicts.json"
_TABLE_FNAME    = "meta_table.csv"
_HIST_FNAME     = "meta_histogram.png"

# Verdict: float in [0, 1] (0.0 = control wins, 1.0 = variable wins), None = missing/failed
Verdict = Optional[float]


# ── Discovery ────────────────────────────────────────────────────────────────

def _norm_frac(s) -> str:
    try:
        return f"{float(s):.2f}"
    except (ValueError, TypeError):
        return str(s)


def _critique_fpath(study_dir: str, prompt_index: int, sparse_k: int, tag: str) -> str:
    return os.path.join(
        study_dir, "critiques",
        f"prompt_{prompt_index:03d}_sk_{sparse_k}_{tag}.json",
    )


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_reply(critique: dict) -> Optional[str]:
    """Return reply text from judge_outputs[0], or None if unavailable/unparsed."""
    outs = critique.get("judge_outputs")
    if not isinstance(outs, list) or not outs:
        return None
    jo = outs[0]
    if not isinstance(jo, dict) or not jo.get("parse_ok"):
        return None
    parsed = jo.get("parsed")
    if not isinstance(parsed, dict):
        return None
    reply = parsed.get("reply")
    return reply if isinstance(reply, str) and reply.strip() else None


def discover_cases(study_dir: str) -> List[Tuple[int, int, str, str]]:
    """Return sorted list of (prompt_index, sparse_k, sparse_fraction, prompt_text).

    Includes every (prompt_index, sparse_k) pair for which at least one of
    control/variable critique files exists. Metadata sourced from whichever
    file is present.
    """
    critiques_dir = os.path.join(study_dir, "critiques")
    if not os.path.isdir(critiques_dir):
        return []

    seen: set = set()
    cases = []

    for fname in sorted(os.listdir(critiques_dir)):
        m = _CTRL_FILE_RE.match(fname) or _VAR_FILE_RE.match(fname)
        if not m:
            continue
        pi, sk = int(m.group(1)), int(m.group(2))
        if (pi, sk) in seen:
            continue
        seen.add((pi, sk))

        ctrl = _load_json(_critique_fpath(study_dir, pi, sk, "control"))
        var  = _load_json(_critique_fpath(study_dir, pi, sk, "variable"))
        src  = ctrl or var
        if src is None:
            continue

        frac = _norm_frac(src.get("sparse_fraction", sk))
        text = src.get("prompt_text", "")
        cases.append((pi, sk, frac, text))

    cases.sort(key=lambda c: (c[0], c[1]))
    return cases


# ── Meta-judge ───────────────────────────────────────────────────────────────

def _run_meta_judge(
    client: OpenAI,
    model: str,
    prompt_text: str,
    control_reply: str,
    variable_reply: str,
    label: str,
    *,
    system_prompt: str = _META_SYS_PROMPT,
    judge_attempts: int = 3,
    judge_temperature: float = 0.1,
    judge_timeout_seconds: float = 180.0,
    judge_per_chunk_timeout_seconds: float = 30.0,
    judge_max_tokens: int = 4096,
) -> Verdict:
    """Continuous judge: float 0.0 (control) to 1.0 (variable), None on failure."""
    user_msg = (
        f"Query: {prompt_text}\n\n"
        f"Reply 0:\n{control_reply}\n\n"
        f"Reply 1:\n{variable_reply}\n\n"
        "Reply with a two-digit floating point number between 0.00 and 1.00."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg},
    ]

    _RETRY_BASE = 1.0
    _RETRY_MAX  = 10.0

    for attempt in range(1, judge_attempts + 1):
        if attempt > 1:
            delay = min(_RETRY_BASE * 2 ** (attempt - 2), _RETRY_MAX)
            logger.info("meta-judge retry %d/%d — waiting %.1fs", attempt, judge_attempts, delay)
            time.sleep(delay)

        atl = label if attempt == 1 else f"{label} attempt {attempt}/{judge_attempts}"
        try:
            content = _stream_judge(
                client=client,
                model=model,
                messages=messages,
                label=atl,
                total_timeout=judge_timeout_seconds,
                per_chunk_timeout=judge_per_chunk_timeout_seconds,
                max_tokens=judge_max_tokens,
                temperature=judge_temperature,
            )
        except StreamAbort as exc:
            logger.warning("meta-judge stream abort: %s", exc)
            continue
        except Exception as exc:
            logger.warning("meta-judge transport error (%s): %s", type(exc).__name__, exc)
            continue

        stripped = _strip_wrappers(content).strip()

        # Strict: entire content is a float in [0, 1]
        try:
            v = float(stripped)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            pass

        # Lenient: find first float token in [0, 1]
        for m in _FLOAT_RE.finditer(stripped):
            try:
                v = float(m.group(0))
                if 0.0 <= v <= 1.0:
                    logger.info("meta-judge lenient parse: took %r from %r", m.group(0), stripped[:60])
                    return v
            except ValueError:
                continue

        logger.warning("meta-judge unparseable (attempt %d): %r", attempt, stripped[:80])

    return None


# ── Verdict cache ────────────────────────────────────────────────────────────

def _vkey(prompt_index: int, sparse_k: int) -> str:
    return f"p{prompt_index}_sk{sparse_k}"


def load_verdicts(meta_dir: str) -> Dict[str, Verdict]:
    path = os.path.join(meta_dir, _VERDICTS_FNAME)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, Verdict] = {}
    n_coerced = 0
    for k, v in raw.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        elif v == "0":
            out[k] = 0.0
            n_coerced += 1
        elif v == "1":
            out[k] = 1.0
            n_coerced += 1
        else:
            out[k] = None
    if n_coerced:
        logger.warning("Coerced %d binary verdicts ('0'/'1') to 0.0/1.0 — consider re-running", n_coerced)
    return out


def save_verdicts(meta_dir: str, verdicts: Dict[str, Verdict]) -> None:
    os.makedirs(meta_dir, exist_ok=True)
    path = os.path.join(meta_dir, _VERDICTS_FNAME)
    fd, tmp = tempfile.mkstemp(dir=meta_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(verdicts, f, indent=2)
        os.rename(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Table ────────────────────────────────────────────────────────────────────

def build_table(
    cases: List[Tuple[int, int, str, str]],
    verdicts: Dict[str, Verdict],
) -> Tuple[List[str], List[Tuple[int, str, List[Verdict]]]]:
    """
    Returns:
        fractions: sorted unique sparse_fraction strings e.g. ["0.00", "0.17", ...]
        rows:      (prompt_index, prompt_text, [verdict_per_fraction]) per prompt
    """
    frac_set: set = set()
    prompt_map: Dict[int, str] = {}

    for pi, sk, frac, text in cases:
        frac_set.add(frac)
        prompt_map.setdefault(pi, text)

    fractions = sorted(frac_set, key=float)

    # frac → prompt_index → sparse_k
    frac_sk: Dict[str, Dict[int, int]] = {}
    for pi, sk, frac, _ in cases:
        frac_sk.setdefault(frac, {})[pi] = sk

    rows = []
    for pi in sorted(prompt_map):
        row_verdicts: List[Verdict] = []
        for frac in fractions:
            sk = frac_sk.get(frac, {}).get(pi)
            if sk is None:
                row_verdicts.append(None)
            else:
                row_verdicts.append(verdicts.get(_vkey(pi, sk)))
        rows.append((pi, prompt_map[pi], row_verdicts))

    return fractions, rows


def _col_agg(col_verdicts: List[Verdict]) -> str:
    """'Variable preference (%)' cell: mean score ×100 truncated to 1 decimal, '-' if all null."""
    valid = [v for v in col_verdicts if v is not None]
    if not valid:
        return "-"
    mean_score = sum(valid) / len(valid)
    pct = int(mean_score * 1000) / 10.0  # truncate, not round
    return f"{pct:.1f}"


def write_table_csv(
    meta_dir: str,
    fractions: List[str],
    rows: List[Tuple[int, str, List[Verdict]]],
) -> str:
    path = os.path.join(meta_dir, _TABLE_FNAME)
    os.makedirs(meta_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt"] + fractions)
        for pi, text, vrow in rows:
            label = f"{pi:03d}: {text[:80].replace(chr(10), ' ')}"
            cells = [f"{v:.2f}" if v is not None else "-" for v in vrow]
            writer.writerow([label] + cells)
        agg = [_col_agg([row[2][i] for row in rows]) for i in range(len(fractions))]
        writer.writerow(["Variable preference (%)"] + agg)
    return path


def print_table(
    fractions: List[str],
    rows: List[Tuple[int, str, List[Verdict]]],
) -> None:
    col_w = [5] + [max(5, len(f)) for f in fractions]
    header_cells = ["Prmpt"] + fractions
    sep = "  ".join("-" * w for w in col_w)

    print("  ".join(h.ljust(w) for h, w in zip(header_cells, col_w)))
    print(sep)
    for pi, _, vrow in rows:
        cells = [f"{pi:03d}"] + [f"{v:.2f}" if v is not None else "-" for v in vrow]
        print("  ".join(c.ljust(w) for c, w in zip(cells, col_w)))
    print(sep)
    agg = [_col_agg([row[2][i] for row in rows]) for i in range(len(fractions))]
    agg_cells = ["var%"] + agg
    print("  ".join(c.ljust(w) for c, w in zip(agg_cells, col_w)))


# ── Histogram ────────────────────────────────────────────────────────────────

def plot_histogram(
    meta_dir: str,
    fractions: List[str],
    rows: List[Tuple[int, str, List[Verdict]]],
) -> str:
    """Bar chart of mean preference score (0-1) per sparse fraction.

    0.0 = full control preference, 1.0 = full variable preference, 0.5 = neutral.
    Null counts annotated in red above each bar.
    """
    mean_scores = []
    null_counts  = []
    for col_idx in range(len(fractions)):
        col = [row[2][col_idx] for row in rows]
        valid = [v for v in col if v is not None]
        mean_scores.append(sum(valid) / len(valid) if valid else float("nan"))
        null_counts.append(sum(1 for v in col if v is None))

    x = np.arange(len(fractions))
    bar_w = 0.6

    fig, ax = plt.subplots(figsize=(max(6, len(fractions) * 1.1 + 2), 5))

    colors = ["#4C72B0" if (np.isnan(s) or s >= 0.5) else "#DD8452"
              for s in mean_scores]
    bars = ax.bar(x, mean_scores, width=bar_w, color=colors)

    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1.0, label="Neutral (0.50)")

    # Mean score labels inside bars
    for bar, s in zip(bars, mean_scores):
        if not np.isnan(s):
            bx = bar.get_x() + bar.get_width() / 2
            ty = s / 2
            ax.text(bx, ty, f"{s:.2f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    # Null counts annotated in red above bars
    for bar, s, nn in zip(bars, mean_scores, null_counts):
        if nn > 0:
            bx = bar.get_x() + bar.get_width() / 2
            top = s if not np.isnan(s) else 0.0
            ax.text(bx, top + 0.02, f"null={nn}", ha="center", va="bottom",
                    fontsize=7, color="#CC3333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(fractions, fontsize=9)
    ax.set_xlabel("Sparse fraction")
    ax.set_ylabel("Mean preference score (0=control, 1=variable)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Meta-study: mean preference score by sparse fraction")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    os.makedirs(meta_dir, exist_ok=True)
    path = os.path.join(meta_dir, _HIST_FNAME)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Meta-study pairwise judgment: control vs variable retrieval."
    )
    p.add_argument("study_dir",
                   help="Study run dir (contains config.json + critiques/)")

    # Judge flags — default to study config.json values
    p.add_argument("--judge-model",                     default=None)
    p.add_argument("--judge-base-url",                  default=None)
    p.add_argument("--judge-api-key",                   default=None)
    p.add_argument("--judge-temperature",               type=float, default=None)
    p.add_argument("--judge-timeout-seconds",           type=float, default=None)
    p.add_argument("--judge-per-chunk-timeout-seconds", type=float, default=None)
    p.add_argument("--judge-max-tokens",                type=int,   default=None)
    p.add_argument("--judge-attempts",                  type=int,   default=None)
    p.add_argument("--system-prompt-file", default=None,
                   help="Path to a text file whose contents replace the default meta-judge system prompt")

    p.add_argument("--dry-run", action="store_true",
                   help="Discover and show cases without running the judge")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    study_dir = os.path.abspath(args.study_dir)
    meta_dir  = os.path.join(study_dir, "meta")

    cfg = read_config(study_dir) or {}

    def _cfg(flag, key, default):
        val = getattr(args, flag.replace("-", "_"))
        return val if val is not None else cfg.get(key, default)

    judge_model    = args.judge_model    or cfg.get("judge_model", "")
    judge_base_url = args.judge_base_url or cfg.get("judge_base_url", "")
    judge_api_key  = (args.judge_api_key or cfg.get("judge_api_key", "")
                      or os.environ.get("JUDGE_API_KEY", "not-set"))
    judge_temp     = _cfg("judge_temperature",               "judge_temperature",               0.1)
    judge_timeout  = _cfg("judge_timeout_seconds",           "judge_timeout_seconds",           180.0)
    judge_chunk_to = _cfg("judge_per_chunk_timeout_seconds", "judge_per_chunk_timeout_seconds", 30.0)
    judge_max_tok  = _cfg("judge_max_tokens",                "judge_max_tokens",                4096)
    judge_attempts = _cfg("judge_attempts",                  "judge_attempts",                  3)

    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as _f:
            meta_sys_prompt = _f.read().strip()
        logger.info("Using system prompt from %s", args.system_prompt_file)
    else:
        meta_sys_prompt = _META_SYS_PROMPT

    if not judge_model or not judge_base_url:
        sys.exit("ERROR: judge_model and judge_base_url required (from config.json or CLI)")

    cases = discover_cases(study_dir)
    if not cases:
        sys.exit(f"ERROR: no critique files found in {study_dir}/critiques/")

    n_prompts = len({pi for pi, *_ in cases})
    logger.info("Discovered %d cases across %d prompts", len(cases), n_prompts)

    if args.dry_run:
        fractions, rows = build_table(cases, {})
        print(f"Study dir : {study_dir}")
        print(f"Cases     : {len(cases)}")
        print(f"Prompts   : {len(rows)}")
        print(f"Fractions : {fractions}")
        return

    verdicts: Dict[str, Verdict] = {}
    client = OpenAI(base_url=judge_base_url, api_key=judge_api_key)

    n_run = n_null_data = n_failed = 0

    for pi, sk, frac, prompt_text in cases:
        key = _vkey(pi, sk)

        ctrl = _load_json(_critique_fpath(study_dir, pi, sk, "control"))
        var  = _load_json(_critique_fpath(study_dir, pi, sk, "variable"))
        ctrl_reply = _extract_reply(ctrl) if ctrl else None
        var_reply  = _extract_reply(var)  if var  else None

        if ctrl_reply is None or var_reply is None:
            logger.info("p%03d sk=%d: missing reply — ctrl=%s var=%s, verdict=null",
                        pi, sk, ctrl_reply is not None, var_reply is not None)
            verdicts[key] = None
            n_null_data += 1
            save_verdicts(meta_dir, verdicts)
            continue

        label = f"meta p{pi:03d} sk={sk} ({frac})"
        logger.info("Running %s", label)

        verdict = _run_meta_judge(
            client=client,
            model=judge_model,
            prompt_text=prompt_text,
            control_reply=ctrl_reply,
            variable_reply=var_reply,
            label=label,
            system_prompt=meta_sys_prompt,
            judge_attempts=judge_attempts,
            judge_temperature=judge_temp,
            judge_timeout_seconds=judge_timeout,
            judge_per_chunk_timeout_seconds=judge_chunk_to,
            judge_max_tokens=judge_max_tok,
        )

        verdicts[key] = verdict
        n_run += 1
        if verdict is None:
            n_failed += 1
        save_verdicts(meta_dir, verdicts)

    logger.info(
        "Done — run=%d cached=%d no_data=%d judge_failed=%d",
        n_run, len(verdicts) - n_run - n_null_data, n_null_data, n_failed,
    )

    fractions, rows = build_table(cases, verdicts)

    csv_path  = write_table_csv(meta_dir, fractions, rows)
    hist_path = plot_histogram(meta_dir, fractions, rows)

    print()
    print_table(fractions, rows)
    print()
    print(f"CSV:  {csv_path}")
    print(f"Plot: {hist_path}")


if __name__ == "__main__":
    main()
