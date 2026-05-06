"""Plotting functions: per-prompt bar charts + aggregate contour."""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sparse_fraction_to_float(s: str) -> float:
    return float(s)


def _norm_frac(s: Any) -> str:
    try:
        return f"{float(s):.2f}"
    except (ValueError, TypeError):
        return str(s)


def _iter_judge_outputs(critique: Dict):
    """Yield judge_output dicts. Supports new ``judge_outputs`` list and
    legacy single ``judge_output`` field."""
    outs = critique.get("judge_outputs")
    if isinstance(outs, list):
        for o in outs:
            if isinstance(o, dict):
                yield o
        return
    one = critique.get("judge_output")
    if isinstance(one, dict):
        yield one


def _get_satisfaction(critique: Dict) -> float:
    """Mean satisfaction across all parsed judgements, or -1 if none parsed."""
    vals: List[float] = []
    for jo in _iter_judge_outputs(critique):
        if not jo.get("parse_ok"):
            continue
        parsed = jo.get("parsed", {})
        if not isinstance(parsed, dict):
            continue
        sat = parsed.get("satisfaction")
        if isinstance(sat, (int, float)) and 1 <= sat <= 10:
            vals.append(float(sat))
    if not vals:
        return -1
    return float(np.mean(vals))


def _get_avg_relevance(critique: Dict) -> float:
    """Mean relevance pooled across all judgements × all chunks. NaN if none."""
    vals: List[float] = []
    for jo in _iter_judge_outputs(critique):
        if not jo.get("parse_ok"):
            continue
        parsed = jo.get("parsed", {})
        if not isinstance(parsed, dict):
            continue
        for ch in parsed.get("chunks") or []:
            if not isinstance(ch, dict):
                continue
            r = ch.get("relevance")
            if isinstance(r, (int, float)) and 1 <= r <= 10:
                vals.append(float(r))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _get_avg_relevance_by_source(critique: Dict) -> Tuple[float, float]:
    """Return (avg_relevance_dense, avg_relevance_sparse). NaN if absent.

    Pools across all judgements × chunks. Map judge.parsed.chunks[].id
    through critique.chunks[].docket_id (legacy: judge_id) to recover
    source. Older legacy: id may be the 1-based rank.
    """
    chunks_meta = critique.get("chunks") or []
    docket_to_source: Dict[str, str] = {}
    rank_to_source: Dict[int, str] = {}
    for c in chunks_meta:
        if not isinstance(c, dict):
            continue
        src = c.get("source", "")
        did = c.get("docket_id") or c.get("judge_id")
        if isinstance(did, str) and did:
            docket_to_source[did] = src
        rank = c.get("rank")
        if isinstance(rank, int):
            rank_to_source[rank] = src

    dense_vals: List[float] = []
    sparse_vals: List[float] = []
    for jo in _iter_judge_outputs(critique):
        if not jo.get("parse_ok"):
            continue
        parsed = jo.get("parsed", {})
        if not isinstance(parsed, dict):
            continue
        for ch in parsed.get("chunks") or []:
            if not isinstance(ch, dict):
                continue
            r = ch.get("relevance")
            rid = ch.get("id")
            if not isinstance(r, (int, float)) or not (1 <= r <= 10):
                continue
            src = ""
            if isinstance(rid, str):
                src = docket_to_source.get(rid, "")
            elif isinstance(rid, (int, float)):
                src = rank_to_source.get(int(rid), "")
            if src == "dense":
                dense_vals.append(float(r))
            elif src in ("sparse", "sparse_resolved"):
                sparse_vals.append(float(r))
    d = float(np.mean(dense_vals)) if dense_vals else float("nan")
    s = float(np.mean(sparse_vals)) if sparse_vals else float("nan")
    return d, s


def per_prompt_bar(
    run_dir: str,
    prompt_index: int,
    critiques: List[Dict[str, Any]],
    all_sparse_fractions: List[str],
    topk: int,
) -> str:
    """Per-prompt grouped bar: avg satisfaction + avg relevance, both 1-10."""
    sat_by_frac: Dict[str, List[float]] = {}
    rel_by_frac: Dict[str, List[float]] = {}
    for c in critiques:
        if c.get("prompt_index") != prompt_index:
            continue
        frac = _norm_frac(c.get("sparse_fraction", ""))
        s = _get_satisfaction(c)
        if s >= 1:
            sat_by_frac.setdefault(frac, []).append(float(s))
        r = _get_avg_relevance(c)
        if not np.isnan(r):
            rel_by_frac.setdefault(frac, []).append(r)

    fracs = sorted(set(list(sat_by_frac.keys()) + list(rel_by_frac.keys())),
                   key=_sparse_fraction_to_float)
    sat_avgs = [np.mean(sat_by_frac[f]) if sat_by_frac.get(f) else np.nan for f in fracs]
    rel_avgs = [np.mean(rel_by_frac[f]) if rel_by_frac.get(f) else np.nan for f in fracs]

    fig, ax = plt.subplots(figsize=(len(fracs) * 1.0 + 2, 4))
    if fracs:
        x = np.arange(len(fracs))
        w = 0.38
        b1 = ax.bar(x - w / 2, sat_avgs, width=w, color="#4C72B0",
                    edgecolor="white", label="Satisfaction")
        b2 = ax.bar(x + w / 2, rel_avgs, width=w, color="#DD8452",
                    edgecolor="white", label="Avg relevance")
        for bar, val in zip(b1, sat_avgs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(b2, rel_avgs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{float(f):.2f}" for f in fracs], fontsize=9)

    ax.set_ylim(0, 11)
    ax.set_xlabel("Sparse fraction")
    ax.set_ylabel("")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Prompt {prompt_index:03d} - satisfaction vs avg relevance")

    fig.tight_layout()
    path = os.path.join(run_dir, "report_assets", f"prompt_{prompt_index:03d}_bars.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def per_prompt_table_csv(
    run_dir: str,
    prompt_index: int,
    critiques: List[Dict[str, Any]],
    all_sparse_fractions: List[str],
    topk: int,
) -> str:
    """Per-prompt CSV: sparse_frac, satisfaction, avg_relevance, dense_rel, sparse_rel, reply."""
    path = os.path.join(run_dir, "report_assets", f"per_prompt_table_{prompt_index:03d}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sparse_frac", "satisfaction", "avg_relevance",
                         "avg_relevance_dense", "avg_relevance_sparse", "reply"])
        for c in critiques:
            if c.get("prompt_index") != prompt_index:
                continue
            frac = c.get("sparse_fraction", "")
            sat = _get_satisfaction(c)
            rel = _get_avg_relevance(c)
            dense_r, sparse_r = _get_avg_relevance_by_source(c)
            jout = c.get("judge_output")
            reply = ""
            if jout and jout.get("parse_ok"):
                parsed = jout.get("parsed", {})
                if isinstance(parsed, dict):
                    reply = parsed.get("reply", "")
            writer.writerow([
                f"{float(frac):.2f}",
                sat if sat >= 1 else "",
                "" if np.isnan(rel) else f"{rel:.2f}",
                "" if np.isnan(dense_r) else f"{dense_r:.2f}",
                "" if np.isnan(sparse_r) else f"{sparse_r:.2f}",
                reply,
            ])
    return path


def aggregate_contour(
    run_dir: str,
    all_critiques: List[Dict[str, Any]],
    all_sparse_fractions: List[str],
    n_prompts: int,
) -> str:
    """Aggregate smoothed heatmap over scattered (frac, rel, sat) points.

    X-axis: sparse fraction
    Y-axis: avg relevance (1-10)
    Color: satisfaction, color scale forced to 1-10

    This version:
      - aggregates duplicate (frac, rel) points
      - builds a dense grid
      - uses anisotropic Gaussian kernel smoothing directly on the grid
      - masks outside the convex hull
      - overlays raw points so actual observations remain visible
    """
    from collections import defaultdict

    import matplotlib.tri as mtri

    raw_points: List[Tuple[float, float, float]] = []

    for c in all_critiques:
        sat = _get_satisfaction(c)
        if sat < 1:
            continue

        rel = _get_avg_relevance(c)
        if np.isnan(rel):
            continue

        try:
            frac = float(_norm_frac(c.get("sparse_fraction", "")))
        except (ValueError, TypeError):
            continue

        raw_points.append((frac, rel, float(sat)))

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("turbo")

    rendered_surface = False

    if len(raw_points) >= 3:
        xs = np.array([p[0] for p in raw_points], dtype=float)
        ys = np.array([p[1] for p in raw_points], dtype=float)
        zs = np.array([p[2] for p in raw_points], dtype=float)

        unique_x = len(np.unique(xs))
        unique_y = len(np.unique(ys))

        if unique_x >= 2 and unique_y >= 2:
            try:
                # Aggregate duplicate coordinate pairs before smoothing.
                buckets: Dict[Tuple[float, float], List[float]] = defaultdict(list)
                for x, y, z in zip(xs, ys, zs):
                    buckets[(round(float(x), 6), round(float(y), 6))].append(float(z))

                pts = np.array([[x, y] for (x, y) in buckets.keys()], dtype=float)
                vals = np.array([np.mean(v) for v in buckets.values()], dtype=float)

                if len(pts) >= 3 and len(np.unique(pts[:, 0])) >= 2 and len(np.unique(pts[:, 1])) >= 2:
                    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
                    y_min, y_max = 0.5, 10.5

                    # Dense regular grid for rendering.
                    grid_nx = 300
                    grid_ny = 300
                    xg = np.linspace(x_min, x_max, grid_nx)
                    yg = np.linspace(y_min, y_max, grid_ny)
                    X, Y = np.meshgrid(xg, yg)

                    # Kernel bandwidths:
                    # - Lower bw_x => less sideways bleed across sparse-fraction columns
                    # - Higher bw_y => more vertical smoothing
                    #
                    # Starting point:
                    #   bw_x = 0.06 to 0.10 usually works well if your columns are around 0.16 apart
                    #   bw_y = 0.35 to 0.60 depending on how much vertical smoothing you want
                    bw_x = 0.07
                    bw_y = 0.45

                    Z_num = np.zeros_like(X, dtype=float)
                    Z_den = np.zeros_like(X, dtype=float)

                    # Gaussian kernel smoother
                    for px, py, pz in zip(pts[:, 0], pts[:, 1], vals):
                        w = np.exp(
                            -0.5 * (
                                ((X - px) / bw_x) ** 2 +
                                ((Y - py) / bw_y) ** 2
                            )
                        )
                        Z_num += w * pz
                        Z_den += w

                    with np.errstate(invalid="ignore", divide="ignore"):
                        Z_smooth = Z_num / Z_den

                    # Mask outside convex hull so corners are not hallucinated.
                    tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
                    finder = tri.get_trifinder()
                    outside_hull = finder(X, Y) == -1

                    Z_smooth[Z_den == 0] = np.nan
                    Z_smooth[outside_hull] = np.nan
                    Z_smooth = np.clip(Z_smooth, 1.0, 10.0)

                    im = ax.imshow(
                        Z_smooth,
                        origin="lower",
                        extent=[x_min, x_max, y_min, y_max],
                        aspect="auto",
                        cmap=cmap,
                        vmin=1.0,
                        vmax=10.0,
                        interpolation="bilinear",
                    )

                    cbar = plt.colorbar(im, ax=ax, ticks=range(1, 11))
                    cbar.set_label("Mean satisfaction (1-10)")
                    rendered_surface = True

            except Exception as e:  # noqa: BLE001
                print(f"Smoothed surface failed: {e}; falling back to scatter")

        # Overlay actual observations
        ax.scatter(
            xs,
            ys,
            c=zs,
            cmap=cmap,
            vmin=1.0,
            vmax=10.0,
            s=36,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )

        if not rendered_surface:
            sc = ax.scatter(xs, ys, c=zs, cmap=cmap, vmin=1.0, vmax=10.0, s=0)
            cbar = plt.colorbar(sc, ax=ax, ticks=range(1, 11))
            cbar.set_label("Mean satisfaction (1-10)")

    ax.set_xlabel("Sparse fraction")
    ax.set_ylabel("Avg relevance (1-10)")
    ax.set_title(
        f"Aggregate - sparse fraction vs avg relevance, color=satisfaction ({n_prompts} prompts)"
    )
    ax.set_ylim(0.5, 10.5)

    if raw_points:
        xs_arr = np.array([p[0] for p in raw_points])
        x_min, x_max = float(xs_arr.min()), float(xs_arr.max())
        pad = max(0.02, 0.03 * (x_max - x_min if x_max > x_min else 1.0))
        ax.set_xlim(x_min - pad, x_max + pad)

    fig.tight_layout()

    path = os.path.join(run_dir, "report_assets", "aggregate_contour.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def aggregate_table_csv(
    run_dir: str,
    all_critiques: List[Dict[str, Any]],
    all_sparse_fractions: List[str],
    topk: int = 0,
) -> str:
    """Aggregate CSV: sparse_frac, satisfaction, avg_relevance, dense_rel, sparse_rel."""
    path = os.path.join(run_dir, "report_assets", "aggregate_table.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sparse_frac", "satisfaction", "avg_relevance",
                         "avg_relevance_dense", "avg_relevance_sparse"])
        fracs = sorted({_norm_frac(c.get("sparse_fraction", "")) for c in all_critiques},
                       key=_sparse_fraction_to_float)
        for frac in fracs:
            sat_vals: List[float] = []
            rel_vals: List[float] = []
            d_vals: List[float] = []
            s_vals: List[float] = []
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
                    s_vals.append(sp)
            writer.writerow([
                frac,
                f"{np.mean(sat_vals):.2f}" if sat_vals else "",
                f"{np.mean(rel_vals):.2f}" if rel_vals else "",
                f"{np.mean(d_vals):.2f}" if d_vals else "",
                f"{np.mean(s_vals):.2f}" if s_vals else "",
            ])
    return path
