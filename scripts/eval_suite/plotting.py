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


def _get_mass_metrics(critique: Dict) -> Tuple[float, float]:
    """Return (sparse_mass, dense_mass) averaged across judge outputs.

    sparse_mass = mean(sum of relevance% for sparse/sparse_resolved chunks / 100)
    dense_mass  = mean(sum of relevance% for dense chunks / 100)
    """
    chunks_meta = critique.get("chunks") or []
    docket_to_source: Dict[str, str] = {}
    for ch in chunks_meta:
        if not isinstance(ch, dict):
            continue
        did = ch.get("docket_id") or ch.get("judge_id")
        src = ch.get("source", "")
        if isinstance(did, str) and did:
            docket_to_source[did] = src

    sparse_masses: List[float] = []
    dense_masses: List[float] = []
    for jo in _iter_judge_outputs(critique):
        if not jo.get("parse_ok"):
            continue
        parsed = jo.get("parsed", {})
        if not isinstance(parsed, dict):
            continue
        chunks = parsed.get("chunks") or []
        if not chunks:
            continue
        run_sparse = 0.0
        run_dense = 0.0
        for ch in chunks:
            if not isinstance(ch, dict):
                continue
            r = ch.get("relevance")
            if not isinstance(r, (int, float)):
                continue
            cid = ch.get("id")
            src = docket_to_source.get(cid, "") if isinstance(cid, str) else ""
            if src in ("sparse", "sparse_resolved"):
                run_sparse += r
            elif src == "dense":
                run_dense += r
        sparse_masses.append(run_sparse / 100.0)
        dense_masses.append(run_dense / 100.0)

    if not sparse_masses:
        return float("nan"), float("nan")
    return float(np.mean(sparse_masses)), float(np.mean(dense_masses))


def _get_sparse_lift(critique: Dict) -> float:
    """sparse_mass / sparse_fraction. NaN when sparse_fraction is 0 or 1."""
    try:
        sf = float(critique.get("sparse_fraction", 0))
    except (ValueError, TypeError):
        return float("nan")
    if sf <= 0.0 or sf >= 1.0:
        return float("nan")
    sparse_mass, _ = _get_mass_metrics(critique)
    if np.isnan(sparse_mass):
        return float("nan")
    return sparse_mass / sf


def _get_sparse_satisfaction_component(critique: Dict) -> float:
    """satisfaction * sparse_mass."""
    sat = _get_satisfaction(critique)
    if sat < 1:
        return float("nan")
    sparse_mass, _ = _get_mass_metrics(critique)
    if np.isnan(sparse_mass):
        return float("nan")
    return sat * sparse_mass


def _get_avg_relevance_by_source(critique: Dict) -> Tuple[float, float]:
    """Return (avg_relevance_dense, avg_relevance_sparse). NaN if absent.

    Pools across all judgements × chunks. Map judge.parsed.chunks[].id
    through critique.chunks[].docket_id (legacy: judge_id) to recover
    source. Older legacy: id may be the 1-based rank.
    """
    chunks_meta = critique.get("chunks") or []
    docket_to_sources: Dict[str, List[str]] = {}
    rank_to_sources: Dict[int, List[str]] = {}
    for c in chunks_meta:
        if not isinstance(c, dict):
            continue
        # Prefer the multi-source list; fall back to the single source string.
        srcs = c.get("sources") or ([c.get("source", "")] if c.get("source") else [])
        did = c.get("docket_id") or c.get("judge_id")
        if isinstance(did, str) and did:
            docket_to_sources[did] = srcs
        rank = c.get("rank")
        if isinstance(rank, int):
            rank_to_sources[rank] = srcs

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
            srcs: List[str] = []
            if isinstance(rid, str):
                srcs = docket_to_sources.get(rid, [])
            elif isinstance(rid, (int, float)):
                srcs = rank_to_sources.get(int(rid), [])
            # Attribute to every source that retrieved this chunk so dual hits
            # don't unfairly credit only the first-seen path.
            if "dense" in srcs:
                dense_vals.append(float(r))
            if any(s in ("sparse", "sparse_resolved") for s in srcs):
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
    suffix: str = "",
) -> str:
    """Aggregate smoothed heatmap over scattered (frac, sparse_lift, sat) points.

    X-axis: sparse fraction (excludes 0 and 1)
    Y-axis: sparse_lift = sparse_mass / sparse_fraction, clipped to [0, 1]
    Color: satisfaction, color scale forced to 1-10

    Smoothing:
      - builds a dense grid
      - uses anisotropic Gaussian kernel smoothing
      - masks outside the convex hull
      - overlays raw points so actual observations remain visible
    """
    import matplotlib.tri as mtri

    raw_points: List[Tuple[float, float, float]] = []

    for c in all_critiques:
        try:
            frac = float(_norm_frac(c.get("sparse_fraction", "")))
        except (ValueError, TypeError):
            continue
        if frac <= 0.0 or frac >= 1.0:
            continue
        sat = _get_satisfaction(c)
        if sat < 1:
            continue
        lift = _get_sparse_lift(c)
        if np.isnan(lift):
            continue
        raw_points.append((frac, min(lift, 2.0), float(sat)))

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
                pts = np.column_stack([xs, ys])
                vals = zs

                if len(pts) >= 3 and len(np.unique(pts[:, 0])) >= 2 and len(np.unique(pts[:, 1])) >= 2:
                    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
                    y_min, y_max = 0.0, 2.0

                    grid_nx = 300
                    grid_ny = 300
                    xg = np.linspace(x_min, x_max, grid_nx)
                    yg = np.linspace(y_min, y_max, grid_ny)
                    X, Y = np.meshgrid(xg, yg)

                    bw_x = 0.07
                    bw_y = 0.08

                    Z_num = np.zeros_like(X, dtype=float)
                    Z_den = np.zeros_like(X, dtype=float)

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

                    # Mask outside convex hull. Add tiny fixed-seed jitter so
                    # duplicate (x, y) coordinates don't crash Triangulation.
                    rng = np.random.default_rng(0)
                    jitter = rng.uniform(-1e-4, 1e-4, len(pts))
                    tri = mtri.Triangulation(pts[:, 0], pts[:, 1] + jitter)
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
                    cbar.set_label("Satisfaction")
                    rendered_surface = True

            except Exception as e:  # noqa: BLE001
                print(f"Smoothed surface failed: {e}; falling back to scatter")

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
            cbar.set_label("Satisfaction")

    ax.set_xlabel("Sparse fraction")
    ax.set_ylabel("Sparse contribution lift")
    ax.set_title(
        f"Aggregate - sparse fraction vs sparse lift, color=satisfaction ({n_prompts} prompts)"
    )
    ax.set_ylim(0.0, 2.0)

    if raw_points:
        xs_arr = np.array([p[0] for p in raw_points])
        x_min, x_max = float(xs_arr.min()), float(xs_arr.max())
        pad = max(0.02, 0.03 * (x_max - x_min if x_max > x_min else 1.0))
        ax.set_xlim(x_min - pad, x_max + pad)

    fig.tight_layout()

    fname = f"aggregate_contour{'_' + suffix if suffix else ''}.png"
    path = os.path.join(run_dir, "report_assets", fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def aggregate_table_csv(
    run_dir: str,
    all_critiques: List[Dict[str, Any]],
    all_sparse_fractions: List[str],
    topk: int = 0,
    suffix: str = "",
) -> str:
    """Aggregate CSV: sparse_frac, satisfaction, sparse_mass, dense_mass, sparse_lift, sparse_satisfaction_component."""
    fname = f"aggregate_table{'_' + suffix if suffix else ''}.csv"
    path = os.path.join(run_dir, "report_assets", fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sparse_frac", "satisfaction", "sparse_mass", "dense_mass",
                         "sparse_lift", "sparse_satisfaction_component"])
        fracs = sorted(
            {_norm_frac(c.get("sparse_fraction", "")) for c in all_critiques},
            key=_sparse_fraction_to_float,
        )
        for frac in fracs:
            sat_vals: List[float] = []
            sm_vals: List[float] = []
            dm_vals: List[float] = []
            lift_vals: List[float] = []
            ssc_vals: List[float] = []
            for c in all_critiques:
                if _norm_frac(c.get("sparse_fraction", "")) != frac:
                    continue
                s = _get_satisfaction(c)
                if s >= 1:
                    sat_vals.append(float(s))
                sm, dm = _get_mass_metrics(c)
                if not np.isnan(sm):
                    sm_vals.append(sm)
                if not np.isnan(dm):
                    dm_vals.append(dm)
                lift = _get_sparse_lift(c)
                if not np.isnan(lift):
                    lift_vals.append(lift)
                ssc = _get_sparse_satisfaction_component(c)
                if not np.isnan(ssc):
                    ssc_vals.append(ssc)
            writer.writerow([
                frac,
                f"{np.mean(sat_vals):.3f}" if sat_vals else "",
                f"{np.mean(sm_vals):.3f}" if sm_vals else "",
                f"{np.mean(dm_vals):.3f}" if dm_vals else "",
                f"{np.mean(lift_vals):.3f}" if lift_vals else "",
                f"{np.mean(ssc_vals):.3f}" if ssc_vals else "",
            ])
    return path


def aggregate_bars(
    run_dir: str,
    all_critiques: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Two grouped bar charts comparing sparse collections.

    Bar 1: x=sparse_fraction, grouped=sparse_collection, y=mean satisfaction
    Bar 2: x=sparse_fraction, grouped=sparse_collection, y=sparse_satisfaction_component
    Returns (path_satisfaction, path_sparse_satisfaction_component).
    """
    collections: List[str] = []
    seen_cols: set = set()
    for c in all_critiques:
        col = c.get("sparse_collection", "")
        if col not in seen_cols:
            seen_cols.add(col)
            collections.append(col)

    fracs = sorted(
        {_norm_frac(c.get("sparse_fraction", "")) for c in all_critiques},
        key=_sparse_fraction_to_float,
    )

    sat_data: Dict[str, Dict[str, List[float]]] = {col: {} for col in collections}
    ssc_data: Dict[str, Dict[str, List[float]]] = {col: {} for col in collections}

    for c in all_critiques:
        col = c.get("sparse_collection", "")
        frac = _norm_frac(c.get("sparse_fraction", ""))
        sat = _get_satisfaction(c)
        if sat >= 1:
            sat_data[col].setdefault(frac, []).append(sat)
        ssc = _get_sparse_satisfaction_component(c)
        if not np.isnan(ssc):
            ssc_data[col].setdefault(frac, []).append(ssc)

    os.makedirs(os.path.join(run_dir, "report_assets"), exist_ok=True)

    def _draw_grouped_bar(
        data: Dict[str, Dict[str, List[float]]],
        ylabel: str,
        title: str,
        fname: str,
        ylim: Optional[Tuple[float, float]],
    ) -> str:
        n_cols = len(collections)
        x = np.arange(len(fracs))
        width = 0.8 / max(n_cols, 1)
        offsets = np.linspace(-(n_cols - 1) / 2, (n_cols - 1) / 2, n_cols) * width
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        fig, ax = plt.subplots(figsize=(max(len(fracs) * 1.2 + 2, 6), 5))
        for i, col in enumerate(collections):
            means = [
                float(np.mean(data[col][f])) if data[col].get(f) else float("nan")
                for f in fracs
            ]
            bars = ax.bar(x + offsets[i], means, width=width, color=colors[i % len(colors)],
                          edgecolor="white", label=col)
            for bar, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{float(f):.2f}" for f in fracs], fontsize=9)
        ax.set_xlabel("Sparse fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9)
        if ylim is not None:
            ax.set_ylim(*ylim)
        fig.tight_layout()
        path = os.path.join(run_dir, "report_assets", fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    path1 = _draw_grouped_bar(
        sat_data,
        ylabel="Satisfaction",
        title="Mean satisfaction by sparse fraction and collection",
        fname="aggregate_bar_satisfaction.png",
        ylim=(0, 10),
    )
    path2 = _draw_grouped_bar(
        ssc_data,
        ylabel="Sparse-supported satisfaction",
        title="Sparse satisfaction component by sparse fraction and collection",
        fname="aggregate_bar_sparse_satisfaction_component.png",
        ylim=None,
    )
    return path1, path2


def aggregate_satisfaction_histograms(
    run_dir: str,
    all_critiques: List[Dict[str, Any]],
) -> str:
    """Satisfaction histogram per collection, pooled across all sparse fractions."""
    collections: List[str] = []
    seen_cols: set = set()
    for c in all_critiques:
        col = c.get("sparse_collection", "")
        if col not in seen_cols:
            seen_cols.add(col)
            collections.append(col)

    sat_by_col: Dict[str, List[float]] = {col: [] for col in collections}
    for c in all_critiques:
        col = c.get("sparse_collection", "")
        sat = _get_satisfaction(c)
        if sat >= 1:
            sat_by_col[col].append(sat)

    n_cols = len(collections)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bins = np.arange(0.5, 11.5, 1.0)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, col, color in zip(axes, collections, colors):
        vals = np.array(sat_by_col[col])
        ax.hist(vals, bins=bins, color=color, edgecolor="white", linewidth=0.6)
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Satisfaction")
        ax.set_title(col)
        n = len(vals)
        if n:
            ax.axvline(float(np.mean(vals)), color="black", linestyle="--",
                       linewidth=1.2, label=f"mean {np.mean(vals):.2f}")
            ax.legend(fontsize=8)
        ax.text(0.02, 0.97, f"n={n}", transform=ax.transAxes,
                va="top", ha="left", fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle("Satisfaction distribution by collection (all sparse fractions)")
    fig.tight_layout()

    path = os.path.join(run_dir, "report_assets", "aggregate_hist_satisfaction.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
