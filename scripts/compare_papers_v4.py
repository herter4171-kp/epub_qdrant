#!/usr/bin/env python3
"""
Cross-method retrieval comparison: dense-only (papers) vs hybrid dense+sparse (papers-named).

Runs the same queries against both collection pairs, computes metrics purely from
the results (no LLM judge), and generates scatter plots + JSON output.

Metrics:
  1. keyword_hit_rate  — fraction of top-K chunks where query terms appear verbatim
  2. section_diversity — unique section_titles in top-K / K
  3. citation_density  — fraction of chunks containing academic citations
  4. papers_coverage   — fraction of results from papers (vs books)
  5. score_spread      — max_score - min_score in top-K (confidence signal)
  6. token_efficiency  — avg token_count per chunk (content depth proxy)

Usage:
    python scripts/compare_papers_v4.py
    python scripts/compare_papers_v4.py --top-k 10 --output results/v4_comparison.json
"""

from __future__ import annotations

import json
import re
import sys
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Project path setup ────────────────────────────────────────────────────────
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from servers.mcp_server.retriever import Retriever, ChunkResult

# ── Queries (same 30 as evaluate.py) ──────────────────────────────────────────
QUERIES: List[Dict[str, Any]] = [
    {"query": "Help me design a planning module for an LLM agent that can break a long task into executable subtasks without overplanning.", "tags": ["cross", "planning"]},
    {"query": "What is the current best practice for giving an agent reasoning structure without forcing brittle chain-of-thought templates?", "tags": ["cross", "reasoning"]},
    {"query": "Show me how to build an agent that decides when to use a tool versus when to answer directly.", "tags": ["cross", "tool-use"]},
    {"query": "How should I architect multi-tool orchestration so an agent can chain search, code execution, and document drafting safely?", "tags": ["cross", "tool-use"]},
    {"query": "What scaffolding do strong terminal-based coding agents use around the model itself?", "tags": ["cross", "scaffolding"]},
    {"query": "What is a good memory design for an agent that needs short-term working memory and long-term user memory?", "tags": ["cross", "memory"]},
    {"query": "How do I stop an agent's memory from turning into a junk drawer full of redundant or low-value facts?", "tags": ["cross", "memory"]},
    {"query": "I want an agent that can localize bugs in a large repository. What retrieval and hypothesis-testing loop should I start with?", "tags": ["papers", "retrieval"]},
    {"query": "How would you benchmark an agent that performs refactors rather than one-off bug fixes?", "tags": ["papers", "evaluation"]},
    {"query": "I want an agent that can notice its own mistakes and retry with a different strategy. What self-correction loop should I use?", "tags": ["cross", "self-correction"]},
    {"query": "How do I make a coding agent ask clarifying questions only when needed instead of either guessing wildly or stopping constantly?", "tags": ["cross", "tool-use"]},
    {"query": "What should I measure to catch post-merge quality problems in agent-generated pull requests?", "tags": ["cross", "evaluation"]},
    {"query": "How would you build a GUI agent that can operate a clunky enterprise web app with inconsistent layouts?", "tags": ["papers", "gui"]},
    {"query": "What are the main failure modes of web agents in realistic browsing tasks, and how should I evaluate them?", "tags": ["papers", "evaluation"]},
    {"query": "I want a mobile agent that can carry out a multi-step task across apps. What architecture would make that robust?", "tags": ["cross", "mobile"]},
    {"query": "How should an enterprise agent handle permissions, approvals, and audit logs when acting on behalf of employees?", "tags": ["books", "enterprise"]},
    {"query": "What is the cleanest way to build an API agent that can discover schema details and recover from malformed tool responses?", "tags": ["cross", "api"]},
    {"query": "I need a data agent that can query a warehouse, validate the SQL, and then generate charts with commentary. How would you structure it?", "tags": ["cross", "data"]},
    {"query": "What would a serious research agent pipeline look like for literature review, source ranking, note synthesis, and citation tracking?", "tags": ["cross", "research"]},
    {"query": "How do deep research agents verify intermediate conclusions instead of just producing polished nonsense?", "tags": ["papers", "research"]},
    {"query": "I'm interested in agentic scientific simulation. How would an agent iteratively propose, run, and revise model configurations?", "tags": ["cross", "science"]},
    {"query": "What are the tradeoffs between a single AI scientist agent and a multi-agent scientific discovery pipeline?", "tags": ["cross", "science"]},
    {"query": "How should sub-agent creation work in a larger orchestration system so I do not end up with agent sprawl?", "tags": ["cross", "orchestration"]},
    {"query": "I want to experiment with multi-agent problem solving where agents debate, specialize, and then converge. What patterns are worth trying first?", "tags": ["cross", "multi-agent"]},
    {"query": "How do self-evolving agents accumulate reusable skills without drifting into unstable behavior?", "tags": ["papers", "self-evolution"]},
    {"query": "What safeguards are needed when agents can modify their own prompts, memories, or tool policies over time?", "tags": ["cross", "safety"]},
    {"query": "Help me think through safety for agents that can browse the web, run code, and call external APIs.", "tags": ["cross", "safety"]},
    {"query": "How does agent tuning look like in practice for a smaller open model that needs to behave more like a reliable operator?", "tags": ["papers", "tuning"]},
    {"query": "How should I evaluate agents in production beyond task success rate—latency, cost, recovery rate, human override rate, what else?", "tags": ["cross", "evaluation"]},
    {"query": "Can you sketch a benchmark for comparing a data-analysis agent against human analysts on realistic business tasks?", "tags": ["papers", "evaluation"]},
]

# ── Citation patterns ─────────────────────────────────────────────────────────
_CITATION_PATTERNS = [
    re.compile(r'\[\d[\d, ]*\]'),           # [1], [1, 2, 3]
    re.compile(r'\([A-Z][a-z]+(?:, [A-Z][a-z]+)*, \d{4}\)'),  # (Smith, 2024)
    re.compile(r'(?:et al\.|et al)\s*,\s*\d{4}'),  # et al., 2024
    re.compile(r'arXiv[:\s]*\d+\.\d{4,}'),  # arXiv:2603.07444
]

# ── Metric functions ──────────────────────────────────────────────────────────

def _extract_query_terms(query: str, min_len: int = 3) -> set:
    """Extract meaningful words from query (ignore stop words, short tokens)."""
    stop = {"how", "what", "when", "where", "who", "why", "the", "and", "for",
            "with", "that", "this", "have", "has", "had", "not", "but", "are",
            "been", "was", "were", "been", "being", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall", "an",
            "a", "in", "to", "of", "or", "on", "at", "by", "from", "as", "is",
            "it", "its", "i", "me", "my", "we", "our", "you", "your"}
    words = re.findall(r'[A-Za-z]+', query.lower())
    return {w for w in words if len(w) >= min_len and w not in stop}


def keyword_hit_rate(chunks: List[ChunkResult], query: str) -> float:
    """Fraction of top-K chunks where query terms appear verbatim in text."""
    if not chunks:
        return 0.0
    terms = _extract_query_terms(query)
    if not terms:
        return 1.0  # no meaningful terms to match
    hits = 0
    for c in chunks:
        text_lower = c.text.lower()
        if any(t in text_lower for t in terms):
            hits += 1
    return hits / len(chunks)


def section_diversity(chunks: List[ChunkResult]) -> float:
    """Unique section_titles / total chunks (1 = all different sections)."""
    if not chunks:
        return 0.0
    sections = set()
    for c in chunks:
        st = c.section_title or c.section or c.title or ""
        if st:
            sections.add(st)
    return len(sections) / len(chunks)


def citation_density(chunks: List[ChunkResult]) -> float:
    """Fraction of chunks containing academic citation markers."""
    if not chunks:
        return 0.0
    hits = 0
    for c in chunks:
        for pat in _CITATION_PATTERNS:
            if pat.search(c.text):
                hits += 1
                break
    return hits / len(chunks)


def papers_coverage(chunks: List[ChunkResult]) -> float:
    """Fraction of results from papers (doc_type=paper or has arxiv_id)."""
    if not chunks:
        return 0.0
    paper_count = sum(1 for c in chunks if c.arxiv_id or (c.doc_type == "paper"))
    return paper_count / len(chunks)


def score_spread(chunks: List[ChunkResult]) -> float:
    """Max score minus min score in top-K (higher = more confidence gradient)."""
    if len(chunks) < 2:
        return 0.0
    scores = [c.score for c in chunks]
    return max(scores) - min(scores)


def token_efficiency(chunks: List[ChunkResult]) -> float:
    """Average token count per chunk (proxy for content depth)."""
    if not chunks:
        return 0.0
    return sum(c.token_count for c in chunks) / len(chunks)


# ── Collection runner ─────────────────────────────────────────────────────────

def run_query_dense(
    retriever: Retriever,
    query: str,
    collections: List[str],
    top_k: int,
) -> List[ChunkResult]:
    """Dense-only retrieval: call _storage.search per collection, no grouping/expansion."""
    from servers.embedding_server.client import get_dense_vectors

    all_points: List[Tuple[float, Dict]] = []  # (score, payload)
    for col in collections:
        query_vec = get_dense_vectors([query])[0]
        raw = retriever._storage.search(col, query, top_k=top_k)
        for r in raw:
            all_points.append((r["score"], r))
    # Sort globally by score descending, take top_k
    all_points.sort(key=lambda x: -x[0])
    results = []
    for score, payload in all_points[:top_k]:
        results.append(ChunkResult(
            score=score,
            text=payload.get("text", ""),
            source_file=payload.get("source_file", ""),
            title=payload.get("title", "") or payload.get("book_title", "") or payload.get("section_title", ""),
            section=payload.get("section", "") or payload.get("section_title", ""),
            doc_type=payload.get("doc_type", ""),
            book_title=payload.get("book_title", ""),
            section_title=payload.get("section_title", ""),
            chapter_index=payload.get("chapter_index", 0),
            section_index=payload.get("section_index", 0),
            chunk_index=payload.get("chunk_index", 0),
            token_count=payload.get("token_count", 0),
            publisher=payload.get("publisher"),
            language=payload.get("language"),
            isbn=payload.get("isbn"),
            arxiv_id=payload.get("arxiv_id"),
            category=payload.get("category"),
            subcategory=payload.get("subcategory"),
            publish_date=payload.get("publish_date"),
            authors=payload.get("authors"),
            year=payload.get("year"),
        ))
    return results


def run_query_hybrid(
    retriever: Retriever,
    query: str,
    collections: List[str],
    top_k: int,
) -> List[ChunkResult]:
    """Hybrid retrieval: call hybrid_search per collection, combine, normalize, top_k."""
    all_points: List[Tuple[float, ChunkResult]] = []  # (rrf_score, ChunkResult)
    for col in collections:
        hybrid_results = retriever.hybrid_search(query, col, top_k=top_k)
        for cr in hybrid_results:
            all_points.append((cr.score, cr))

    if not all_points:
        return []

    # Normalize RRF scores to 0-1 range within combined results
    # (hybrid_search returns per-collection results; we need global normalization)
    max_score = max(s for s, _ in all_points)
    min_score = min(s for s, _ in all_points)
    score_range = max_score - min_score if max_score != min_score else 1.0

    results = []
    # Sort by raw RRF score desc, take top_k
    all_points.sort(key=lambda x: -x[0])
    for _raw_score, cr in all_points[:top_k]:
        normalized = (max_score - _raw_score) / score_range
        results.append(ChunkResult(
            score=normalized,
            text=cr.text,
            source_file=cr.source_file,
            title=cr.title,
            section=cr.section,
            doc_type=cr.doc_type,
            book_title=cr.book_title,
            section_title=cr.section_title,
            chapter_index=cr.chapter_index,
            section_index=cr.section_index,
            chunk_index=cr.chunk_index,
            token_count=cr.token_count,
            publisher=cr.publisher,
            language=cr.language,
            isbn=cr.isbn,
            arxiv_id=cr.arxiv_id,
            category=cr.category,
            subcategory=cr.subcategory,
            publish_date=cr.publish_date,
            authors=cr.authors,
            year=cr.year,
            point_id=cr.point_id,
        ))
    return results


def run_query(
    retriever: Retriever,
    query: str,
    collections: List[str],
    top_k: int,
    is_hybrid: bool = False,
) -> List[ChunkResult]:
    """Run retrieval — dispatches to dense or hybrid based on collection naming."""
    if is_hybrid:
        return run_query_hybrid(retriever, query, collections, top_k)
    return run_query_dense(retriever, query, collections, top_k)


def compute_all_metrics(
    chunks: List[ChunkResult],
    query: str,
) -> Dict[str, float]:
    """Compute all 6 metrics for a result set."""
    return {
        "keyword_hit_rate": round(keyword_hit_rate(chunks, query), 4),
        "section_diversity": round(section_diversity(chunks), 4),
        "citation_density": round(citation_density(chunks), 4),
        "papers_coverage": round(papers_coverage(chunks), 4),
        "score_spread": round(score_spread(chunks), 4),
        "token_efficiency": round(token_efficiency(chunks), 2),
    }


# ── Jaccard overlap (document-level) ─────────────────────────────────────────

def doc_ids(chunks: List[ChunkResult]) -> List[str]:
    """Extract stable document-level IDs."""
    ids = []
    for c in chunks:
        if c.arxiv_id:
            ids.append(f"paper:{c.arxiv_id}")
        elif c.book_title:
            ids.append(f"book:{c.book_title.strip().lower()[:80]}")
        else:
            ids.append(f"chunk:{hash(c.text[:100])}")
    return ids


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _try_import_matplotlib():
    """Return True if matplotlib is available, False otherwise."""
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def generate_plots(
    results: List[Dict],
    output_dir: Path,
    force: bool = False,
) -> List[str]:
    """Generate scatter plots and bar charts. Returns list of saved filenames."""
    if not _try_import_matplotlib() and not force:
        return []

    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as GridSpec

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec.GridSpec(3, 3, wspace=0.35, hspace=0.3)

    metrics = [
        ("keyword_hit_rate", "Keyword Hit Rate"),
        ("section_diversity", "Section Diversity"),
        ("citation_density", "Citation Density"),
        ("papers_coverage", "Papers Coverage"),
        ("score_spread", "Score Spread"),
        ("token_efficiency", "Token Efficiency"),
    ]

    saved_files = []

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        baseline_vals = []
        hybrid_vals = []
        for r in results:
            b = r["baseline"].get(metric_key, 0)
            h = r["hybrid"].get(metric_key, 0)
            baseline_vals.append(b)
            hybrid_vals.append(h)

        # Limits for diagonal
        all_vals = baseline_vals + hybrid_vals
        lo = min(min(all_vals), 0) if all_vals else 0
        hi = max(max(all_vals), 1) if all_vals else 1
        if metric_key == "token_efficiency":
            hi = hi * 1.1

        ax.scatter(baseline_vals, hybrid_vals, c="#2563eb", alpha=0.6, s=60, zorder=3)
        ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5, label="y=x")
        ax.set_xlabel(f"Dense-only ({metric_label})", fontsize=9)
        ax.set_ylabel(f"Hybrid ({metric_label})", fontsize=9)
        ax.set_title(f"Baseline vs Hybrid: {metric_label}", fontsize=10)
        ax.axhline(y=0, color="k", linewidth=0.3)
        ax.axvline(x=0, color="k", linewidth=0.3)
        ax.legend(fontsize=7)

    # Bar chart: aggregate means
    ax_bar = fig.add_subplot(gs[2, 1:])
    metric_labels_short = [m[1] for m in metrics[:4]]  # first 4 for bar chart
    baseline_means = []
    hybrid_means = []
    for mk in metric_labels_short:
        b_vals = [r["baseline"].get(mk, 0) for r in results]
        h_vals = [r["hybrid"].get(mk, 0) for r in results]
        baseline_means.append(sum(b_vals) / len(b_vals) if b_vals else 0)
        hybrid_means.append(sum(h_vals) / len(h_vals) if h_vals else 0)

    x = range(len(metric_labels_short))
    w = 0.35
    ax_bar.bar([i - w/2 for i in x], baseline_means, w, label="Dense-only", color="#2563eb", alpha=0.8)
    ax_bar.bar([i + w/2 for i in x], hybrid_means, w, label="Hybrid", color="#dc2626", alpha=0.8)
    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels(metric_labels_short, fontsize=8)
    ax_bar.set_title("Aggregate Means (all 30 queries)", fontsize=10)
    ax_bar.legend()

    plt.suptitle("Dense-Only vs Hybrid Retrieval: Cross-Collection Comparison", fontsize=14, y=1.01)
    out_path = output_dir / "comparison_v4_plots.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(str(out_path))

    # Second figure: per-query keyword hit rate bar chart
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    query_labels = []
    b_khr = []
    h_khr = []
    for i, r in enumerate(results):
        q = r["query"][:45] + "..." if len(q) > 48 else r["query"]
        query_labels.append(q)
        b_khr.append(r["baseline"].get("keyword_hit_rate", 0))
        h_khr.append(r["hybrid"].get("keyword_hit_rate", 0))

    x2 = range(len(query_labels))
    w2 = 0.2
    ax2.bar([i - w2 for i in x2], b_khr, w2, label="Dense-only", color="#2563eb", alpha=0.8)
    ax2.bar([i + w2 for i in x2], h_khr, w2, label="Hybrid", color="#dc2626", alpha=0.8)
    ax2.set_xticks(list(x2))
    ax2.set_xticklabels(query_labels, rotation=45, ha="right", fontsize=6)
    ax2.set_ylabel("Keyword Hit Rate")
    ax2.set_title("Per-Query: Keyword Hit Rate (Dense vs Hybrid)", fontsize=11)
    ax2.legend()
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    out_path2 = output_dir / "comparison_v4_keyword_hitrate.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    saved_files.append(str(out_path2))

    return saved_files


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10, help="Results per query (default: 10)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--plots", action="store_true", help="Generate plots (requires matplotlib)")
    args = parser.parse_args()

    top_k = args.top_k
    retriever = Retriever()

    dense_collections = ["books", "papers"]
    hybrid_collections = ["books-named", "papers-named"]

    out_path = Path(args.output) if args.output else _project_root / "results" / "comparison_v4.json"
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"\n{'Query':<60} | {'KHR':>5} {'SDiv':>5} {'Cite':>5} {'Paper':>5} {'Spd':>7} | {'KHR':>5} {'SDiv':>5} {'Cite':>5} {'Paper':>5} {'Spd':>7}")
    print("-" * 115)

    for qi, qd in enumerate(QUERIES):
        query = qd["query"]

        dense_chunks = run_query(retriever, query, dense_collections, top_k, is_hybrid=False)
        hybrid_chunks = run_query(retriever, query, hybrid_collections, top_k, is_hybrid=True)

        b_metrics = compute_all_metrics(dense_chunks, query)
        h_metrics = compute_all_metrics(hybrid_chunks, query)

        b_docs = doc_ids(dense_chunks)
        h_docs = doc_ids(hybrid_chunks)
        overlap = jaccard(b_docs, h_docs)

        entry = {
            "query": query,
            "tags": qd.get("tags", []),
            "top_k": top_k,
            "baseline": {**b_metrics, "doc_ids": b_docs, "n_chunks": len(dense_chunks)},
            "hybrid": {**h_metrics, "doc_ids": h_docs, "n_chunks": len(hybrid_chunks)},
            "jaccard_overlap": round(overlap, 4),
        }
        results.append(entry)

        short_q = query[:58] + "..." if len(query) > 61 else query
        print(f"{short_q:<58} "
              f"{b_metrics['keyword_hit_rate']:>6.2f} {b_metrics['section_diversity']:>6.2f} "
              f"{b_metrics['citation_density']:>6.2f} {b_metrics['papers_coverage']:>6.2f} "
              f"{b_metrics['score_spread']:>8.4f}  "
              f"{h_metrics['keyword_hit_rate']:>6.2f} {h_metrics['section_diversity']:>6.2f} "
              f"{h_metrics['citation_density']:>6.2f} {h_metrics['papers_coverage']:>6.2f} "
              f"{h_metrics['score_spread']:>8.4f}")

    # ── Aggregates ────────────────────────────────────────────────────────
    def _mean(dicts: List[Dict], key: str) -> float:
        vals = [d.get(key, 0) for d in dicts]
        return sum(vals) / len(vals) if vals else 0

    b_all = [r["baseline"] for r in results]
    h_all = [r["hybrid"] for r in results]

    aggregates = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "num_queries": len(QUERIES),
        "top_k": top_k,
        "dense_collections": dense_collections,
        "hybrid_collections": hybrid_collections,
        "baseline": {},
        "hybrid": {},
    }

    for mk in ["keyword_hit_rate", "section_diversity", "citation_density",
               "papers_coverage", "score_spread", "token_efficiency"]:
        aggregates["baseline"][mk] = round(_mean(b_all, mk), 4)
        aggregates["hybrid"][mk] = round(_mean(h_all, mk), 4)

    # Improvement direction: positive means hybrid is better
    improvements = {}
    for mk in ["keyword_hit_rate", "citation_density", "papers_coverage"]:
        improvements[mk] = round(_mean(h_all, mk) - _mean(b_all, mk), 4)
    for mk in ["section_diversity", "score_spread", "token_efficiency"]:
        improvements[mk] = round(_mean(h_all, mk) - _mean(b_all, mk), 4)
    aggregates["improvement"] = improvements

    avg_overlap = _mean([r["jaccard_overlap"] for r in results], "jaccard_overlap")
    aggregates["avg_jaccard_overlap"] = round(avg_overlap, 4)

    high_overlap = sum(1 for r in results if r["jaccard_overlap"] >= 0.7)
    mid_overlap = sum(1 for r in results if 0.3 <= r["jaccard_overlap"] < 0.7)
    low_overlap = sum(1 for r in results if r["jaccard_overlap"] < 0.3)
    aggregates["overlap_distribution"] = {
        "high_ge_70": high_overlap,
        "mid_30_to_70": mid_overlap,
        "low_lt_30": low_overlap,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "version": "1.0",
        "description": "Dense-only (papers) vs Hybrid dense+sparse (papers-named) comparison. No LLM judge — all metrics computed from retrieval results.",
        **aggregates,
        "per_query": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  AGGREGATE SUMMARY")
    print("=" * 80)
    for mk in ["keyword_hit_rate", "section_diversity", "citation_density",
               "papers_coverage", "score_spread", "token_efficiency"]:
        b = aggregates["baseline"][mk]
        h = aggregates["hybrid"][mk]
        imp = improvements[mk]
        arrow = "+" if imp > 0 else ("-" if imp < 0 else "=")
        print(f"  {mk:<22}  baseline={b:.4f}  hybrid={h:.4f}  delta={arrow}{imp:.4f}")
    print(f"\n  Avg Jaccard Overlap: {avg_overlap:.3f}")
    print(f"  Overlap distribution: high(>=70%)={high_overlap}  mid(30-70%)={mid_overlap}  low(<30%)={low_overlap}")
    print("=" * 80)

    # ── Generate plots ────────────────────────────────────────────────────
    if args.plots:
        print("\nGenerating plots...")
        plots = generate_plots(results, out_dir)
        if plots:
            for p in plots:
                print(f"  Saved: {p}")
        else:
            print("  matplotlib not installed. Install with: pip install matplotlib")
            print("  Or run with --plots-force to generate PNGs anyway.")


if __name__ == "__main__":
    main()