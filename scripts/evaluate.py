#!/usr/bin/env python3
"""Phase-agnostic evaluation harness for retrieval quality.

Uses LLM-as-judge pairwise comparison against a baseline method.
Generates per-query metrics and aggregate win-rate statistics.

Results are written to ``results.json`` in the project root, accumulating
across runs so you can track score evolution by prompt.

Usage:
    python scripts/evaluate.py baseline
    python scripts/evaluate.py phase_0
    python scripts/evaluate.py phase_2_hybrid
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ─── Project imports ───────────────────────────────────────────────────────

# Ensure project root and MCP server paths are on sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from servers.mcp_server.retriever import Retriever, ChunkResult
from servers.mcp_server.config import settings
from servers.mcp_server.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate")

# ─── Query Set (30 agent-architecture-focused queries) ─────────────────────

QUERIES: List[Dict[str, Any]] = [
    # Planning & task decomposition (5)
    {"query": "Help me design a planning module for an LLM agent that can break a long task into executable subtasks without overplanning.", "tags": ["cross", "planning"]},
    #{"query": "What is the current best practice for giving an agent reasoning structure without forcing brittle chain-of-thought templates?", "tags": ["cross", "reasoning"]},
    #{"query": "Show me how to build an agent that decides when to use a tool versus when to answer directly.", "tags": ["cross", "tool-use"]},
    #{"query": "How should I architect multi-tool orchestration so an agent can chain search, code execution, and document drafting safely?", "tags": ["cross", "tool-use"]},
    #{"query": "What scaffolding do strong terminal-based coding agents use around the model itself?", "tags": ["cross", "scaffolding"]},

    # Memory design (4)
    {"query": "What is a good memory design for an agent that needs short-term working memory and long-term user memory?", "tags": ["cross", "memory"]},
    #{"query": "How do I stop an agent's memory from turning into a junk drawer full of redundant or low-value facts?", "tags": ["cross", "memory"]},
    #{"query": "I want an agent that can localize bugs in a large repository. What retrieval and hypothesis-testing loop should I start with?", "tags": ["papers", "retrieval"]},
    #{"query": "How would you benchmark an agent that performs refactors rather than one-off bug fixes?", "tags": ["papers", "evaluation"]},

    # Self-correction & evaluation (4)
    {"query": "I want an agent that can notice its own mistakes and retry with a different strategy. What self-correction loop should I use?", "tags": ["cross", "self-correction"]},
    #{"query": "How do I make a coding agent ask clarifying questions only when needed instead of either guessing wildly or stopping constantly?", "tags": ["cross", "tool-use"]},
    #{"query": "What should I measure to catch post-merge quality problems in agent-generated pull requests?", "tags": ["cross", "evaluation"]},
    #{"query": "How would you build a GUI agent that can operate a clunky enterprise web app with inconsistent layouts?", "tags": ["papers", "gui"]},

    # Domain-specific agents (6)
    {"query": "What are the main failure modes of web agents in realistic browsing tasks, and how should I evaluate them?", "tags": ["papers", "evaluation"]},
    #{"query": "I want a mobile agent that can carry out a multi-step task across apps. What architecture would make that robust?", "tags": ["cross", "mobile"]},
    #{"query": "How should an enterprise agent handle permissions, approvals, and audit logs when acting on behalf of employees?", "tags": ["books", "enterprise"]},
    #{"query": "What is the cleanest way to build an API agent that can discover schema details and recover from malformed tool responses?", "tags": ["cross", "api"]},
    #{"query": "I need a data agent that can query a warehouse, validate the SQL, and then generate charts with commentary. How would you structure it?", "tags": ["cross", "data"]},
    #{"query": "What would a serious research agent pipeline look like for literature review, source ranking, note synthesis, and citation tracking?", "tags": ["cross", "research"]},

    # Deep research & scientific (4)
    {"query": "How do deep research agents verify intermediate conclusions instead of just producing polished nonsense?", "tags": ["papers", "research"]},
    #{"query": "I'm interested in agentic scientific simulation. How would an agent iteratively propose, run, and revise model configurations?", "tags": ["cross", "science"]},
    #{"query": "What are the tradeoffs between a single AI scientist agent and a multi-agent scientific discovery pipeline?", "tags": ["cross", "science"]},
    #{"query": "How should sub-agent creation work in a larger orchestration system so I do not end up with agent sprawl?", "tags": ["cross", "orchestration"]},

    # Multi-agent & self-evolution (4)
    {"query": "I want to experiment with multi-agent problem solving where agents debate, specialize, and then converge. What patterns are worth trying first?", "tags": ["cross", "multi-agent"]},
    #{"query": "How do self-evolving agents accumulate reusable skills without drifting into unstable behavior?", "tags": ["papers", "self-evolution"]},
    #{"query": "What safeguards are needed when agents can modify their own prompts, memories, or tool policies over time?", "tags": ["cross", "safety"]},
    #{"query": "Help me think through safety for agents that can browse the web, run code, and call external APIs.", "tags": ["cross", "safety"]},

    # Production & tuning (3)
    {"query": "How does agent tuning look like in practice for a smaller open model that needs to behave more like a reliable operator?", "tags": ["papers", "tuning"]},
    #{"query": "How should I evaluate agents in production beyond task success rate—latency, cost, recovery rate, human override rate, what else?", "tags": ["cross", "evaluation"]},
    #{"query": "Can you sketch a benchmark for comparing a data-analysis agent against human analysts on realistic business tasks?", "tags": ["papers", "evaluation"]},
]

# ─── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Result for a single query under a single method."""
    query: str
    method: str
    chunks: List[ChunkResult]
    top5_avg_score: float = 0.0
    top1_category: str = ""
    cross_collection_ratio: float = 0.0
    has_metadata_match: bool = False
    judge_winner: Optional[str] = None
    judge_reason: str = ""


@dataclass
class EvaluationResult:
    """Complete evaluation output."""
    version: str = "1.0"
    model: str = ""
    evaluated_at: str = ""
    baseline_method: str = ""
    num_queries: int = 0
    per_query_scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aggregate: Dict[str, Any] = field(default_factory=dict)


# ─── Metric Computation ────────────────────────────────────────────────────

def compute_metrics(chunks: List[ChunkResult]) -> Dict[str, Any]:
    """Compute per-query metrics from a list of ChunkResults."""
    if not chunks:
        return {
            "top5_avg_score": 0.0,
            "top1_category": "",
            "cross_collection_ratio": 0.0,
            "has_metadata_match": False,
        }

    top5 = chunks[:5]
    scores = [c.score for c in top5]
    top5_avg = sum(scores) / len(scores) if scores else 0.0

    # Most common category (case-insensitive)
    categories: Dict[str, int] = {}
    collection_counts: Dict[str, int] = {}
    for c in chunks:
        cat = (c.category or c.subcategory or c.publisher or "").strip().lower()
        if cat:
            categories[cat] = categories.get(cat, 0) + 1
        # Track per-collection
        doc = c.doc_type or ""
        if c.book_title:
            doc = "book"
        elif c.arxiv_id:
            doc = "paper"
        collection_counts[doc] = collection_counts.get(doc, 0) + 1

    top1_cat = max(categories, key=categories.get) if categories else ""

    # Cross-collection ratio: how balanced is the result set
    if len(collection_counts) <= 1:
        cross_ratio = 0.0
    else:
        total = sum(collection_counts.values())
        # 1 - (max_fraction - min_fraction) → 1 = perfectly balanced, 0 = all from one
        fractions = [v / total for v in collection_counts.values()]
        cross_ratio = 1.0 - (max(fractions) - min(fractions))

    # Metadata match: any chunk whose metadata fields appear in query terms
    return {
        "top5_avg_score": round(top5_avg, 4),
        "top1_category": top1_cat,
        "cross_collection_ratio": round(cross_ratio, 4),
        "has_metadata_match": False,  # computed externally
    }


# ─── Retriever Wrapper ─────────────────────────────────────────────────────

class RetrieverRunner:
    """Runs retrieval for a given method name and extracts metrics."""

    def __init__(self):
        self._retriever: Optional[Retriever] = None

    def _get_retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever

    def run(
        self,
        query: str,
        method: str = "baseline",
    ) -> QueryResult:
        """Run retrieval for a query and return a QueryResult.

        For phase_2_hybrid: targets books-named/papers-named collections
        which have dense+sparse named vectors and use RRF fusion.
        """
        retriever = self._get_retriever()

        # Baseline explicitly targets original collections (dense-only + z-score)
        # Hybrid explicitly targets -named collections (dense+sparse + RRF)
        # This ensures they use different code paths.
        collections = None
        if method == "phase_2_hybrid":
            collections = ["books-named", "papers-named"]
        elif method == "semantic":
            collections = ["books-semantic", "papers"]
        elif method == "baseline":
            collections = ["books", "papers"]

        # Use cross-collection search for full coverage
        bundle = retriever.search_collections(
            query=query,
            top_k=20,  # grab more than 5 to compute meaningful metrics
            collections=collections,
        )

        # Flatten groups into a single list, sorted by score
        all_chunks: List[ChunkResult] = []
        for g in bundle.groups:
            for c in g.results:
                all_chunks.append(c)

        all_chunks.sort(key=lambda x: x.score, reverse=True)

        metrics = compute_metrics(all_chunks)

        return QueryResult(
            query=query,
            method=method,
            chunks=all_chunks,
            top5_avg_score=metrics["top5_avg_score"],
            top1_category=metrics["top1_category"],
            cross_collection_ratio=metrics["cross_collection_ratio"],
            has_metadata_match=metrics["has_metadata_match"],
        )


# ─── LLM Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """Compares two sets of results via LLM pairwise judgment."""

    def __init__(self, model: str = "qwen36"):
        import os
        # Prefer OPENAI_API_BASE from .env if LITELLM_API_URL is the default (unconfigured)
        litellm_url = settings.LITELLM_API_URL
        openai_base = os.getenv("OPENAI_API_BASE")
        if openai_base:
            self._api_url = openai_base.rstrip("/")
        else:
            self._api_url = litellm_url
        self._api_key = os.getenv("LITELLM_API_KEY") or settings.LITELLM_API_KEY
        # When using an OpenAI-compatible endpoint, LiteLLM needs a provider prefix.
        # If the endpoint is our local OpenAI-compatible server, prefix with "openai/"
        if openai_base and not model.startswith(("openai/", "azure/", "anthropic/", "google/", "groq/")):
            self._model = f"openai/{model}"
        else:
            self._model = model

    async def judge(
        self,
        query: str,
        baseline_chunks: List[ChunkResult],
        new_chunks: List[ChunkResult],
    ) -> Dict[str, str]:
        """Ask the LLM which result set is better.

        Returns {"winner": "baseline"|"new"|"tie", "reason": "..."}.
        """
        def format_context(results: List[ChunkResult], prefix: str) -> str:
            lines = []
            for i, r in enumerate(results[:5], 1):
                meta_parts = []
                if r.category:
                    meta_parts.append(f"cat:{r.category}")
                if r.publisher:
                    meta_parts.append(f"pub:{r.publisher}")
                if r.arxiv_id:
                    meta_parts.append(f"arxiv:{r.arxiv_id}")
                if r.book_title:
                    meta_parts.append(f"book:{r.book_title}")
                meta = " ".join(meta_parts)
                lines.append(
                    f"{prefix} {i}. [score:{r.score:.3f}] "
                    f"{r.text[:200]}... [{meta}]"
                )
            return "\n".join(lines) if lines else "(no results)"

        prompt = f"""You are a retrieval quality judge. Given a query and two result sets,
decide which is more relevant and useful.

Query: {query}

Result Set A (baseline — dense semantic + z-score normalization):
{format_context(baseline_chunks, "A")}

Result Set B (hybrid — dense semantic + sparse keyword + RRF fusion):
{format_context(new_chunks, "B")}

Which result set is more relevant to the query? Consider:
- Semantic relevance of results to the query
- Quality and informativeness of retrieved text
- Diversity across sources/collections
- Metadata alignment (e.g. if query mentions a publisher, do results match?)

Respond as JSON only: {{"winner": "A"|"B"|"tie", "reason": "brief explanation"}}"""

        try:
            client = LLMClient(
                api_url=self._api_url,
                api_key=self._api_key,
                model=self._model,
            )
            answer = await client.answer(query=query, context=prompt)
        except Exception as e:
            logger.error(f"LLM judge failed for '{query}': {e}")
            return {"winner": "tie", "reason": f"judge_error: {e}"}

        # Parse JSON from LLM response
        import re
        json_match = re.search(r"\{[^{}]*" + '"winner"' + r"[^{}]*\}", answer, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {"winner": "tie", "reason": f"could not parse LLM response: {answer[:100]}"}


# ─── Evaluation Logic ──────────────────────────────────────────────────────

async def evaluate_phase(
    runner: RetrieverRunner,
    judge: LLMJudge,
    phase_name: str,
    baseline_name: str = "baseline",
    baseline_results: Optional[Dict[str, QueryResult]] = None,
) -> EvaluationResult:
    """Run the full evaluation for one phase.

    If baseline_results is None, uses the current implementation as baseline
    (runs baseline first, then the new method).
    """
    logger.info(f"Evaluating phase: {phase_name}")
    start = time.time()

    er = EvaluationResult()
    er.version = "1.0"
    er.model = judge._model
    er.evaluated_at = datetime.now(timezone.utc).isoformat()
    er.num_queries = len(QUERIES)

    # Accumulators
    wins: Dict[str, int] = {baseline_name: 0, phase_name: 0}
    ties = 0

    for i, q in enumerate(QUERIES):
        query = q["query"]
        logger.info(f"[{i+1}/{len(QUERIES)}] Query: {query!r}")

        # Get baseline result
        if baseline_results and query in baseline_results:
            baseline_qr = baseline_results[query]
        else:
            baseline_qr = runner.run(query, method=baseline_name)

        # Get new method result
        new_qr = runner.run(query, method=phase_name)

        # LLM judge
        judgment = await judge.judge(query, baseline_qr.chunks, new_qr.chunks)
        winner = judgment.get("winner", "tie")

        # Map judge's "A"/"B" result set labels to our method names.
        # The judge prompt labels sets as "A (baseline)" and "B (new method)",
        # so "A" → baseline, "B" → new method.
        if winner in ("A", baseline_name):
            wins[baseline_name] += 1
        elif winner in ("B", phase_name):
            wins[phase_name] += 1
        else:
            ties += 1

        # Store per-query scores
        er.per_query_scores[query] = {
            baseline_name: {
                "top5_avg_score": baseline_qr.top5_avg_score,
                "top1_category": baseline_qr.top1_category,
                "cross_collection_ratio": baseline_qr.cross_collection_ratio,
                "has_metadata_match": baseline_qr.has_metadata_match,
            },
            phase_name: {
                "top5_avg_score": new_qr.top5_avg_score,
                "top1_category": new_qr.top1_category,
                "cross_collection_ratio": new_qr.cross_collection_ratio,
                "has_metadata_match": new_qr.has_metadata_match,
                "judge_wins_against_baseline": winner == phase_name,
                "judge_reason": judgment.get("reason", ""),
            },
            "judgment": {
                "winner": winner,
                "reason": judgment.get("reason", ""),
            },
        }

    elapsed = time.time() - start
    total = wins[baseline_name] + wins[phase_name] + ties
    win_rate = wins[phase_name] / total if total > 0 else 0.0

    # Compute average score improvements
    score_improvements = []
    cross_improvements = []
    for query_data in er.per_query_scores.values():
        baseline_score = query_data.get(baseline_name, {}).get("top5_avg_score", 0)
        new_score = query_data.get(phase_name, {}).get("top5_avg_score", 0)
        score_improvements.append(new_score - baseline_score)
        baseline_cross = query_data.get(baseline_name, {}).get("cross_collection_ratio", 0)
        new_cross = query_data.get(phase_name, {}).get("cross_collection_ratio", 0)
        cross_improvements.append(new_cross - baseline_cross)

    avg_score_imp = sum(score_improvements) / len(score_improvements) if score_improvements else 0
    avg_cross_imp = sum(cross_improvements) / len(cross_improvements) if cross_improvements else 0

    er.aggregate[phase_name] = {
        "wins_against_baseline": wins[phase_name],
        "losses_against_baseline": wins[baseline_name],
        "ties": ties,
        "win_rate": round(win_rate, 4),
        "total_comparisons": total,
        "avg_score_improvement": round(avg_score_imp, 4),
        "avg_cross_collection_improvement": round(avg_cross_imp, 4),
        "elapsed_seconds": round(elapsed, 1),
    }

    logger.info(
        f"Phase {phase_name} complete: wins={wins[phase_name]}, "
        f"losses={wins[baseline_name]}, ties={ties}, "
        f"win_rate={win_rate:.2%}, time={elapsed:.1f}s"
    )

    return er


def load_previous_results(path: Path) -> Optional[EvaluationResult]:
    """Load existing results.json if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        er = EvaluationResult()
        er.version = data.get("version", "1.0")
        er.model = data.get("model", "")
        er.evaluated_at = data.get("evaluated_at", "")
        er.baseline_method = data.get("baseline_method", "")
        er.num_queries = data.get("num_queries", 0)
        er.per_query_scores = data.get("per_query_scores", {})
        er.aggregate = data.get("aggregate", {})
        return er
    except Exception as e:
        logger.warning(f"Could not load existing results: {e}")
        return None


def save_results(er: EvaluationResult, path: Path) -> None:
    """Save (or append) evaluation results to JSON."""
    with open(path, "w") as f:
        json.dump(vars(er), f, indent=2, default=str)
    logger.info(f"Results saved to {path}")


# ─── Main ──────────────────────────────────────────────────────────────────

async def run_evaluation(
    phase_name: str,
    baseline_name: str = "baseline",
    output_path: Optional[str] = None,
) -> EvaluationResult:
    """Run a full evaluation phase and return results."""
    runner = RetrieverRunner()
    judge = LLMJudge()

    # Load previous results if any
    path = Path(output_path) if output_path else _project_root / "results.json"
    previous = load_previous_results(path)

    # If we have previous results and this isn't the baseline, use them as baseline
    baseline_results: Optional[Dict[str, QueryResult]] = None
    if previous and baseline_name in previous.aggregate:
        baseline_name = baseline_name
        logger.info(f"Using previous aggregate: {baseline_name}")

    # Run evaluation
    er = await evaluate_phase(runner, judge, phase_name, baseline_name, baseline_results)

    # Merge with previous results: keep old aggregate data (previous phases),
    # but ONLY for the current phase+baseline keys — don't overwrite with stale data.
    # previous.per_query_scores contains OLD baseline+OLD phase data that we don't want.
    # We only want previous aggregate entries for OTHER phases.
    if previous:
        for k, v in previous.aggregate.items():
            if k != baseline_name and k != phase_name:
                er.aggregate[k] = v
        # Do NOT merge per_query_scores — those are all fresh from this run

    # Set baseline method name
    er.baseline_method = baseline_name

    # Save
    save_results(er, path)

    return er


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument(
        "phase",
        nargs="?",
        default="baseline",
        help="Phase name (e.g. baseline, phase_0, phase_2_hybrid)",
    )
    parser.add_argument(
        "--baseline",
        default="baseline",
        help="Baseline phase name to compare against",
    )
    parser.add_argument(
        "--output",
        default=str(_project_root / "results.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.phase, args.baseline, args.output))


if __name__ == "__main__":
    main()