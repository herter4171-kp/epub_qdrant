#!/usr/bin/env python3
"""Single-book sparse/dense weight sweep.

Finds the optimal sparse weight for hybrid search on one EPUB book,
using that book's actual chunk content to drive query generation and
relevance evaluation.

Workflow:
  1. Pick one EPUB (default: masteringretrieval-augmentedgeneration.epub)
  2. Fetch all its chunks from Qdrant (books-named, filtered by source_file)
  3. Generate 10-20 natural-language queries from chunk snippets via LLM
  4. Fetch dense + sparse rank lists per query with source_file filter
  5. LLM-score each (query, chunk) pair for relevance (1-3)
  6. Sweep sparse_weight 0.0 -> 2.0 in 0.25 steps via RRF simulation
  7. Report avg_relevance@k per weight -> tells you where to set the multiplier

Run:
    python3 scripts/sweep_single_book.py
    python3 scripts/sweep_single_book.py --book "masteringretrieval-augmentedgeneration.epub" --top-k 5 --num-queries 15
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_mcp_server_dir = _project_root / "mcp_servers" / "retrieval"
if str(_mcp_server_dir) not in sys.path:
    sys.path.insert(0, str(_mcp_server_dir))

from mcp_server.retriever import Retriever
from mcp_server.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sweep_single_book")

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BOOK = "masteringretrieval-augmentedgeneration.epub"
HYBRID_COLLECTION = "books-named"
SPARSE_WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0]
K_RRF = 60
SCORE_BATCH_SIZE = 5
QUERY_GEN_BATCHES = 2
QUERY_PER_BATCH = 2


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A chunk from the target book."""
    point_id: str
    text: str
    source_file: str
    title: str
    section_title: str = ""
    chunk_index: int = 0


@dataclass
class RankedChunk:
    """A chunk with its dense rank, sparse rank, and relevance score."""
    point_id: str
    doc_id: str
    text: str
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    relevance_score: Optional[int] = None


@dataclass
class QueryData:
    query: str
    chunks: Dict[str, RankedChunk] = field(default_factory=dict)


# ── Step 1: Fetch chunks from Qdrant ─────────────────────────────────────────

def fetch_book_chunks(
    book_filename: str,
    collection: str = HYBRID_COLLECTION,
    top_k: int = 2000,
) -> List[Chunk]:
    """Fetch all chunks for a single book from Qdrant, filtered by source_file.

    Note: This version of qdrant-client's scroll() does not accept query_filter,
    so we filter by source_file in Python after fetching.
    """
    retriever = Retriever(collection=collection)
    client = retriever._client

    chunks = []
    offset = None
    batch_limit = 256

    while True:
        points = client.scroll(
            collection_name=collection,
            limit=batch_limit,
            offset=offset,
            with_payload=["source_file", "title", "section_title", "chunk_index", "text",
                          "book_title", "doc_type", "publisher"],
            with_vectors=False,
        )
        batch = points[0]
        if not batch:
            break

        for hit in batch:
            payload = hit.payload or {}
            source = payload.get("source_file", "")
            if source != book_filename:
                continue
            chunks.append(Chunk(
                point_id=str(hit.id),
                text=payload.get("text", ""),
                source_file=source,
                title=payload.get("title", "") or payload.get("book_title", ""),
                section_title=payload.get("section_title", ""),
                chunk_index=payload.get("chunk_index", 0),
            ))

        next_offset = points[1]
        if next_offset is None or len(batch) == 0:
            break
        offset = next_offset

        if len(chunks) >= top_k:
            break

    logger.info(f"Fetched {len(chunks)} chunks for {book_filename} (from {points[1] if points[1] else 'end'} offset)")
    return chunks


# ── Step 2: Generate queries from chunk snippets ─────────────────────────────

def select_representative_chunks(chunks: List[Chunk], num_batches: int = QUERY_GEN_BATCHES) -> List[str]:
    """Select representative chunks spread across the book's chunk index range."""
    if len(chunks) <= num_batches:
        return [c.text[:800] for c in chunks]
    step = max(1, len(chunks) // num_batches)
    selected = []
    for i in range(0, len(chunks), step):
        chunk = chunks[i]
        if chunk.text.strip():
            selected.append(chunk.text[:800])
    return selected[:num_batches]


QUERY_GENERATION_PROMPT = """You are generating retrieval queries for a book about {book_topic}.
Generate {num} diverse natural-language queries that a user would ask to retrieve this content.

Requirements:
- Mix query types: some should favor sparse/keyword matching, some dense/semantic, some hybrid
- Use natural phrasing (not keyword strings)
- Vary specificity: some very specific, some broad
- Do NOT include any instructions or meta-text — only the queries

Reply with ONLY a JSON array of strings, e.g.:
["query 1", "query 2", "query 3"]

Snippet:
{snippet}"""


async def generate_queries(
    llm: LLMClient,
    chunks: List[Chunk],
    book_topic: str = "retrieval-augmented generation (RAG), embedding models, and LLM applications",
    num_batches: int = QUERY_GEN_BATCHES,
    queries_per_batch: int = QUERY_PER_BATCH,
) -> List[str]:
    """Generate queries from representative chunk snippets."""
    snippet_batches = select_representative_chunks(chunks, num_batches)
    all_queries: set = set()

    logger.info(f"Generating {num_batches * queries_per_batch} queries from {len(snippet_batches)} chunk snippets...")

    for i, snippet in enumerate(snippet_batches):
        prompt = QUERY_GENERATION_PROMPT.format(
            book_topic=book_topic,
            num=queries_per_batch,
            snippet=snippet,
        )
        try:
            response = await llm.answer(query="generate queries", context=prompt)
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                for q in parsed:
                    if isinstance(q, str) and len(q.strip()) > 10:
                        all_queries.add(q.strip())
                logger.info(f"  Batch {i+1}/{len(snippet_batches)}: generated {len(parsed)} queries")
            else:
                logger.warning(f"  Batch {i+1}: no JSON found in response (len={len(response)})")
        except Exception as e:
            logger.error(f"  Batch {i+1}: LLM query generation failed: {e}")

    queries = sorted(all_queries)
    logger.info(f"Total unique queries: {len(queries)}")
    return queries


# ── Step 3: Fetch dense + sparse rank lists ──────────────────────────────────

def fetch_rank_lists(
    retriever: Retriever,
    query: str,
    book_filename: str,
    top_k: int,
) -> QueryData:
    """Fetch dense and sparse rank lists for a query, filtered to the target book."""
    from qdrant_client.models import SparseVector, FieldCondition, MatchValue, Filter

    qdata = QueryData(query=query)
    client = retriever._client

    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="source_file",
                match=MatchValue(value=book_filename),
            )
        ]
    )

    dense_vector = retriever._embed(query)
    sparse_query = retriever._embed_sparse(query)

    for collection in [HYBRID_COLLECTION]:
        # Dense-only search
        try:
            dense_hits = client.query_points(
                collection_name=collection,
                query=dense_vector,
                using="dense",
                limit=top_k,
                query_filter=qdrant_filter,
            )
            for rank, hit in enumerate(dense_hits.points):
                pid = str(hit.id)
                payload = hit.payload or {}
                if pid not in qdata.chunks:
                    qdata.chunks[pid] = RankedChunk(
                        point_id=pid,
                        doc_id=payload.get("title", payload.get("book_title", "")),
                        text=payload.get("text", ""),
                    )
                if qdata.chunks[pid].dense_rank is None:
                    qdata.chunks[pid].dense_rank = rank
        except Exception as e:
            logger.warning(f"Dense search failed: {e}")

        # Sparse-only search
        try:
            sparse_hits = client.query_points(
                collection_name=collection,
                query=SparseVector(
                    indices=sparse_query["indices"],
                    values=sparse_query["values"],
                ),
                using="sparse",
                limit=top_k,
                query_filter=qdrant_filter,
            )
            for rank, hit in enumerate(sparse_hits.points):
                pid = str(hit.id)
                payload = hit.payload or {}
                if pid not in qdata.chunks:
                    qdata.chunks[pid] = RankedChunk(
                        point_id=pid,
                        doc_id=payload.get("title", payload.get("book_title", "")),
                        text=payload.get("text", ""),
                    )
                if qdata.chunks[pid].sparse_rank is None:
                    qdata.chunks[pid].sparse_rank = rank
        except Exception as e:
            logger.warning(f"Sparse search failed: {e}")

    dense_count = sum(1 for c in qdata.chunks.values() if c.dense_rank is not None)
    sparse_count = sum(1 for c in qdata.chunks.values() if c.sparse_rank is not None)
    logger.info(f"  Fetched {len(qdata.chunks)} unique chunks ({dense_count} dense, {sparse_count} sparse)")
    return qdata


# ── Step 4: LLM relevance scoring ────────────────────────────────────────────

SCORING_PROMPT = """Rate how relevant this text is to the query. Reply with ONLY a single digit: 1, 2, or 3.

1 = Not relevant (off-topic or too generic)
2 = Somewhat relevant (related but doesn't directly address the query)
3 = Directly relevant (directly addresses what the query is asking)

Query: {query}

Text: {text}

Reply with only 1, 2, or 3."""


async def score_chunk(llm: LLMClient, query: str, chunk: RankedChunk) -> int:
    """Rate a single chunk's relevance to the query on a 1-3 scale."""
    prompt = SCORING_PROMPT.format(query=query, text=chunk.text[:800])
    try:
        response = await llm.answer(query="score", context=prompt)
        for char in response.strip():
            if char in ("1", "2", "3"):
                return int(char)
        logger.warning(f"Could not parse score from: {response[:50]}")
        return 1
    except Exception as e:
        logger.error(f"LLM scoring failed: {e}")
        return 1


async def score_all_chunks(llm: LLMClient, all_query_data: List[QueryData]) -> None:
    """Score all unique (query, chunk) pairs. Modifies in place."""
    tasks = []
    refs = []

    for qi, qdata in enumerate(all_query_data):
        for pid, chunk in qdata.chunks.items():
            tasks.append(score_chunk(llm, qdata.query, chunk))
            refs.append((qi, pid))

    total = len(tasks)
    if total == 0:
        logger.warning("No chunks to score")
        return

    logger.info(f"Scoring {total} (query, chunk) pairs with LLM...")

    batch_size = SCORE_BATCH_SIZE
    for i in range(0, total, batch_size):
        batch = tasks[i:i + batch_size]
        batch_refs = refs[i:i + batch_size]
        results = await asyncio.gather(*batch)
        for (qi, pid), score in zip(batch_refs, results):
            all_query_data[qi].chunks[pid].relevance_score = score
        logger.info(f"  Scored {min(i + batch_size, total)}/{total} pairs")


# ── Step 5: RRF simulation + main sweep ──────────────────────────────────────

def simulate_rrf(qdata: QueryData, sparse_weight: float, top_k: int) -> List[RankedChunk]:
    """Re-run RRF fusion with a given sparse weight multiplier."""
    scores: Dict[str, float] = defaultdict(float)

    for pid, chunk in qdata.chunks.items():
        if chunk.dense_rank is not None:
            scores[pid] += 1.0 / (K_RRF + chunk.dense_rank + 1)
        if chunk.sparse_rank is not None:
            scores[pid] += sparse_weight * (1.0 / (K_RRF + chunk.sparse_rank + 1))

    ranked = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
    return [qdata.chunks[pid] for pid in ranked[:top_k]]


def compute_avg_relevance_at_k(qdata: QueryData, sparse_weight: float, top_k: int) -> float:
    top_chunks = simulate_rrf(qdata, sparse_weight, top_k)
    scores = [c.relevance_score or 1 for c in top_chunks]
    return sum(scores) / len(scores) if scores else 0.0


async def run_sweep(
    book_filename: str = DEFAULT_BOOK,
    top_k: int = 5,
    num_queries: int = QUERY_GEN_BATCHES * QUERY_PER_BATCH,
    output_path: Optional[str] = None,
) -> None:
    """Run the full single-book sweep."""

    # Build LLM client
    openai_base = os.getenv("OPENAI_API_BASE")
    api_url = openai_base.rstrip("/") if openai_base else "http://localhost:11434"
    api_key = os.getenv("LITELLM_API_KEY") or "dummy"
    model = "openai/qwen36"
    llm = LLMClient(api_url=api_url, api_key=api_key, model=model)

    # Build retriever
    retriever = Retriever(collection=HYBRID_COLLECTION)

    # Step 1: Fetch book chunks
    logger.info(f"Fetching chunks for {book_filename} from {HYBRID_COLLECTION}...")
    chunks = fetch_book_chunks(book_filename, HYBRID_COLLECTION, top_k=2000)
    if not chunks:
        logger.error(f"No chunks found for {book_filename}. Check collection and filename.")
        return

    # Step 2: Generate queries
    logger.info("Generating queries from chunk snippets...")
    queries = await generate_queries(
        llm, chunks,
        book_topic="retrieval-augmented generation and LLM applications",
    )
    if not queries:
        logger.error("No queries generated. Exiting.")
        return

    # Limit to requested number
    queries = queries[:num_queries]
    logger.info(f"Using {len(queries)} queries:")
    for q in queries:
        logger.info(f"  - {q[:70]}...")

    # Step 3: Fetch rank lists for all queries
    logger.info("\nFetching dense + sparse rank lists...")
    all_query_data = []
    for query in queries:
        logger.info(f"  Query: {query[:60]}...")
        qdata = fetch_rank_lists(retriever, query, book_filename, top_k=top_k * 3)
        all_query_data.append(qdata)

    # Step 4: LLM score all (query, chunk) pairs
    await score_all_chunks(llm, all_query_data)

    # Step 5: Sweep sparse weights
    logger.info("\nSweeping sparse weights...")
    weight_results: Dict[float, List[float]] = {w: [] for w in SPARSE_WEIGHTS}

    for qdata in all_query_data:
        for w in SPARSE_WEIGHTS:
            avg_rel = compute_avg_relevance_at_k(qdata, w, top_k)
            weight_results[w].append(avg_rel)

    # Step 6: Print summary table
    print(f"\n{'Weight':>8}  {'Avg Relevance@' + str(top_k):>18}  {'vs weight=1.0':>14}")
    print("-" * 46)

    baseline_avg = sum(weight_results[1.0]) / len(weight_results[1.0])
    best_weight = max(SPARSE_WEIGHTS, key=lambda w: sum(weight_results[w]))

    for w in SPARSE_WEIGHTS:
        avg = sum(weight_results[w]) / len(weight_results[w])
        delta = avg - baseline_avg
        marker = "  <- best" if w == best_weight else ""
        print(f"{w:>8.2f}  {avg:>18.4f}  {delta:>+14.4f}{marker}")

    print(f"\nBest sparse weight: {best_weight}")
    print(f"Book: {book_filename}")
    print(f"Chunks: {len(chunks)}")
    print(f"Queries: {len(queries)}")
    print(f"Top-K: {top_k}")

    # Per-query breakdown
    print(f"\nPer-query avg relevance@{top_k} at best weight ({best_weight}):")
    for qdata in all_query_data:
        avg = compute_avg_relevance_at_k(qdata, best_weight, top_k)
        print(f"  {qdata.query[:65]:<65}  {avg:.3f}")

    # Save results
    if output_path is None:
        safe_name = book_filename.replace(".epub", "").replace("-", "_")
        output_path = str(_project_root / "results" / f"sweep_{safe_name}.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "book": book_filename,
        "collection": HYBRID_COLLECTION,
        "top_k": top_k,
        "num_chunks": len(chunks),
        "num_queries": len(queries),
        "queries": queries,
        "sparse_weights_tested": SPARSE_WEIGHTS,
        "best_weight": best_weight,
        "weight_summary": {
            str(w): {
                "avg_relevance_at_k": round(sum(weight_results[w]) / len(weight_results[w]), 4),
                "per_query": [round(v, 4) for v in weight_results[w]],
            }
            for w in SPARSE_WEIGHTS
        },
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Single-book sparse/dense weight sweep")
    parser.add_argument("--book", type=str, default=DEFAULT_BOOK,
                        help="EPUB filename to sweep (default: masteringretrieval-augmentedgeneration.epub)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K for relevance evaluation (default: 5)")
    parser.add_argument("--num-queries", type=int, default=QUERY_GEN_BATCHES * QUERY_PER_BATCH,
                        help="Number of queries to generate (default: 15)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/sweep_<book>.json)")
    args = parser.parse_args()
    asyncio.run(run_sweep(
        book_filename=args.book,
        top_k=args.top_k,
        num_queries=args.num_queries,
        output_path=args.output,
    ))
    print("\nDone.")
