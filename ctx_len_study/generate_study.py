#!/usr/bin/env python3
"""Generate ctx_len_study markdown files.

Compares retrieval from Qdrant collections plus Bedrock KB across 50 prompts,
for papers discovered from the first collection in the list.

Usage:
    python3 generate_study.py --collections "papers,papers-2048ctx-SAE" --topk 8
    python3 generate_study.py --collections "papers" --topk 4 --limit 10
    bash run_study.sh                        # defaults to papers + papers-2048ctx-SAE
    bash run_study.sh --collections "papers" --limit 5  # override defaults

Collections are specified via --collections (comma-separated, required).
Bedrock KB is always queried as a fourth contender.
Named-vector collections use hybrid dense+sparse (topk//2 each).
Other collections use single dense query.
Auto-detect named-vector collections via Qdrant API unless --named-vectors is set.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient

# ── Config ───────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent))
from servers.embedding_server.client import get_dense_vectors, get_sparse_vectors
from bedrock_compare.bedrock_client import BedrockKBClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector

QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.68.75:6333")
BEDROCK_KB_ID = os.getenv("BEDROCK_KB_ID", "RQMBIXUSXH")
OUTPUT_DIR = Path(__file__).parent

# All 50 prompts
PROMPT_IDS = list(range(1, 51))


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_prompts():
    """Load prompts.json -> dict keyed by id."""
    prompts_path = Path(__file__).parent.parent / "bedrock_compare" / "prompts.json"
    with open(prompts_path) as f:
        data = json.load(f)
    return {p["id"]: p for p in data["prompts"]}


def get_bedrock_client():
    """Lazy-init Bedrock KB client."""
    if not hasattr(get_bedrock_client, "_client"):
        get_bedrock_client._client = BedrockKBClient(kb_id=BEDROCK_KB_ID)
    return get_bedrock_client._client


def query_bedrock(query, top_k=8):
    """Query Bedrock KB directly and return list of (score, text) tuples."""
    client = get_bedrock_client()
    results = client.query(query, number_of_results=top_k)
    tuples = []
    for r in results:
        content = r.get("content", {})
        text = content.get("text", "")
        score = r.get("score", 0.0)
        tuples.append((score, text))
    return tuples


def discover_papers_from_collection(collection, qdrant_url):
    """Discover distinct source_file values from a collection.

    Returns a sorted list of paper IDs (underscore format, e.g. '2010_03768').
    """
    client = QdrantClient(url=qdrant_url)
    source_files = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=["source_file"],
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            sf = p.payload.get("source_file", "")
            if sf.endswith(".pdf"):
                stem = sf[:-4]
                source_files.add(stem)
        if offset is None:
            break
    return sorted(source_files)


def query_qdrant_collection(collection, query, paper_ids, top_k, named_vector_collections=None):
    """Query a single Qdrant collection, filtered to specific papers.

    Named vector collections use hybrid dense+sparse (topk//2 each).
    Unnamed collections use single dense query.

    Args:
        collection: Qdrant collection name.
        query: Natural language query string.
        paper_ids: List of paper IDs (underscore format) to filter on.
        top_k: Number of results to return per vector.
        named_vector_collections: Set of collection names that use named vectors.
                                  If None, auto-detects via Qdrant API.

    Returns list of dicts with score, text, source_file.
    """
    if named_vector_collections is None:
        named_vector_collections = set()

    client = QdrantClient(url=QDRANT_URL)

    source_conditions = [
        FieldCondition(key="source_file", match=MatchValue(value=f"{pid}.pdf"))
        for pid in paper_ids
    ]
    query_filter = Filter(should=source_conditions)

    results = []

    use_hybrid = False
    try:
        use_hybrid = collection in named_vector_collections
    except Exception:
        pass  # set comparison failed -> fall through to dense-only

    if use_hybrid:
        half_k = max(1, top_k // 2)

        # Dense query
        query_vec = get_dense_vectors([query])[0]
        dense_points = client.query_points(
            collection_name=collection,
            query=query_vec,
            using="dense",
            limit=half_k,
            query_filter=query_filter,
        ).points

        # Sparse query
        sparse_vec = get_sparse_vectors([query], is_query=True)[0]
        sparse_points = client.query_points(
            collection_name=collection,
            query=SparseVector(
                indices=sparse_vec["indices"],
                values=sparse_vec["values"],
            ),
            using="sparse",
            limit=half_k,
            query_filter=query_filter,
        ).points

        for point in dense_points + sparse_points:
            payload = point.payload or {}
            results.append({
                "score": point.score,
                "text": payload.get("text", ""),
                "source_file": payload.get("source_file", ""),
            })
    else:
        # Unnamed dense-only collection
        query_vec = get_dense_vectors([query])[0]
        points = client.query_points(
            collection_name=collection,
            query=query_vec,
            limit=top_k,
            query_filter=query_filter,
        )

        for point in points.points:
            payload = point.payload or {}
            results.append({
                "score": point.score,
                "text": payload.get("text", ""),
                "source_file": payload.get("source_file", ""),
            })

    return results


# ── Markdown generation ─────────────────────────────────────────────────────

def _source_display_name(coll):
    """Human-readable display name for a collection."""
    names = {
        "papers": "Papers",
        "papers-2048ctx-SAE": "Papers-2048ctx-SAE",
    }
    return names.get(coll, coll)


def _source_key(coll):
    """Internal key for results dict."""
    return coll


def _bedrock_display_name():
    return "Bedrock"


def intro_md(paper_list):
    sources = []
    for c in QDRANT_COLLECTIONS:
        names = {
            "papers": "Papers",
            "papers-2048ctx-SAE": "Papers-2048ctx-SAE",
        }
        sources.append((names.get(c, c), f"Qdrant `{c}` ({len(paper_list)} papers)"))
    source_rows = "| ".join([
        "| Source | Description |",
        "|--------|-------------|"
    ]) + "\n"
    for s, desc in sources:
        source_rows += f"| **{s}** | {desc} |\n"

    return f"""# Context Window Length Study

## Overview

Comparing retrieval from Qdrant collections plus Bedrock KB across {len(PROMPT_IDS)} prompts:

{source_rows}

## {len(paper_list)} Papers Studied

""" + "\n".join(f"- {pid.replace('_', '.')} ({pid})" for pid in paper_list) + f"""

## Caveats

- **Scores are NOT comparable across collections.** Different embedding models, different chunk sizes, different collection sizes.
- This study focuses on **which chunks appear** (qualitative overlap), not their scores.
- {len(PROMPT_IDS)} prompts used across 5 categories x 5 proficiency levels.

---

"""


def paper_md(paper_id, prompts, all_results):
    """Generate markdown for a single paper."""
    md = f"""# {paper_id}

## Metadata

- **Arxiv ID:** {paper_id.replace('_', '.')}
- **Source File:** {paper_id}.pdf

---

"""

    for i, pid in enumerate(PROMPT_IDS, 1):
        prompt = prompts.get(pid, {})
        prompt_text = prompt.get("prompt", f"Prompt {pid}")
        prompt_cat = prompt.get("category", "unknown")
        prompt_prof = prompt.get("proficiency", "?")

        md += f"""## Prompt {i}: [{prompt_cat} p{prompt_prof}]

**Query:** {prompt_text}

"""

        md += "| Source | Score | Text |\n"
        md += "|--------|-------|------|\n"

        # Qdrant collections first, then Bedrock
        for source_key in QDRANT_COLLECTIONS:
            source_name = _source_display_name(source_key)
            rows = all_results.get((paper_id, pid, source_key), [])
            if not rows:
                md += f"| {source_name} | - | (no results) |\n"
            else:
                for score, text in rows:
                    display_text = text.replace('\n', ' ')
                    display_text = display_text.replace('|', '\\|')
                    md += f"| {source_name} | {score:.4f} | {display_text} |\n"

        # Bedrock always last
        bedrock_rows = all_results.get((paper_id, pid, "bedrock"), [])
        bedrock_name = _bedrock_display_name()
        if not bedrock_rows:
            md += f"| {bedrock_name} | - | (no results) |\n"
        else:
            for score, text in bedrock_rows:
                display_text = text.replace('\n', ' ')
                display_text = display_text.replace('|', '\\|')
                md += f"| {bedrock_name} | {score:.4f} | {display_text} |\n"

        md += "\n---\n\n"

    return md


def main():
    parser = argparse.ArgumentParser(description="Generate ctx_len_study markdown files")
    parser.add_argument("--collections", type=str, default="",
                        help="Comma-separated list of Qdrant collections to query (e.g. 'papers,papers-2048ctx-SAE')")
    parser.add_argument("--named-vectors", type=str, default="",
                        help="Comma-separated list of collections that use named vectors (dense+sparse). All others use dense-only. If empty, auto-detects via Qdrant API.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of papers to process (sorted, first N)")
    parser.add_argument("--topk", type=int, default=8,
                        help="Results per Qdrant query per collection. MUST be divisible by 4 for named-vector (hybrid) collections.")
    args = parser.parse_args()

    # Parse collections
    if not args.collections:
        print("\033[31mERROR: --collections is required. Provide comma-separated collection names.\033[0m")
        sys.exit(1)
    QDRANT_COLLECTIONS = [c.strip() for c in args.collections.split(",") if c.strip()]
    if not QDRANT_COLLECTIONS:
        print("\033[31mERROR: no valid collections provided.\033[0m")
        sys.exit(1)

    # Parse named vector collections
    if args.named_vectors:
        NAMED_VECTOR_COLLECTIONS = set(c.strip() for c in args.named_vectors.split(",") if c.strip())
    else:
        # Auto-detect: check each collection's vector config
        NAMED_VECTOR_COLLECTIONS = set()
        print("Auto-detecting named-vector collections via Qdrant API...")
        client = QdrantClient(url=QDRANT_URL)
        for coll in QDRANT_COLLECTIONS:
            try:
                info = client.get_collection(coll)
                vec_config = info.config.params.vectors
                is_named = False
                # Unnamed single vector: Vectors object with size but no named sub-config
                if isinstance(vec_config, dict):
                    # Named multi-vector config: {"dense": VectorParams(...), "sparse": SparseVectorParams(...)}
                    # Check if values are VectorParams/SparseVectorParams (named vectors)
                    # vs sparse/vector_config (legacy unnamed)
                    is_named = True
                elif vec_config is not None:
                    # Could be a Vectors object (unnamed) or NamedVectors object
                    # Vectors has 'size' attribute; NamedVectors has 'vectors' dict
                    has_named_attr = hasattr(vec_config, 'named_vectors') or (
                        hasattr(vec_config, 'vectors') and isinstance(getattr(vec_config, 'vectors', None), dict)
                    )
                    if has_named_attr:
                        is_named = True

                if is_named:
                    NAMED_VECTOR_COLLECTIONS.add(coll)
                    print(f"  {coll} -> named-vector collection")
                else:
                    print(f"  {coll} -> single-vector collection")
            except Exception as e:
                print(f"  {coll} -> warning: could not inspect config ({e}), assuming single-vector")
        print(f"  Named-vector collections: {NAMED_VECTOR_COLLECTIONS or '(none)'}")

    # Validate topk is divisible by 4 for named-vector collections
    if NAMED_VECTOR_COLLECTIONS:
        if args.topk % 4 != 0:
            msg = (
                "\033[31m"
                f"\n{'='*70}\n"
                f"  ERROR: --topk {args.topk} is not divisible by 4.\n"
                f"  Named-vector collections ({', '.join(NAMED_VECTOR_COLLECTIONS)})\n"
                f"  split topk into topk//2 dense + topk//2 sparse.\n"
                f"  Fix: use --topk 8, 12, 16, 20, etc.\n"
                f"  {'='*70}\033[0m"
            )
            print(msg)
            sys.exit(1)
        print(f"  Named-vector collections will use hybrid search (topk//{4 // 4 * 2} = {args.topk // 2} dense + {args.topk // 2} sparse each)")

    topk = args.topk

    print("Loading prompts...")
    prompts = load_prompts()
    print(f"  Loaded {len(prompts)} prompts")

    print(f"Initializing Bedrock KB client (kb={BEDROCK_KB_ID})...")
    get_bedrock_client()
    print("  Bedrock client ready")

    # Discover papers from first collection
    discover_coll = QDRANT_COLLECTIONS[0]
    print(f"Discovering papers from {discover_coll}...")
    all_paper_ids = discover_papers_from_collection(discover_coll, QDRANT_URL)
    print(f"  Found {len(all_paper_ids)} papers")

    if args.limit:
        paper_ids = all_paper_ids[:args.limit]
        print(f"  Limiting to first {args.limit}: {', '.join(paper_ids)}")
    else:
        paper_ids = all_paper_ids

    # all_results: (paper_id, prompt_id, source_key) -> list of (score, text)
    all_results = defaultdict(list)

    for pid in PROMPT_IDS:
        prompt = prompts[pid]
        query = prompt["prompt"]
        print(f"  [{pid:2d}] {query[:55]}...")

        # Query ALL qdrant collections for this prompt
        for coll in QDRANT_COLLECTIONS:
            coll_texts = query_qdrant_collection(coll, query, paper_ids, topk, NAMED_VECTOR_COLLECTIONS)

            # Group by source_file -> paper_id
            by_paper = defaultdict(list)
            for t in coll_texts:
                sf = t["source_file"]
                for pid_check in paper_ids:
                    if sf.endswith(f"{pid_check}.pdf"):
                        by_paper[pid_check].append((t["score"], t["text"]))
                        break

            for pid_check in paper_ids:
                all_results[(pid_check, pid, coll)] = by_paper.get(pid_check, [])

        # Query Bedrock (same for all papers)
        bedrock_texts = query_bedrock(query, top_k=8)
        for pid_check in paper_ids:
            all_results[(pid_check, pid, "bedrock")] = bedrock_texts

    # Clean old output files
    print("\nCleaning old output files...")
    for f in OUTPUT_DIR.glob("*.md"):
        if f.name in ("generate_study.py", "named-vector-query-strategy.md"):
            continue
        f.unlink()
        print(f"  Removed: {f.name}")
    if (OUTPUT_DIR / "compile.sh").exists():
        (OUTPUT_DIR / "compile.sh").unlink()
        print(f"  Removed: compile.sh")

    print("Writing intro.md...")
    with open(OUTPUT_DIR / "intro.md", "w") as f:
        f.write(intro_md(paper_ids))

    print(f"Writing {len(paper_ids)} paper files...")
    for paper_id in paper_ids:
        md = paper_md(paper_id, prompts, all_results)
        fname = f"{paper_id[:4]}_{paper_id[4:]}.md"
        with open(OUTPUT_DIR / fname, "w") as f:
            f.write(md)
        print(f"  Written: {fname}")

    # Write compile.sh
    print("Writing compile.sh...")
    compile_script = "#!/bin/bash\n# Compile ctx_len_study\nset -e\ncd $(dirname \"$0\")\n"
    compile_script += "echo 'Compiling ctx_len_study.md...'\n"
    compile_script += "echo '# Context Window Length Study' > ctx_len_study.md\n"
    compile_script += "echo '' >> ctx_len_study.md\n"
    compile_script += "cat intro.md >> ctx_len_study.md\n"
    for paper_id in paper_ids:
        fname = f"{paper_id[:4]}_{paper_id[4:]}.md"
        compile_script += f"cat {fname} >> ctx_len_study.md\n"
    compile_script += "echo '' >> ctx_len_study.md\n"
    compile_script += "echo 'Done: ctx_len_study.md' "

    with open(OUTPUT_DIR / "compile.sh", "w") as f:
        f.write(compile_script)
    os.chmod(OUTPUT_DIR / "compile.sh", 0o755)

    print(f"\nDone! {len(PROMPT_IDS)} prompts x {len(paper_ids)} papers x {len(QDRANT_COLLECTIONS)} Qdrant collections + Bedrock")
    print(f"Run: bash compile.sh")


if __name__ == "__main__":
    main()