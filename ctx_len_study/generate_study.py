#!/usr/bin/env python3
"""Generate ctx_len_study markdown files.

Compares retrieval from `papers` vs `papers-2048ctx` across 50 prompts,
for papers discovered from the `papers-2048ctx` collection.

Reads papers_semantic data from bedrock_compare/query_results/*.json
(which contains agent-lookup MCP results for the `papers` collection).
Queries `papers-2048ctx` via qdrant-client.

Writes one md file per paper + intro.md + compile.sh.
"""

import argparse
import json
import os
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
PAPERS_2048CTX = "papers-2048ctx"
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
    # Scroll with limit=100 to get unique source files
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
                stem = sf[:-4]  # remove .pdf
                source_files.add(stem)
        if offset is None:
            break
    return sorted(source_files)


def query_papers_collection(query, paper_ids, top_k=8):
    """Query `papers` collection via qdrant-client, filtered to specific papers.

    Returns list of dicts with score, text, source_file.
    Papers collection gets k=topk/2 results (papers-2048ctx gets topk via hybrid).
    """
    client = QdrantClient(url=QDRANT_URL)
    query_vec = get_dense_vectors([query])[0]

    # Build should filter: source_file matches any of the paper PDF names
    source_conditions = [
        FieldCondition(key="source_file", match=MatchValue(value=f"{pid}.pdf"))
        for pid in paper_ids
    ]
    query_filter = Filter(should=source_conditions)

    points = client.query_points(
        collection_name="papers",
        query=query_vec,
        limit=top_k,
        query_filter=query_filter,
    )

    results = []
    for point in points.points:
        payload = point.payload or {}
        source_file = payload.get("source_file", "")
        results.append({
            "score": point.score,
            "text": payload.get("text", ""),
            "source_file": source_file,
        })
    return results


def query_papers_2048ctx(query, paper_ids, top_k=8):
    """Query papers-2048ctx collection via qdrant-client with hybrid dense+sparse.

    Queries both dense and sparse named vectors, each fetching top_k//2 results
    so the combined total equals top_k.

    Args:
        query: The query string.
        paper_ids: Set of paper IDs (underscore format) to filter to.
        top_k: Total number of results to return.

    Returns:
        List of dicts with score, text, source_file.
    """
    client = QdrantClient(url=QDRANT_URL)

    source_conditions = [
        FieldCondition(key="source_file", match=MatchValue(value=f"{pid}.pdf"))
        for pid in paper_ids
    ]
    query_filter = Filter(should=source_conditions)

    half_k = top_k // 2

    # Dense query: fetch half_k results
    query_vec = get_dense_vectors([query])[0]
    dense_points = client.query_points(
        collection_name=PAPERS_2048CTX,
        query=query_vec,
        using="dense",
        limit=half_k,
        query_filter=query_filter,
    ).points

    # Sparse query: fetch half_k results
    sparse_vec = get_sparse_vectors([query], is_query=True)[0]
    sparse_points = client.query_points(
        collection_name=PAPERS_2048CTX,
        query=SparseVector(
            indices=sparse_vec["indices"],
            values=sparse_vec["values"],
        ),
        using="sparse",
        limit=half_k,
        query_filter=query_filter,
    ).points

    # Concatenate dense first, then sparse (total = top_k)
    results = []
    for point in dense_points + sparse_points:
        payload = point.payload or {}
        results.append({
            "score": point.score,
            "text": payload.get("text", ""),
            "source_file": payload.get("source_file", ""),
        })
    return results


# ── Markdown generation ─────────────────────────────────────────────────────

def intro_md(paper_list):
    return f"""# Context Window Length Study

## Overview

Comparing retrieval from two Qdrant collections across 50 prompts:

| Source | Description |
|--------|-------------|
| **Papers** | Qdrant `papers` collection (~96K points, smaller chunks), dense-only |
| **Papers-2048ctx** | Qdrant `papers-2048ctx` (692 points, 2048-token chunks), hybrid dense+sparse |

## {len(paper_list)} Papers Studied

""" + "\n".join(f"- {pid.replace('_', '.')} ({pid})" for pid in paper_list) + f"""

## Caveats

- **Scores are NOT comparable across collections.** Different embedding models, different chunk sizes, different collection sizes.
- This study focuses on **which chunks appear** (qualitative overlap), not their scores.
- 50 prompts used across 5 categories x 5 proficiency levels.

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

        for source_name, source_key in [("Papers", "papers"), ("Papers-2048ctx", "papers-2048ctx"), ("Bedrock", "bedrock")]:
            rows = all_results.get((paper_id, pid, source_key), [])
            if not rows:
                md += f"| {source_name} | - | (no results) |\n"
            else:
                for score, text in rows:
                    display_text = text.replace('\n', ' ')
                    display_text = display_text.replace('|', '\\|')
                    md += f"| {source_name} | {score:.4f} | {display_text} |\n"

        md += "\n---\n\n"

    return md


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate ctx_len_study markdown files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of papers to process (sorted, first N)")
    args = parser.parse_args()

    print("Loading prompts...")
    prompts = load_prompts()
    print(f"  Loaded {len(prompts)} prompts")

    print(f"Initializing Bedrock KB client (kb={BEDROCK_KB_ID})...")
    get_bedrock_client()
    print("  Bedrock client ready")

    print(f"Discovering papers from {PAPERS_2048CTX}...")
    all_paper_ids = discover_papers_from_collection(PAPERS_2048CTX, QDRANT_URL)
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

        # Query papers collection filtered to our papers
        papers_texts = query_papers_collection(query, paper_ids)

        # Query papers-2048ctx (hybrid dense+sparse), filtered to our papers
        p2048_results = query_papers_2048ctx(query, paper_ids)

        # Group papers results by source_file
        papers_by_paper = defaultdict(list)
        for t in papers_texts:
            sf = t["source_file"]
            for pid_check in paper_ids:
                if sf.endswith(f"{pid_check}.pdf"):
                    papers_by_paper[pid_check].append((t["score"], t["text"]))
                    break

        # Group papers-2048ctx results by source_file
        p2048_by_paper = defaultdict(list)
        for t in p2048_results:
            sf = t["source_file"]
            for pid_check in paper_ids:
                if sf.endswith(f"{pid_check}.pdf"):
                    p2048_by_paper[pid_check].append((t["score"], t["text"]))
                    break

        # Store results for each paper
        for pid_check in paper_ids:
            all_results[(pid_check, pid, "papers")] = papers_by_paper.get(pid_check, [])
            all_results[(pid_check, pid, "papers-2048ctx")] = p2048_by_paper.get(pid_check, [])

        # Query bedrock directly from KB (same results for all papers, this prompt)
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

    print(f"\nDone! {len(PROMPT_IDS)} prompts x {len(paper_ids)} papers")
    print(f"Run: bash compile.sh")

if __name__ == "__main__":
    main()