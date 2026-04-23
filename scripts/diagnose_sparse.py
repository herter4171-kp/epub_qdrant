#!/usr/bin/env python3
"""Diagnose whether MiniCOIL sparse vectors contribute to hybrid search.

Usage:
    python3 scripts/diagnose_sparse.py
"""
import json
from collections import defaultdict

OLLAMA_URL = "http://192.168.68.75:11434"
MINICOIL_URL = "http://192.168.68.75:9000"
QDRANT_URL = "http://192.168.68.75:6333"

TEST_QUERIES = [
    ("agentic ai patterns", "semantic"),
    ("Apress books on AI", "keyword+metadata"),
    ("Springer ISBN published 2023", "pure keyword"),
]


def get_dense_vec(query):
    import urllib.request
    data = json.dumps({"model": "embeddinggemma:300m", "input": query}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        d = json.load(resp)
    return d["embeddings"][0]


def get_sparse_vec(query):
    import urllib.request
    data = json.dumps({"texts": [query], "is_query": True}).encode()
    req = urllib.request.Request(
        f"{MINICOIL_URL}/embed",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        d = json.load(resp)
    return d["vectors"][0]


def qdrant_search(query_vec, using, limit=20):
    import urllib.request
    payload = {
        "query": query_vec,
        "using": using,
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/books-named/points/query",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def main():
    for query, label in TEST_QUERIES:
        print(f"\n{'='*70}")
        print(f"QUERY ({label}): {query!r}")
        print(f"{'='*70}")

        # Get embeddings
        dense_vec = get_dense_vec(query)
        sparse_vec = get_sparse_vec(query)

        print(f"Dense dims: {len(dense_vec)}")
        print(f"Sparse non-zero: {len(sparse_vec['indices'])}")
        print(f"Sparse indices: {sparse_vec['indices']}")
        print(f"Sparse values: {[round(v,3) for v in sparse_vec['values']]}")
        print()

        # Dense search
        dense_resp = qdrant_search(dense_vec, "dense")
        dense_pts = dense_resp["result"]["points"]

        # Sparse search
        sparse_resp = qdrant_search(sparse_vec, "sparse")
        sparse_pts = sparse_resp["result"]["points"]

        # Extract results
        dense_ids = {}
        for i, p in enumerate(dense_pts):
            pid = p["id"]
            dense_ids[pid] = {"rank": i+1, "score": p["score"], "payload": p["payload"]}

        sparse_ids = {}
        for i, p in enumerate(sparse_pts):
            pid = p["id"]
            sparse_ids[pid] = {"rank": i+1, "score": p["score"], "payload": p["payload"]}

        # Print dense results
        print("DENSE-ONLY TOP 20:")
        for i, p in enumerate(dense_pts):
            pid = p["id"]
            score = p["score"]
            title = (p["payload"].get("book_title") or p["payload"].get("title", ""))[:40]
            section = (p["payload"].get("section_title") or "")[:30]
            publisher = p["payload"].get("publisher", "")
            in_sparse = "✓" if pid in sparse_ids else " "
            print(f"  {i+1:2d}. [{in_sparse}] id={pid} score={score:.4f} {title} | {publisher} | {section}")

        print()

        # Print sparse results
        print("SPARSE-ONLY TOP 20:")
        for i, p in enumerate(sparse_pts):
            pid = p["id"]
            score = p["score"]
            title = (p["payload"].get("book_title") or p["payload"].get("title", ""))[:40]
            section = (p["payload"].get("section_title") or "")[:30]
            publisher = p["payload"].get("publisher", "")
            in_dense = "✓" if pid in dense_ids else " "
            print(f"  {i+1:2d}. [{in_dense}] id={pid} score={score:.4f} {title} | {publisher} | {section}")

        # Overlap analysis
        dense_set = set(dense_ids.keys())
        sparse_set = set(sparse_ids.keys())
        overlap = dense_set & sparse_set
        dense_only = dense_set - sparse_set
        sparse_only = sparse_set - dense_set

        print()
        print(f"OVERLAP ANALYSIS:")
        print(f"  Dense count: {len(dense_pts)}, Sparse count: {len(sparse_pts)}")
        print(f"  In both: {len(overlap)}")
        print(f"  Dense only: {len(dense_only)}")
        print(f"  Sparse only: {len(sparse_only)}")

        if dense_only:
            print(f"\n  Points in dense but NOT sparse: {sorted(dense_only)[:10]}")
        if sparse_only:
            print(f"\n  Points in sparse but NOT dense: {sorted(sparse_only)[:10]}")

        # Rank comparison for overlapping points
        if overlap:
            print(f"\n  RANK CHANGES (for overlapping points):")
            rank_diffs = []
            for pid in sorted(overlap)[:10]:
                dr = dense_ids[pid]["rank"]
                sr = sparse_ids[pid]["rank"]
                rank_diffs.append((pid, dr, sr, sr - dr))
            rank_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
            for pid, dr, sr, diff in rank_diffs:
                title = dense_ids[pid]["payload"].get("book_title", "")[:30]
                print(f"    id={pid} dense_rank={dr} sparse_rank={sr} diff={diff:+d} {title}")


if __name__ == "__main__":
    main()