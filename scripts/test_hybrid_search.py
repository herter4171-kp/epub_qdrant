"""Test hybrid search end-to-end using the project .venv.

Connects to Qdrant on GPU box (192.168.68.75) and MiniCOIL embedding server there too.
"""

import sys
sys.path.insert(0, "/Users/justinherter/projects/epub_qdrant")
sys.path.insert(0, "/Users/justinherter/projects/epub_qdrant/mcp_servers/retrieval")

import logging
logging.basicConfig(level=logging.WARNING)

from qdrant_client import QdrantClient
from mcp_servers.minicoil_server.client import get_sparse_vectors
from qdrant_client.models import SparseVector
from collections import defaultdict

QDRANT_URL = "http://192.168.68.75:6333"

client = QdrantClient(url=QDRANT_URL)

# Check collection configs
print("=== COLLECTION CONFIGS ===")
for col in ["books", "papers", "books-named", "papers-named"]:
    try:
        info = client.get_collection(col)
        print(f"\n{col}:")
        print(f"  Points: {info.points_count}")
        print(f"  Vectors: {info.config.params.vectors}")
        sv = getattr(info.config.params, 'sparse_vectors', None)
        print(f"  Sparse vectors: {sv}")
    except Exception as e:
        print(f"\n{col}: NOT FOUND ({e})")

# Test queries
queries = [
    "agentic ai patterns",
    "Apress books on AI",
    "Springer published 2023",
]

for query in queries:
    print(f"\n{'='*70}")
    print(f"QUERY: '{query}'")
    print(f"{'='*70}")

    # Embed dense query via Ollama on GPU box
    from src.embedder import Embedder
    embedder = Embedder("http://192.168.68.75:11434", "embeddinggemma:300m")
    dense_vec = embedder.embed_single(query)
    print(f"Dense vector dim: {len(dense_vec)}")

    # Sparse query via MiniCOIL
    sparse_q = get_sparse_vectors([query], is_query=True)[0]
    print(f"Sparse query: {len(sparse_q['indices'])} non-zero indices")

    # Dense search on books-named
    dense_results = client.query_points(
        collection_name="books-named",
        query=dense_vec,
        using="dense",
        limit=20,
    )

    # Sparse search on books-named
    sparse_hits = client.query_points(
        collection_name="books-named",
        query=SparseVector(indices=sparse_q["indices"], values=sparse_q["values"]),
        using="sparse",
        limit=20,
    )

    print(f"\nDense top 10:")
    for i, pt in enumerate(dense_results.points[:10]):
        title = (pt.payload.get("book_title") or pt.payload.get("title", ""))[:50]
        score = pt.score
        print(f"  [{i+1:2d}] score={score:.4f} id={pt.id} title={title}")

    print(f"\nSparse top 10:")
    for i, pt in enumerate(sparse_hits.points[:10]):
        title = (pt.payload.get("book_title") or pt.payload.get("title", ""))[:50]
        score = pt.score
        print(f"  [{i+1:2d}] score={score:.4f} id={pt.id} title={title}")

    # RRF fusion
    rrf_scores = defaultdict(float)
    k_rrf = 60
    for rank, hit in enumerate(dense_results.points):
        rrf_scores[hit.id] += 1.0 / (k_rrf + rank + 1)
    for rank, hit in enumerate(sparse_hits.points):
        rrf_scores[hit.id] += 1.0 / (k_rrf + rank + 1)

    id_to_point = {}
    for p in list(dense_results.points) + list(sparse_hits.points):
        id_to_point[p.id] = p

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:20]

    print(f"\nRRF Fusion top 10:")
    for i, pid in enumerate(sorted_ids[:10]):
        pt = id_to_point[pid]
        title = (pt.payload.get("book_title") or pt.payload.get("title", ""))[:50]
        dense_rank = next((r for r, h in enumerate(dense_results.points) if h.id == pid), None)
        sparse_rank = next((r for r, h in enumerate(sparse_hits.points) if h.id == pid), None)
        print(f"  [{i+1:2d}] RRF={rrf_scores[pid]:.4f} dense={dense_rank} sparse={sparse_rank} id={pid} title={title}")

    # Comparison
    dense_top10 = set(pt.id for pt in dense_results.points[:10])
    sparse_top10 = set(pt.id for pt in sparse_hits.points[:10])
    hybrid_top10 = set(sorted_ids[:10])
    print(f"\nDense top 10 IDs:   {sorted(dense_top10)}")
    print(f"Sparse top 10 IDs:  {sorted(sparse_top10)}")
    print(f"Hybrid top 10 IDs:  {sorted(hybrid_top10)}")
    print(f"Dense intersection Sparse: {sorted(dense_top10 & sparse_top10)}")
    print(f"Dense-only in hybrid: {sorted(dense_top10 - hybrid_top10)}")
    print(f"Sparse-only in hybrid: {sorted(sparse_top10 - hybrid_top10)}")

    # Check overlap in top 20
    dense_top20 = set(pt.id for pt in dense_results.points)
    sparse_top20 = set(pt.id for pt in sparse_hits.points)
    overlap_20 = dense_top20 & sparse_top20
    print(f"Dense intersect Sparse top 20: {len(overlap_20)}/{len(dense_top20 | sparse_top20)}")
    if len(overlap_20) < 5:
        print("  *** VERY LOW OVERLAP - sparse retrieves completely different points! ***")
    elif len(overlap_20) < 10:
        print("  *** MODERATE OVERLAP - sparse contributes distinct signal ***")
    else:
        print("  HIGH OVERLAP - dense dominates")