"""Smoke test: query sae-sparse and verify sensible results.

Run:
    python3 scripts/test_sae_query.py
"""

import sys
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings

QDRANT_URL = settings.QDRANT_URL
EMB_URL = "http://localhost:8100"

# Test queries — pick topics that should have relevant papers
TEST_QUERIES = [
    "attention mechanisms in transformers",
    "sparse autoencoders for interpretability",
    "hybrid search reciprocal rank fusion",
    "dense retrieval dual encoder BERT",
]


def main():
    client = QdrantClient(url=QDRANT_URL)

    # Verify collection exists
    try:
        info = client.get_collection("sae-sparse")
        print(f"Collection 'sae-sparse': {info.points_count} points")
    except Exception as e:
        print(f"ERROR: collection 'sae-sparse' not found: {e}")
        return

    for query in TEST_QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # Get SAE vector
        resp = requests.post(
            f"{EMB_URL}/embed_sae",
            json={"texts": [query], "is_query": True},
            timeout=30,
        )
        resp.raise_for_status()
        vec = resp.json()["vectors"][0]
        print(f"  SAE vector: {len(vec['indices'])} non-zeros, "
              f"values range [{min(vec['values']):.4f}, {max(vec['values']):.4f}]")

        # Search sae-sparse
        hits = client.query_points(
            collection_name="sae-sparse",
            query=SparseVector(indices=vec["indices"], values=vec["values"]),
            using="sparse",
            limit=5,
        )

        print(f"  Top 5 hits:")
        for hit in hits.points:
            print(f"    id={hit.id}  score={hit.score:.4f}")

        # Fetch payloads from books for display
        ids = [h.id for h in hits.points]
        try:
            payloads = client.retrieve(
                collection_name="books",
                ids=ids,
                with_payload=True,
            )
            for p in payloads:
                title = p.payload.get("title", p.payload.get("book_title", "N/A"))
                text = p.payload.get("text", p.payload.get("content", ""))[:120]
                print(f"    → {title}: {text}...")
        except Exception as e:
            print(f"    (could not fetch payloads: {e})")


if __name__ == "__main__":
    main()