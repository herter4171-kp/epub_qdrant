"""Integration tests: signal isolation via MCP calls + point_id in results.

Requires a running MCP server against a real Qdrant collection with both
dense and sparse named vectors. Skips gracefully if server unreachable.
"""

import os
import requests
import pytest

MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8090"))
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"

# Use a known hybrid collection — override via env var if needed
COLLECTION = os.getenv("TEST_COLLECTION", "books-semantic")


def _mcp_call(tool: str, args: dict, timeout: int = 30) -> dict:
    """JSON-RPC 2.0 tools/call to MCP server."""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool, "arguments": args},
        "id": 1,
    }
    r = requests.post(MCP_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    result = body.get("result", body)
    # Handle content wrapper
    if isinstance(result, dict) and "content" in result:
        import json as _json
        text = result["content"][0]["text"]
        return _json.loads(text)
    return result


def _server_reachable() -> bool:
    try:
        base = MCP_URL.rsplit("/mcp", 1)[0]
        r = requests.get(f"{base}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=f"MCP server not reachable at {MCP_URL}",
)


class TestSignalIsolation:
    """Verify dense-only and sparse-only calls return different results."""

    def test_dense_only_vs_sparse_only_differ(self):
        """Dense-only (sparse_weight=0) and sparse-only (dense_weight=0) should differ."""
        query = "retrieval augmented generation"

        dense_result = _mcp_call("query", {
            "query": query,
            "collection": COLLECTION,
            "mode": "search",
            "top_k": 5,
            "sparse_weight": 0,
            "dense_weight": 1,
        })

        sparse_result = _mcp_call("query", {
            "query": query,
            "collection": COLLECTION,
            "mode": "search",
            "top_k": 5,
            "sparse_weight": 1,
            "dense_weight": 0,
        })

        # Both should return valid grouped results
        assert "groups" in dense_result
        assert "groups" in sparse_result
        assert len(dense_result["groups"]) > 0, "Dense signal returned no groups"
        assert len(sparse_result["groups"]) > 0, "Sparse signal returned no groups"

        # Extract chunk texts from both
        def _chunk_texts(result):
            texts = []
            for g in result["groups"]:
                for c in g.get("chunks", []):
                    texts.append(c.get("text", "")[:100])
            return texts

        dense_texts = _chunk_texts(dense_result)
        sparse_texts = _chunk_texts(sparse_result)

        # They should not be identical (different signals → different rankings)
        assert dense_texts != sparse_texts, (
            "Dense-only and sparse-only returned identical results — "
            "signal isolation may not be working"
        )

    def test_both_signals_return_chunks(self):
        """Both isolated signals should return valid chunks with text."""
        query = "transformer attention mechanism"

        for label, sw, dw in [("dense", 0, 1), ("sparse", 1, 0)]:
            result = _mcp_call("query", {
                "query": query,
                "collection": COLLECTION,
                "mode": "search",
                "top_k": 3,
                "sparse_weight": sw,
                "dense_weight": dw,
            })

            total = 0
            for g in result.get("groups", []):
                for c in g.get("chunks", []):
                    assert c.get("text"), f"{label} signal returned chunk with empty text"
                    total += 1
            assert total > 0, f"{label} signal returned no chunks"


class TestPointIdInResults:
    """Verify point_id is present in MCP search results."""

    def test_point_id_present_in_all_chunks(self):
        """Every chunk in search results should have a non-null point_id."""
        result = _mcp_call("query", {
            "query": "neural network training",
            "collection": COLLECTION,
            "mode": "search",
            "top_k": 5,
        })

        assert "groups" in result
        assert len(result["groups"]) > 0, "No groups returned"

        for g in result["groups"]:
            for c in g.get("chunks", []):
                assert "point_id" in c, f"Chunk missing point_id field: {list(c.keys())}"
                assert c["point_id"] is not None, "point_id is None"
                assert isinstance(c["point_id"], str), f"point_id not string: {type(c['point_id'])}"
                assert len(c["point_id"]) > 0, "point_id is empty string"
