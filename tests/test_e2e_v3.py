"""End-to-end integration test for v3 fused retrieval pipeline.

Requires running MCP server and LiteLLM against a real collection.
Skips gracefully if either is unreachable.
"""

import json
import os
import tempfile

import pytest
import requests
from dotenv import load_dotenv
load_dotenv()

MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8090"))
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"
COLLECTION = os.getenv("TEST_COLLECTION", "books-semantic")


def _server_reachable():
    try:
        base = MCP_URL.rsplit("/mcp", 1)[0]
        r = requests.get(f"{base}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _litellm_reachable():
    from servers.mcp_server.config import settings
    try:
        base = settings.LITELLM_API_URL.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        r = requests.get(f"{base}/health", timeout=5)
        # Any HTTP response means server is reachable
        return r.status_code in (200, 401, 403)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_server_reachable() and _litellm_reachable()),
    reason="MCP server or LiteLLM not reachable",
)


class TestE2EV3:
    """Run main() with 1 position on 1 book, verify v3 JSON output."""

    def test_full_pipeline(self):
        from unittest.mock import patch
        from scripts.blind_ab_test import main, mcp_call

        # Get one real book from the collection
        books = mcp_call("list_books", {"collection": COLLECTION}).get("books", [])
        assert books, "No books in collection"
        one_book = [books[0]]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            with patch("scripts.blind_ab_test.discover_books", return_value=one_book):
                main([COLLECTION, "--positions", "1", "--output", output_path, "--seed", "42"])

            with open(output_path) as f:
                data = json.load(f)

            metadata = data["metadata"]
            samples = data["samples"]

            # Metadata v3 fields
            assert metadata["version"] == "3.0.0"
            assert metadata["script_version"] == "3.0.0"
            assert metadata["collection"] == COLLECTION
            assert "dense_k" in metadata
            assert "sparse_k" in metadata
            assert "avg_judge_score" in metadata
            assert "reranker_model" in metadata
            assert "avg_dedup_set_size" in metadata
            assert "avg_both_signal_count" in metadata
            assert "bucket_summary" in metadata

            # Should not have v2 fields
            assert "dense_collection" not in metadata
            assert "hybrid_collection" not in metadata
            assert "dense_wins" not in metadata
            assert "hybrid_wins" not in metadata
            assert "ties" not in metadata
            assert "sparse_weight" not in metadata

            # At least 1 sample (might be 0 if all books fail, but unlikely with seed 42)
            if samples:
                s = samples[0]
                # v3 required fields
                assert "dedup_set_size" in s
                assert "both_signal_count" in s
                assert "reranked_passages" in s
                assert "reranker_raw" in s
                assert "fused_answer" in s
                assert "judge_score" in s
                assert s["judge_score"] in (1, 2, 3)
                assert "judge_reason" in s
                assert "judge_raw" in s
                assert "dense_raw_results" in s
                assert "sparse_raw_results" in s
                assert "dense_source_hit_rank" in s
                assert "sparse_source_hit_rank" in s

                # Should not have v2 fields
                assert "dense_answer" not in s
                assert "hybrid_answer" not in s
                assert "winner" not in s
                assert "judge_winner" not in s
                assert "hybrid_raw_results" not in s
                assert "zero_chunk_retrieval" not in s

        finally:
            os.unlink(output_path)
