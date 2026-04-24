"""Unit tests for dense_weight parameter in hybrid_search() RRF fusion."""

from dataclasses import dataclass
from unittest.mock import patch, MagicMock

import pytest

from servers.mcp_server.retriever import (
    Retriever,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_SPARSE_WEIGHT,
)


# ─── Helpers ─────────────────────────────────────────────────────────


@dataclass
class FakePoint:
    id: str
    payload: dict


class FakeQueryResult:
    def __init__(self, points):
        self.points = points


def _make_retriever():
    """Build a Retriever with mocked storage so no real Qdrant needed."""
    with patch("servers.mcp_server.retriever.Storage"):
        r = Retriever(collection="test-col")
    # _client is a property that returns self._storage._client
    r._storage._client = MagicMock()
    return r


def _make_dense_hits():
    """3 dense hits with known IDs."""
    return FakeQueryResult([
        FakePoint("d1", {"text": "dense hit 1", "source_file": "a.epub"}),
        FakePoint("d2", {"text": "dense hit 2", "source_file": "a.epub"}),
        FakePoint("shared", {"text": "shared hit", "source_file": "a.epub"}),
    ])


def _make_sparse_hits():
    """3 sparse hits — 'shared' overlaps with dense."""
    return FakeQueryResult([
        FakePoint("s1", {"text": "sparse hit 1", "source_file": "a.epub"}),
        FakePoint("shared", {"text": "shared hit", "source_file": "a.epub"}),
        FakePoint("s2", {"text": "sparse hit 2", "source_file": "a.epub"}),
    ])


# ─── Tests ───────────────────────────────────────────────────────────


class TestDenseWeightZero:
    """dense_weight=0 should zero out dense RRF contribution."""

    @patch("servers.mcp_server.retriever.get_sparse_vectors")
    @patch("servers.mcp_server.retriever.get_dense_vectors")
    def test_dense_weight_zero_only_sparse_scores(self, mock_dense_vec, mock_sparse_vec):
        mock_dense_vec.return_value = [[0.1] * 768]
        mock_sparse_vec.return_value = [{"indices": [1], "values": [0.5]}]

        retriever = _make_retriever()
        retriever._storage._client.query_points = MagicMock(
            side_effect=[_make_dense_hits(), _make_sparse_hits()]
        )

        results = retriever.hybrid_search(
            "test query", "test-col", top_k=10,
            sparse_weight=1.0, dense_weight=0.0,
        )

        scores = {r.point_id: r.score for r in results}

        # d1, d2 only in dense → with dense_weight=0, score = 0
        assert scores["d1"] == 0.0
        assert scores["d2"] == 0.0

        # s1 only in sparse → nonzero
        assert scores["s1"] > 0.0

        # shared in both → only sparse contribution
        assert scores["shared"] > 0.0


class TestSparseWeightZero:
    """sparse_weight=0 should zero out sparse RRF contribution."""

    @patch("servers.mcp_server.retriever.get_sparse_vectors")
    @patch("servers.mcp_server.retriever.get_dense_vectors")
    def test_sparse_weight_zero_only_dense_scores(self, mock_dense_vec, mock_sparse_vec):
        mock_dense_vec.return_value = [[0.1] * 768]
        mock_sparse_vec.return_value = [{"indices": [1], "values": [0.5]}]

        retriever = _make_retriever()
        retriever._storage._client.query_points = MagicMock(
            side_effect=[_make_dense_hits(), _make_sparse_hits()]
        )

        results = retriever.hybrid_search(
            "test query", "test-col", top_k=10,
            sparse_weight=0.0, dense_weight=1.0,
        )

        scores = {r.point_id: r.score for r in results}

        # s1, s2 only in sparse → with sparse_weight=0, score = 0
        assert scores["s1"] == 0.0
        assert scores["s2"] == 0.0

        # d1 only in dense → nonzero
        assert scores["d1"] > 0.0

        # shared in both → only dense contribution
        assert scores["shared"] > 0.0


class TestDefaultWeightsUnchanged:
    """Default weights should produce same behavior as before."""

    @patch("servers.mcp_server.retriever.get_sparse_vectors")
    @patch("servers.mcp_server.retriever.get_dense_vectors")
    def test_defaults_both_signals_contribute(self, mock_dense_vec, mock_sparse_vec):
        mock_dense_vec.return_value = [[0.1] * 768]
        mock_sparse_vec.return_value = [{"indices": [1], "values": [0.5]}]

        retriever = _make_retriever()
        retriever._storage._client.query_points = MagicMock(
            side_effect=[_make_dense_hits(), _make_sparse_hits()]
        )

        results = retriever.hybrid_search(
            "test query", "test-col", top_k=10,
            sparse_weight=DEFAULT_SPARSE_WEIGHT,
            dense_weight=DEFAULT_DENSE_WEIGHT,
        )

        scores = {r.point_id: r.score for r in results}

        # All IDs should have nonzero scores
        for pid in ("d1", "d2", "s1", "s2", "shared"):
            assert scores[pid] > 0.0, f"{pid} should have nonzero score"

        # shared should have highest score (both signals)
        assert scores["shared"] > scores["d1"]
        assert scores["shared"] > scores["s1"]


class TestPointIdPopulated:
    """hybrid_search results should have point_id set."""

    @patch("servers.mcp_server.retriever.get_sparse_vectors")
    @patch("servers.mcp_server.retriever.get_dense_vectors")
    def test_point_id_in_results(self, mock_dense_vec, mock_sparse_vec):
        mock_dense_vec.return_value = [[0.1] * 768]
        mock_sparse_vec.return_value = [{"indices": [1], "values": [0.5]}]

        retriever = _make_retriever()
        retriever._storage._client.query_points = MagicMock(
            side_effect=[_make_dense_hits(), _make_sparse_hits()]
        )

        results = retriever.hybrid_search("test query", "test-col", top_k=10)

        for r in results:
            assert r.point_id is not None
            assert isinstance(r.point_id, str)
            assert len(r.point_id) > 0

        point_ids = {r.point_id for r in results}
        assert "d1" in point_ids
        assert "shared" in point_ids
        assert "s1" in point_ids
