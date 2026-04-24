"""Unit tests for v3 pure functions in blind_ab_test.py."""

import pytest

from scripts.blind_ab_test import (
    _flatten_with_rank,
    dedup_and_normalize,
    _u_shape_order,
    _parse_reranker_response,
    _parse_judge_response_v3,
)


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_mcp_results(chunks):
    """Wrap chunk dicts into MCP-shaped result with one group."""
    return {"groups": [{"chunks": chunks}]}


def _chunk(pid, score, text="some text", source_file="a.epub"):
    return {
        "point_id": pid,
        "score": score,
        "text": text,
        "source_file": source_file,
        "title": "Book",
        "section_title": "Ch1",
        "chunk_index": 0,
    }


# ─── _flatten_with_rank ─────────────────────────────────────────────


class TestFlattenWithRank:

    def test_multi_group(self):
        results = {
            "groups": [
                {"chunks": [_chunk("a", 0.9), _chunk("b", 0.7)]},
                {"chunks": [_chunk("c", 0.8)]},
            ]
        }
        flat = _flatten_with_rank(results)
        assert len(flat) == 3
        assert flat[0]["point_id"] == "a"
        assert flat[0]["rank"] == 1
        assert flat[1]["point_id"] == "c"
        assert flat[1]["rank"] == 2
        assert flat[2]["point_id"] == "b"
        assert flat[2]["rank"] == 3

    def test_empty_groups(self):
        assert _flatten_with_rank({"groups": []}) == []
        assert _flatten_with_rank({}) == []

    def test_empty_text_filtered(self):
        results = _make_mcp_results([
            _chunk("a", 0.9, text="real text"),
            _chunk("b", 0.8, text=""),
            _chunk("c", 0.7, text="more text"),
        ])
        flat = _flatten_with_rank(results)
        assert len(flat) == 2
        pids = [c["point_id"] for c in flat]
        assert "b" not in pids

    def test_ranks_consecutive(self):
        results = _make_mcp_results([_chunk(f"p{i}", 1.0 - i * 0.1) for i in range(5)])
        flat = _flatten_with_rank(results)
        ranks = [c["rank"] for c in flat]
        assert ranks == [1, 2, 3, 4, 5]


# ─── dedup_and_normalize ─────────────────────────────────────────────


class TestDedupAndNormalize:

    def test_overlapping_sets(self):
        """Shared point_ids get signal='both'."""
        dense = _make_mcp_results([_chunk("shared", 0.9), _chunk("d1", 0.7)])
        sparse = _make_mcp_results([_chunk("shared", 0.02), _chunk("s1", 0.01)])
        candidates = dedup_and_normalize(dense, sparse)

        by_pid = {c["point_id"]: c for c in candidates}
        assert len(candidates) == 3  # shared + d1 + s1
        assert by_pid["shared"]["signal"] == "both"
        assert by_pid["shared"]["dense_score"] == 0.9
        assert by_pid["shared"]["sparse_score"] == 0.02
        assert by_pid["d1"]["signal"] == "dense"
        assert by_pid["s1"]["signal"] == "sparse"

    def test_disjoint_sets(self):
        """No overlap → all dense or sparse."""
        dense = _make_mcp_results([_chunk("d1", 0.9)])
        sparse = _make_mcp_results([_chunk("s1", 0.01)])
        candidates = dedup_and_normalize(dense, sparse)
        assert len(candidates) == 2
        signals = {c["point_id"]: c["signal"] for c in candidates}
        assert signals["d1"] == "dense"
        assert signals["s1"] == "sparse"

    def test_empty_one_signal(self):
        """One signal empty → all from other signal."""
        dense = _make_mcp_results([_chunk("d1", 0.9)])
        sparse = _make_mcp_results([])
        candidates = dedup_and_normalize(dense, sparse)
        assert len(candidates) == 1
        assert candidates[0]["signal"] == "dense"

    def test_identical_scores_normalize_to_one(self):
        """All same score → normalized to 1.0."""
        dense = _make_mcp_results([_chunk("d1", 0.5), _chunk("d2", 0.5)])
        sparse = _make_mcp_results([])
        candidates = dedup_and_normalize(dense, sparse)
        for c in candidates:
            assert c["dense_score_norm"] == 1.0

    def test_empty_text_excluded(self):
        """Chunks with empty text excluded."""
        dense = _make_mcp_results([_chunk("d1", 0.9, text=""), _chunk("d2", 0.8)])
        sparse = _make_mcp_results([])
        candidates = dedup_and_normalize(dense, sparse)
        assert len(candidates) == 1
        assert candidates[0]["point_id"] == "d2"

    def test_normalization_bounds(self):
        """Normalized scores in [0, 1], min→0, max→1."""
        dense = _make_mcp_results([
            _chunk("d1", 0.9), _chunk("d2", 0.5), _chunk("d3", 0.1),
        ])
        sparse = _make_mcp_results([])
        candidates = dedup_and_normalize(dense, sparse)
        norms = {c["point_id"]: c["dense_score_norm"] for c in candidates}
        assert norms["d1"] == 1.0
        assert norms["d3"] == 0.0
        assert 0.0 < norms["d2"] < 1.0

    def test_signal_score_consistency(self):
        """dense-only → sparse fields None, sparse-only → dense fields None."""
        dense = _make_mcp_results([_chunk("d1", 0.9)])
        sparse = _make_mcp_results([_chunk("s1", 0.01)])
        candidates = dedup_and_normalize(dense, sparse)
        by_pid = {c["point_id"]: c for c in candidates}

        d = by_pid["d1"]
        assert d["sparse_score"] is None
        assert d["sparse_rank"] is None
        assert d["sparse_score_norm"] is None

        s = by_pid["s1"]
        assert s["dense_score"] is None
        assert s["dense_rank"] is None
        assert s["dense_score_norm"] is None


# ─── _u_shape_order ──────────────────────────────────────────────────


class TestUShapeOrder:

    def test_empty(self):
        assert _u_shape_order([]) == []

    def test_single(self):
        assert _u_shape_order([1]) == [1]

    def test_two_unchanged(self):
        assert _u_shape_order([1, 2]) == [1, 2]

    def test_three(self):
        result = _u_shape_order([1, 2, 3])
        assert result == [1, 2, 3]  # ceil(3/2)=2 top, [3] reversed = [3]

    def test_eight(self):
        result = _u_shape_order([1, 2, 3, 4, 5, 6, 7, 8])
        assert result == [1, 2, 3, 4, 8, 7, 6, 5]

    def test_first_element_preserved(self):
        items = list(range(10))
        result = _u_shape_order(items)
        assert result[0] == 0

    def test_same_length_same_elements(self):
        items = list(range(7))
        result = _u_shape_order(items)
        assert len(result) == len(items)
        assert sorted(result) == sorted(items)


# ─── _parse_reranker_response ────────────────────────────────────────


class TestParseRerankerResponse:

    def test_valid_json(self):
        raw = '{"ranked_indices": [2, 0, 1]}'
        assert _parse_reranker_response(raw, 3) == [2, 0, 1]

    def test_invalid_json(self):
        assert _parse_reranker_response("not json", 3) == [0, 1, 2]

    def test_partial_indices(self):
        """Missing indices appended at end."""
        raw = '{"ranked_indices": [2]}'
        result = _parse_reranker_response(raw, 3)
        assert result == [2, 0, 1]

    def test_duplicate_indices(self):
        raw = '{"ranked_indices": [1, 1, 0]}'
        result = _parse_reranker_response(raw, 3)
        assert result == [1, 0, 2]

    def test_out_of_range_indices(self):
        raw = '{"ranked_indices": [5, 0, -1, 1]}'
        result = _parse_reranker_response(raw, 3)
        assert result == [0, 1, 2]

    def test_markdown_fenced_json(self):
        raw = '```json\n{"ranked_indices": [1, 0]}\n```'
        assert _parse_reranker_response(raw, 2) == [1, 0]

    def test_empty_string(self):
        assert _parse_reranker_response("", 3) == [0, 1, 2]

    def test_always_permutation(self):
        """Result is always a permutation of range(n)."""
        for raw in ['{}', '{"ranked_indices": []}', 'garbage', '{"ranked_indices": [99]}']:
            result = _parse_reranker_response(raw, 4)
            assert sorted(result) == [0, 1, 2, 3]


# ─── _parse_judge_response_v3 ───────────────────────────────────────


class TestParseJudgeResponseV3:

    def test_valid_score_1(self):
        raw = '{"score": 1, "reason": "unfaithful"}'
        result = _parse_judge_response_v3(raw)
        assert result["score"] == 1
        assert result["reason"] == "unfaithful"
        assert result["judge_raw"] == raw

    def test_valid_score_2(self):
        result = _parse_judge_response_v3('{"score": 2, "reason": "partial"}')
        assert result["score"] == 2

    def test_valid_score_3(self):
        result = _parse_judge_response_v3('{"score": 3, "reason": "faithful"}')
        assert result["score"] == 3

    def test_out_of_range_score(self):
        result = _parse_judge_response_v3('{"score": 5, "reason": "oops"}')
        assert result["score"] == 2

    def test_invalid_json(self):
        result = _parse_judge_response_v3("not json at all")
        assert result["score"] == 2
        assert result["reason"] == "judge_error"

    def test_empty_string(self):
        result = _parse_judge_response_v3("")
        assert result["score"] == 2
        assert result["reason"] == "judge_error"

    def test_markdown_fenced_json(self):
        raw = '```json\n{"score": 3, "reason": "good"}\n```'
        result = _parse_judge_response_v3(raw)
        assert result["score"] == 3
        assert result["reason"] == "good"

    def test_always_includes_judge_raw(self):
        for raw in ['{"score": 1}', "bad", "", '```\n{"score":3}\n```']:
            result = _parse_judge_response_v3(raw)
            assert "judge_raw" in result
            assert result["judge_raw"] == raw
