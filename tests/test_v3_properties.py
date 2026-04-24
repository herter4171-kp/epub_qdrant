"""Property-based tests for v3 pure functions (Properties 1-5, 8-9)."""

import json
import math

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from scripts.blind_ab_test import (
    _flatten_with_rank,
    dedup_and_normalize,
    _u_shape_order,
    _parse_reranker_response,
    _parse_judge_response_v3,
)


# ─── Strategies ──────────────────────────────────────────────────────

# Generate a chunk dict with random point_id and score
chunk_st = st.fixed_dictionaries({
    "point_id": st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
    "score": st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    "text": st.text(min_size=1, max_size=50),
    "source_file": st.just("test.epub"),
    "title": st.just("Test"),
    "section_title": st.just("Ch1"),
    "chunk_index": st.integers(min_value=0, max_value=100),
})


def mcp_results_st(chunk_strategy=chunk_st):
    """Strategy for MCP-shaped result dicts."""
    return st.fixed_dictionaries({
        "groups": st.lists(
            st.fixed_dictionaries({
                "chunks": st.lists(chunk_strategy, min_size=0, max_size=5),
            }),
            min_size=0, max_size=3,
        ),
    })


# ─── Property 1: Dedup set-union correctness ─────────────────────────


class TestProperty1DedupSetUnion:

    @given(dense=mcp_results_st(), sparse=mcp_results_st())
    @settings(max_examples=100)
    def test_output_size_equals_union(self, dense, sparse):
        candidates = dedup_and_normalize(dense, sparse)

        dense_pids = {c["point_id"] for g in dense.get("groups", [])
                      for c in g.get("chunks", []) if c.get("text") and c.get("point_id")}
        sparse_pids = {c["point_id"] for g in sparse.get("groups", [])
                       for c in g.get("chunks", []) if c.get("text") and c.get("point_id")}
        expected_union = dense_pids | sparse_pids

        output_pids = {c["point_id"] for c in candidates}
        assert output_pids == expected_union

    @given(dense=mcp_results_st(), sparse=mcp_results_st())
    @settings(max_examples=100)
    def test_signal_labels_correct(self, dense, sparse):
        candidates = dedup_and_normalize(dense, sparse)

        dense_pids = {c["point_id"] for g in dense.get("groups", [])
                      for c in g.get("chunks", []) if c.get("text") and c.get("point_id")}
        sparse_pids = {c["point_id"] for g in sparse.get("groups", [])
                       for c in g.get("chunks", []) if c.get("text") and c.get("point_id")}

        for c in candidates:
            pid = c["point_id"]
            in_dense = pid in dense_pids
            in_sparse = pid in sparse_pids
            if in_dense and in_sparse:
                assert c["signal"] == "both"
            elif in_dense:
                assert c["signal"] == "dense"
            else:
                assert c["signal"] == "sparse"


# ─── Property 2: Score normalization bounds ──────────────────────────


class TestProperty2NormalizationBounds:

    @given(dense=mcp_results_st(), sparse=mcp_results_st())
    @settings(max_examples=100)
    def test_normalized_scores_in_unit_interval(self, dense, sparse):
        candidates = dedup_and_normalize(dense, sparse)
        for c in candidates:
            if c["dense_score_norm"] is not None:
                assert 0.0 <= c["dense_score_norm"] <= 1.0
            if c["sparse_score_norm"] is not None:
                assert 0.0 <= c["sparse_score_norm"] <= 1.0

    @given(scores=st.lists(st.floats(min_value=0.0, max_value=10.0,
                                      allow_nan=False, allow_infinity=False),
                           min_size=2, max_size=10))
    @settings(max_examples=50)
    def test_min_maps_to_zero_max_to_one(self, scores):
        assume(max(scores) > min(scores))  # skip identical
        chunks = [{"point_id": f"p{i}", "score": s, "text": "t",
                    "source_file": "a.epub", "title": "T", "section_title": "S",
                    "chunk_index": i} for i, s in enumerate(scores)]
        dense = {"groups": [{"chunks": chunks}]}
        sparse = {"groups": []}
        candidates = dedup_and_normalize(dense, sparse)
        norms = [c["dense_score_norm"] for c in candidates]
        assert min(norms) == 0.0
        assert max(norms) == 1.0

    @given(val=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20)
    def test_identical_scores_all_one(self, val):
        chunks = [{"point_id": f"p{i}", "score": val, "text": "t",
                    "source_file": "a.epub", "title": "T", "section_title": "S",
                    "chunk_index": i} for i in range(3)]
        dense = {"groups": [{"chunks": chunks}]}
        sparse = {"groups": []}
        candidates = dedup_and_normalize(dense, sparse)
        for c in candidates:
            assert c["dense_score_norm"] == 1.0


# ─── Property 3: Reranker output is a permutation ───────────────────


class TestProperty3RerankerPermutation:

    @given(raw=st.text(max_size=200), n=st.integers(min_value=1, max_value=20))
    @settings(max_examples=100)
    def test_always_permutation(self, raw, n):
        result = _parse_reranker_response(raw, n)
        assert sorted(result) == list(range(n))

    @given(n=st.integers(min_value=1, max_value=20))
    @settings(max_examples=50)
    def test_valid_json_preserves_order(self, n):
        indices = list(range(n))
        import random as rng
        rng.shuffle(indices)
        raw = json.dumps({"ranked_indices": indices})
        result = _parse_reranker_response(raw, n)
        assert result == indices


# ─── Property 4: Judge score domain ──────────────────────────────────


class TestProperty4JudgeScoreDomain:

    @given(raw=st.text(max_size=200))
    @settings(max_examples=100)
    def test_score_always_in_domain(self, raw):
        result = _parse_judge_response_v3(raw)
        assert result["score"] in (1, 2, 3)
        assert "judge_raw" in result
        assert "reason" in result


# ─── Property 5: Signal label consistency ────────────────────────────


class TestProperty5SignalLabelConsistency:

    @given(dense=mcp_results_st(), sparse=mcp_results_st())
    @settings(max_examples=100)
    def test_signal_score_field_consistency(self, dense, sparse):
        candidates = dedup_and_normalize(dense, sparse)
        for c in candidates:
            if c["signal"] == "both":
                assert c["dense_score"] is not None
                assert c["sparse_score"] is not None
                assert c["dense_score_norm"] is not None
                assert c["sparse_score_norm"] is not None
            elif c["signal"] == "dense":
                assert c["dense_score"] is not None
                assert c["sparse_score"] is None
                assert c["sparse_rank"] is None
                assert c["sparse_score_norm"] is None
            elif c["signal"] == "sparse":
                assert c["sparse_score"] is not None
                assert c["dense_score"] is None
                assert c["dense_rank"] is None
                assert c["dense_score_norm"] is None


# ─── Property 8: Flatten preserves all non-empty chunks ─────────────


class TestProperty8FlattenPreservesChunks:

    @given(results=mcp_results_st())
    @settings(max_examples=100)
    def test_all_nonempty_chunks_present(self, results):
        expected = []
        for g in results.get("groups", []):
            for c in g.get("chunks", []):
                if c.get("text"):
                    expected.append(c["point_id"])

        flat = _flatten_with_rank(results)
        flat_pids = [c["point_id"] for c in flat]

        # All expected present (may have dupes if same pid in multiple groups)
        for pid in set(expected):
            assert pid in flat_pids

    @given(results=mcp_results_st())
    @settings(max_examples=100)
    def test_ranks_consecutive_1_based(self, results):
        flat = _flatten_with_rank(results)
        if flat:
            ranks = [c["rank"] for c in flat]
            assert ranks == list(range(1, len(flat) + 1))


# ─── Property 9: U-shape ordering preserves elements and structure ───


class TestProperty9UShapeOrdering:

    @given(items=st.lists(st.integers(), min_size=0, max_size=20))
    @settings(max_examples=100)
    def test_same_length_same_elements(self, items):
        result = _u_shape_order(items)
        assert len(result) == len(items)
        assert sorted(result) == sorted(items)

    @given(items=st.lists(st.integers(), min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_first_element_preserved(self, items):
        result = _u_shape_order(items)
        assert result[0] == items[0]

    @given(items=st.lists(st.integers(), min_size=3, max_size=20))
    @settings(max_examples=100)
    def test_structural_invariants(self, items):
        n = len(items)
        mid = (n + 1) // 2
        result = _u_shape_order(items)
        # Top half in original order
        assert result[:mid] == items[:mid]
        # Bottom half reversed
        assert result[mid:] == items[mid:][::-1]


# ─── Property 7: Aggregate score computation ─────────────────────────

from scripts.blind_ab_test import compute_aggregates


def _sample_st():
    """Strategy for sample dicts with random judge_score and bucket."""
    return st.fixed_dictionaries({
        "judge_score": st.sampled_from([1, 2, 3]),
        "query_bucket": st.sampled_from(["trivia", "conceptual", "operational"]),
        "dense_source_hit_rank": st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
        "sparse_source_hit_rank": st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
        "dedup_set_size": st.integers(min_value=1, max_value=20),
        "both_signal_count": st.integers(min_value=0, max_value=10),
    })


class TestProperty7AggregateScoreComputation:

    @given(samples=st.lists(_sample_st(), min_size=1, max_size=30))
    @settings(max_examples=100)
    def test_avg_judge_score_is_mean(self, samples):
        agg = compute_aggregates(samples)
        expected = sum(s["judge_score"] for s in samples) / len(samples)
        assert abs(agg["avg_judge_score"] - round(expected, 2)) < 0.01

    @given(samples=st.lists(_sample_st(), min_size=1, max_size=30))
    @settings(max_examples=100)
    def test_per_bucket_avg_correct(self, samples):
        agg = compute_aggregates(samples)
        for bucket in ("trivia", "conceptual", "operational"):
            bucket_samples = [s for s in samples if s["query_bucket"] == bucket]
            if bucket_samples:
                expected = sum(s["judge_score"] for s in bucket_samples) / len(bucket_samples)
                assert abs(agg["bucket_summary"][bucket]["avg_score"] - expected) < 0.01

    @given(samples=st.lists(_sample_st(), min_size=1, max_size=30))
    @settings(max_examples=100)
    def test_hit_rates_correct(self, samples):
        agg = compute_aggregates(samples)
        total = len(samples)
        dense_hits = sum(1 for s in samples if s["dense_source_hit_rank"] is not None)
        sparse_hits = sum(1 for s in samples if s["sparse_source_hit_rank"] is not None)
        assert abs(agg["dense_hit_rate"] - round(dense_hits / total, 3)) < 0.001
        assert abs(agg["sparse_hit_rate"] - round(sparse_hits / total, 3)) < 0.001
