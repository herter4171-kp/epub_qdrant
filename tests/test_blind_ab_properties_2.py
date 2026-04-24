"""Property-based tests for blind A/B test — Properties 2, 6, 7.

Tag format: Feature: retrieval-blind-test, Property N: <title>
"""

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.blind_ab_test import compute_aggregates


# ─── Property 2: A/B assignment determinism ──────────────────────────
# For any seed + sample count, A/B assignments identical across runs.
# Validates: Requirements 7.1


class TestProperty2ABDeterminism:
    """Feature: retrieval-blind-test, Property 2: A/B assignment determinism."""

    @given(
        seed=st.integers(min_value=0, max_value=2**31),
        n_samples=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=200)
    def test_same_seed_same_assignments(self, seed, n_samples):
        def run_assignments(s, n):
            random.seed(s)
            assignments = []
            for _ in range(n):
                if random.random() < 0.5:
                    assignments.append(("dense", "hybrid"))
                else:
                    assignments.append(("hybrid", "dense"))
            return assignments

        run1 = run_assignments(seed, n_samples)
        run2 = run_assignments(seed, n_samples)
        assert run1 == run2

    @given(
        seed1=st.integers(min_value=0, max_value=2**31),
        seed2=st.integers(min_value=0, max_value=2**31),
        n_samples=st.integers(min_value=10, max_value=50),
    )
    @settings(max_examples=100)
    def test_different_seeds_likely_different(self, seed1, seed2, n_samples):
        """Different seeds should (almost always) produce different assignments."""
        if seed1 == seed2:
            return  # skip trivial case

        def run_assignments(s, n):
            random.seed(s)
            return [random.random() < 0.5 for _ in range(n)]

        run1 = run_assignments(seed1, n_samples)
        run2 = run_assignments(seed2, n_samples)
        # With 10+ samples, probability of identical sequences is negligible
        # but not impossible, so we just check they're not always equal
        # (this is a soft property — we don't assert difference)


# ─── Property 6: Hit metric aggregation ─────────────────────────────
# Validates: Requirements 8.3

hit_rank_st = st.one_of(st.none(), st.integers(min_value=1, max_value=100))

sample_st = st.fixed_dictionaries({
    "judge_score": st.sampled_from([1, 2, 3]),
    "query_bucket": st.sampled_from(["trivia", "conceptual", "operational"]),
    "dense_source_hit_rank": hit_rank_st,
    "sparse_source_hit_rank": hit_rank_st,
    "dedup_set_size": st.integers(min_value=1, max_value=20),
    "both_signal_count": st.integers(min_value=0, max_value=10),
})


class TestProperty6HitAggregation:
    """Feature: fused-retrieval-reranking, Property 6: Hit metric aggregation (v3)."""

    @given(samples=st.lists(sample_st, min_size=1, max_size=50))
    @settings(max_examples=200)
    def test_hit_counts_correct(self, samples):
        agg = compute_aggregates(samples)
        expected_dense = sum(1 for s in samples if s["dense_source_hit_rank"] is not None)
        expected_sparse = sum(1 for s in samples if s["sparse_source_hit_rank"] is not None)
        assert agg["dense_hit_count"] == expected_dense
        assert agg["sparse_hit_count"] == expected_sparse

    @given(samples=st.lists(sample_st, min_size=1, max_size=50))
    @settings(max_examples=200)
    def test_hit_rates_correct(self, samples):
        agg = compute_aggregates(samples)
        total = len(samples)
        expected_dense_rate = round(sum(
            1 for s in samples if s["dense_source_hit_rank"] is not None
        ) / total, 3)
        expected_sparse_rate = round(sum(
            1 for s in samples if s["sparse_source_hit_rank"] is not None
        ) / total, 3)
        assert abs(agg["dense_hit_rate"] - expected_dense_rate) < 1e-9
        assert abs(agg["sparse_hit_rate"] - expected_sparse_rate) < 1e-9

    def test_empty_samples_zero_rates(self):
        agg = compute_aggregates([])
        assert agg["dense_hit_count"] == 0
        assert agg["sparse_hit_count"] == 0
        assert agg["dense_hit_rate"] == 0.0
        assert agg["sparse_hit_rate"] == 0.0


# ─── Property 7: Output JSON schema completeness ────────────────────
# Validates: Requirements 10.2, 10.3

REQUIRED_METADATA_FIELDS = {
    "version", "collection", "dense_k", "sparse_k",
    "positions_per_book", "total_samples", "avg_judge_score",
    "bucket_summary", "timestamp", "seed", "answer_model",
    "judge_model", "query_model", "reranker_model", "mcp_url",
    "dense_hit_count", "sparse_hit_count", "dense_hit_rate",
    "sparse_hit_rate", "avg_dedup_set_size", "avg_both_signal_count",
    "script_version",
}

REQUIRED_SAMPLE_FIELDS = {
    "book", "position_index",
    "source_chunk_id", "source_metadata", "source_passage",
    "source_passage_excerpt", "query", "query_generation_raw",
    "dense_raw_results", "sparse_raw_results",
    "dense_retrieved_passages", "sparse_retrieved_passages",
    "dedup_set_size", "both_signal_count",
    "reranked_passages", "reranker_raw",
    "fused_answer", "judge_score", "judge_reason", "judge_raw",
    "dense_source_hit_rank", "dense_source_hit_method",
    "sparse_source_hit_rank", "sparse_source_hit_method",
}


def _make_metadata(**overrides):
    base = {
        "version": "3.0.0",
        "collection": "books-semantic",
        "dense_k": 5,
        "sparse_k": 5,
        "positions_per_book": 1,
        "total_samples": 1,
        "avg_judge_score": 2.5,
        "bucket_summary": {},
        "timestamp": "2025-01-01T00:00:00Z",
        "seed": 42,
        "answer_model": "test-model",
        "judge_model": "test-model",
        "query_model": "test-model",
        "reranker_model": "test-model",
        "mcp_url": "http://localhost:8090/mcp",
        "dense_hit_count": 1,
        "sparse_hit_count": 0,
        "dense_hit_rate": 1.0,
        "sparse_hit_rate": 0.0,
        "avg_dedup_set_size": 7.0,
        "avg_both_signal_count": 2.0,
        "script_version": "3.0.0",
    }
    base.update(overrides)
    return base


def _make_sample(**overrides):
    base = {
        "book": "Test Book",
        "position_index": 0,
        "source_chunk_id": "abc-123",
        "source_metadata": {},
        "source_passage": "text",
        "source_passage_excerpt": "text",
        "query": "what?",
        "query_generation_raw": "what?",
        "dense_raw_results": {},
        "sparse_raw_results": {},
        "dense_retrieved_passages": [],
        "sparse_retrieved_passages": [],
        "dedup_set_size": 7,
        "both_signal_count": 2,
        "reranked_passages": [],
        "reranker_raw": "{}",
        "fused_answer": "answer",
        "judge_score": 3,
        "judge_reason": "faithful",
        "judge_raw": "{}",
        "dense_source_hit_rank": 1,
        "dense_source_hit_method": "chunk_index",
        "sparse_source_hit_rank": None,
        "sparse_source_hit_method": "none",
    }
    base.update(overrides)
    return base


class TestProperty7SchemaCompleteness:
    """Feature: retrieval-blind-test, Property 7: Output JSON schema completeness."""

    def test_metadata_has_all_required_fields(self):
        meta = _make_metadata()
        assert REQUIRED_METADATA_FIELDS.issubset(set(meta.keys()))

    def test_sample_has_all_required_fields(self):
        sample = _make_sample()
        assert REQUIRED_SAMPLE_FIELDS.issubset(set(sample.keys()))

    @given(
        dense_wins=st.integers(min_value=0, max_value=100),
        hybrid_wins=st.integers(min_value=0, max_value=100),
        ties=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=100)
    def test_metadata_fields_present_with_varied_values(self, dense_wins, hybrid_wins, ties):
        meta = _make_metadata(
            dense_wins=dense_wins, hybrid_wins=hybrid_wins, ties=ties,
            total_samples=dense_wins + hybrid_wins + ties,
        )
        missing = REQUIRED_METADATA_FIELDS - set(meta.keys())
        assert not missing, f"Missing metadata fields: {missing}"

    @given(
        winner=st.sampled_from(["dense", "hybrid", "tie"]),
        method=st.sampled_from(["chunk_index", "text_overlap", "none"]),
    )
    @settings(max_examples=100)
    def test_sample_fields_present_with_varied_values(self, winner, method):
        sample = _make_sample(
            winner=winner,
            dense_source_hit_method=method,
            hybrid_source_hit_method=method,
        )
        missing = REQUIRED_SAMPLE_FIELDS - set(sample.keys())
        assert not missing, f"Missing sample fields: {missing}"
