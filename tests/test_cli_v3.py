"""Unit tests for v3 CLI parsing and compute_aggregates."""

import pytest

from scripts.blind_ab_test import (
    parse_args,
    compute_aggregates,
    DEFAULT_DENSE_K,
    DEFAULT_SPARSE_K,
)


# ─── parse_args ──────────────────────────────────────────────────────


class TestParseArgs:

    def test_single_positional_collection(self):
        args = parse_args(["books-semantic"])
        assert args.collection == "books-semantic"

    def test_dense_k_default(self):
        args = parse_args(["col"])
        assert args.dense_k == DEFAULT_DENSE_K

    def test_sparse_k_default(self):
        args = parse_args(["col"])
        assert args.sparse_k == DEFAULT_SPARSE_K

    def test_dense_k_override(self):
        args = parse_args(["col", "--dense-k", "10"])
        assert args.dense_k == 10

    def test_sparse_k_override(self):
        args = parse_args(["col", "--sparse-k", "8"])
        assert args.sparse_k == 8

    def test_positions_default(self):
        args = parse_args(["col"])
        assert args.positions == 1

    def test_positions_override(self):
        args = parse_args(["col", "--positions", "3"])
        assert args.positions == 3

    def test_output_default(self):
        args = parse_args(["col"])
        assert args.output == "results/blind_ab_test.json"

    def test_output_override(self):
        args = parse_args(["col", "--output", "/tmp/test.json"])
        assert args.output == "/tmp/test.json"

    def test_seed_default_none(self):
        args = parse_args(["col"])
        assert args.seed is None

    def test_seed_override(self):
        args = parse_args(["col", "--seed", "42"])
        assert args.seed == 42

    def test_no_top_k_arg(self):
        """--top-k should not exist in v3."""
        with pytest.raises(SystemExit):
            parse_args(["col", "--top-k", "10"])

    def test_no_sparse_weight_arg(self):
        """--sparse-weight should not exist in v3."""
        with pytest.raises(SystemExit):
            parse_args(["col", "--sparse-weight", "1.2"])

    def test_no_dense_collection_positional(self):
        """Old two-positional-arg form should fail."""
        with pytest.raises(SystemExit):
            parse_args(["books", "books-semantic"])


# ─── compute_aggregates ─────────────────────────────────────────────


def _sample(score, bucket, dense_hit=None, sparse_hit=None, dedup=7, both=2):
    return {
        "judge_score": score,
        "query_bucket": bucket,
        "dense_source_hit_rank": dense_hit,
        "sparse_source_hit_rank": sparse_hit,
        "dedup_set_size": dedup,
        "both_signal_count": both,
    }


class TestComputeAggregates:

    def test_mixed_scores(self):
        samples = [
            _sample(3, "trivia", dense_hit=1),
            _sample(2, "trivia"),
            _sample(1, "conceptual", sparse_hit=2),
            _sample(3, "operational", dense_hit=1, sparse_hit=1),
        ]
        agg = compute_aggregates(samples)

        assert agg["avg_judge_score"] == 2.25
        assert agg["dense_hit_count"] == 2
        assert agg["sparse_hit_count"] == 2
        assert agg["dense_hit_rate"] == 0.5
        assert agg["sparse_hit_rate"] == 0.5

        # Trivia bucket
        trivia = agg["bucket_summary"]["trivia"]
        assert trivia["avg_score"] == 2.5
        assert trivia["samples"] == 2
        assert trivia["score_distribution"] == {1: 0, 2: 1, 3: 1}

        # Conceptual bucket
        conceptual = agg["bucket_summary"]["conceptual"]
        assert conceptual["avg_score"] == 1.0
        assert conceptual["samples"] == 1

    def test_empty_samples(self):
        agg = compute_aggregates([])
        assert agg["avg_judge_score"] == 0.0
        assert agg["dense_hit_rate"] == 0.0
        assert agg["sparse_hit_rate"] == 0.0
        assert agg["avg_dedup_set_size"] == 0.0
        assert agg["avg_both_signal_count"] == 0.0
        for bucket in ("trivia", "conceptual", "operational"):
            assert agg["bucket_summary"][bucket]["avg_score"] == 0.0
            assert agg["bucket_summary"][bucket]["samples"] == 0

    def test_all_fields_present(self):
        agg = compute_aggregates([_sample(3, "trivia")])
        expected_keys = {
            "avg_judge_score", "dense_hit_count", "sparse_hit_count",
            "dense_hit_rate", "sparse_hit_rate", "avg_dedup_set_size",
            "avg_both_signal_count", "bucket_summary",
        }
        assert set(agg.keys()) == expected_keys

    def test_dedup_and_both_averages(self):
        samples = [
            _sample(3, "trivia", dedup=8, both=3),
            _sample(2, "trivia", dedup=6, both=1),
        ]
        agg = compute_aggregates(samples)
        assert agg["avg_dedup_set_size"] == 7.0
        assert agg["avg_both_signal_count"] == 2.0
