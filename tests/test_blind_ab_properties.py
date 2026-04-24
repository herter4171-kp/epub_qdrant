"""Property-based tests for blind A/B test pure functions.

Properties 1, 3, 4, 5 (from design doc).
Tag format: Feature: retrieval-blind-test, Property N: <title>
"""

import json
import random

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from scripts.blind_ab_test import (
    intersect_books,
    parse_judge_response,
    map_verdict,
    compute_hit_rank,
)


# ─── Strategies ──────────────────────────────────────────────────────

source_file_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="._-/"),
    min_size=1,
    max_size=30,
)

book_dict_st = st.fixed_dictionaries({
    "source_file": source_file_st,
    "book_title": st.text(min_size=0, max_size=50),
    "chunk_count": st.integers(min_value=0, max_value=10000),
})

book_list_st = st.lists(book_dict_st, min_size=0, max_size=20)


# ─── Property 1: Book intersection correctness ──────────────────────
# For any two lists of book dicts, intersection returns exactly those
# books whose source_file appears in both lists, and no others.
# Validates: Requirements 2.3, 2.4


class TestProperty1BookIntersection:
    """Feature: retrieval-blind-test, Property 1: Book intersection correctness."""

    @given(dense=book_list_st, hybrid=book_list_st)
    @settings(max_examples=200)
    def test_intersection_returns_only_common_source_files(self, dense, hybrid):
        result = intersect_books(dense, hybrid)
        result_sfs = {b["source_file"] for b in result}
        dense_sfs = {b["source_file"] for b in dense}
        hybrid_sfs = {b["source_file"] for b in hybrid}
        expected = dense_sfs & hybrid_sfs
        assert result_sfs == expected

    @given(dense=book_list_st, hybrid=book_list_st)
    @settings(max_examples=200)
    def test_intersection_excludes_non_common(self, dense, hybrid):
        result = intersect_books(dense, hybrid)
        dense_sfs = {b["source_file"] for b in dense}
        hybrid_sfs = {b["source_file"] for b in hybrid}
        for b in result:
            assert b["source_file"] in dense_sfs
            assert b["source_file"] in hybrid_sfs

    @given(books=book_list_st)
    @settings(max_examples=100)
    def test_intersection_with_self_returns_all_unique(self, books):
        result = intersect_books(books, books)
        result_sfs = {b["source_file"] for b in result}
        expected_sfs = {b["source_file"] for b in books}
        assert result_sfs == expected_sfs


# ─── Property 3: Judge response parsing ─────────────────────────────
# Valid JSON with winner/reason → correct extraction.
# Invalid JSON → tie/judge_error fallback.
# Validates: Requirements 7.5, 7.7

winner_st = st.sampled_from(["A", "B", "tie"])
reason_st = st.text(min_size=1, max_size=100)


class TestProperty3JudgeParsing:
    """Feature: retrieval-blind-test, Property 3: Judge response parsing."""

    @given(winner=winner_st, reason=reason_st)
    @settings(max_examples=200)
    def test_valid_json_extracted_correctly(self, winner, reason):
        raw = json.dumps({"winner": winner, "reason": reason})
        result = parse_judge_response(raw)
        assert result["winner"] == winner
        assert result["reason"] == reason
        assert result["judge_raw"] == raw

    @given(garbage=st.text(min_size=1, max_size=200))
    @settings(max_examples=200)
    def test_invalid_json_returns_tie_judge_error(self, garbage):
        # Make sure it's not accidentally valid judge JSON
        try:
            parsed = json.loads(garbage)
            if isinstance(parsed, dict) and "winner" in parsed:
                assume(False)
        except (json.JSONDecodeError, TypeError):
            pass
        result = parse_judge_response(garbage)
        assert result["winner"] == "tie"
        assert result["reason"] == "judge_error"
        assert result["judge_raw"] == garbage

    @given(winner=st.text(min_size=1, max_size=20).filter(lambda x: x not in ("A", "B", "tie")))
    @settings(max_examples=100)
    def test_invalid_winner_value_becomes_tie(self, winner):
        raw = json.dumps({"winner": winner, "reason": "test"})
        result = parse_judge_response(raw)
        assert result["winner"] == "tie"


# ─── Property 4: A/B → collection mapping ───────────────────────────
# For all combos of a_src/b_src and verdict, verify correct mapping.
# Validates: Requirements 7.6

ab_source_st = st.sampled_from(["dense", "hybrid"])


class TestProperty4ABMapping:
    """Feature: retrieval-blind-test, Property 4: A/B → collection mapping."""

    @given(verdict=winner_st, a_src=ab_source_st)
    @settings(max_examples=200)
    def test_mapping_correctness(self, verdict, a_src):
        b_src = "hybrid" if a_src == "dense" else "dense"
        result = map_verdict(verdict, a_src, b_src)
        if verdict == "A":
            assert result == a_src
        elif verdict == "B":
            assert result == b_src
        else:
            assert result == "tie"

    @given(a_src=ab_source_st)
    @settings(max_examples=100)
    def test_tie_always_returns_tie(self, a_src):
        b_src = "hybrid" if a_src == "dense" else "dense"
        assert map_verdict("tie", a_src, b_src) == "tie"


# ─── Property 5: Hit rank computation ───────────────────────────────
# Validates: Requirements 8.1, 8.2

chunk_index_st = st.integers(min_value=0, max_value=1000)
score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


class TestProperty5HitRank:
    """Feature: retrieval-blind-test, Property 5: Hit rank computation."""

    @given(
        seed_idx=chunk_index_st,
        range_lo=st.integers(min_value=0, max_value=500),
        range_size=st.integers(min_value=0, max_value=10),
        retrieved_indices=st.lists(chunk_index_st, min_size=1, max_size=20),
        scores=st.lists(score_st, min_size=1, max_size=20),
    )
    @settings(max_examples=200)
    def test_chunk_index_overlap_detected(self, seed_idx, range_lo, range_size,
                                          retrieved_indices, scores):
        range_hi = range_lo + range_size
        source_file = "test.epub"
        source_chunk = {
            "source_file": source_file,
            "seed_chunk_index": seed_idx,
            "chunk_range": [range_lo, range_hi],
            "text": "some source text here for testing",
        }

        # Pad scores to match indices
        while len(scores) < len(retrieved_indices):
            scores.append(0.5)

        chunks = [
            {"source_file": source_file, "chunk_index": idx, "score": sc, "text": f"chunk {idx}"}
            for idx, sc in zip(retrieved_indices, scores)
        ]
        results = {"groups": [{"chunks": chunks}]}

        result = compute_hit_rank(source_chunk, results)

        # Check: if any retrieved chunk overlaps range, should find it
        has_overlap = any(range_lo <= idx <= range_hi for idx in retrieved_indices)
        if has_overlap:
            assert result["rank"] is not None
            assert result["match_method"] == "chunk_index"
            assert result["rank"] >= 1
        # If no overlap, could still match via text or be none

    @given(data=st.data())
    @settings(max_examples=100)
    def test_no_match_returns_none(self, data):
        source_chunk = {
            "source_file": "book_a.epub",
            "seed_chunk_index": 50,
            "chunk_range": [48, 52],
            "text": "completely unique source text xyz123",
        }
        # Retrieved chunks from different file, no text overlap
        chunks = [
            {"source_file": "book_b.epub", "chunk_index": i, "score": 0.5,
             "text": f"unrelated content number {i} about different topic"}
            for i in range(5)
        ]
        results = {"groups": [{"chunks": chunks}]}
        result = compute_hit_rank(source_chunk, results)
        assert result["rank"] is None
        assert result["match_method"] == "none"

    def test_text_overlap_fallback(self):
        """When chunk_index doesn't match but text overlaps >50%, use text_overlap."""
        words = "the quick brown fox jumps over the lazy dog near the river bank"
        source_chunk = {
            "source_file": "book.epub",
            "seed_chunk_index": 10,
            "chunk_range": [10, 10],
            "text": words,
        }
        # Different file so chunk_index won't match, but same text
        chunks = [
            {"source_file": "other.epub", "chunk_index": 99, "score": 0.9, "text": words},
        ]
        results = {"groups": [{"chunks": chunks}]}
        result = compute_hit_rank(source_chunk, results)
        assert result["rank"] == 1
        assert result["match_method"] == "text_overlap"

    def test_empty_results(self):
        source_chunk = {
            "source_file": "book.epub",
            "seed_chunk_index": 5,
            "chunk_range": [5, 5],
            "text": "some text",
        }
        result = compute_hit_rank(source_chunk, {"groups": []})
        assert result["rank"] is None
        assert result["match_method"] == "none"
