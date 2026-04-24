"""Unit tests for v3 LLM-calling functions: prompt construction and contracts.

All tests mock llm_call to capture prompts — no real LLM calls.
"""

import re
from unittest.mock import patch, MagicMock

import pytest

from scripts.blind_ab_test import (
    rerank,
    generate_fused_answer,
    judge_faithfulness,
    RERANKER_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT_V3,
    JUDGE_SYSTEM_PROMPT_V3,
    SIGNAL_LABELS,
    _u_shape_order,
)


# ─── Helpers ─────────────────────────────────────────────────────────


def _candidate(pid, text="passage text", signal="dense", score=0.5):
    return {
        "point_id": pid,
        "text": text,
        "source_file": "test.epub",
        "title": "Test Book",
        "section_title": "Ch1",
        "chunk_index": 0,
        "signal": signal,
        "dense_score": score if signal in ("dense", "both") else None,
        "sparse_score": score if signal in ("sparse", "both") else None,
        "dense_rank": 1 if signal in ("dense", "both") else None,
        "sparse_rank": 1 if signal in ("sparse", "both") else None,
        "dense_score_norm": 0.8 if signal in ("dense", "both") else None,
        "sparse_score_norm": 0.6 if signal in ("sparse", "both") else None,
    }


# ─── rerank() ────────────────────────────────────────────────────────


class TestRerank:

    @patch("scripts.blind_ab_test.llm_call")
    def test_no_scores_in_user_message(self, mock_llm):
        """Reranker prompt must NOT contain any numeric scores."""
        mock_llm.return_value = '{"ranked_indices": [0, 1, 2]}'
        candidates = [
            _candidate("a", "text a", "dense", 0.95),
            _candidate("b", "text b", "sparse", 0.03),
            _candidate("c", "text c", "both", 0.72),
        ]
        rerank("test query", candidates)

        user_msg = mock_llm.call_args[0][1]  # second positional arg
        # No raw scores
        assert "0.95" not in user_msg
        assert "0.03" not in user_msg
        assert "0.72" not in user_msg
        # No normalized scores
        assert "0.8" not in user_msg
        assert "0.6" not in user_msg
        assert "score_norm" not in user_msg
        assert "dense_score" not in user_msg
        assert "sparse_score" not in user_msg

    @patch("scripts.blind_ab_test.llm_call")
    def test_signal_labels_present(self, mock_llm):
        """Reranker prompt must contain signal labels."""
        mock_llm.return_value = '{"ranked_indices": [0, 1]}'
        candidates = [
            _candidate("a", "text a", "dense"),
            _candidate("b", "text b", "both"),
        ]
        rerank("test query", candidates)

        user_msg = mock_llm.call_args[0][1]
        assert "semantic" in user_msg
        assert "both (semantic + keyword)" in user_msg

    @patch("scripts.blind_ab_test.llm_call")
    def test_passage_text_present(self, mock_llm):
        """Reranker prompt must contain passage text."""
        mock_llm.return_value = '{"ranked_indices": [0]}'
        candidates = [_candidate("a", "unique passage content xyz")]
        rerank("test query", candidates)

        user_msg = mock_llm.call_args[0][1]
        assert "unique passage content xyz" in user_msg

    @patch("scripts.blind_ab_test.llm_call")
    def test_uses_reranker_system_prompt(self, mock_llm):
        mock_llm.return_value = '{"ranked_indices": [0]}'
        rerank("q", [_candidate("a")])
        system_msg = mock_llm.call_args[0][0]
        assert system_msg == RERANKER_SYSTEM_PROMPT

    @patch("scripts.blind_ab_test.llm_call")
    def test_temperature_zero(self, mock_llm):
        mock_llm.return_value = '{"ranked_indices": [0]}'
        rerank("q", [_candidate("a")])
        assert mock_llm.call_args.kwargs["temperature"] == 0.0

    @patch("scripts.blind_ab_test.llm_call")
    def test_returns_reordered_candidates(self, mock_llm):
        mock_llm.return_value = '{"ranked_indices": [1, 0]}'
        candidates = [_candidate("a"), _candidate("b")]
        result, raw = rerank("q", candidates)
        assert result[0]["point_id"] == "b"
        assert result[1]["point_id"] == "a"

    @patch("scripts.blind_ab_test.llm_call")
    def test_exception_returns_original_order(self, mock_llm):
        mock_llm.side_effect = Exception("LLM down")
        candidates = [_candidate("a"), _candidate("b")]
        result, raw = rerank("q", candidates)
        assert result[0]["point_id"] == "a"
        assert "LLM down" in raw

    def test_empty_candidates(self):
        result, raw = rerank("q", [])
        assert result == []
        assert raw == ""


# ─── generate_fused_answer() ─────────────────────────────────────────


class TestGenerateFusedAnswer:

    @patch("scripts.blind_ab_test.llm_call")
    def test_no_signal_tags_in_user_message(self, mock_llm):
        """Answer prompt must NOT contain signal tags."""
        mock_llm.return_value = "The answer is..."
        passages = [
            _candidate("a", "text a", "dense"),
            _candidate("b", "text b", "sparse"),
            _candidate("c", "text c", "both"),
        ]
        generate_fused_answer("test query", passages)

        user_msg = mock_llm.call_args[0][1]
        # No signal labels
        assert "semantic" not in user_msg.lower()
        assert "keyword" not in user_msg.lower()
        assert "Signal:" not in user_msg
        assert "dense" not in user_msg.lower()
        assert "sparse" not in user_msg.lower()

    @patch("scripts.blind_ab_test.llm_call")
    def test_u_shape_ordering_applied(self, mock_llm):
        """Passages should be in U-shape order in the prompt."""
        mock_llm.return_value = "answer"
        # 4 passages: U-shape of [a,b,c,d] → [a,b,d,c]
        passages = [
            _candidate("a", "TEXT_A"),
            _candidate("b", "TEXT_B"),
            _candidate("c", "TEXT_C"),
            _candidate("d", "TEXT_D"),
        ]
        generate_fused_answer("q", passages)

        user_msg = mock_llm.call_args[0][1]
        # Find positions of each text in the message
        pos_a = user_msg.index("TEXT_A")
        pos_b = user_msg.index("TEXT_B")
        pos_d = user_msg.index("TEXT_D")
        pos_c = user_msg.index("TEXT_C")
        # U-shape: A before B before D before C
        assert pos_a < pos_b < pos_d < pos_c

    @patch("scripts.blind_ab_test.llm_call")
    def test_numbered_list_format(self, mock_llm):
        """Passages should be numbered [1], [2], etc."""
        mock_llm.return_value = "answer"
        passages = [_candidate("a", "text a"), _candidate("b", "text b")]
        generate_fused_answer("q", passages)

        user_msg = mock_llm.call_args[0][1]
        assert "[1]" in user_msg
        assert "[2]" in user_msg

    @patch("scripts.blind_ab_test.llm_call")
    def test_uses_answer_system_prompt_v3(self, mock_llm):
        mock_llm.return_value = "answer"
        generate_fused_answer("q", [_candidate("a")])
        system_msg = mock_llm.call_args[0][0]
        assert system_msg == ANSWER_SYSTEM_PROMPT_V3

    @patch("scripts.blind_ab_test.llm_call")
    def test_empty_passages_sends_empty_msg(self, mock_llm):
        mock_llm.return_value = "no info"
        generate_fused_answer("q", [])
        user_msg = mock_llm.call_args[0][1]
        assert "No passages were retrieved" in user_msg

    @patch("scripts.blind_ab_test.llm_call")
    def test_exception_returns_none(self, mock_llm):
        mock_llm.side_effect = Exception("LLM down")
        result = generate_fused_answer("q", [_candidate("a")])
        assert result is None


# ─── judge_faithfulness() ────────────────────────────────────────────


class TestJudgeFaithfulness:

    @patch("scripts.blind_ab_test.llm_call")
    def test_includes_all_context(self, mock_llm):
        """Judge prompt must include source, query, passages, and answer."""
        mock_llm.return_value = '{"score": 3, "reason": "good"}'
        passages = [_candidate("a", "retrieved text")]
        judge_faithfulness("source text here", "the query", passages, "the answer")

        user_msg = mock_llm.call_args[0][1]
        assert "source text here" in user_msg
        assert "the query" in user_msg
        assert "retrieved text" in user_msg
        assert "the answer" in user_msg

    @patch("scripts.blind_ab_test.llm_call")
    def test_uses_judge_system_prompt_v3(self, mock_llm):
        mock_llm.return_value = '{"score": 3, "reason": "ok"}'
        judge_faithfulness("src", "q", [], "ans")
        system_msg = mock_llm.call_args[0][0]
        assert system_msg == JUDGE_SYSTEM_PROMPT_V3

    @patch("scripts.blind_ab_test.llm_call")
    def test_temperature_zero(self, mock_llm):
        mock_llm.return_value = '{"score": 3, "reason": "ok"}'
        judge_faithfulness("src", "q", [], "ans")
        assert mock_llm.call_args.kwargs["temperature"] == 0.0

    @patch("scripts.blind_ab_test.llm_call")
    def test_returns_parsed_result(self, mock_llm):
        mock_llm.return_value = '{"score": 1, "reason": "unfaithful"}'
        result = judge_faithfulness("src", "q", [], "ans")
        assert result["score"] == 1
        assert result["reason"] == "unfaithful"
        assert "judge_raw" in result

    @patch("scripts.blind_ab_test.llm_call")
    def test_exception_returns_judge_error(self, mock_llm):
        mock_llm.side_effect = Exception("LLM down")
        result = judge_faithfulness("src", "q", [], "ans")
        assert result["score"] == 2
        assert result["reason"] == "judge_error"
        assert "LLM down" in result["judge_raw"]

    @patch("scripts.blind_ab_test.llm_call")
    def test_empty_passages_handled(self, mock_llm):
        mock_llm.return_value = '{"score": 2, "reason": "no passages"}'
        judge_faithfulness("src", "q", [], "ans")
        user_msg = mock_llm.call_args[0][1]
        assert "(no passages retrieved)" in user_msg

    @patch("scripts.blind_ab_test.llm_call")
    def test_passages_joined_with_separator(self, mock_llm):
        mock_llm.return_value = '{"score": 3, "reason": "ok"}'
        passages = [_candidate("a", "text1"), _candidate("b", "text2")]
        judge_faithfulness("src", "q", passages, "ans")
        user_msg = mock_llm.call_args[0][1]
        assert "---" in user_msg
        assert "text1" in user_msg
        assert "text2" in user_msg


# ─── Prompt verbatim checks ─────────────────────────────────────────


class TestPromptsMatchDesign:
    """Verify system prompts match design.md verbatim."""

    def test_reranker_prompt_starts_correctly(self):
        assert RERANKER_SYSTEM_PROMPT.startswith(
            "You are a relevance reranker for a technical knowledge base."
        )
        assert '{"ranked_indices": [3, 1, 5, 2, 4, ...]}' in RERANKER_SYSTEM_PROMPT

    def test_answer_prompt_starts_correctly(self):
        assert ANSWER_SYSTEM_PROMPT_V3.startswith(
            "You are a precise technical research assistant."
        )
        assert "not an explanation of how you found it." in ANSWER_SYSTEM_PROMPT_V3

    def test_judge_prompt_starts_correctly(self):
        assert JUDGE_SYSTEM_PROMPT_V3.startswith(
            "You are an impartial evaluator of retrieval quality."
        )
        assert '{"score": 1|2|3, "reason": "one sentence"}' in JUDGE_SYSTEM_PROMPT_V3

    def test_signal_labels_mapping(self):
        assert SIGNAL_LABELS == {
            "dense": "semantic",
            "sparse": "keyword",
            "both": "both (semantic + keyword)",
        }
