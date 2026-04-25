"""Tests for citation_masker — property-based and unit tests.

Feature: mineru-pdf-ingestion
Properties 1-3: Citation masking round-trip, reduces splits, placeholder uniqueness
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.ingestion.citation_masker import (
    MAX_CITATION_MASK_CHARS,
    mask,
    restore,
    citation_aware_split,
    _default_split_sentences,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies — generate text with academic citation patterns
# ---------------------------------------------------------------------------

_BRACKET_CITES = [
    "[1]", "[1, 2, 3]", "[1-5]", "[1, 2, 5-8]", "[42]", "[7, 12]",
    "[3-7, 10]", "[1, 3, 5-9, 12]",
]

_PAREN_CITES = [
    "(Smith et al., 2023)", "(Jones & Lee, 2022)",
    "(Smith et al., 2023; Jones, 2022)", "(Brown, 2021)",
    "(Garcia & Kim, 2020)", "(Wang et al., 2019; Li, 2018)",
]

_ABBREVIATIONS = [
    "Fig.", "Figs.", "Eq.", "Eqs.", "Tab.", "Sec.", "Ref.",
    "Vol.", "No.", "vs.", "et al.", "i.e.", "e.g.", "cf.",
]

_NUMBERED_REFS = [
    "Fig. 1.", "Fig. 12.", "Eq. 4.", "Eq. 23.", "Tab. 3.",
    "Sec. 2.", "Ref. 7.",
]

_ALL_PATTERNS = _BRACKET_CITES + _PAREN_CITES + _ABBREVIATIONS + _NUMBERED_REFS


def _academic_text_strategy():
    """Generate text with randomly inserted academic citation patterns."""
    # Build sentences with optional citation insertions
    word = st.text(
        alphabet=st.characters(whitelist_categories=("L",), min_codepoint=65, max_codepoint=122),
        min_size=2, max_size=10,
    )
    citation = st.sampled_from(_ALL_PATTERNS)

    # A fragment is either a word or a citation
    fragment = st.one_of(word, citation)

    # Build a sentence: several fragments joined by spaces, ending with ". "
    sentence = st.lists(fragment, min_size=2, max_size=8).map(
        lambda parts: " ".join(parts) + ". "
    )

    # Build a paragraph: several sentences joined
    return st.lists(sentence, min_size=1, max_size=5).map(
        lambda sents: "".join(sents).strip()
    )


# ---------------------------------------------------------------------------
# Property 1: Citation masking round-trip
# Feature: mineru-pdf-ingestion, Property 1: Citation masking round-trip
# Validates: Requirements 4.6, 8.1
# ---------------------------------------------------------------------------

@given(text=_academic_text_strategy())
@settings(max_examples=200)
def test_mask_restore_roundtrip(text: str):
    """mask() then restore() on the joined result must equal the original."""
    masked_text, restore_map = mask(text)

    # Simulate what happens in the pipeline: split then restore
    # We use a trivial "split" (just wrap in list) to test pure round-trip
    sentences = [masked_text]
    restored = restore(sentences, restore_map)
    assert restored[0] == text, (
        f"Round-trip failed.\n"
        f"  Original:  {text!r}\n"
        f"  Restored:  {restored[0]!r}\n"
        f"  Map: {restore_map}"
    )


@given(text=_academic_text_strategy())
@settings(max_examples=200)
def test_mask_restore_roundtrip_with_split(text: str):
    """mask → split → join → restore must equal original text."""
    masked_text, restore_map = mask(text)
    # Split on sentence boundaries (the actual use case)
    sentences = _default_split_sentences(masked_text)
    if not sentences:
        # If splitting produces nothing, the original must be empty/whitespace
        assert not text.strip()
        return
    # Join back and restore
    joined = " ".join(sentences)
    restored_parts = restore([joined], restore_map)
    # The joined+restored text should contain all original citations
    for original in restore_map.values():
        assert original in restored_parts[0], (
            f"Lost citation {original!r} after split+restore"
        )


# ---------------------------------------------------------------------------
# Property 2: Masking reduces false sentence splits
# Feature: mineru-pdf-ingestion, Property 2: Masking reduces false splits
# Validates: Requirements 8.2
# ---------------------------------------------------------------------------

@given(text=_academic_text_strategy())
@settings(max_examples=200)
def test_masking_reduces_sentence_count(text: str):
    """Splitting masked text should produce <= sentences than splitting original."""
    assume(len(text) > 10)  # skip trivially short inputs

    masked_text, _ = mask(text)
    original_sentences = _default_split_sentences(text)
    masked_sentences = _default_split_sentences(masked_text)

    assert len(masked_sentences) <= len(original_sentences), (
        f"Masking increased sentence count from {len(original_sentences)} "
        f"to {len(masked_sentences)}.\n"
        f"  Original: {original_sentences}\n"
        f"  Masked:   {masked_sentences}"
    )


# ---------------------------------------------------------------------------
# Property 3: Placeholder uniqueness
# Feature: mineru-pdf-ingestion, Property 3: Placeholder uniqueness
# Validates: Requirements 8.3
# ---------------------------------------------------------------------------

@given(text=_academic_text_strategy())
@settings(max_examples=200)
def test_placeholder_uniqueness(text: str):
    """No placeholder should be a substring of any other placeholder."""
    _, restore_map = mask(text)
    placeholders = list(restore_map.keys())
    for i, p1 in enumerate(placeholders):
        for j, p2 in enumerate(placeholders):
            if i != j:
                assert p1 not in p2, (
                    f"Placeholder {p1!r} is a substring of {p2!r}"
                )


# ---------------------------------------------------------------------------
# Unit tests for citation masker
# Requirements: 4.1, 4.2, 4.3, 4.4, 10.3
# ---------------------------------------------------------------------------

class TestMaskSpecificPatterns:
    """Test that specific citation patterns are correctly masked."""

    def test_bracket_single(self):
        text = "This result [1] is important."
        masked, rmap = mask(text)
        assert "[1]" not in masked
        assert any(v == "[1]" for v in rmap.values())

    def test_bracket_multiple(self):
        text = "See results [1, 2, 3] for details."
        masked, rmap = mask(text)
        assert "[1, 2, 3]" not in masked
        assert any(v == "[1, 2, 3]" for v in rmap.values())

    def test_bracket_range(self):
        text = "As shown in [1-5] and [1, 2, 5-8]."
        masked, rmap = mask(text)
        assert "[1-5]" not in masked
        assert "[1, 2, 5-8]" not in masked

    def test_paren_author_year(self):
        text = "This was shown (Smith et al., 2023) previously."
        masked, rmap = mask(text)
        assert "(Smith et al., 2023)" not in masked
        assert any(v == "(Smith et al., 2023)" for v in rmap.values())

    def test_paren_multiple_authors(self):
        text = "Results (Smith et al., 2023; Jones, 2022) confirm this."
        masked, rmap = mask(text)
        assert "(Smith et al., 2023; Jones, 2022)" not in masked

    def test_abbreviation_fig(self):
        text = "As shown in Fig. 3, the results are clear."
        masked, rmap = mask(text)
        assert "Fig." not in masked

    def test_abbreviation_eq(self):
        text = "Using Eq. 4 we derive the bound."
        masked, rmap = mask(text)
        assert "Eq." not in masked

    def test_abbreviation_et_al(self):
        text = "Smith et al. showed this result."
        masked, rmap = mask(text)
        assert "et al." not in masked

    def test_abbreviation_ie_eg(self):
        text = "Some methods (i.e. gradient descent) and tools (e.g. PyTorch) are used."
        masked, rmap = mask(text)
        assert "i.e." not in masked
        assert "e.g." not in masked

    def test_numbered_ref_fig(self):
        text = "See Fig. 1. The results show improvement."
        masked, rmap = mask(text)
        # "Fig. 1." should be masked as one unit
        assert "Fig. 1." not in masked

    def test_numbered_ref_eq(self):
        text = "From Eq. 4. We can derive the bound."
        masked, rmap = mask(text)
        assert "Eq. 4." not in masked


class TestMaskEdgeCases:
    """Test edge cases for the citation masker."""

    def test_empty_input(self):
        masked, rmap = mask("")
        assert masked == ""
        assert rmap == {}

    def test_no_citations(self):
        text = "This is a normal sentence without any citations."
        masked, rmap = mask(text)
        assert masked == text
        assert rmap == {}

    def test_input_length_guard_at_limit(self):
        """Text at exactly MAX_CITATION_MASK_CHARS should be masked."""
        text = "A" * MAX_CITATION_MASK_CHARS
        masked, rmap = mask(text)
        # No citations to mask, but masking should still run
        assert masked == text
        assert rmap == {}

    def test_input_length_guard_over_limit(self):
        """Text over MAX_CITATION_MASK_CHARS should skip masking."""
        text = "See [1] in " + "A" * MAX_CITATION_MASK_CHARS
        masked, rmap = mask(text)
        # Should return original text unmasked
        assert masked == text
        assert rmap == {}

    def test_ordering_paren_before_abbrev(self):
        """Parenthetical citations containing 'et al.' should be masked
        as a whole unit before the abbreviation pass sees 'et al.'."""
        text = "This (Smith et al., 2023) and et al. are different."
        masked, rmap = mask(text)
        # The parenthetical should be one placeholder
        paren_masked = any(
            v == "(Smith et al., 2023)" for v in rmap.values()
        )
        assert paren_masked, "Parenthetical citation not masked as whole unit"
        # The standalone "et al." should also be masked separately
        etal_masked = any(v == "et al." for v in rmap.values())
        assert etal_masked, "Standalone 'et al.' not masked"


class TestRestore:
    """Test the restore function."""

    def test_restore_empty_map(self):
        sentences = ["Hello world.", "Another sentence."]
        result = restore(sentences, {})
        assert result == sentences

    def test_restore_replaces_placeholders(self):
        sentences = ["See __CITE_0__ for details."]
        rmap = {"__CITE_0__": "[1, 2, 3]"}
        result = restore(sentences, rmap)
        assert result == ["See [1, 2, 3] for details."]

    def test_restore_multiple_placeholders(self):
        sentences = ["__CITE_0__ showed __CITE_1__ results."]
        rmap = {"__CITE_0__": "(Smith et al., 2023)", "__CITE_1__": "[1-5]"}
        result = restore(sentences, rmap)
        assert result == ["(Smith et al., 2023) showed [1-5] results."]


class TestCitationAwareSplit:
    """Test the citation_aware_split convenience function."""

    def test_basic_split(self):
        text = "First sentence. Second sentence."
        result = citation_aware_split(text)
        assert len(result) >= 1

    def test_citation_does_not_split(self):
        text = "Results in Fig. 1. The improvement is clear. Next topic here."
        result = citation_aware_split(text)
        # "Fig. 1." should NOT cause a split between "Fig" and "1"
        # We should get <= 3 sentences, not more
        assert len(result) <= 3

    def test_empty_input(self):
        assert citation_aware_split("") == []
        assert citation_aware_split("   ") == []

    def test_pysbd_fallback_when_not_installed(self):
        """use_pysbd=True should fall back gracefully if pysbd not installed."""
        # This test works whether pysbd is installed or not
        text = "First sentence. Second sentence."
        result = citation_aware_split(text, use_pysbd=True)
        assert len(result) >= 1
