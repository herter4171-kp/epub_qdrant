"""Tests for md_section_splitter — property-based and unit tests.

Feature: mineru-pdf-ingestion
Properties 5-6: Section splitting structural correctness, excluded sections filtered
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.ingestion.md_section_splitter import (
    MarkdownSection,
    split_markdown_sections,
    _is_excluded,
    EXCLUDED_SECTIONS,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies — generate Markdown with headings
# ---------------------------------------------------------------------------

_NORMAL_TITLES = [
    "Introduction", "Methodology", "Experiments", "Results",
    "Discussion", "Conclusion", "Background", "Related Work",
    "Evaluation", "Analysis", "Approach", "Framework",
    "Implementation", "Architecture", "Data Collection",
]

_EXCLUDED_TITLES = ["References", "Bibliography", "Appendix"]

_EXCLUDED_WITH_PREFIXES = [
    "References", "Bibliography", "Appendix",
    "5. References", "A. Appendix", "6. Bibliography",
    "7. References", "B. Bibliography", "C. Appendix",
]


def _content_strategy():
    """Generate non-empty section content (at least one word)."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
        min_size=5,
        max_size=200,
    ).filter(lambda t: t.strip())


def _heading_level_strategy():
    """Generate heading level 1-3."""
    return st.integers(min_value=1, max_value=3)


def _markdown_with_headings_strategy(min_headings=2, max_headings=8):
    """Generate Markdown with N headings at random levels with content."""
    return st.integers(
        min_value=min_headings, max_value=max_headings
    ).flatmap(lambda n: st.lists(
        st.tuples(
            _heading_level_strategy(),
            st.sampled_from(_NORMAL_TITLES),
            _content_strategy(),
        ),
        min_size=n,
        max_size=n,
    )).map(_build_markdown)


def _build_markdown(sections: list[tuple[int, str, str]]) -> str:
    """Build a Markdown string from (level, title, content) tuples."""
    parts = []
    for level, title, content in sections:
        hashes = "#" * level
        parts.append(f"{hashes} {title}\n\n{content}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Property 5: Markdown section splitting structural correctness
# Feature: mineru-pdf-ingestion, Property 5: Section splitting structural correctness
# ---------------------------------------------------------------------------

@given(data=_markdown_with_headings_strategy(min_headings=2, max_headings=8))
@settings(max_examples=100)
def test_section_splitting_correctness(data: str):
    """For Markdown with 2+ headings, verify structural properties.

    - titles match heading text
    - heading_level matches # count
    - section_index sequential from 0
    - all content non-empty
    """
    sections = split_markdown_sections(data)

    # Must produce at least one section (could be fewer than input if some empty)
    assert len(sections) >= 1

    # section_index sequential from 0
    indices = [s.section_index for s in sections]
    assert indices == list(range(len(sections)))

    # All content non-empty
    for s in sections:
        assert s.content, f"Section '{s.title}' has empty content"

    # heading_level in valid range
    for s in sections:
        assert s.heading_level in (1, 2, 3), f"Invalid heading_level {s.heading_level}"

    # Verify titles and levels match what's in the Markdown.
    # Headings in source order, excluding empty-content and excluded titles.
    headings_in_md = re.findall(r"^(#{1,3})\s+(.+)", data, re.MULTILINE)
    non_excluded = [(len(h), t.strip()) for h, t in headings_in_md if not _is_excluded(t.strip())]

    # Build expected (title, level) pairs by replaying the splitter logic:
    # sections are produced in source order, so match positionally against
    # non_excluded headings that have non-empty content between them.
    # We only check that each returned section exists somewhere in the
    # non-excluded list with matching level (accounting for duplicates).
    remaining = list(non_excluded)
    for s in sections:
        found = False
        for idx, (level, title) in enumerate(remaining):
            if title == s.title and level == s.heading_level:
                remaining.pop(idx)
                found = True
                break
        assert found, (
            f"Section (title='{s.title}', level={s.heading_level}) "
            f"not found in remaining non-excluded headings: {remaining}"
        )


# ---------------------------------------------------------------------------
# Property 6: Excluded sections are filtered
# Feature: mineru-pdf-ingestion, Property 6: Excluded sections are filtered
# ---------------------------------------------------------------------------

def _markdown_with_mixed_headings_strategy():
    """Generate Markdown with mix of normal and excluded headings."""
    return st.lists(
        st.tuples(
            _heading_level_strategy(),
            st.sampled_from(_NORMAL_TITLES + _EXCLUDED_WITH_PREFIXES),
            _content_strategy(),
        ),
        min_size=3,
        max_size=8,
    ).map(_build_markdown)


@given(data=_markdown_with_mixed_headings_strategy())
@settings(max_examples=100)
def test_excluded_sections_filtered(data: str):
    """No returned section title matches References/Bibliography/Appendix."""
    sections = split_markdown_sections(data)

    for s in sections:
        if s.title == "Full Text":
            continue  # fallback section, skip
        assert not _is_excluded(s.title), (
            f"Excluded section '{s.title}' found in output"
        )



# ---------------------------------------------------------------------------
# Unit tests for md_section_splitter (Task 5.4)
# ---------------------------------------------------------------------------

SAMPLE_MARKDOWN = """\
# Introduction

This paper presents a novel approach to document parsing.

## Related Work

Prior work includes MinerU and Docling for PDF extraction.

## Methodology

We propose a three-stage pipeline for processing.

### Data Collection

We collected 500 academic PDFs from arXiv.

### Preprocessing

Each PDF was converted to Markdown using MinerU.

## Results

Our method achieves state-of-the-art performance.

## Conclusion

We have demonstrated effective PDF ingestion.

## References

[1] Xue et al. MinerU evaluation. arXiv 2601.15170.
[2] AutoPage. arXiv 2510.19600.
"""


class TestSplitMarkdownSections:
    """Unit tests for split_markdown_sections."""

    def test_splits_at_heading_boundaries(self):
        sections = split_markdown_sections(SAMPLE_MARKDOWN)
        titles = [s.title for s in sections]
        assert "Introduction" in titles
        assert "Methodology" in titles
        assert "Results" in titles

    def test_excludes_references(self):
        sections = split_markdown_sections(SAMPLE_MARKDOWN)
        titles = [s.title for s in sections]
        assert "References" not in titles

    def test_sequential_indices(self):
        sections = split_markdown_sections(SAMPLE_MARKDOWN)
        indices = [s.section_index for s in sections]
        assert indices == list(range(len(sections)))

    def test_non_empty_content(self):
        sections = split_markdown_sections(SAMPLE_MARKDOWN)
        for s in sections:
            assert s.content, f"Section '{s.title}' has empty content"

    def test_heading_levels(self):
        sections = split_markdown_sections(SAMPLE_MARKDOWN)
        level_map = {s.title: s.heading_level for s in sections}
        assert level_map["Introduction"] == 1
        assert level_map["Related Work"] == 2
        assert level_map["Data Collection"] == 3

    def test_nested_headings_split(self):
        """### within ## should be separate sections."""
        sections = split_markdown_sections(SAMPLE_MARKDOWN)
        titles = [s.title for s in sections]
        assert "Data Collection" in titles
        assert "Preprocessing" in titles

    def test_zero_headings_fallback(self):
        text = "Just plain text with no headings at all."
        sections = split_markdown_sections(text)
        assert len(sections) == 1
        assert sections[0].title == "Full Text"
        assert sections[0].heading_level == 1
        assert sections[0].section_index == 0

    def test_single_heading_fallback(self):
        text = "# Only One Heading\n\nSome content here."
        sections = split_markdown_sections(text)
        assert len(sections) == 1
        assert sections[0].title == "Full Text"

    def test_bibliography_exclusion(self):
        md = "## Introduction\n\nText here.\n\n## Bibliography\n\n[1] Ref.\n\n## Conclusion\n\nDone."
        sections = split_markdown_sections(md)
        titles = [s.title for s in sections]
        assert "Bibliography" not in titles
        assert "Introduction" in titles
        assert "Conclusion" in titles

    def test_appendix_exclusion(self):
        md = "## Introduction\n\nText.\n\n## Appendix\n\nExtra tables.\n\n## Discussion\n\nMore text."
        sections = split_markdown_sections(md)
        titles = [s.title for s in sections]
        assert "Appendix" not in titles

    def test_numbered_references_exclusion(self):
        md = "## Introduction\n\nText.\n\n## 5. References\n\n[1] Ref.\n\n## Conclusion\n\nDone."
        sections = split_markdown_sections(md)
        titles = [s.title for s in sections]
        assert "5. References" not in titles

    def test_lettered_appendix_exclusion(self):
        md = "## Introduction\n\nText.\n\n## A. Appendix\n\nExtra.\n\n## Conclusion\n\nDone."
        sections = split_markdown_sections(md)
        titles = [s.title for s in sections]
        assert "A. Appendix" not in titles

    def test_case_insensitive_exclusion(self):
        md = "## Introduction\n\nText.\n\n## REFERENCES\n\n[1] Ref.\n\n## Conclusion\n\nDone."
        sections = split_markdown_sections(md)
        titles = [s.title.lower() for s in sections]
        assert "references" not in titles

    def test_empty_content_between_headings_excluded(self):
        md = "## First\n\n## Second\n\nActual content here."
        sections = split_markdown_sections(md)
        # "First" has no content between it and "Second", should be excluded
        titles = [s.title for s in sections]
        assert "First" not in titles
        assert "Second" in titles

    def test_empty_input(self):
        sections = split_markdown_sections("")
        assert len(sections) == 1
        assert sections[0].title == "Full Text"

    def test_none_like_empty(self):
        sections = split_markdown_sections("   ")
        assert len(sections) == 1
        assert sections[0].title == "Full Text"

    def test_mixed_heading_levels(self):
        md = (
            "# Top Level\n\nIntro text.\n\n"
            "## Section A\n\nContent A.\n\n"
            "### Subsection A1\n\nContent A1.\n\n"
            "## Section B\n\nContent B.\n"
        )
        sections = split_markdown_sections(md)
        assert len(sections) == 4
        levels = [s.heading_level for s in sections]
        assert levels == [1, 2, 3, 2]


class TestIsExcluded:
    """Tests for _is_excluded helper."""

    def test_references(self):
        assert _is_excluded("References") is True

    def test_bibliography(self):
        assert _is_excluded("Bibliography") is True

    def test_appendix(self):
        assert _is_excluded("Appendix") is True

    def test_numbered_references(self):
        assert _is_excluded("5. References") is True

    def test_lettered_appendix(self):
        assert _is_excluded("A. Appendix") is True

    def test_case_insensitive(self):
        assert _is_excluded("REFERENCES") is True
        assert _is_excluded("bibliography") is True

    def test_introduction_not_excluded(self):
        assert _is_excluded("Introduction") is False

    def test_methodology_not_excluded(self):
        assert _is_excluded("3. Methodology") is False

    def test_results_not_excluded(self):
        assert _is_excluded("Results") is False
