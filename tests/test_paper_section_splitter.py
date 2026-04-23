"""Unit tests for paper section splitter."""

import pytest

from src.ingestion.paper_section_splitter import (
    PaperSection,
    split_paper_sections,
    _is_excluded,
)


SAMPLE_PAPER = """\
Abstract
This paper presents a novel approach to retrieval-augmented generation.

Introduction
Large language models have shown remarkable capabilities in recent years.

Related Work
Prior work on RAG systems includes DPR and REALM.

Methodology
We propose a three-stage pipeline for document processing.

Experiments
We evaluate our approach on three benchmark datasets.

Results
Our method achieves state-of-the-art performance on all benchmarks.

Discussion
The results demonstrate the effectiveness of our approach.

Conclusion
We have presented a novel approach to RAG that outperforms baselines.

References
[1] Lewis et al. Retrieval-Augmented Generation. NeurIPS 2020.
[2] Karpukhin et al. Dense Passage Retrieval. EMNLP 2020.
"""


class TestSplitPaperSections:
    """Tests for split_paper_sections."""

    def test_splits_standard_academic_headers(self):
        sections = split_paper_sections(SAMPLE_PAPER)
        titles = [s.title for s in sections]
        assert "Abstract" in titles
        assert "Introduction" in titles
        assert "Methodology" in titles
        assert "Conclusion" in titles

    def test_excludes_references(self):
        sections = split_paper_sections(SAMPLE_PAPER)
        titles = [s.title for s in sections]
        assert "References" not in titles

    def test_sequential_indices(self):
        sections = split_paper_sections(SAMPLE_PAPER)
        indices = [s.section_index for s in sections]
        assert indices == list(range(len(sections)))

    def test_non_empty_content(self):
        sections = split_paper_sections(SAMPLE_PAPER)
        for s in sections:
            assert s.content, f"Section '{s.title}' has empty content"

    def test_numbered_section_headers(self):
        text = """\
Abstract
This is the abstract.

1. Introduction
This is the introduction.

2. Related Work
This is related work.

3. Methodology
This is the methodology.

References
[1] Some reference.
"""
        sections = split_paper_sections(text)
        titles = [s.title for s in sections]
        # Should recognize numbered headers
        assert any("Introduction" in t for t in titles)
        assert any("Methodology" in t for t in titles)
        # References excluded
        assert not any("References" in t for t in titles)

    def test_fallback_fewer_than_two_headers(self):
        text = "This is just a plain text document with no recognizable headers at all."
        sections = split_paper_sections(text)
        assert len(sections) == 1
        assert sections[0].title == "Full Text"
        assert sections[0].section_index == 0

    def test_fallback_single_header(self):
        text = "Abstract\nThis paper is about something interesting but has only one header."
        sections = split_paper_sections(text)
        assert len(sections) == 1
        assert sections[0].title == "Full Text"

    def test_bibliography_exclusion(self):
        text = """\
Abstract
The abstract text here.

Introduction
The introduction text here.

Bibliography
[1] Author et al. Title. 2020.
"""
        sections = split_paper_sections(text)
        titles = [s.title for s in sections]
        assert "Bibliography" not in titles

    def test_appendix_exclusion(self):
        text = """\
Abstract
The abstract text here.

Introduction
The introduction text here.

Appendix
Additional tables and figures.
"""
        sections = split_paper_sections(text)
        titles = [s.title for s in sections]
        assert "Appendix" not in titles

    def test_case_insensitive_exclusion(self):
        """Exclusion should work regardless of case."""
        text = """\
Abstract
The abstract.

Introduction
The intro.

REFERENCES
[1] Some ref.
"""
        sections = split_paper_sections(text)
        titles = [s.title.lower() for s in sections]
        assert "references" not in titles

    def test_no_content_lost_from_non_excluded_sections(self):
        """Content from non-excluded sections should appear in output."""
        sections = split_paper_sections(SAMPLE_PAPER)
        all_content = " ".join(s.content for s in sections)
        assert "novel approach" in all_content
        assert "remarkable capabilities" in all_content
        assert "three-stage pipeline" in all_content


class TestIsExcluded:
    """Tests for _is_excluded helper."""

    def test_references(self):
        assert _is_excluded("References") is True

    def test_numbered_references(self):
        assert _is_excluded("7. References") is True

    def test_bibliography(self):
        assert _is_excluded("Bibliography") is True

    def test_appendix(self):
        assert _is_excluded("Appendix") is True

    def test_introduction_not_excluded(self):
        assert _is_excluded("Introduction") is False

    def test_methodology_not_excluded(self):
        assert _is_excluded("3. Methodology") is False
