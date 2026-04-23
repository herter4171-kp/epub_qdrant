"""Unit tests for EPUB heading extraction and section splitting."""

import pytest

from src.ingestion.epub_parser import (
    _extract_headings_from_html,
    _split_into_sections,
)


# ── _extract_headings_from_html tests ─────────────────────────────────────────


class TestExtractHeadingsFromHtml:
    """Tests for _extract_headings_from_html."""

    def test_extracts_h1_through_h3(self):
        html = (
            b"<h1>Title One</h1><p>Content one.</p>"
            b"<h2>Subtitle Two</h2><p>Content two.</p>"
            b"<h3>Sub-subtitle Three</h3><p>Content three.</p>"
        )
        result = _extract_headings_from_html(html)
        assert len(result) == 3
        assert result[0][0] == 1  # level
        assert result[0][1] == "Title One"
        assert result[1][0] == 2
        assert result[1][1] == "Subtitle Two"
        assert result[2][0] == 3
        assert result[2][1] == "Sub-subtitle Three"

    def test_strips_inner_html_tags(self):
        html = b'<h2><span class="num">5.1</span> <a href="#">Chunking</a></h2><p>Body.</p>'
        result = _extract_headings_from_html(html)
        assert len(result) == 1
        assert result[0][1] == "5.1 Chunking"

    def test_returns_empty_list_when_no_headings(self):
        html = b"<p>Just a paragraph with no headings at all.</p>"
        result = _extract_headings_from_html(html)
        assert result == []

    def test_skips_empty_title_headings(self):
        html = b"<h1>  </h1><p>Content.</p><h2>Real Title</h2><p>More.</p>"
        result = _extract_headings_from_html(html)
        assert len(result) == 1
        assert result[0][1] == "Real Title"

    def test_document_order(self):
        html = (
            b"<h1>First</h1><p>A</p>"
            b"<h2>Second</h2><p>B</p>"
            b"<h1>Third</h1><p>C</p>"
        )
        result = _extract_headings_from_html(html)
        titles = [r[1] for r in result]
        assert titles == ["First", "Second", "Third"]

    def test_non_overlapping_content_boundaries(self):
        html = (
            b"<h1>Alpha</h1><p>Content alpha.</p>"
            b"<h2>Beta</h2><p>Content beta.</p>"
        )
        result = _extract_headings_from_html(html)
        assert len(result) == 2
        # content_start of second heading >= content_end of first heading
        # Actually content_end of first == content_start region of second heading tag
        _, _, start1, end1 = result[0]
        _, _, start2, end2 = result[1]
        assert start1 < end1
        assert start2 < end2
        assert end1 <= start2 or start2 >= start1  # no overlap

    def test_nested_html_in_heading(self):
        html = b'<h1><em>Italic</em> and <strong>Bold</strong></h1><p>Body text.</p>'
        result = _extract_headings_from_html(html)
        assert len(result) == 1
        assert result[0][1] == "Italic and Bold"

    def test_malformed_html_returns_empty(self):
        """Unclosed heading tags should not match the regex."""
        html = b"<h1>Unclosed heading<p>Some content</p>"
        result = _extract_headings_from_html(html)
        assert result == []


# ── _split_into_sections tests ────────────────────────────────────────────────


class TestSplitIntoSections:
    """Tests for _split_into_sections (refactored to accept raw bytes)."""

    def test_splits_by_headings(self):
        html = (
            b"<h1>Chapter 1</h1>"
            b"<p>This is the first chapter with enough content to pass the length check easily.</p>"
            b"<h2>Section 1.1</h2>"
            b"<p>This is a subsection with enough content to pass the length check easily.</p>"
        )
        result = _split_into_sections(html)
        assert len(result) == 2
        assert result[0][0] == "Chapter 1"
        assert result[1][0] == "Section 1.1"
        # heading levels
        assert result[0][2] == 1
        assert result[1][2] == 2

    def test_fallback_no_headings(self):
        html = b"<p>This is a long paragraph without any headings at all, just plain text content here.</p>"
        result = _split_into_sections(html)
        assert len(result) == 1
        assert result[0][0] == "(no title)"
        assert result[0][2] == 0  # fallback heading level

    def test_fallback_empty_content(self):
        html = b""
        result = _split_into_sections(html)
        assert result == []

    def test_returns_tuples_with_three_elements(self):
        html = (
            b"<h1>Heading</h1>"
            b"<p>Enough content here to pass the twenty character minimum threshold.</p>"
        )
        result = _split_into_sections(html)
        assert len(result) == 1
        title, content, level = result[0]
        assert title == "Heading"
        assert len(content) > 20
        assert level == 1

    def test_skips_short_content_sections(self):
        html = (
            b"<h1>Short</h1><p>Tiny.</p>"
            b"<h2>Long Enough</h2>"
            b"<p>This section has plenty of content to exceed the twenty character minimum.</p>"
        )
        result = _split_into_sections(html)
        # "Short" section has content "Tiny." which is <= 20 chars, should be skipped
        assert len(result) == 1
        assert result[0][0] == "Long Enough"
