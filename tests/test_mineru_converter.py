"""Tests for mineru_converter — property-based and unit tests.

Feature: mineru-pdf-ingestion
Property 4: Heading count validation
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

from src.ingestion.mineru_converter import (
    validate_heading_structure,
    convert_pdf_to_markdown,
    _extract_markdown,
    MINERU_TIMEOUT_SECONDS,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies — generate Markdown with known heading counts
# ---------------------------------------------------------------------------

def _markdown_with_known_headings_strategy():
    """Generate (markdown_string, expected_counts) tuples."""
    return st.tuples(
        st.integers(min_value=0, max_value=5),  # h1 count
        st.integers(min_value=0, max_value=5),  # h2 count
        st.integers(min_value=0, max_value=5),  # h3 count
    ).map(_build_markdown_with_counts)


def _build_markdown_with_counts(
    counts: tuple[int, int, int],
) -> tuple[str, dict[str, int]]:
    """Build Markdown with exact heading counts and return expected dict."""
    h1, h2, h3 = counts
    lines = []
    for i in range(h1):
        lines.append(f"# Heading H1 {i}")
        lines.append(f"Content for h1 section {i}.")
        lines.append("")
    for i in range(h2):
        lines.append(f"## Heading H2 {i}")
        lines.append(f"Content for h2 section {i}.")
        lines.append("")
    for i in range(h3):
        lines.append(f"### Heading H3 {i}")
        lines.append(f"Content for h3 section {i}.")
        lines.append("")
    markdown = "\n".join(lines)
    expected = {"h1": h1, "h2": h2, "h3": h3}
    return markdown, expected


# ---------------------------------------------------------------------------
# Property 4: Heading count validation
# Feature: mineru-pdf-ingestion, Property 4: Heading count validation
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

@given(data=_markdown_with_known_headings_strategy())
@h_settings(max_examples=100)
def test_heading_count_validation(data: tuple[str, dict[str, int]]):
    """validate_heading_structure() counts must match actual heading counts."""
    markdown, expected = data
    result = validate_heading_structure(markdown)
    assert result == expected, (
        f"Heading count mismatch.\n"
        f"  Expected: {expected}\n"
        f"  Got:      {result}\n"
        f"  Markdown (first 200 chars): {markdown[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Unit tests for validate_heading_structure
# ---------------------------------------------------------------------------

class TestValidateHeadingStructure:
    """Unit tests for validate_heading_structure."""

    def test_empty_string(self):
        assert validate_heading_structure("") == {"h1": 0, "h2": 0, "h3": 0}

    def test_no_headings(self):
        md = "Just plain text with no headings."
        assert validate_heading_structure(md) == {"h1": 0, "h2": 0, "h3": 0}

    def test_single_h1(self):
        md = "# Title\n\nSome content."
        assert validate_heading_structure(md) == {"h1": 1, "h2": 0, "h3": 0}

    def test_mixed_levels(self):
        md = "# Title\n\n## Section\n\n### Subsection\n\nContent."
        assert validate_heading_structure(md) == {"h1": 1, "h2": 1, "h3": 1}

    def test_multiple_same_level(self):
        md = "## A\n\n## B\n\n## C\n\nContent."
        assert validate_heading_structure(md) == {"h1": 0, "h2": 3, "h3": 0}

    def test_h3_not_counted_as_h2(self):
        md = "### Only subsection\n\nContent."
        result = validate_heading_structure(md)
        assert result == {"h1": 0, "h2": 0, "h3": 1}

    def test_h2_not_counted_as_h1(self):
        md = "## Section\n\nContent."
        result = validate_heading_structure(md)
        assert result == {"h1": 0, "h2": 1, "h3": 0}

    def test_realistic_paper(self):
        md = (
            "# ReAct: Synergizing Reasoning and Acting\n\n"
            "## Abstract\n\nWe propose...\n\n"
            "## 1. Introduction\n\nLarge language models...\n\n"
            "## 2. Related Work\n\nPrior work...\n\n"
            "### 2.1 Reasoning\n\nChain of thought...\n\n"
            "### 2.2 Acting\n\nTool use...\n\n"
            "## 3. Methodology\n\nOur approach...\n\n"
            "## 4. Experiments\n\nWe evaluate...\n\n"
            "## 5. Conclusion\n\nWe have shown...\n"
        )
        result = validate_heading_structure(md)
        assert result == {"h1": 1, "h2": 6, "h3": 2}


# ---------------------------------------------------------------------------
# Unit tests for _extract_markdown
# ---------------------------------------------------------------------------

class TestExtractMarkdown:
    """Test markdown extraction from various MinerU response formats."""

    def test_direct_md_content(self):
        data = {"md_content": "# Hello\n\nWorld"}
        assert _extract_markdown(data, "test.pdf") == "# Hello\n\nWorld"

    def test_direct_markdown_key(self):
        data = {"markdown": "# Hello\n\nWorld"}
        assert _extract_markdown(data, "test.pdf") == "# Hello\n\nWorld"

    def test_nested_results(self):
        data = {"results": [{"md_content": "# Hello"}]}
        assert _extract_markdown(data, "test.pdf") == "# Hello"

    def test_list_response(self):
        data = [{"md_content": "# Hello"}]
        assert _extract_markdown(data, "test.pdf") == "# Hello"

    def test_unknown_format_raises(self):
        data = {"something_else": "value"}
        with pytest.raises(RuntimeError, match="Cannot extract markdown"):
            _extract_markdown(data, "test.pdf")


# ---------------------------------------------------------------------------
# Unit tests for convert_pdf_to_markdown error handling
# ---------------------------------------------------------------------------

class TestConvertPdfToMarkdown:
    """Unit tests for convert_pdf_to_markdown error paths."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not_a_real_file"):
            convert_pdf_to_markdown("/tmp/not_a_real_file.pdf")

    def test_connection_error(self, tmp_path):
        """Should raise ConnectionError when MinerU service unreachable."""
        import requests as req_lib

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_post = MagicMock(
            side_effect=req_lib.exceptions.ConnectionError("Connection refused")
        )
        with patch("src.ingestion.mineru_converter.requests.post", mock_post):
            with pytest.raises(ConnectionError, match="Cannot reach MinerU"):
                convert_pdf_to_markdown(str(pdf))

    def test_timeout_error(self, tmp_path):
        """Should raise TimeoutError on request timeout."""
        import requests as req_lib

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_post = MagicMock(side_effect=req_lib.exceptions.Timeout("timed out"))
        with patch("src.ingestion.mineru_converter.requests.post", mock_post):
            with pytest.raises(TimeoutError, match="timed out"):
                convert_pdf_to_markdown(str(pdf))

    def test_http_error(self, tmp_path):
        """Should raise RuntimeError on non-200 response."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post = MagicMock(return_value=mock_resp)

        with patch("src.ingestion.mineru_converter.requests.post", mock_post):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                convert_pdf_to_markdown(str(pdf))

    def test_success(self, tmp_path):
        """Should return markdown on successful response."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"md_content": "# Title\n\n## Section\n\nContent."}
        mock_post = MagicMock(return_value=mock_resp)

        with patch("src.ingestion.mineru_converter.requests.post", mock_post):
            result = convert_pdf_to_markdown(str(pdf))
            assert "# Title" in result
            assert "## Section" in result
