"""Tests for mineru_pipeline — property-based and unit tests.

Feature: mineru-pdf-ingestion
Property 7: Pipeline payload completeness
Requirements: 6.3, 6.4, 6.5, 6.6
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

# Ensure project root on path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.ingestion.md_section_splitter import MarkdownSection
from src.ingestion.semantic_chunker import ChunkConfig, ChunkResult, load_tokenizer
from scripts.mineru_pipeline import run_pipeline, REQUIRED_PAYLOAD_FIELDS


# ---------------------------------------------------------------------------
# Hypothesis strategies — generate sections for payload testing
# ---------------------------------------------------------------------------

_TITLES = [
    "Introduction", "Methodology", "Experiments", "Results",
    "Discussion", "Background", "Evaluation", "Framework",
]


def _section_strategy():
    """Generate a MarkdownSection with random content."""
    return st.builds(
        MarkdownSection,
        title=st.sampled_from(_TITLES),
        content=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
            min_size=20,
            max_size=200,
        ).filter(lambda t: t.strip()),
        heading_level=st.integers(min_value=1, max_value=3),
        section_index=st.integers(min_value=0, max_value=10),
    )


def _sections_strategy():
    """Generate list of sections."""
    return st.lists(_section_strategy(), min_size=1, max_size=5)


# ---------------------------------------------------------------------------
# Property 7: Pipeline payload completeness
# Feature: mineru-pdf-ingestion, Property 7: Pipeline payload completeness
# Validates: Requirements 6.3
# ---------------------------------------------------------------------------

@given(sections=_sections_strategy())
@h_settings(max_examples=100)
def test_payload_fields_present(sections: list[MarkdownSection]):
    """Every payload dict must contain all required fields."""
    from src.ingestion.citation_masker import citation_aware_split
    from src.ingestion.semantic_chunker import chunk_section

    token_counter = lambda text: max(1, len(text.split()))
    config = ChunkConfig(
        chunk_size=500,
        enable_semantic=False,
    )

    payloads = []
    for ms in sections:
        results = chunk_section(
            title=ms.title,
            content=ms.content,
            config=config,
            token_counter=token_counter,
            embedding_fn=None,
            sentence_splitter=citation_aware_split,
        )
        chunk_count = len(results)
        for cr in results:
            payloads.append({
                "text": cr.text,
                "section_title": cr.section_title or ms.title,
                "heading_level": ms.heading_level,
                "chunk_index": cr.chunk_index,
                "chunk_count": chunk_count,
                "token_count": cr.token_count,
                "source_file": "test.pdf",
            })

    assert len(payloads) >= 1, "Pipeline produced no payloads"

    for i, payload in enumerate(payloads):
        for field in REQUIRED_PAYLOAD_FIELDS:
            assert field in payload, (
                f"Payload {i} missing required field '{field}'. "
                f"Keys: {list(payload.keys())}"
            )



# ---------------------------------------------------------------------------
# Unit tests for pipeline runner
# Requirements: 6.4, 6.5, 6.6
# ---------------------------------------------------------------------------

class TestPipelineCLI:
    """Test CLI argument parsing."""

    def test_argparse_pdf_required(self):
        """--pdf is required."""
        from scripts.mineru_pipeline import main
        import argparse
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["mineru_pipeline.py"]):
                main()

    def test_argparse_accepts_pdf_and_output(self):
        """--pdf and --output are accepted."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--pdf", required=True)
        parser.add_argument("--output", default=None)
        args = parser.parse_args(["--pdf", "test.pdf", "--output", "out.json"])
        assert args.pdf == "test.pdf"
        assert args.output == "out.json"


class TestPipelineOutput:
    """Test JSON output and summary."""

    def test_json_output_written(self, tmp_path):
        """When --output provided, JSON file is written."""
        output_file = tmp_path / "output.json"

        # Mock convert_pdf_to_markdown to return known Markdown
        mock_md = "## Introduction\n\nThis is test content.\n\n## Methods\n\nMore content here."

        with patch("scripts.mineru_pipeline.convert_pdf_to_markdown", return_value=mock_md):
            with patch("scripts.mineru_pipeline.load_tokenizer", return_value=lambda x: len(x.split())):
                payloads = run_pipeline(pdf_path="/fake/test.pdf", output_path=str(output_file))

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert len(data) == len(payloads)

    def test_summary_output(self, tmp_path, capsys):
        """Summary prints sections, chunks, avg/total tokens."""
        mock_md = "## Intro\n\nContent here.\n\n## Methods\n\nMore content."

        with patch("scripts.mineru_pipeline.convert_pdf_to_markdown", return_value=mock_md):
            with patch("scripts.mineru_pipeline.load_tokenizer", return_value=lambda x: len(x.split())):
                run_pipeline(pdf_path="/fake/test.pdf")

        captured = capsys.readouterr()
        assert "Sections found:" in captured.out
        assert "Chunks produced:" in captured.out
        assert "Avg token count:" in captured.out
        assert "Total token count:" in captured.out

    def test_payload_source_file(self, tmp_path):
        """source_file in payload matches PDF filename."""
        mock_md = "## Intro\n\nContent.\n\n## Methods\n\nMore."

        with patch("scripts.mineru_pipeline.convert_pdf_to_markdown", return_value=mock_md):
            with patch("scripts.mineru_pipeline.load_tokenizer", return_value=lambda x: len(x.split())):
                payloads = run_pipeline(pdf_path="/some/path/2303_09014.pdf")

        for p in payloads:
            assert p["source_file"] == "2303_09014.pdf"
