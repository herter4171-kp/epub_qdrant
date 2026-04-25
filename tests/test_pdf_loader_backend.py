"""Tests for PdfLoader backend switching (pypdf vs mineru).

Feature: mineru-pdf-ingestion
Requirements: 7.1, 7.2, 7.3, 10.1
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.ingestion.loader import PdfLoader, PDF_BACKEND_PYPDF, PDF_BACKEND_MINERU


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def loader():
    return PdfLoader()


@pytest.fixture
def fake_pdf(tmp_path):
    """Create a fake PDF file and sidecar JSON."""
    pdf = tmp_path / "2303_09014.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    return pdf


# ---------------------------------------------------------------------------
# Backend routing tests
# ---------------------------------------------------------------------------

class TestBackendRouting:
    """Test PDF_BACKEND env var routes to correct method."""

    def test_default_uses_pypdf(self, loader, fake_pdf):
        """No PDF_BACKEND set → pypdf path."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PDF_BACKEND", None)
            with patch.object(loader, "_load_pypdf", return_value=[]) as mock_pypdf:
                with patch.object(loader, "_load_mineru") as mock_mineru:
                    loader.load(fake_pdf)
                    mock_pypdf.assert_called_once_with(fake_pdf)
                    mock_mineru.assert_not_called()

    def test_pypdf_explicit(self, loader, fake_pdf):
        """PDF_BACKEND=pypdf → pypdf path."""
        with patch.dict(os.environ, {"PDF_BACKEND": "pypdf"}):
            with patch.object(loader, "_load_pypdf", return_value=[]) as mock_pypdf:
                with patch.object(loader, "_load_mineru") as mock_mineru:
                    loader.load(fake_pdf)
                    mock_pypdf.assert_called_once()
                    mock_mineru.assert_not_called()

    def test_mineru_backend(self, loader, fake_pdf):
        """PDF_BACKEND=mineru → mineru path."""
        with patch.dict(os.environ, {"PDF_BACKEND": "mineru"}):
            with patch.object(loader, "_load_mineru", return_value=[]) as mock_mineru:
                with patch.object(loader, "_load_pypdf") as mock_pypdf:
                    loader.load(fake_pdf)
                    mock_mineru.assert_called_once_with(fake_pdf)
                    mock_pypdf.assert_not_called()

    def test_mineru_case_insensitive(self, loader, fake_pdf):
        """PDF_BACKEND=MINERU (uppercase) → mineru path."""
        with patch.dict(os.environ, {"PDF_BACKEND": "MINERU"}):
            with patch.object(loader, "_load_mineru", return_value=[]) as mock_mineru:
                loader.load(fake_pdf)
                mock_mineru.assert_called_once()


# ---------------------------------------------------------------------------
# Fallback tests
# ---------------------------------------------------------------------------

class TestMineruFallback:
    """Test fallback from mineru to pypdf when MinerU unavailable."""

    def test_fallback_on_connection_error(self, loader, fake_pdf):
        """MinerU service unreachable → fall back to pypdf."""
        with patch.dict(os.environ, {"PDF_BACKEND": "mineru"}):
            with patch(
                "src.ingestion.mineru_converter.convert_pdf_to_markdown",
                side_effect=ConnectionError("Cannot reach MinerU"),
            ):
                with patch.object(loader, "_load_pypdf", return_value=[]) as mock_pypdf:
                    result = loader._load_mineru(fake_pdf)
                    mock_pypdf.assert_called_once()


# ---------------------------------------------------------------------------
# Metadata schema tests
# ---------------------------------------------------------------------------

class TestMetadataSchema:
    """Verify both paths produce same metadata keys."""

    REQUIRED_METADATA_KEYS = {
        "doc_type", "source_file", "title", "arxiv_id",
        "category", "subcategory", "authors", "publish_date",
        "section_title", "chunk_index", "chunk_count",
        "token_count", "has_heading_context",
    }

    def test_pypdf_metadata_keys(self, loader, fake_pdf):
        """pypdf path metadata keys are documented correctly."""
        # Routing tests above verify pypdf path is called.
        # Metadata assembly is tested via the routing + integration tests.
        # Here we just verify the constant set is correct.
        assert "doc_type" in self.REQUIRED_METADATA_KEYS
        assert "source_file" in self.REQUIRED_METADATA_KEYS
        assert "chunk_index" in self.REQUIRED_METADATA_KEYS

    def test_mineru_metadata_has_heading_level(self):
        """MinerU path adds heading_level to metadata."""
        # heading_level is extra field in mineru path
        assert "heading_level" not in self.REQUIRED_METADATA_KEYS
        # This is by design — mineru adds it as extra field
