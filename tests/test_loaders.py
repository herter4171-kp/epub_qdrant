"""Tests for updated EpubLoader and PdfLoader using semantic chunker.

These tests mock the embedding server (get_dense_vectors) so they run
locally without a GPU box. The real tokenizer.json in the project root
is used for token counting.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.ingestion.loader import DocumentChunk, DocumentLoader, EpubLoader, PdfLoader


# ── DocumentLoader.for_path ────────────────────────────────────────────────

class TestDocumentLoaderForPath:
    def test_epub_returns_epub_loader(self):
        loader = DocumentLoader.for_path(Path("test.epub"))
        assert isinstance(loader, EpubLoader)

    def test_pdf_returns_pdf_loader(self):
        loader = DocumentLoader.for_path(Path("test.pdf"))
        assert isinstance(loader, PdfLoader)

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError):
            DocumentLoader.for_path(Path("test.docx"))


# ── DocumentChunk interface unchanged ──────────────────────────────────────

class TestDocumentChunk:
    def test_fields(self):
        dc = DocumentChunk(text="hello", metadata={"k": "v"})
        assert dc.text == "hello"
        assert dc.metadata == {"k": "v"}
        assert dc.dense_vector is None

    def test_with_vector(self):
        dc = DocumentChunk(text="hi", metadata={}, dense_vector=[1.0, 2.0])
        assert dc.dense_vector == [1.0, 2.0]


# ── EpubLoader ─────────────────────────────────────────────────────────────

class TestEpubLoader:
    """Test EpubLoader with mocked epub parser and embedding server."""

    def _make_mock_book(self):
        """Create a mock Book with real section titles."""
        from dataclasses import dataclass
        from typing import Optional, List

        @dataclass
        class Section:
            title: str
            content: str
            chapter_index: int
            section_index: int
            raw_heading_level: int = 2

        @dataclass
        class Book:
            title: str
            creator: str
            sections: List[Section]
            source_file: str
            publisher: Optional[str] = None
            publication_date: Optional[str] = None
            language: Optional[str] = None
            rights: Optional[str] = None
            isbn: Optional[str] = None

        return Book(
            title="Test Book",
            creator="Author",
            sections=[
                Section(
                    title="Introduction",
                    content="This is the introduction. " * 30,
                    chapter_index=0,
                    section_index=0,
                    raw_heading_level=1,
                ),
                Section(
                    title="Chapter 1: Methods",
                    content="Methods section content here. " * 30,
                    chapter_index=1,
                    section_index=0,
                    raw_heading_level=2,
                ),
            ],
            source_file="test.epub",
            publisher="TestPub",
            language="en",
            isbn="978-0-000-00000-0",
        )

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("src.ingestion.epub_parser.parse_epub")
    def test_produces_chunks_with_real_titles(self, mock_parse, mock_embed):
        mock_parse.return_value = self._make_mock_book()
        # Embedding fn will be called but we mock it to avoid network
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        loader = EpubLoader()
        chunks = loader.load(Path("test.epub"))

        assert len(chunks) > 0
        titles = {c.metadata["section_title"] for c in chunks}
        assert "Introduction" in titles
        assert "Chapter 1: Methods" in titles
        # No (no title) since our mock has real headings
        assert "(no title)" not in titles

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("src.ingestion.epub_parser.parse_epub")
    def test_metadata_fields_present(self, mock_parse, mock_embed):
        mock_parse.return_value = self._make_mock_book()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        loader = EpubLoader()
        chunks = loader.load(Path("test.epub"))

        required_keys = {
            "doc_type", "source_file", "book_title", "section_title",
            "chapter_index", "section_index", "chunk_index", "chunk_count",
            "token_count", "publisher", "language", "isbn",
            "has_heading_context", "heading_level",
        }
        for c in chunks:
            assert required_keys.issubset(c.metadata.keys()), (
                f"Missing keys: {required_keys - c.metadata.keys()}"
            )
            assert c.metadata["doc_type"] == "book"
            assert c.metadata["book_title"] == "Test Book"
            assert c.metadata["publisher"] == "TestPub"
            assert c.metadata["token_count"] > 0

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("src.ingestion.epub_parser.parse_epub")
    def test_has_heading_context(self, mock_parse, mock_embed):
        mock_parse.return_value = self._make_mock_book()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        loader = EpubLoader()
        chunks = loader.load(Path("test.epub"))

        for c in chunks:
            if c.metadata["section_title"] and c.metadata["section_title"] != "(no title)":
                assert c.metadata["has_heading_context"] is True
                assert c.text.startswith("## ")


# ── PdfLoader ──────────────────────────────────────────────────────────────

class TestPdfLoader:
    """Test PdfLoader with mocked PDF reader and embedding server."""

    def _make_mock_reader(self):
        """Create a mock PdfReader with academic paper text."""
        text = (
            "Abstract\n\n"
            "This paper presents a novel approach. " * 20 + "\n\n"
            "Introduction\n\n"
            "We introduce our method here. " * 20 + "\n\n"
            "Related Work\n\n"
            "Previous work has shown. " * 20 + "\n\n"
            "Methodology\n\n"
            "Our methodology is as follows. " * 20 + "\n\n"
            "Results\n\n"
            "The results demonstrate. " * 20 + "\n\n"
            "Conclusion\n\n"
            "We conclude that. " * 20 + "\n\n"
            "References\n\n"
            "[1] Smith et al. 2020. Some paper title.\n"
            "[2] Jones et al. 2021. Another paper.\n"
        )
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = text
        mock_reader.pages = [mock_page]
        return mock_reader

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("pypdf.PdfReader")
    def test_produces_chunks_with_section_titles(self, mock_pdf_cls, mock_embed):
        mock_pdf_cls.return_value = self._make_mock_reader()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        loader = PdfLoader()
        # Need a sidecar json — mock _read_sidecar
        with patch.object(PdfLoader, "_read_sidecar", return_value={
            "arxiv_id": "2303.09014",
            "title": "Test Paper",
            "category": "test",
        }):
            chunks = loader.load(Path("downloads/2303_09014.pdf"))

        assert len(chunks) > 0
        titles = {c.metadata.get("section_title", "") for c in chunks}
        # Should have real section titles from paper splitter
        assert any("Abstract" in t for t in titles)

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("pypdf.PdfReader")
    def test_no_references_chunks(self, mock_pdf_cls, mock_embed):
        """No chunks with References/Bibliography section_title."""
        mock_pdf_cls.return_value = self._make_mock_reader()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        loader = PdfLoader()
        with patch.object(PdfLoader, "_read_sidecar", return_value={}):
            chunks = loader.load(Path("downloads/test.pdf"))

        for c in chunks:
            st = c.metadata.get("section_title", "").lower()
            assert "references" not in st
            assert "bibliography" not in st
            assert "appendix" not in st

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("pypdf.PdfReader")
    def test_metadata_fields_present(self, mock_pdf_cls, mock_embed):
        mock_pdf_cls.return_value = self._make_mock_reader()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        loader = PdfLoader()
        with patch.object(PdfLoader, "_read_sidecar", return_value={
            "arxiv_id": "2303.09014",
            "title": "Test Paper",
            "category": "test",
            "subcategory": "reasoning",
            "authors": "Smith et al.",
            "publish_date": "2023-03-16",
        }):
            chunks = loader.load(Path("downloads/2303_09014.pdf"))

        required_keys = {
            "doc_type", "source_file", "title", "arxiv_id",
            "category", "subcategory", "authors", "publish_date",
            "section_title", "chunk_index", "chunk_count", "token_count",
            "has_heading_context",
        }
        for c in chunks:
            assert required_keys.issubset(c.metadata.keys()), (
                f"Missing keys: {required_keys - c.metadata.keys()}"
            )
            assert c.metadata["doc_type"] == "paper"
            assert c.metadata["token_count"] > 0
