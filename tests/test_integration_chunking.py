"""Integration tests for end-to-end chunking pipeline.

Tests the full flow: parse → split → chunk → verify metadata.
Mocks embedding server since it's on the GPU box.
Uses real tokenizer.json from project root.
"""

import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.ingestion.loader import EpubLoader, PdfLoader
from src.ingestion.semantic_chunker import (
    ChunkConfig, ChunkResult, chunk_section, load_tokenizer,
)

TOKENIZER_PATH = "tokenizer.json"


# ── EPUB end-to-end (task 14.1) ───────────────────────────────────────────

class TestEpubIntegration:
    """End-to-end EPUB ingestion with mocked embedding server."""

    def _mock_book(self):
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
            title="Integration Test Book",
            creator="Test Author",
            sections=[
                Section("Preface", "Short preface content.", 0, 0, 1),
                Section(
                    "Chapter 1: Deep Learning",
                    "Deep learning is a subset of machine learning. " * 80,
                    1, 0, 2,
                ),
                Section(
                    "Chapter 2: Transformers",
                    "Transformers use self-attention mechanisms. " * 80,
                    2, 0, 2,
                ),
            ],
            source_file="test.epub",
            publisher="TestPub",
            language="en",
            isbn="978-0-000-00000-0",
        )

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("src.ingestion.epub_parser.parse_epub")
    def test_real_section_titles(self, mock_parse, mock_embed):
        mock_parse.return_value = self._mock_book()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        chunks = EpubLoader().load(Path("test.epub"))
        titles = {c.metadata["section_title"] for c in chunks}
        assert "Chapter 1: Deep Learning" in titles
        assert "Chapter 2: Transformers" in titles

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("src.ingestion.epub_parser.parse_epub")
    def test_token_counts_in_bounds(self, mock_parse, mock_embed):
        mock_parse.return_value = self._mock_book()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        chunks = EpubLoader().load(Path("test.epub"))
        for c in chunks:
            # 10% tolerance + small buffer for heading overhead
            assert c.metadata["token_count"] <= 500 * 1.1 + 10

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("src.ingestion.epub_parser.parse_epub")
    def test_heading_context_prepended(self, mock_parse, mock_embed):
        mock_parse.return_value = self._mock_book()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        chunks = EpubLoader().load(Path("test.epub"))
        for c in chunks:
            if c.metadata["section_title"] and c.metadata["section_title"] != "(no title)":
                assert c.text.startswith("## ")
                assert c.metadata["has_heading_context"] is True

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("src.ingestion.epub_parser.parse_epub")
    def test_all_metadata_present(self, mock_parse, mock_embed):
        mock_parse.return_value = self._mock_book()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        chunks = EpubLoader().load(Path("test.epub"))
        for c in chunks:
            m = c.metadata
            assert m["doc_type"] == "book"
            assert m["book_title"] == "Integration Test Book"
            assert m["publisher"] == "TestPub"
            assert m["language"] == "en"
            assert m["isbn"] == "978-0-000-00000-0"
            assert "chunk_index" in m
            assert "chunk_count" in m
            assert "heading_level" in m



# ── PDF end-to-end (task 14.2) ────────────────────────────────────────────

class TestPdfIntegration:
    """End-to-end PDF ingestion with mocked PDF reader and embedding server."""

    def _mock_reader(self):
        text = (
            "Abstract\n\n"
            "We present a novel framework for agent-based systems. " * 30 + "\n\n"
            "Introduction\n\n"
            "Recent advances in large language models have enabled. " * 30 + "\n\n"
            "2. Related Work\n\n"
            "Previous approaches to agent design include. " * 30 + "\n\n"
            "3. Methodology\n\n"
            "Our approach consists of three components. " * 30 + "\n\n"
            "4. Experiments\n\n"
            "We evaluate on three benchmarks. " * 30 + "\n\n"
            "5. Conclusion\n\n"
            "We have demonstrated that our framework. " * 30 + "\n\n"
            "References\n\n"
            "[1] Brown et al. Language Models are Few-Shot Learners. 2020.\n"
            "[2] Wei et al. Chain-of-Thought Prompting. 2022.\n"
        )
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = text
        mock_reader.pages = [mock_page]
        return mock_reader

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("pypdf.PdfReader")
    def test_section_splitting(self, mock_pdf_cls, mock_embed):
        mock_pdf_cls.return_value = self._mock_reader()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        with patch.object(PdfLoader, "_read_sidecar", return_value={
            "arxiv_id": "2401.00001", "title": "Agent Framework",
            "category": "agent-frameworks", "authors": "Smith et al.",
        }):
            chunks = PdfLoader().load(Path("downloads/2401_00001.pdf"))

        titles = {c.metadata.get("section_title", "") for c in chunks}
        assert any("Abstract" in t for t in titles)
        assert any("Methodology" in t or "3." in t for t in titles)

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("pypdf.PdfReader")
    def test_references_excluded(self, mock_pdf_cls, mock_embed):
        mock_pdf_cls.return_value = self._mock_reader()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        with patch.object(PdfLoader, "_read_sidecar", return_value={}):
            chunks = PdfLoader().load(Path("downloads/test.pdf"))

        for c in chunks:
            st = c.metadata.get("section_title", "").lower()
            assert "references" not in st
            assert "bibliography" not in st

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("pypdf.PdfReader")
    def test_token_counts_in_bounds(self, mock_pdf_cls, mock_embed):
        mock_pdf_cls.return_value = self._mock_reader()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        with patch.object(PdfLoader, "_read_sidecar", return_value={}):
            chunks = PdfLoader().load(Path("downloads/test.pdf"))

        for c in chunks:
            assert c.metadata["token_count"] <= 500 * 1.1 + 10

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("pypdf.PdfReader")
    def test_all_metadata_present(self, mock_pdf_cls, mock_embed):
        mock_pdf_cls.return_value = self._mock_reader()
        mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768] * len(texts)

        with patch.object(PdfLoader, "_read_sidecar", return_value={
            "arxiv_id": "2401.00001", "title": "Agent Framework",
            "category": "agent-frameworks", "subcategory": "reasoning",
            "authors": "Smith et al.", "publish_date": "2024-01-01",
        }):
            chunks = PdfLoader().load(Path("downloads/2401_00001.pdf"))

        for c in chunks:
            m = c.metadata
            assert m["doc_type"] == "paper"
            assert m["arxiv_id"] == "2401.00001"
            assert m["title"] == "Agent Framework"
            assert "section_title" in m
            assert "has_heading_context" in m


# ── Embedding server fallback (task 14.3) ─────────────────────────────────

class TestEmbeddingFallbackIntegration:
    """Verify chunking completes when embedding server is unavailable."""

    @patch("servers.embedding_server.client.get_dense_vectors")
    @patch("src.ingestion.epub_parser.parse_epub")
    def test_epub_fallback_on_connection_error(self, mock_parse, mock_embed):
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

        mock_parse.return_value = Book(
            title="Fallback Test",
            creator="Author",
            sections=[Section(
                "Chapter 1",
                "Content for fallback testing. " * 50,
                0, 0, 2,
            )],
            source_file="test.epub",
        )
        mock_embed.side_effect = ConnectionError("server down")

        # Should NOT raise — graceful fallback
        chunks = EpubLoader().load(Path("test.epub"))
        assert len(chunks) > 0
        # Chunks still have structural splitting
        for c in chunks:
            assert c.metadata["section_title"] == "Chapter 1"

    def test_chunk_section_fallback_logged(self, caplog):
        """Warning logged when embedding fails."""
        counter = load_tokenizer(path=TOKENIZER_PATH)

        def bad_embed(texts):
            raise ConnectionError("unreachable")

        cfg = ChunkConfig(
            chunk_size=50,
            enable_semantic=True,
            min_sentences_for_semantic=3,
        )
        text = "Test sentence for logging. " * 20

        with caplog.at_level(logging.WARNING):
            results = chunk_section("Test", text, cfg, counter, embedding_fn=bad_embed)

        assert len(results) >= 1
        assert any("Embedding server error" in r.message for r in caplog.records)
