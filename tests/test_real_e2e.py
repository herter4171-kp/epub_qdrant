"""Real end-to-end test: one EPUB + one PDF through the full semantic chunking
pipeline with the LIVE embedding server on 192.168.68.75:8100.

Requires:
- Embedding server running and healthy
- tokenizer.json in project root
- test_books/ with at least one .epub
- downloads/ with at least one .pdf + .json pair

Skip gracefully if embedding server is unreachable.
"""

import pytest
import requests
from pathlib import Path

from src.ingestion.loader import EpubLoader, PdfLoader

EMBEDDING_URL = "http://192.168.68.75:8100"

# embeddinggemma-300m max input context length
MODEL_MAX_TOKENS = 2048


def _server_healthy() -> bool:
    try:
        r = requests.get(f"{EMBEDDING_URL}/health", timeout=5)
        data = r.json()
        return data.get("dense") is True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_healthy(),
    reason="Embedding server not reachable",
)


class TestRealEpubE2E:
    """Load a real EPUB through EpubLoader with live embedding server."""

    def _find_epub(self) -> Path:
        epubs = sorted(Path("test_books").glob("*.epub"))
        if not epubs:
            pytest.skip("No .epub files in test_books/")
        # Pick the RAG book — good heading structure
        for e in epubs:
            if "retrieval" in e.name.lower():
                return e
        return epubs[0]

    def test_produces_chunks(self):
        path = self._find_epub()
        loader = EpubLoader()
        chunks = loader.load(path)

        assert len(chunks) > 0, f"No chunks from {path.name}"
        print(f"\n  EPUB: {path.name} → {len(chunks)} chunks")

    def test_real_section_titles(self):
        path = self._find_epub()
        chunks = EpubLoader().load(path)

        titles = {c.metadata["section_title"] for c in chunks}
        # Should have at least some real titles (not all "(no title)")
        non_empty = {t for t in titles if t and t != "(no title)"}
        print(f"\n  Section titles: {len(non_empty)} unique real titles")
        print(f"  Sample: {list(non_empty)[:5]}")
        assert len(non_empty) > 0, "All sections are (no title)"

    def test_heading_context(self):
        path = self._find_epub()
        chunks = EpubLoader().load(path)

        with_heading = [c for c in chunks if c.metadata.get("has_heading_context")]
        print(f"\n  {len(with_heading)}/{len(chunks)} chunks have heading context")
        assert len(with_heading) > 0

    def test_token_bounds(self):
        path = self._find_epub()
        chunks = EpubLoader().load(path)

        max_tokens = max(c.metadata["token_count"] for c in chunks)
        min_tokens = min(c.metadata["token_count"] for c in chunks)
        print(f"\n  Token range: {min_tokens} – {max_tokens}")
        # chunk_size=500, but semchunk can overshoot on content with
        # non-breaking spaces (\xa0) or other tokenizer edge cases.
        # Use generous bound; model max test catches real problems.
        max_allowed = 500 * 1.5
        over = [c for c in chunks if c.metadata["token_count"] > max_allowed]
        if over:
            for c in over:
                print(f"  OVER: {c.metadata['token_count']} tokens in '{c.metadata['section_title']}'")
        assert len(over) == 0, f"{len(over)} chunks exceed {max_allowed} tokens"

    def test_no_chunk_exceeds_model_max(self):
        """No chunk should exceed embeddinggemma-300m's 2048 token context window."""
        path = self._find_epub()
        chunks = EpubLoader().load(path)

        over = [c for c in chunks if c.metadata["token_count"] > MODEL_MAX_TOKENS]
        if over:
            for c in over:
                print(f"  OVER MODEL MAX: {c.metadata['token_count']} tokens in '{c.metadata['section_title']}'")
        assert len(over) == 0, (
            f"{len(over)} chunks exceed model max {MODEL_MAX_TOKENS} tokens"
        )

    def test_metadata_complete(self):
        path = self._find_epub()
        chunks = EpubLoader().load(path)

        required = {"doc_type", "book_title", "section_title", "chunk_index",
                     "chunk_count", "token_count", "has_heading_context"}
        for c in chunks[:5]:
            missing = required - c.metadata.keys()
            assert not missing, f"Missing metadata: {missing}"


class TestRealPdfE2E:
    """Load a real PDF through PdfLoader with live embedding server."""

    def _find_pdf(self) -> Path:
        # Find a PDF that has a matching JSON sidecar
        for pdf in sorted(Path("downloads").glob("*.pdf"))[:20]:
            if pdf.with_suffix(".json").exists():
                return pdf
        pytest.skip("No PDF+JSON pair in downloads/")

    def test_produces_chunks(self):
        path = self._find_pdf()
        loader = PdfLoader()
        chunks = loader.load(path)

        assert len(chunks) > 0, f"No chunks from {path.name}"
        print(f"\n  PDF: {path.name} → {len(chunks)} chunks")

    def test_section_titles_from_splitter(self):
        path = self._find_pdf()
        chunks = PdfLoader().load(path)

        titles = {c.metadata.get("section_title", "") for c in chunks}
        print(f"\n  Section titles: {titles}")
        # Should have at least one real section title
        assert any(t and t != "Full Text" for t in titles), (
            "No real section titles found"
        )

    def test_no_references_chunks(self):
        path = self._find_pdf()
        chunks = PdfLoader().load(path)

        for c in chunks:
            st = c.metadata.get("section_title", "").lower()
            assert "references" not in st, f"Found references chunk: {st}"
            assert "bibliography" not in st

    def test_token_bounds(self):
        path = self._find_pdf()
        chunks = PdfLoader().load(path)

        for c in chunks:
            assert c.metadata["token_count"] <= 500 * 1.1 + 20

    def test_no_chunk_exceeds_model_max(self):
        """No chunk should exceed embeddinggemma-300m's 2048 token context window."""
        path = self._find_pdf()
        chunks = PdfLoader().load(path)

        over = [c for c in chunks if c.metadata["token_count"] > MODEL_MAX_TOKENS]
        if over:
            for c in over:
                print(f"  OVER MODEL MAX: {c.metadata['token_count']} tokens in '{c.metadata['section_title']}'")
        assert len(over) == 0, (
            f"{len(over)} chunks exceed model max {MODEL_MAX_TOKENS} tokens"
        )

    def test_metadata_complete(self):
        path = self._find_pdf()
        chunks = PdfLoader().load(path)

        required = {"doc_type", "title", "arxiv_id", "section_title",
                     "chunk_index", "chunk_count", "token_count",
                     "has_heading_context"}
        for c in chunks[:5]:
            missing = required - c.metadata.keys()
            assert not missing, f"Missing metadata: {missing}"
