"""Polymorphic document loader: uniform interface for EPUBs and PDFs.

Each loader yields a stream of DocumentChunk — a flat dataclass carrying
the text to embed plus a standard metadata dict that goes straight into
the Qdrant payload.  The embedding pipeline never knows what format the
source was; it just sees text + metadata.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


# ── Uniform chunk produced by every loader ────────────────────────────────────

@dataclass
class DocumentChunk:
    """One chunk ready for embedding.  Carries text + flat metadata payload."""
    text: str
    metadata: Dict[str, object]
    # Populated later by the embedding pipeline
    dense_vector: Optional[List[float]] = None


# ── Abstract loader ───────────────────────────────────────────────────────────

class DocumentLoader(ABC):
    """Base class.  Subclasses know how to turn a file into DocumentChunks."""

    @abstractmethod
    def load(self, path: Path) -> List[DocumentChunk]:
        """Parse *path* and return chunks with metadata."""
        ...

    @staticmethod
    def for_path(path: Path) -> "DocumentLoader":
        """Pick the right loader based on file extension."""
        ext = path.suffix.lower()
        if ext == ".epub":
            return EpubLoader()
        if ext == ".pdf":
            return PdfLoader()
        raise ValueError(f"No loader for extension '{ext}'")


# ── EPUB loader ───────────────────────────────────────────────────────────────

class EpubLoader(DocumentLoader):
    """Parse an EPUB, chunk sections, return DocumentChunks.

    Metadata comes entirely from the OPF embedded in the EPUB — no sidecar
    files.
    """

    def load(self, path: Path) -> List[DocumentChunk]:
        from src.ingestion.epub_parser import parse_epub
        from src.ingestion.chunker import chunk_section

        book = parse_epub(str(path))
        logger.info(
            "EPUB  %s — %s (%s, %s) — %d sections",
            path.name, book.title, book.publisher, book.language,
            len(book.sections),
        )

        chunks: List[DocumentChunk] = []
        for section in book.sections:
            raw = chunk_section(
                section,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                book_title=book.title,
                publisher=book.publisher,
                language=book.language,
                isbn=book.isbn,
            )
            for c in raw:
                chunks.append(DocumentChunk(
                    text=c.text,
                    metadata={
                        "doc_type": "book",
                        "source_file": path.name,
                        "book_title": book.title or "",
                        "section_title": c.section_title or "",
                        "chapter_index": c.chapter_index,
                        "section_index": c.section_index,
                        "chunk_index": c.chunk_index,
                        "token_count": c.token_count,
                        "publisher": book.publisher or "",
                        "language": book.language or "",
                        "isbn": book.isbn or "",
                    },
                ))
        return chunks


# ── PDF + JSON loader ─────────────────────────────────────────────────────────

class PdfLoader(DocumentLoader):
    """Extract text from a PDF, chunk it, attach metadata from sidecar JSON.

    Expects ``downloads/<id>.pdf`` with an optional ``downloads/<id>.json``
    containing ``{"metadataAttributes": ["key: value", ...]}``.
    """

    def load(self, path: Path) -> List[DocumentChunk]:
        from pypdf import PdfReader
        from src.ingestion.paper_loader import chunk_paper

        # ── extract text (re-download if corrupt) ─────────────────────
        try:
            reader = PdfReader(str(path))
        except Exception as e:
            logger.warning("PDF %s corrupt (%s) — attempting re-download", path.name, e)
            if not self._redownload(path):
                return []
            try:
                reader = PdfReader(str(path))
            except Exception as e2:
                logger.warning("PDF %s still corrupt after re-download — skipping: %s", path.name, e2)
                return []

        pages = [p.extract_text() for p in reader.pages if p.extract_text()]
        text = "\n\n".join(pages)
        if len(text.strip()) < 100:
            logger.warning("PDF %s yielded <100 chars — skipping", path.name)
            return []

        # ── sidecar metadata ──────────────────────────────────────────
        meta = self._read_sidecar(path)
        arxiv_id = meta.get("arxiv_id", path.stem.replace("_", "."))
        title = meta.get("title", path.stem)
        category = meta.get("category", "")
        subcategory = meta.get("subcategory", "")
        authors = meta.get("authors", "")
        publish_date = meta.get("publish_date", "")
        abstract = meta.get("abstract", "")

        logger.info(
            "PDF   %s — %s (cat=%s) — %d chars",
            path.name, title[:60], category, len(text),
        )

        raw = chunk_paper(
            text=text,
            arxiv_id=arxiv_id,
            title=title,
            category=category,
            subcategory=subcategory,
            authors=authors,
            publish_date=publish_date,
            abstract=abstract,
            source_file=path.name,
        )

        chunks: List[DocumentChunk] = []
        for c in raw:
            chunks.append(DocumentChunk(
                text=c.text,
                metadata={
                    "doc_type": "paper",
                    "source_file": path.name,
                    "title": c.title or "",
                    "arxiv_id": c.arxiv_id or "",
                    "category": c.category or "",
                    "subcategory": c.subcategory or "",
                    "authors": c.authors or "",
                    "publish_date": c.publish_date or "",
                    "chunk_index": c.chunk_index,
                    "chunk_count": c.chunk_count,
                    "token_count": c.token_count,
                },
            ))
        return chunks

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _redownload(pdf_path: Path) -> bool:
        """Re-download a paper PDF from arxiv.  Filename convention: 2206_10498.pdf → arxiv id 2206.10498."""
        import requests as _req

        arxiv_id = pdf_path.stem.replace("_", ".")
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        logger.info("  Downloading %s → %s", url, pdf_path.name)
        try:
            resp = _req.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            logger.info("  Re-downloaded %s (%d bytes)", pdf_path.name, pdf_path.stat().st_size)
            return True
        except Exception as e:
            logger.warning("  Re-download failed for %s: %s", pdf_path.name, e)
            return False

    @staticmethod
    def _read_sidecar(pdf_path: Path) -> Dict[str, str]:
        json_path = pdf_path.with_suffix(".json")
        if not json_path.exists():
            return {}
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            result: Dict[str, str] = {}
            for attr in data.get("metadataAttributes", []):
                if ": " in attr:
                    k, v = attr.split(": ", 1)
                    result[k] = v
                elif ":" in attr:
                    k, v = attr.split(":", 1)
                    result[k.strip()] = v.strip()
            return result
        except Exception as e:
            logger.warning("Failed to parse %s: %s", json_path.name, e)
            return {}
