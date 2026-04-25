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

import os

logger = logging.getLogger(__name__)

# PDF backend constants
PDF_BACKEND_PYPDF = "pypdf"
PDF_BACKEND_MINERU = "mineru"
PDF_BACKEND_MINERU_JSON = "mineru_json"


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

    @staticmethod
    def _read_sidecar(path: Path) -> Dict[str, str]:
        """Read sidecar metadata JSON adjacent to an EPUB or PDF.

        Convention: {filename}.metadata.json
          - EPUB: book.epub.metadata.json
          - PDF:  2010_03768.pdf.metadata.json

        Schema: {"metadataAttributes": {"key": "value", ...}}
        """
        json_path = path.with_name(path.name + ".metadata.json")
        if not json_path.exists():
            return {}
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            attrs = data.get("metadataAttributes", {})
            if isinstance(attrs, dict):
                return {k: str(v) for k, v in attrs.items()}
            # Backward compat: list of "key: value" strings
            if isinstance(attrs, list):
                result: Dict[str, str] = {}
                for attr in attrs:
                    if ": " in attr:
                        k, v = attr.split(": ", 1)
                        result[k] = v
                    elif ":" in attr:
                        k, v = attr.split(":", 1)
                        result[k.strip()] = v.strip()
                return result
            return {}
        except Exception as e:
            logger.warning("Failed to parse %s: %s", json_path.name, e)
            return {}


# ── EPUB loader ───────────────────────────────────────────────────────────────

class EpubLoader(DocumentLoader):
    """Parse an EPUB, read optional sidecar metadata, chunk sections with semantic chunker, return DocumentChunks."""

    def load(self, path: Path) -> List[DocumentChunk]:
        from src.ingestion.epub_parser import parse_epub
        from src.ingestion.semantic_chunker import ChunkConfig, chunk_section, load_tokenizer
        from servers.embedding_server.client import get_dense_vectors

        book = parse_epub(str(path))

        # ── sidecar metadata (optional) ───────────────────────────────
        meta = DocumentLoader._read_sidecar(path)

        # Override EPUB OPF values when sidecar provides them
        if meta.get("title"):
            book.title = meta["title"]
        if meta.get("authors"):
            book.creator = meta["authors"]
        if meta.get("publisher"):
            book.publisher = meta["publisher"]
        if meta.get("language"):
            book.language = meta["language"]
        if meta.get("isbn"):
            book.isbn = meta["isbn"]
        if meta.get("publish_date"):
            book.publication_date = meta["publish_date"]

        logger.info(
            "EPUB  %s — %s (%s, %s) — %d sections",
            path.name, book.title, book.publisher, book.language,
            len(book.sections),
        )

        token_counter = load_tokenizer(settings.TOKENIZER_JSON or None)
        config = ChunkConfig(
            chunk_size=settings.CHUNK_SIZE,
            overlap_ratio=settings.CHUNK_OVERLAP_RATIO,
            similarity_percentile=settings.SIMILARITY_PERCENTILE,
            min_distance_floor=settings.MIN_DISTANCE_FLOOR,
            min_sentences_for_semantic=settings.MIN_SENTENCES_FOR_SEMANTIC,
            min_chunk_tokens=settings.MIN_CHUNK_TOKENS,
            enable_semantic=settings.SEMANTIC_CHUNKING_ENABLED,
            tokenizer_path=settings.TOKENIZER_JSON or None,
        )

        chunks: List[DocumentChunk] = []
        for section in book.sections:
            results = chunk_section(
                title=section.title,
                content=section.content,
                config=config,
                token_counter=token_counter,
                embedding_fn=get_dense_vectors,
            )
            chunk_count = len(results)
            for cr in results:
                chunks.append(DocumentChunk(
                    text=cr.text,
                    metadata={
                        "doc_type": "book",
                        "source_file": path.name,
                        "book_title": book.title or "",
                        "section_title": cr.section_title or "",
                        "chapter_index": section.chapter_index,
                        "section_index": section.section_index,
                        "chunk_index": cr.chunk_index,
                        "chunk_count": chunk_count,
                        "token_count": cr.token_count,
                        "publisher": book.publisher or "",
                        "language": book.language or "",
                        "isbn": book.isbn or "",
                        "has_heading_context": cr.has_heading_context,
                        "heading_level": getattr(section, "raw_heading_level", 0),
                    },
                ))
        return chunks


# ── PDF + JSON loader ─────────────────────────────────────────────────────────

class PdfLoader(DocumentLoader):
    """Extract text from a PDF, chunk with semantic chunker, attach sidecar metadata.

    Supports two backends via ``PDF_BACKEND`` env var:
    - ``"pypdf"`` (default): existing pypdf extraction + paper_section_splitter
    - ``"mineru"``: MinerU layout-aware PDF→Markdown + md_section_splitter + citation masking

    Expects ``downloads/<id>.pdf`` with an optional ``downloads/<id>.json``
    containing ``{"metadataAttributes": ["key: value", ...]}``.
    """

    def load(self, path: Path) -> List[DocumentChunk]:
        backend = os.getenv("PDF_BACKEND", PDF_BACKEND_PYPDF).lower()
        if backend == PDF_BACKEND_MINERU:
            return self._load_mineru(path)
        if backend == PDF_BACKEND_MINERU_JSON:
            return self._load_mineru_json(path)
        return self._load_pypdf(path)

    def _load_pypdf(self, path: Path) -> List[DocumentChunk]:
        """Existing pypdf extraction path — unchanged behavior."""
        from pypdf import PdfReader
        from src.ingestion.paper_section_splitter import split_paper_sections
        from src.ingestion.semantic_chunker import ChunkConfig, chunk_section, load_tokenizer
        from servers.embedding_server.client import get_dense_vectors

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

        logger.info(
            "PDF   %s — %s (cat=%s) — %d chars",
            path.name, title[:60], category, len(text),
        )

        # ── split into sections, then chunk each ─────────────────────
        token_counter = load_tokenizer(settings.TOKENIZER_JSON or None)
        config = ChunkConfig(
            chunk_size=settings.CHUNK_SIZE,
            overlap_ratio=settings.CHUNK_OVERLAP_RATIO,
            similarity_percentile=settings.SIMILARITY_PERCENTILE,
            min_distance_floor=settings.MIN_DISTANCE_FLOOR,
            min_sentences_for_semantic=settings.MIN_SENTENCES_FOR_SEMANTIC,
            min_chunk_tokens=settings.MIN_CHUNK_TOKENS,
            enable_semantic=settings.SEMANTIC_CHUNKING_ENABLED,
            tokenizer_path=settings.TOKENIZER_JSON or None,
        )

        sections = split_paper_sections(text)
        chunks: List[DocumentChunk] = []

        for ps in sections:
            results = chunk_section(
                title=ps.title,
                content=ps.content,
                config=config,
                token_counter=token_counter,
                embedding_fn=get_dense_vectors,
            )
            chunk_count = len(results)
            for cr in results:
                chunks.append(DocumentChunk(
                    text=cr.text,
                    metadata={
                        "doc_type": "paper",
                        "source_file": path.name,
                        "title": title or "",
                        "arxiv_id": arxiv_id or "",
                        "category": category or "",
                        "subcategory": subcategory or "",
                        "authors": authors or "",
                        "publish_date": publish_date or "",
                        "section_title": cr.section_title or "",
                        "chunk_index": cr.chunk_index,
                        "chunk_count": chunk_count,
                        "token_count": cr.token_count,
                        "has_heading_context": cr.has_heading_context,
                    },
                ))
        return chunks

    def _load_mineru(self, path: Path) -> List[DocumentChunk]:
        """MinerU path: convert → split by Markdown headings → chunk with citation masking."""
        from src.ingestion.semantic_chunker import ChunkConfig, chunk_section, load_tokenizer
        from servers.embedding_server.client import get_dense_vectors

        try:
            from src.ingestion.mineru_converter import convert_pdf_to_markdown
            from src.ingestion.md_section_splitter import split_markdown_sections
            from src.ingestion.citation_masker import citation_aware_split
        except ImportError:
            logger.warning(
                "MinerU dependencies not available for %s — falling back to pypdf",
                path.name,
            )
            return self._load_pypdf(path)

        # ── convert PDF → Markdown ────────────────────────────────────
        try:
            markdown = convert_pdf_to_markdown(str(path))
        except ConnectionError:
            logger.warning(
                "MinerU service unreachable for %s — falling back to pypdf. "
                "Start with: mineru-api --host 0.0.0.0 --port 8010",
                path.name,
            )
            return self._load_pypdf(path)

        # ── sidecar metadata ──────────────────────────────────────────
        meta = self._read_sidecar(path)
        arxiv_id = meta.get("arxiv_id", path.stem.replace("_", "."))
        title = meta.get("title", path.stem)
        category = meta.get("category", "")
        subcategory = meta.get("subcategory", "")
        authors = meta.get("authors", "")
        publish_date = meta.get("publish_date", "")

        logger.info(
            "PDF   %s [mineru] — %s (cat=%s) — %d chars",
            path.name, title[:60], category, len(markdown),
        )

        # ── split into sections, then chunk each ─────────────────────
        token_counter = load_tokenizer(settings.TOKENIZER_JSON or None)
        config = ChunkConfig(
            chunk_size=settings.CHUNK_SIZE,
            overlap_ratio=settings.CHUNK_OVERLAP_RATIO,
            similarity_percentile=settings.SIMILARITY_PERCENTILE,
            min_distance_floor=settings.MIN_DISTANCE_FLOOR,
            min_sentences_for_semantic=settings.MIN_SENTENCES_FOR_SEMANTIC,
            min_chunk_tokens=settings.MIN_CHUNK_TOKENS,
            enable_semantic=settings.SEMANTIC_CHUNKING_ENABLED,
            tokenizer_path=settings.TOKENIZER_JSON or None,
        )

        md_sections = split_markdown_sections(markdown)
        chunks: List[DocumentChunk] = []

        for ms in md_sections:
            results = chunk_section(
                title=ms.title,
                content=ms.content,
                config=config,
                token_counter=token_counter,
                embedding_fn=get_dense_vectors,
                sentence_splitter=citation_aware_split,
            )
            chunk_count = len(results)
            for cr in results:
                chunks.append(DocumentChunk(
                    text=cr.text,
                    metadata={
                        "doc_type": "paper",
                        "source_file": path.name,
                        "title": title or "",
                        "arxiv_id": arxiv_id or "",
                        "category": category or "",
                        "subcategory": subcategory or "",
                        "authors": authors or "",
                        "publish_date": publish_date or "",
                        "section_title": cr.section_title or "",
                        "chunk_index": cr.chunk_index,
                        "chunk_count": chunk_count,
                        "token_count": cr.token_count,
                        "has_heading_context": cr.has_heading_context,
                        "heading_level": ms.heading_level,
                    },
                ))
        return chunks

    def _load_mineru_json(self, path: Path) -> List[DocumentChunk]:
        """JSON path: resolve sidecar → parse → chunk → DocumentChunk assembly.

        Falls back to _load_pypdf() if:
          - JSON path cannot be resolved (logs warning with arxiv_id + base_dir)
          - JSON file fails to parse (logs warning with parse error)
        """
        from src.ingestion.mineru_json_parser import resolve_json_path, parse_content_list
        from src.ingestion.semantic_chunker import ChunkConfig, chunk_section, load_tokenizer
        from src.ingestion.citation_masker import citation_aware_split
        from servers.embedding_server.client import get_dense_vectors

        # Derive arxiv_id from PDF stem: "2603_07444.pdf" → "2603_07444"
        import re
        m = re.match(r'^(\d+_\d+)', path.stem)
        if not m:
            logger.warning(
                "PDF %s stem '%s' does not match expected arxiv format (NNNN_NNNNN) — falling back to pypdf",
                path.name, path.stem,
            )
            return self._load_pypdf(path)
        arxiv_id = m.group(1)

        # Resolve JSON sidecar
        json_path = resolve_json_path(arxiv_id)
        if json_path is None:
            base_dir = os.getenv("MINERU_OUTPUT_DIR", "mineru_output")
            logger.warning(
                "JSON not found for arxiv_id=%s, base_dir=%s — falling back to pypdf",
                arxiv_id, base_dir,
            )
            return self._load_pypdf(path)

        # Read sidecar metadata (same as pypdf/mineru paths)
        meta = self._read_sidecar(path)

        # Bug 3 fix: prefer dot-format arxiv_id from sidecar (e.g. "2201.11903")
        # over the underscore directory key (e.g. "2201_11903").
        arxiv_id_dot = meta.get("arxiv_id", arxiv_id.replace("_", "."))

        title = meta.get("title", path.stem)
        category = meta.get("category", "")
        subcategory = meta.get("subcategory", "")
        authors = meta.get("authors", "")
        publish_date = meta.get("publish_date", "")

        logger.info(
            "PDF   %s [mineru_json] — %s (cat=%s, arxiv=%s) — JSON=%s",
            path.name, title[:60], category, arxiv_id_dot, json_path,
        )

        # Parse JSON → sections
        try:
            json_sections = parse_content_list(json_path)
        except (ValueError, Exception) as e:
            logger.warning(
                "JSON parse failed for %s (%s) — falling back to pypdf",
                json_path, e,
            )
            return self._load_pypdf(path)

        if not json_sections:
            logger.warning(
                "JSON %s produced 0 sections — falling back to pypdf",
                json_path,
            )
            return self._load_pypdf(path)

        # Chunk each section
        token_counter = load_tokenizer(settings.TOKENIZER_JSON or None)
        config = ChunkConfig(
            chunk_size=settings.CHUNK_SIZE,
            overlap_ratio=settings.CHUNK_OVERLAP_RATIO,
            similarity_percentile=settings.SIMILARITY_PERCENTILE,
            min_distance_floor=settings.MIN_DISTANCE_FLOOR,
            min_sentences_for_semantic=settings.MIN_SENTENCES_FOR_SEMANTIC,
            min_chunk_tokens=settings.MIN_CHUNK_TOKENS,
            enable_semantic=settings.SEMANTIC_CHUNKING_ENABLED,
            tokenizer_path=settings.TOKENIZER_JSON or None,
        )

        chunks: List[DocumentChunk] = []
        for js in json_sections:
            results = chunk_section(
                title=js.title,
                content=js.content,
                config=config,
                token_counter=token_counter,
                embedding_fn=get_dense_vectors,
                sentence_splitter=citation_aware_split,
            )
            chunk_count = len(results)
            for cr in results:
                chunks.append(DocumentChunk(
                    text=cr.text,
                    metadata={
                        "doc_type": "paper",
                        "source_file": path.name,
                        "title": title or "",
                        "arxiv_id": arxiv_id_dot or "",
                        "category": category or "",
                        "subcategory": subcategory or "",
                        "authors": authors or "",
                        "publish_date": publish_date or "",
                        "section_title": cr.section_title or "",
                        "chunk_index": cr.chunk_index,
                        "chunk_count": chunk_count,
                        "token_count": cr.token_count,
                        "has_heading_context": cr.has_heading_context,
                        "heading_level": js.heading_level,
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

