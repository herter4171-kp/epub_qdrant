"""Chunk PDF paper text into sized pieces with overlap."""

import re
from dataclasses import dataclass
from typing import List, Optional

from src.config import settings


@dataclass
class PaperChunk:
    """A chunk of text from a PDF paper ready for embedding."""
    id: str
    text: str
    arxiv_id: str
    title: str
    category: str
    subcategory: str
    authors: str
    publish_date: str
    chunk_index: int
    chunk_count: int
    token_count: int
    abstract: Optional[str] = None
    source_file: Optional[str] = None
    vector: Optional[List[float]] = None


def _count_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English text."""
    return len(text) // 4


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by double newlines."""
    parts = re.split(r'\n\s*\n', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_paper(
    text: str,
    arxiv_id: str,
    title: str = "",
    category: str = "",
    subcategory: str = "",
    authors: str = "",
    publish_date: str = "",
    abstract: str = "",
    source_file: str = "",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[PaperChunk]:
    """Split a PDF paper's text into chunks.

    Strategy:
    - Split text into paragraphs
    - Group paragraphs into chunks close to chunk_size tokens
    - Overlap by chunk_overlap tokens between adjacent chunks

    Args:
        text: Raw extracted text from the PDF.
        arxiv_id: arXiv identifier for the paper.
        title: Paper title.
        category: Top-level category (e.g., 'application-papers').
        subcategory: Sub-category (e.g., 'embodied-agents').
        authors: Comma-separated author names.
        publish_date: Publication date string.
        abstract: Paper abstract.
        source_file: Relative path to the PDF file.
        chunk_size: Target token count per chunk.
        chunk_overlap: Token overlap between adjacent chunks.

    Returns:
        List of PaperChunk objects.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    paragraphs = _split_paragraphs(text)

    if not paragraphs:
        return []

    chunks: list[PaperChunk] = []
    current_text_parts: list[str] = []
    current_token_count = 0
    chunk_index = 0

    def _flush(buffer: list[str], total: int) -> None:
        nonlocal chunk_index
        text = "\n\n".join(buffer)
        chunks.append(PaperChunk(
            id=f"{arxiv_id}_chunk_{chunk_index}",
            text=text,
            arxiv_id=arxiv_id,
            title=title,
            category=category,
            subcategory=subcategory,
            authors=authors,
            publish_date=publish_date,
            chunk_index=chunk_index,
            chunk_count=total,
            token_count=_count_tokens(text),
            abstract=abstract if chunk_index == 0 else None,
            source_file=source_file,
        ))
        chunk_index += 1

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        # If this paragraph alone exceeds chunk_size, split on sentences
        if para_tokens > chunk_size:
            if current_text_parts:
                _flush(current_text_parts, 999)  # placeholder total
                current_text_parts = []
                current_token_count = 0

            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_buffer: list[str] = []
            sent_count = 0
            for sent in sentences:
                sent_tokens = _count_tokens(sent)
                if sent_count + sent_tokens > chunk_size and sent_buffer:
                    _flush(sent_buffer, 999)
                    overlap_buffer: list[str] = []
                    ov_count = 0
                    for s in reversed(sent_buffer):
                        st = _count_tokens(s)
                        if ov_count + st > chunk_overlap:
                            break
                        overlap_buffer.insert(0, s)
                        ov_count += st
                    sent_buffer = overlap_buffer + [sent]
                    sent_count = ov_count + sent_tokens
                else:
                    sent_buffer.append(sent)
                    sent_count += sent_tokens

            if sent_buffer:
                _flush(sent_buffer, 999)

        # If adding this paragraph exceeds chunk_size, flush
        elif current_token_count + para_tokens > chunk_size and current_text_parts:
            _flush(current_text_parts, 999)

            overlap_buffer: list[str] = []
            ov_count = 0
            for p in reversed(current_text_parts):
                pt = _count_tokens(p)
                if ov_count + pt > chunk_overlap:
                    break
                overlap_buffer.insert(0, p)
                ov_count += pt

            current_text_parts = overlap_buffer + [para]
            current_token_count = ov_count + para_tokens
        else:
            current_text_parts.append(para)
            current_token_count += para_tokens

    # Flush remaining
    if current_text_parts:
        _flush(current_text_parts, 999)

    # Update chunk_count on all chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.chunk_count = total

    return chunks