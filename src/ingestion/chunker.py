"""Chunk text into sized pieces with overlap, respecting section boundaries."""

import re
from dataclasses import dataclass

from typing import List, Optional

from src.ingestion.epub_parser import Section


@dataclass
class Chunk:
    """A chunk of text ready for embedding."""
    id: str
    text: str
    book_title: str
    chapter_index: int
    section_title: str
    section_index: int
    chunk_index: int
    token_count: int
    # Book-level metadata for payload
    publisher: Optional[str] = None
    language: Optional[str] = None
    isbn: Optional[str] = None
    vector: Optional[List[float]] = None


def _count_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English text."""
    return len(text) // 4


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by double newlines."""
    parts = re.split(r'\n\s*\n', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_section(
    section: Section,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    book_title: str = "",
    publisher: Optional[str] = None,
    language: Optional[str] = None,
    isbn: Optional[str] = None,
) -> list[Chunk]:
    """Split a single section into chunks.

    Strategy:
    - Split section into paragraphs
    - Group paragraphs into chunks close to chunk_size tokens
    - Overlap by chunk_overlap tokens between adjacent chunks
    - Keep code blocks and short paragraphs together when possible

    Args:
        section: A parsed EPUB section.
        chunk_size: Target token count per chunk.
        chunk_overlap: Token overlap between adjacent chunks.

    Returns:
        List of Chunk objects.
    """
    paragraphs = _split_paragraphs(section.content)

    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_text_parts: list[str] = []
    current_token_count = 0
    chunk_index = 0
    global_counter = 0

    def _flush(text: str) -> None:
        nonlocal chunk_index, global_counter
        chunks.append(_make_chunk(
            text, section, chunk_index, global_counter,
            book_title=book_title,
            publisher=publisher,
            language=language,
            isbn=isbn,
        ))
        chunk_index += 1
        global_counter += 1

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        # If this paragraph alone exceeds chunk_size, split on sentences
        if para_tokens > chunk_size:
            # Flush current buffer first
            if current_text_parts:
                full_text = "\n\n".join(current_text_parts)
                _flush(full_text)
                current_text_parts = []
                current_token_count = 0

            # Split long paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_buffer: list[str] = []
            sent_count = 0
            for sent in sentences:
                sent_tokens = _count_tokens(sent)
                if sent_count + sent_tokens > chunk_size and sent_buffer:
                    full = " ".join(sent_buffer)
                    _flush(full)
                    # Keep overlap
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
                _flush(" ".join(sent_buffer))

        # If adding this paragraph exceeds chunk_size, flush
        elif current_token_count + para_tokens > chunk_size and current_text_parts:
            full_text = "\n\n".join(current_text_parts)
            _flush(full_text)

            # Build overlap buffer from end
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
        full_text = "\n\n".join(current_text_parts)
        _flush(full_text)

    return chunks


def _make_chunk(
    text: str,
    section: Section,
    chunk_index: int,
    global_counter: int,
    book_title: str = "",
    publisher: Optional[str] = None,
    language: Optional[str] = None,
    isbn: Optional[str] = None,
) -> Chunk:
    """Create a Chunk from text and section metadata."""
    # Qdrant requires integer or UUID IDs
    return Chunk(
        id=str(global_counter),
        text=text,
        book_title=book_title or section.title,
        chapter_index=section.chapter_index,
        section_title=section.title,
        section_index=section.section_index,
        chunk_index=chunk_index,
        token_count=_count_tokens(text),
        publisher=publisher,
        language=language,
        isbn=isbn,
    )
