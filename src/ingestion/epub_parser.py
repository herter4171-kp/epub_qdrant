"""Extract text content from EPUB files, preserving chapter/section structure."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import epub


@dataclass
class Section:
    """A section of text from an EPUB, with metadata."""
    title: str
    content: str
    chapter_index: int
    section_index: int


@dataclass
class Book:
    """Represent a parsed EPUB book with full OPF metadata."""
    title: str
    creator: str
    sections: List[Section]
    source_file: str
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    language: Optional[str] = None
    rights: Optional[str] = None
    isbn: Optional[str] = None


def _clean_html_text(html_text: str) -> str:
    """Strip HTML tags and clean up whitespace."""
    # Remove script and style elements
    html_text = re.sub(r'<script[^>]*>.*?</script>', '', html_text, flags=re.DOTALL)
    html_text = re.sub(r'<style[^>]*>.*?</style>', '', html_text, flags=re.DOTALL)

    # Decode HTML entities using standard library
    import html as html_module
    html_text = html_module.unescape(html_text)

    # Remove remaining HTML tags
    html_text = re.sub(r'<[^>]+>', '', html_text)

    # Collapse whitespace
    html_text = re.sub(r'\n+', '\n', html_text)
    html_text = re.sub(r' {2,}', ' ', html_text)
    html_text = html_text.strip()

    return html_text


def _extract_text_from_html(raw_bytes: bytes) -> str:
    """Extract clean text from HTML/xhtml content bytes."""
    text = raw_bytes.decode('utf-8', errors='replace')
    return _clean_html_text(text)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (heading_title, content) pairs by <h1>-<h3> headings."""
    # Find all heading blocks and their content
    pattern = r'<h([1-3])[^>]*>(.*?)</h\1>\s*((?:.(?!<h[1-3]))*)'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

    if matches:
        sections = []
        for _, heading_text, content in matches:
            title = re.sub(r'<[^>]+>', '', heading_text).strip()
            section_text = _clean_html_text(content)
            if section_text:
                sections.append((title, section_text))

        if sections:
            return sections

    # Fallback: treat entire content as one section
    cleaned = _clean_html_text(text)
    if cleaned:
        return [("(no title)", cleaned)]

    return []


def parse_epub(epub_path: str) -> Book:
    """Parse an EPUB file and extract structured sections.

    Args:
        epub_path: Path to the EPUB file.

    Returns:
        A Book object with title, creator, and list of sections.

    Raises:
        FileNotFoundError: If the EPUB file does not exist.
    """
    path = Path(epub_path)
    if not path.exists():
        raise FileNotFoundError(f"EPUB not found: {epub_path}")

    # Open the EPUB using the correct API
    book_file = epub.open_epub(str(path))

    # Extract metadata from OPF
    meta = book_file.opf.metadata
    titles = meta.titles
    creators = meta.creators
    book_title = titles[0][0] if titles else path.stem
    book_creator = creators[0][0] if creators else "Unknown"

    # Extract additional metadata
    publisher = meta.publisher or None
    dates = meta.dates
    publication_date = dates[0][0] if dates else None
    languages = meta.languages
    language = languages[0] if languages else None
    rights = meta.right or None

    # Get ISBN from identifiers (try to find isbn scheme first, fall back to any)
    isbn = None
    for ident_id, scheme, props in meta.identifiers:
        if scheme and 'isbn' in scheme.lower():
            isbn = ident_id
            break
    # Fallback: use first identifier if no ISBN found
    if not isbn and meta.identifiers:
        isbn = meta.identifiers[0][0]

    # Iterate through spine items
    sections: List[Section] = []
    chapter_index = 0

    # spine.itemrefs is a list of (idref, linear) tuples
    for idref, linear in book_file.opf.spine.itemrefs:
        manifest_item = book_file.opf.manifest.get(idref, None)
        if manifest_item is None:
            continue

        try:
            raw_content = book_file.read_item(manifest_item)
        except Exception:
            continue

        html_text = _extract_text_from_html(raw_content)
        if not html_text:
            continue

        # Split content into sections by headings
        sub_sections = _split_into_sections(html_text)

        for section_index, (title, content) in enumerate(sub_sections):
            if content and len(content) > 20:
                sections.append(Section(
                    title=title,
                    content=content,
                    chapter_index=chapter_index,
                    section_index=section_index,
                ))

        chapter_index += 1

    book_file.close()

    return Book(
        title=book_title,
        creator=book_creator,
        sections=sections,
        source_file=str(path.name),
        publisher=publisher,
        publication_date=publication_date,
        language=language,
        rights=rights,
        isbn=isbn,
    )
