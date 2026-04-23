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
    raw_heading_level: int = 0  # 1-3 for h1-h3, 0 for fallback


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


def _extract_headings_from_html(raw_bytes: bytes) -> List[tuple]:
    """Extract heading level, title text, content_start, content_end from raw HTML.

    Runs regex against raw HTML before any tag stripping so headings are
    reliably detected even when _clean_html_text would collapse them.

    Returns:
        List of (level, title, content_start, content_end) tuples in
        document order.  Empty list when no h1–h3 headings are found.
    """
    html_text = raw_bytes.decode('utf-8', errors='replace')

    heading_pattern = re.compile(
        r'<h([1-3])[^>]*>(.*?)</h\1>',
        re.IGNORECASE | re.DOTALL,
    )

    matches = list(heading_pattern.finditer(html_text))
    if not matches:
        return []

    results: List[tuple] = []
    for i, match in enumerate(matches):
        level = int(match.group(1))
        title_html = match.group(2)
        # Strip inner HTML tags (e.g. <span>, <a>) from heading text
        title = re.sub(r'<[^>]+>', '', title_html).strip()

        if not title:
            continue

        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(html_text)

        results.append((level, title, content_start, content_end))

    return results


def _split_into_sections(raw_bytes: bytes) -> list[tuple[str, str, int]]:
    """Split raw HTML bytes into (title, cleaned_content, heading_level) triples.

    Uses _extract_headings_from_html on the raw bytes so that heading tags
    are matched before any cleaning.  Falls back to a single ``(no title)``
    section when no headings are found or the HTML is malformed.
    """
    headings = _extract_headings_from_html(raw_bytes)

    if not headings:
        cleaned = _extract_text_from_html(raw_bytes)
        if cleaned:
            return [("(no title)", cleaned, 0)]
        return []

    html_text = raw_bytes.decode('utf-8', errors='replace')
    sections: list[tuple[str, str, int]] = []

    for level, title, start, end in headings:
        raw_content = html_text[start:end]
        cleaned = _clean_html_text(raw_content)
        if cleaned and len(cleaned) > 20:
            sections.append((title, cleaned, level))

    if not sections:
        cleaned = _extract_text_from_html(raw_bytes)
        if cleaned:
            return [("(no title)", cleaned, 0)]
        return []

    return sections


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

        # Quick check: skip items with no meaningful text
        html_text = _extract_text_from_html(raw_content)
        if not html_text:
            continue

        # Split content into sections by headings (operates on raw bytes)
        sub_sections = _split_into_sections(raw_content)

        for section_index, (title, content, heading_level) in enumerate(sub_sections):
            if content and len(content) > 20:
                sections.append(Section(
                    title=title,
                    content=content,
                    chapter_index=chapter_index,
                    section_index=section_index,
                    raw_heading_level=heading_level,
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
