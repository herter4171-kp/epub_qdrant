"""Split MinerU Markdown output into named sections by heading markers.

Analogous to ``paper_section_splitter.py`` for raw PDF text, but operates
on Markdown heading syntax (``#``, ``##``, ``###``) produced by MinerU's
layout-aware PDF-to-Markdown converter.

Informed by MinerU usage in Xue et al. (arXiv:2601.15170), AutoPage
(arXiv:2510.19600), and PaperBanana (arXiv:2601.23265) for structured
Markdown extraction from academic PDFs.

Public API:
    split_markdown_sections(markdown) -> list[MarkdownSection]
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches Markdown headings: #, ##, or ### followed by whitespace and text.
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)", re.MULTILINE)

# Section titles to exclude from output (case-insensitive, after stripping
# leading numbers/letters like "5. References" or "A. Appendix").
EXCLUDED_SECTIONS = {"references", "bibliography", "appendix"}


@dataclass
class MarkdownSection:
    """A named section from MinerU Markdown output."""

    title: str
    content: str
    heading_level: int  # 1 for #, 2 for ##, 3 for ###
    section_index: int


def _is_excluded(title: str) -> bool:
    """Return True if *title* matches an excluded section name.

    Strips leading numbers, letters, and punctuation so that
    ``"5. References"``, ``"A. Appendix"``, and ``"BIBLIOGRAPHY"``
    all match.
    """
    stripped = re.sub(r"^[A-Za-z0-9]+\.?\s*", "", title).strip().lower()
    # Also check the raw title in case it's just "References" with no prefix
    raw = title.strip().lower()
    return stripped in EXCLUDED_SECTIONS or raw in EXCLUDED_SECTIONS


def split_markdown_sections(markdown: str) -> list[MarkdownSection]:
    """Split Markdown text at heading boundaries.

    Each heading (``#``, ``##``, ``###``) becomes a split point. The heading
    text becomes the section title and the ``#`` count determines
    ``heading_level``.

    Sections matching References, Bibliography, or Appendix (case-insensitive,
    with or without leading numbers like ``"5. References"``) are excluded.

    If fewer than 2 headings are found, returns a single ``"Full Text"``
    section with ``heading_level=1``.

    Sections have sequential indices starting from 0 and non-empty content.
    """
    if not markdown or not markdown.strip():
        return [MarkdownSection(title="Full Text", content="", heading_level=1, section_index=0)]

    matches = list(_HEADING_RE.finditer(markdown))

    if len(matches) < 2:
        return [MarkdownSection(
            title="Full Text",
            content=markdown.strip(),
            heading_level=1,
            section_index=0,
        )]

    sections: list[MarkdownSection] = []
    for i, match in enumerate(matches):
        hashes = match.group(1)
        title = match.group(2).strip()
        heading_level = len(hashes)

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = markdown[start:end].strip()

        if _is_excluded(title):
            continue

        if not content:
            continue

        sections.append(MarkdownSection(
            title=title,
            content=content,
            heading_level=heading_level,
            section_index=len(sections),
        ))

    # If all sections were excluded, fall back to full text
    if not sections:
        return [MarkdownSection(
            title="Full Text",
            content=markdown.strip(),
            heading_level=1,
            section_index=0,
        )]

    return sections
