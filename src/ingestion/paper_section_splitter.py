"""Split extracted PDF text into named sections using academic header patterns.

Excludes References, Bibliography, and Appendix sections from the output
so that arxiv IDs and author name lists do not pollute the embedding space.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class PaperSection:
    """A named section from an academic paper."""
    title: str
    content: str
    section_index: int


# Matches common academic section headers (named or numbered).
PAPER_HEADER_PATTERN = re.compile(
    r'^(?:'
    r'(?:Abstract|Introduction|Related\s+Work|Background|'
    r'Methodology|Methods?|Approach|'
    r'Experiments?|Results?|Evaluation|'
    r'Discussion|Analysis|'
    r'Conclusion|Summary|Future\s+Work|'
    r'References|Bibliography|Appendix)'
    r'|'
    r'(?:\d+\.?\s+\w.{2,60})'  # numbered: "3. Methodology"
    r')$',
    re.MULTILINE | re.IGNORECASE,
)

# Section titles to exclude from the output.
EXCLUDED_SECTIONS = {"references", "bibliography", "appendix"}


def _is_excluded(title: str) -> bool:
    """Return True if *title* matches an excluded section name."""
    # Strip leading numbers and punctuation: "7. References" → "references"
    stripped = re.sub(r'^\d+\.?\s*', '', title).strip().lower()
    return stripped in EXCLUDED_SECTIONS


def split_paper_sections(text: str) -> List[PaperSection]:
    """Split PDF paper text into named sections by header regex.

    Sections matching References, Bibliography, or Appendix are dropped.
    If fewer than 2 headers are detected the entire text is returned as a
    single "Full Text" section.

    Args:
        text: Raw extracted text from a PDF paper.

    Returns:
        List of PaperSection with sequential indices and non-empty content.
    """
    if not text or not text.strip():
        return [PaperSection(title="Full Text", content=text or "", section_index=0)]

    matches = list(PAPER_HEADER_PATTERN.finditer(text))

    if len(matches) < 2:
        return [PaperSection(title="Full Text", content=text.strip(), section_index=0)]

    sections: List[PaperSection] = []
    for i, match in enumerate(matches):
        title = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if _is_excluded(title):
            continue

        if content:
            sections.append(PaperSection(
                title=title,
                content=content,
                section_index=len(sections),
            ))

    if not sections:
        return [PaperSection(title="Full Text", content=text.strip(), section_index=0)]

    return sections
