"""Citation masking for academic sentence boundary detection.

Temporarily masks academic citation patterns and abbreviations before
sentence splitting, then restores them after. This prevents false sentence
breaks on patterns like ``[1, 2, 3]``, ``(Smith et al., 2023)``,
``Fig. 1.``, and ``Eq. 4.`` that shatter standard sentence tokenizers.

Informed by NUPunkt/CharBoundary (arXiv:2504.04131) demonstrating cascading
RAG failures from false sentence splits in legal text, and pySBD
(Sadvilkar & Neumann, arXiv:2010.09657) for rule-based SBD.

Public API:
    mask(text) -> (masked_text, restore_map)
    restore(sentences, restore_map) -> sentences
    citation_aware_split(text, use_pysbd=False) -> list[str]
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MAX_CITATION_MASK_CHARS = 500_000

# ---------------------------------------------------------------------------
# Pattern categories — ORDER MATTERS.
# Category 2 (parenthetical author-year) MUST run before category 3
# (abbreviations) because parenthetical citations contain "et al." which
# would otherwise be matched by the abbreviation pattern, causing
# double-masking artifacts.
# ---------------------------------------------------------------------------

# Category 1: Bracketed numeric citations  [1], [1, 2, 3], [1-5], [1, 2, 5-8]
_BRACKET_CITE_RE = re.compile(
    r"\[(?:\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\]"
)

# Category 2: Parenthetical author-year citations
# (Smith et al., 2023), (Jones & Lee, 2022), (Smith et al., 2023; Jones, 2022)
_PAREN_CITE_RE = re.compile(
    r"\("
    r"(?:[A-Z][A-Za-z''\-]+"           # first author surname
    r"(?:\s+(?:et\s+al\.|&\s*[A-Z][A-Za-z''\-]+))?"  # optional et al. or & coauthor
    r",?\s*\d{4}"                       # year
    r"(?:\s*;\s*"                        # semicolon separator for multiple citations
    r"[A-Z][A-Za-z''\-]+"
    r"(?:\s+(?:et\s+al\.|&\s*[A-Z][A-Za-z''\-]+))?"
    r",?\s*\d{4})*"
    r")"
    r"\)"
)

# Category 3: Academic abbreviations with trailing periods
_ABBREV_LIST = [
    "et al.", "i.e.", "e.g.", "cf.",
    "Figs.", "Fig.", "Eqs.", "Eq.", "Tab.", "Sec.", "Ref.",
    "Vol.", "No.", "vs.",
]
# Sort longest-first so "Figs." matches before "Fig." etc.
_ABBREV_LIST.sort(key=len, reverse=True)
_ABBREV_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(a) for a in _ABBREV_LIST) + r")",
    re.IGNORECASE,
)

# Category 4: Numbered figure/equation/table references with trailing period
# e.g. "Fig. 1.", "Eq. 4.", "Table 2."
# After category 3 masks "Fig." → placeholder, this catches "placeholder 1."
# But we apply it on the ORIGINAL text before cat-3, so we need a combined pattern.
_NUMBERED_REF_RE = re.compile(
    r"(?:Fig(?:s)?|Eq(?:s)?|Tab(?:le)?|Sec|Ref)\.\s*\d+\.",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core mask / restore functions
# ---------------------------------------------------------------------------

def mask(text: str) -> tuple[str, dict[str, str]]:
    """Replace citation patterns and abbreviations with unique placeholders.

    Args:
        text: Input text with academic citations.

    Returns:
        Tuple of (masked_text, restore_map) where restore_map maps
        placeholder tokens to original text.

    If ``len(text) > MAX_CITATION_MASK_CHARS``, returns ``(text, {})``
    with a logged warning — skips regex masking to prevent catastrophic
    backtracking on pathologically large inputs.
    """
    if not text:
        return text, {}

    if len(text) > MAX_CITATION_MASK_CHARS:
        logger.warning(
            "Text length %d exceeds MAX_CITATION_MASK_CHARS (%d), "
            "skipping citation masking",
            len(text),
            MAX_CITATION_MASK_CHARS,
        )
        return text, {}

    restore_map: dict[str, str] = {}
    counter = 0

    def _replace(match: re.Match) -> str:
        nonlocal counter
        placeholder = f"__CITE_{counter}__"
        restore_map[placeholder] = match.group(0)
        counter += 1
        return placeholder

    # Apply in strict order: 1 → 2 → 4 → 3
    # Category 4 (numbered refs like "Fig. 1.") must run before category 3
    # (bare abbreviations like "Fig.") so the full "Fig. 1." is captured as
    # one unit rather than having "Fig." stripped first.
    masked = _BRACKET_CITE_RE.sub(_replace, text)       # cat 1
    masked = _PAREN_CITE_RE.sub(_replace, masked)       # cat 2
    masked = _NUMBERED_REF_RE.sub(_replace, masked)     # cat 4 (before cat 3)
    masked = _ABBREV_RE.sub(_replace, masked)            # cat 3

    return masked, restore_map


def restore(sentences: list[str], restore_map: dict[str, str]) -> list[str]:
    """Replace all placeholder tokens in sentences with original text.

    Args:
        sentences: List of sentence strings containing placeholders.
        restore_map: Mapping from placeholder to original text.

    Returns:
        Sentences with all placeholders restored to original content.
    """
    if not restore_map:
        return sentences

    result = []
    for sent in sentences:
        for placeholder, original in restore_map.items():
            sent = sent.replace(placeholder, original)
        result.append(sent)
    return result


# ---------------------------------------------------------------------------
# Citation-aware sentence splitting
# ---------------------------------------------------------------------------

def _default_split_sentences(text: str) -> list[str]:
    """Fallback regex sentence splitter matching semantic_chunker._split_sentences."""
    if not text or not text.strip():
        return []
    # Import the existing splitter to stay DRY
    try:
        from src.ingestion.semantic_chunker import _split_sentences
        return _split_sentences(text)
    except ImportError:
        # Minimal fallback if import fails
        parts = re.split(
            r"(?<=[.!?])(?<!\b\w\.)\s+(?=[A-Z\"'\(])",
            text,
        )
        return [s.strip() for s in parts if s.strip()]


def citation_aware_split(
    text: str,
    use_pysbd: bool = False,
    sentence_splitter: Optional[Callable[[str], list[str]]] = None,
) -> list[str]:
    """Split text into sentences with citation masking.

    Flow:
        1. mask(text) → (masked_text, restore_map)
        2. Split masked_text into sentences (regex or pySBD)
        3. restore(sentences, restore_map) → restored sentences

    Args:
        text: Input text with academic citations.
        use_pysbd: If True, use pySBD instead of regex splitter.
        sentence_splitter: Optional custom sentence splitter function.
            If provided, overrides both regex and pySBD.

    Returns:
        List of sentence strings with original citations intact.
    """
    if not text or not text.strip():
        return []

    masked_text, restore_map = mask(text)

    # Choose sentence splitter
    if sentence_splitter is not None:
        splitter = sentence_splitter
    elif use_pysbd:
        try:
            import pysbd
            seg = pysbd.Segmenter(language="en", clean=False)
            splitter = seg.segment
        except ImportError:
            logger.warning(
                "pySBD not installed, falling back to regex sentence splitter. "
                "Install with: pip install pysbd"
            )
            splitter = _default_split_sentences
    else:
        splitter = _default_split_sentences

    sentences = splitter(masked_text)

    # Strip and filter empty
    sentences = [s.strip() for s in sentences if s.strip()]

    # Restore original citations
    sentences = restore(sentences, restore_map)

    return sentences
