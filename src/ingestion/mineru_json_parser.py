"""MinerU JSON parser: content_list_v2.json → JsonSection list.

Reads a MinerU ``content_list_v2.json`` file and produces a list of
``JsonSection`` objects ready for the existing ``chunk_section`` pipeline.

Block-type policy:
    - **Include:** ``title``, ``paragraph``, ``list``, ``table``,
      ``code``, ``algorithm``, ``page_footnote``
    - **Exclude:** ``page_header``, ``page_number``, ``page_aside_text``,
      ``image``, ``equation_interline``, ``chart``, ``page_footer``

Section exclusion policy:
    - ``references``, ``bibliography`` → excluded at any level
    - ``appendix``, ``appendices`` → excluded only when title is EXACTLY
      that word (case-insensitive) AND heading level ≤ 2

Usage:
    from src.ingestion.mineru_json_parser import (
        JsonSection,
        resolve_json_path,
        parse_content_list,
    )

    path = resolve_json_path("2603.07444")
    if path:
        sections = parse_content_list(path)

# Feature: mineru-json-ingestion
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Block-type inventory ──────────────────────────────────────────────────────

INCLUDED_BLOCK_TYPES: frozenset[str] = frozenset({
    "title", "paragraph", "list", "table", "code", "algorithm", "page_footnote",
})

EXCLUDED_BLOCK_TYPES: frozenset[str] = frozenset({
    "page_header", "page_number", "page_aside_text",
    "image", "equation_interline", "chart", "page_footer",
})

DEFAULT_MINERU_OUTPUT_DIR: str = "./mineru_output"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class JsonSection:
    """One logical section from a MinerU JSON file.

    Structurally compatible with ``MarkdownSection`` — same field names and
    types — so the existing ``chunk_section`` call requires no changes.
    """
    title: str           # assembled title text, e.g. "3. Methodology"
    content: str         # block texts joined by \\n\\n
    heading_level: int   # from title block's level field, >= 1
    section_index: int   # sequential from 0 after exclusion


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assemble_inline(spans: List[Dict]) -> str:
    """Concatenate text and equation_inline spans into a single plain string.

    Joins adjacent spans with a single space where neither span already has
    surrounding whitespace.  Strips leading/trailing whitespace from result.
    Skips spans with unrecognized types (debug log).

    Validates: Requirements 5.1–5.7, Property 3.
    """
    parts: List[str] = []
    for span in spans:
        span_type = span.get("type", "")
        if span_type in ("text", "equation_inline"):
            content = span.get("content", "")
            if content:
                if parts and not parts[-1].endswith((" ", "\n", "\t")) and not content.startswith((" ", "\n", "\t")):
                    parts.append(" ")
                parts.append(content)
        else:
            logger.debug("Unrecognized inline span type: %s", span_type)

    result = "".join(parts)
    return result.strip()


def _render_table(block: Dict) -> str:
    """Render a table block as: ``'{caption}\\n{plain_text_rows}'``.

    Iterates ``<tr>`` elements; joins each row's ``<td>``/``<th>`` cells
    with ``\\t``, then joins rows with ``\\n``.  Returns empty string if
    both caption and html are absent/empty.

    Validates: Requirements 7.1–7.6, Property 7.
    """
    caption_spans = block.get("content", {}).get("table_caption", [])
    caption = _assemble_inline(caption_spans) if caption_spans else ""

    html = block.get("content", {}).get("html", "") or ""
    plain_rows = ""
    if html.strip():
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        for tr in soup.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append("\t".join(cells))
        plain_rows = "\n".join(rows)

    if caption and plain_rows:
        return f"{caption}\n{plain_rows}"
    return caption or plain_rows


def _is_excluded_section(title: str, level: int) -> bool:
    """Return True if this section should be excluded from output.

    Exclusion rules:
      - Title matches 'references' or 'bibliography' (case-insensitive,
        with or without leading section numbers like '5. References')
      - Title is EXACTLY 'appendix' or 'appendices' (case-insensitive,
        no other words) AND level <= 2

    Subsections like 'Appendix B: Experimental Details' are NOT excluded.

    Validates: Requirements 4.4, 4.4a, 4.4b, Property 4.
    """
    # Strip leading section numbers (e.g. "5. References" → "references")
    normalized = title.strip()
    import re
    normalized = re.sub(r'^\d+[\.\)]\s*', '', normalized).strip().lower()

    if normalized in ("references", "bibliography"):
        return True

    # Exact match only — "Appendix" at level 1-2 excluded, "Appendix B: ..." not
    if normalized in ("appendix", "appendices") and level <= 2:
        return True

    return False


# ── Path resolution ───────────────────────────────────────────────────────────

def resolve_json_path(arxiv_id: str) -> Optional[Path]:
    """Locate ``content_list_v2.json`` for a given arxiv ID.

    Normalizes arxiv_id to underscore format, then tries:
      1. Remote tree layout: ``{MINERU_OUTPUT_DIR}/{id}/vlm/{id}_content_list_v2.json``
      2. Flat layout:        ``{MINERU_OUTPUT_DIR}/{id}_content_list_v2.json``

    Returns ``None`` (not raises) if neither path exists.
    Reads ``MINERU_OUTPUT_DIR`` from env, defaulting to ``DEFAULT_MINERU_OUTPUT_DIR``.

    Validates: Requirements 1.1–1.7.
    """
    # Normalize: "2603.07444" → "2603_07444"
    normalized = arxiv_id.replace(".", "_")

    base_dir = os.getenv("MINERU_OUTPUT_DIR", DEFAULT_MINERU_OUTPUT_DIR)
    base = Path(base_dir)

    # Tree layout
    tree_path = base / normalized / "vlm" / f"{normalized}_content_list_v2.json"
    if tree_path.exists():
        return tree_path

    # Flat layout
    flat_path = base / f"{normalized}_content_list_v2.json"
    if flat_path.exists():
        return flat_path

    return None


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_content_list(path: Union[str, Path]) -> List[JsonSection]:
    """Parse a ``content_list_v2.json`` file into ``JsonSection`` objects.

    Raises:
        FileNotFoundError: if path does not exist
        ValueError: if top-level JSON is not a list, or JSON is malformed

    Skips malformed pages/blocks with a logged warning rather than aborting.

    Validates: Requirements 2.1–2.5, 3.1–3.4, 4.1–4.6, 5.1–5.7, 6.1–6.4,
               7.1–7.6, 8.1–8.4, 12.1–12.3, Properties 1, 2, 4, 5, 6.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, list):
        actual_type = type(data).__name__
        raise ValueError(
            f"Expected top-level list in {path}, got {actual_type}"
        )

    # ── Pass 1: Flatten pages → blocks, filter, reconstruct sections ──
    sections: List[JsonSection] = []
    current_section: Optional[JsonSection] = None
    section_content_parts: List[str] = []
    section_index_counter = 0
    preamble_parts: List[str] = []

    for page_idx, page in enumerate(data):
        if not isinstance(page, list):
            logger.warning("Skipping page %d: expected list, got %s", page_idx, type(page).__name__)
            continue

        for block_pos, block in enumerate(page):
            if not isinstance(block, dict):
                logger.warning(
                    "Skipping block at page %d, pos %d: expected dict, got %s",
                    page_idx, block_pos, type(block).__name__,
                )
                continue

            block_type = block.get("type")
            if block_type is None:
                logger.warning(
                    "Skipping block at page %d, pos %d: missing 'type' key",
                    page_idx, block_pos,
                )
                continue

            # ── Excluded block types ──
            if block_type in EXCLUDED_BLOCK_TYPES:
                continue

            # ── Unknown block types ──
            if block_type not in INCLUDED_BLOCK_TYPES:
                logger.debug("Skipping unknown block type '%s' at page %d, pos %d",
                             block_type, page_idx, block_pos)
                continue

            # ── Title block → start new section ──
            if block_type == "title":
                # Finalize any current section
                if current_section is not None:
                    content = "\n\n".join(section_content_parts) if section_content_parts else ""
                    if content.strip():
                        sections.append(JsonSection(
                            title=current_section.title,
                            content=content,
                            heading_level=current_section.heading_level,
                            section_index=current_section.section_index,
                        ))
                    section_index_counter += 1
                elif preamble_parts:
                    # Finalize preamble
                    preamble_content = "\n\n".join(preamble_parts)
                    if preamble_content.strip():
                        sections.append(JsonSection(
                            title="Preamble",
                            content=preamble_content,
                            heading_level=1,
                            section_index=section_index_counter,
                        ))
                        section_index_counter += 1
                    preamble_parts = []

                # Assemble title text
                title_spans = block.get("content", {}).get("title_content", [])
                title_text = _assemble_inline(title_spans) if title_spans else f"Untitled-{len(sections)}"

                # Get heading level
                level = block.get("content", {}).get("level")
                if not isinstance(level, int) or level < 1:
                    logger.warning(
                        "Title block at page %d, pos %d has missing/non-integer level (%s), defaulting to 1",
                        page_idx, block_pos, level,
                    )
                    level = 1

                current_section = JsonSection(
                    title=title_text,
                    content="",
                    heading_level=level,
                    section_index=0,  # placeholder
                )
                section_content_parts = []
                continue

            # ── Paragraph block ──
            if block_type == "paragraph":
                content_data = block.get("content", {})
                spans = content_data.get("paragraph_content") if isinstance(content_data, dict) else None
                if not isinstance(spans, list):
                    logger.warning(
                        "Skipping paragraph block at page %d, pos %d: paragraph_content missing or not a list",
                        page_idx, block_pos,
                    )
                    continue
                text = _assemble_inline(spans)
                if text.strip():
                    if current_section is not None:
                        section_content_parts.append(text)
                    else:
                        preamble_parts.append(text)
                continue

            # ── List block ──
            if block_type == "list":
                content_data = block.get("content", {})
                items = content_data.get("list_items", []) if isinstance(content_data, dict) else []
                list_texts = []
                for item in items:
                    item_spans = item.get("item_content", []) if isinstance(item, dict) else []
                    item_text = _assemble_inline(item_spans) if item_spans else ""
                    if item_text.strip():
                        list_texts.append(item_text)
                if list_texts:
                    list_text = "\n".join(list_texts)
                    if current_section is not None:
                        section_content_parts.append(list_text)
                    else:
                        preamble_parts.append(list_text)
                continue

            # ── Table block ──
            if block_type == "table":
                rendered = _render_table(block)
                if rendered.strip():
                    if current_section is not None:
                        section_content_parts.append(rendered)
                    else:
                        preamble_parts.append(rendered)
                else:
                    logger.debug("Skipping empty table at page %d, pos %d", page_idx, block_pos)
                continue

            # ── Code block ──
            if block_type == "code":
                content_data = block.get("content", {})
                spans = content_data.get("code_content", []) if isinstance(content_data, dict) else []
                text = _assemble_inline(spans) if spans else ""
                if text.strip():
                    if current_section is not None:
                        section_content_parts.append(text)
                    else:
                        preamble_parts.append(text)
                continue

            # ── Algorithm block ──
            if block_type == "algorithm":
                content_data = block.get("content", {})
                spans = content_data.get("algorithm_content", []) if isinstance(content_data, dict) else []
                text = _assemble_inline(spans) if spans else ""
                if text.strip():
                    if current_section is not None:
                        section_content_parts.append(text)
                    else:
                        preamble_parts.append(text)
                continue

            # ── Page footnote block ──
            if block_type == "page_footnote":
                content_data = block.get("content", {})
                spans = content_data.get("page_footnote_content", []) if isinstance(content_data, dict) else []
                text = _assemble_inline(spans) if spans else ""
                if text.strip():
                    if current_section is not None:
                        section_content_parts.append(text)
                    else:
                        preamble_parts.append(text)
                continue

    # ── Finalize last section ──
    if current_section is not None:
        content = "\n\n".join(section_content_parts) if section_content_parts else ""
        if content.strip():
            sections.append(JsonSection(
                title=current_section.title,
                content=content,
                heading_level=current_section.heading_level,
                section_index=section_index_counter,
            ))
            section_index_counter += 1
    elif preamble_parts:
        preamble_content = "\n\n".join(preamble_parts)
        if preamble_content.strip():
            sections.append(JsonSection(
                title="Preamble",
                content=preamble_content,
                heading_level=1,
                section_index=section_index_counter,
            ))
            section_index_counter += 1

    # ── Apply section exclusion filters ──
    filtered: List[JsonSection] = []
    for section in sections:
        if _is_excluded_section(section.title, section.heading_level):
            continue
        filtered.append(section)

    # ── Re-index sequentially after exclusion ──
    for idx, section in enumerate(filtered):
        section.section_index = idx

    # ── Validate: no empty content, no heading_level < 1 ──
    for section in filtered:
        if not section.content.strip():
            logger.warning(
                "Section '%s' has empty content after filtering — excluding",
                section.title,
            )
            continue
        if section.heading_level < 1:
            logger.warning(
                "Section '%s' has heading_level=%d — fixing to 1",
                section.title, section.heading_level,
            )
            section.heading_level = 1

    # Final re-index after empty-content exclusion
    filtered = [s for s in filtered if s.content.strip()]
    for idx, section in enumerate(filtered):
        section.section_index = idx

    # ── Warn if fewer than 2 sections ──
    if len(filtered) < 2:
        logger.warning(
            "Document '%s' has only %d included section(s) — insufficient structure",
            path.name, len(filtered),
        )

    return filtered

