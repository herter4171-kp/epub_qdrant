"""Convert PDF to Markdown via MinerU HTTP API.

MinerU 3.x exposes a FastAPI service (``mineru-api``) with a synchronous
``POST /file_parse`` endpoint that accepts a PDF upload and returns Markdown.
This module calls that endpoint — no ``magic-pdf`` import needed.

Start the service:  ``mineru-api --host 0.0.0.0 --port 8010``
Configure URL:      ``export MINERU_API_URL=http://localhost:8010``

Validated by multiple academic teams:
- Xue et al. (arXiv:2601.15170): MinerU → structured Markdown → LLM analysis
- AutoPage (arXiv:2510.19600): MinerU + Docling → raw Markdown → LLM refinement
- PaperBanana (arXiv:2601.23265): MinerU toolkit for NeurIPS paper parsing

Public API:
    convert_pdf_to_markdown(pdf_path, timeout) -> str
    validate_heading_structure(markdown) -> dict[str, int]
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MINERU_API_URL = os.getenv("MINERU_API_URL", "http://localhost:8010")
MINERU_TIMEOUT_SECONDS = int(os.getenv("MINERU_TIMEOUT", "300"))


def convert_pdf_to_markdown(
    pdf_path: str,
    timeout: int = MINERU_TIMEOUT_SECONDS,
) -> str:
    """Convert a PDF to Markdown via MinerU HTTP API.

    Calls ``POST /file_parse`` on the MinerU service with ``return_md=true``.

    Args:
        pdf_path: Path to input PDF file.
        timeout: Maximum seconds for the HTTP request (default MINERU_TIMEOUT_SECONDS).

    Returns:
        Markdown string with heading hierarchies (#, ##, ###).

    Raises:
        FileNotFoundError: If pdf_path doesn't exist.
        ConnectionError: If MinerU service is unreachable.
        RuntimeError: If MinerU fails to process PDF.
        TimeoutError: If conversion exceeds timeout.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    url = f"{MINERU_API_URL.rstrip('/')}/file_parse"

    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                url,
                files={"files": (os.path.basename(pdf_path), f, "application/pdf")},
                data={"return_md": "true"},
                timeout=timeout,
            )
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot reach MinerU service at {MINERU_API_URL}. "
            f"Start it with: mineru-api --host 0.0.0.0 --port 8010\n"
            f"Error: {exc}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise TimeoutError(
            f"MinerU conversion timed out after {timeout}s for: {pdf_path}"
        ) from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"MinerU returned HTTP {resp.status_code} for {pdf_path}: "
            f"{resp.text[:500]}"
        )

    # Parse response — /file_parse returns JSON with markdown content
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(
            f"MinerU returned non-JSON response for {pdf_path}: "
            f"{resp.text[:500]}"
        )

    # Extract markdown from response structure
    md_content = _extract_markdown(data, pdf_path)

    # Validate heading structure
    heading_counts = validate_heading_structure(md_content)
    total_headings = sum(heading_counts.values())
    if total_headings < 2:
        logger.warning(
            "MinerU output for %s has only %d heading(s) — "
            "insufficient for structural splitting",
            pdf_path,
            total_headings,
        )

    return md_content


def _extract_markdown(data: dict | list, pdf_path: str) -> str:
    """Extract markdown string from MinerU /file_parse response.

    MinerU response format varies by version. Try common structures:
    - {"md_content": "..."} (direct)
    - {"results": [{"md_content": "..."}]} (batch)
    - {"markdown": "..."} (alternate key)
    - list of results: [{"md_content": "..."}]
    """
    # Direct dict with md_content
    if isinstance(data, dict):
        if "md_content" in data:
            return data["md_content"]
        if "markdown" in data:
            return data["markdown"]
        # Nested in results array
        results = data.get("results", [])
        if results and isinstance(results, list):
            first = results[0]
            if isinstance(first, dict):
                if "md_content" in first:
                    return first["md_content"]
                if "markdown" in first:
                    return first["markdown"]

    # Response is a list directly
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if "md_content" in first:
                return first["md_content"]
            if "markdown" in first:
                return first["markdown"]

    raise RuntimeError(
        f"Cannot extract markdown from MinerU response for {pdf_path}. "
        f"Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
    )


# Exclusive heading regexes — each matches exactly its level, not deeper.
_H1_RE = re.compile(r"^#(?!#)\s+", re.MULTILINE)    # # but not ##
_H2_RE = re.compile(r"^##(?!#)\s+", re.MULTILINE)   # ## but not ###
_H3_RE = re.compile(r"^###\s+", re.MULTILINE)        # ###


def validate_heading_structure(markdown: str) -> dict[str, int]:
    """Count Markdown headings at each level.

    Returns:
        Dict mapping heading level to count, e.g. {"h1": 2, "h2": 5, "h3": 3}.
        Counts are exclusive: ``## Foo`` counts as h2 only, not h1.
    """
    if not markdown:
        return {"h1": 0, "h2": 0, "h3": 0}

    return {
        "h1": len(_H1_RE.findall(markdown)),
        "h2": len(_H2_RE.findall(markdown)),
        "h3": len(_H3_RE.findall(markdown)),
    }
