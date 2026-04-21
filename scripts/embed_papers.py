#!/usr/bin/env python3
"""Download arxiv papers from ai-agent-papers markdown indexes.

Scans all markdown files under ai-agent-papers/, extracts arxiv paper links,
downloads PDFs flat per top-level category, and produces one metadata CSV
per category.

Usage:
    python scripts/embed_papers.py
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import arxiv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── arxiv client ───────────────────────────────────────────────────────
# arxiv.Client with num_retries=0 so _fetch_with_backoff handles ALL retries
# with proper exponential backoff (no internal rapid retries from arxiv library).
_ARXIV_CLIENT = arxiv.Client(delay_seconds=2.0, num_retries=0)

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_PAPERS_DIR = PROJECT_ROOT / "ai-agent-papers"
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"

# Top-level categories to process (order matters for reproducibility)
CATEGORIES = [
    "agent-frameworks",
    "application-papers",
    "capability-papers",
    "lectures",
    "newsletters",
]

# CJK range for filtering Chinese papers
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

# arxiv link patterns
ARXIV_ABS_RE = re.compile(r"https?://arxiv\.org/abs/(\d+\.\d+)(?:v\d+)?")
ARXIV_PDF_RE = re.compile(r"https?://arxiv\.org/pdf/(\d+\.\d+)(?:v\d+)?")


def find_arxiv_ids_in_file(filepath: Path) -> list[str]:
    """Extract unique arxiv IDs from a markdown file."""
    text = filepath.read_text(encoding="utf-8")
    ids = set()
    for match in ARXIV_ABS_RE.finditer(text):
        ids.add(match.group(1))
    for match in ARXIV_PDF_RE.finditer(text):
        ids.add(match.group(1))
    return list(ids)


def collect_papers() -> dict[str, dict[str, set[str]]]:
    """Walk source dirs and collect {arxiv_id: set_of_subcategories}.

    Returns:
        {
            "application-papers": {
                "1605.08386": {"deep-research-agents", "deep_research"},
                ...
            },
            ...
        }
    """
    result: dict[str, dict[str, set[str]]] = {cat: {} for cat in CATEGORIES}

    for category in CATEGORIES:
        cat_dir = AI_PAPERS_DIR / category

        if not cat_dir.is_dir():
            log.warning("Source dir missing: %s — skipping", cat_dir)
            continue

        for md_file in sorted(cat_dir.rglob("*.md")):
            subcat = md_file.stem
            ids = find_arxiv_ids_in_file(md_file)
            for aid in ids:
                if aid not in result[category]:
                    result[category][aid] = set()
                result[category][aid].add(subcat)

    return result


def _write_metadata_json(arxiv_id: str, meta: dict, category: str, subcategories: set[str], downloads_dir: Path) -> None:
    """Write per-PDF JSON metadata file.

    Creates downloads/{arxiv_id_with_underscores}.json with a single key
    'metadataAttributes' containing a list of 'key: value' strings.
    """
    import json

    safe_name = arxiv_id.replace(".", "_")
    json_path = downloads_dir / f"{safe_name}.json"

    attributes = []
    for key, value in meta.items():
        if key in ("title", "authors", "publish_date", "abstract", "url", "pdf_url"):
            attributes.append(f"{key}: {value}")

    attributes.append(f"arxiv_id: {arxiv_id}")
    attributes.append(f"category: {category}")
    attributes.append(f"subcategory: {', '.join(sorted(subcategories))}")

    payload = {"metadataAttributes": attributes}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log.info("  [json] Wrote %s", json_path.name)


def is_chinese(title: str, abstract: str) -> bool:
    """Return True if title or abstract contains significant CJK characters."""
    if CJK_RE.search(title) or CJK_RE.search(abstract):
        return True
    return False


def _fetch_with_backoff(arxiv_id: str) -> Optional[arxiv.Result]:
    """Fetch a single paper from arxiv with exponential backoff on failure.

    Delays: ~10s, ~20s, ~40s, ~80s, ~160s, ~320s, ~640s (7 retries max, ~870s total).
    When arXiv rate-limits, this waits long enough between attempts for the block to lift.
    Uses base_delay * 2^attempt with random jitter to avoid thundering herd.

    Returns the paper Result or None.
    """
    import random

    max_retries = 7
    base_delay = 10.0

    for attempt in range(max_retries):
        try:
            results = list(_ARXIV_CLIENT.results(arxiv.Search(id_list=[arxiv_id])))
            if results:
                log.info("  [OK] Got result for %s on attempt %d/%d", arxiv_id, attempt + 1, max_retries)
                return results[0]

            # No results returned — treat as transient failure, retry with backoff
            log.warning("  [empty] No results for %s (attempt %d/%d)", arxiv_id, attempt + 1, max_retries)

        except Exception as exc:
            log.warning("  [fail] %s on attempt %d/%d: %s", arxiv_id, attempt + 1, max_retries, exc)

        # Not the last attempt — back off before retrying
        if attempt < max_retries - 1:
            jitter = random.uniform(0.5, 1.5)
            wait = base_delay * (2 ** attempt) * jitter
            log.info("  [backoff] Waiting %.0fs before retry %d/%d for %s", wait, attempt + 2, max_retries, arxiv_id)
            time.sleep(wait)

    # All retries exhausted
    log.warning("  [done] All %d retries exhausted for %s", max_retries, arxiv_id)
    return None


def fetch_paper_metadata_and_download(arxiv_id: str, downloads_dir: Path, force_refresh: bool = False) -> Optional[tuple[dict, str]]:
    """Fetch metadata and download PDF for a paper.

    Checks for existing PDF before making any API calls. Skips existing papers
    unless force_refresh is True.

    Uses _fetch_with_backoff for exponential backoff.
    Returns (metadata_dict, filename) or None on failure/skip.
    """
    # Check if PDF already exists locally BEFORE making any API calls
    safe_name = arxiv_id.replace(".", "_")
    filename = f"{safe_name}.pdf"
    local_path = downloads_dir / filename

    if local_path.exists() and not force_refresh:
        log.info("  (skip) %s already exists", filename)
        return None
    elif local_path.exists() and force_refresh:
        log.info("  (force-refresh) %s exists but re-fetching metadata", filename)

    paper = _fetch_with_backoff(arxiv_id)
    if paper is None:
        return None

    # Build metadata dict
    meta = {
        "title": paper.title,
        "authors": ", ".join(a.name for a in paper.authors),
        "publish_date": str(paper.published.date()) if paper.published else "",
        "abstract": paper.summary or "",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    }

    # Download PDF (skip if already exists and not force-refresh)
    safe_name = arxiv_id.replace(".", "_")
    filename = f"{safe_name}.pdf"
    local_path = downloads_dir / filename

    if not local_path.exists():
        try:
            paper.download_pdf(filename=str(local_path))
            log.info("  ↓ downloaded %s", filename)
        except Exception as exc:
            log.warning("  ↓ failed to download %s: %s", arxiv_id, exc)
            return None
    else:
        log.info("  (skip download) %s already exists", filename)

    return meta, filename


def write_csv(category: str, rows: list[dict], downloads_dir: Path) -> None:
    """Write metadata CSV for a category."""
    csv_path = downloads_dir / category / "metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "title",
        "authors",
        "publish_date",
        "arxiv_id",
        "source_file",
        "category",
        "subcategory",
        "topics",
        "abstract",
        "url",
        "pdf_url",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %s (%d rows)", csv_path, len(rows))


def process_category(category: str, papers: dict[str, set[str]], downloads_dir: Path, force_refresh: bool = False) -> None:
    """Process all papers for one category."""
    cat_dir = downloads_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_ids: set[str] = set()

    # Pre-scan existing PDFs for O(1) skip lookups
    existing_pdfs: set[str] = set()
    for p in downloads_dir.glob("*.pdf"):
        stem = p.stem.replace("_", ".")
        existing_pdfs.add(stem)
    for p in cat_dir.glob("*.pdf"):
        stem = p.stem.replace("_", ".")
        existing_pdfs.add(stem)

    log.info("[%s] Found %d existing PDFs on disk", category, len(existing_pdfs))

    for arxiv_id, subcategories in sorted(papers.items()):
        if arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)

        # Skip if PDF already exists and not force-refresh
        if arxiv_id in existing_pdfs and not force_refresh:
            log.info("[%s] (skip) %s already exists", category, arxiv_id)
            continue

        log.info("[%s] Processing %s ...", category, arxiv_id)

        result = fetch_paper_metadata_and_download(arxiv_id, downloads_dir, force_refresh)
        if result is None:
            continue

        meta, filename = result
        rel_source = f"downloads/{category}/{filename}"

        # Skip Chinese papers
        if is_chinese(meta["title"], meta["abstract"]):
            log.info("  [skip Chinese] %s", meta["title"][:80])
            continue

        # Write per-PDF JSON metadata
        _write_metadata_json(arxiv_id, meta, category, subcategories, downloads_dir)

        topics = f"{category},{','.join(sorted(subcategories))}"

        rows.append({
            "title": meta["title"],
            "authors": meta["authors"],
            "publish_date": meta["publish_date"],
            "arxiv_id": arxiv_id,
            "source_file": rel_source,
            "category": category,
            "subcategory": "; ".join(sorted(subcategories)),
            "topics": topics,
            "abstract": meta["abstract"],
            "url": meta["url"],
            "pdf_url": meta["pdf_url"],
        })

        # Rate limit between papers
        time.sleep(1)

    if rows:
        write_csv(category, rows, downloads_dir)
    else:
        log.warning("[%s] No papers processed (all skipped or failed)", category)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download arxiv papers from ai-agent-papers indexes.")
    parser.add_argument("--category", nargs="+", default=None, help="Process only these categories")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-fetch metadata even if PDFs already exist locally")
    args = parser.parse_args()

    log.info("ai-agent-papers dir: %s", AI_PAPERS_DIR)
    log.info("Downloads dir: %s", DOWNLOADS_DIR)
    log.info("Force refresh: %s", args.force_refresh)

    if not AI_PAPERS_DIR.is_dir():
        log.error("Source dir not found: %s", AI_PAPERS_DIR)
        sys.exit(1)

    all_papers = collect_papers()
    total_ids = sum(len(p) for p in all_papers.values())
    log.info("Found %d unique paper-category entries", total_ids)

    categories_to_run = args.category if args.category else CATEGORIES
    for cat in categories_to_run:
        if cat not in all_papers:
            log.warning("No papers found for category: %s", cat)
            continue
        process_category(cat, all_papers[cat], DOWNLOADS_DIR, args.force_refresh)

    log.info("Done.")


if __name__ == "__main__":
    main()