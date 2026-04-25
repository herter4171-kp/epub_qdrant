#!/usr/bin/env python3
"""MinerU PDF ingestion pipeline — proof-of-concept.

Chains: MinerU PDF→Markdown → section splitting → citation-aware chunking
→ embedding payload preparation.

Usage:
    python scripts/mineru_pipeline.py --pdf <path> [--output <json_path>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root on sys.path for imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ingestion.mineru_converter import convert_pdf_to_markdown
from src.ingestion.md_section_splitter import split_markdown_sections
from src.ingestion.citation_masker import citation_aware_split
from src.ingestion.semantic_chunker import ChunkConfig, chunk_section, load_tokenizer
from src.config import settings

logger = logging.getLogger(__name__)

REQUIRED_PAYLOAD_FIELDS = (
    "text", "section_title", "heading_level",
    "chunk_index", "chunk_count", "token_count", "source_file",
)


def run_pipeline(
    pdf_path: str,
    output_path: Optional[str] = None,
) -> List[dict]:
    """Run full MinerU ingestion pipeline.

    1. MinerU_Converter → Markdown
    2. Markdown_Section_Splitter → sections
    3. For each section: citation masking → sentence splitting → Semantic_Chunker
    4. Assemble Embedding_Payload dicts

    Returns list of payload dicts.
    """
    # ── Step 1: Convert PDF → Markdown ────────────────────────────
    markdown = convert_pdf_to_markdown(pdf_path)
    source_file = Path(pdf_path).name

    # ── Step 2: Split into sections ───────────────────────────────
    sections = split_markdown_sections(markdown)
    logger.info("Sections found: %d", len(sections))

    # ── Step 3: Chunk each section ────────────────────────────────
    token_counter = load_tokenizer(settings.TOKENIZER_JSON or None)
    config = ChunkConfig(
        chunk_size=settings.CHUNK_SIZE,
        overlap_ratio=settings.CHUNK_OVERLAP_RATIO,
        similarity_percentile=settings.SIMILARITY_PERCENTILE,
        min_distance_floor=settings.MIN_DISTANCE_FLOOR,
        min_sentences_for_semantic=settings.MIN_SENTENCES_FOR_SEMANTIC,
        min_chunk_tokens=settings.MIN_CHUNK_TOKENS,
        enable_semantic=False,  # No embedding server for PoC
        tokenizer_path=settings.TOKENIZER_JSON or None,
    )

    payloads: List[dict] = []

    for ms in sections:
        results = chunk_section(
            title=ms.title,
            content=ms.content,
            config=config,
            token_counter=token_counter,
            embedding_fn=None,
            sentence_splitter=citation_aware_split,
        )
        chunk_count = len(results)
        for cr in results:
            payloads.append({
                "text": cr.text,
                "section_title": cr.section_title or ms.title,
                "heading_level": ms.heading_level,
                "chunk_index": cr.chunk_index,
                "chunk_count": chunk_count,
                "token_count": cr.token_count,
                "source_file": source_file,
            })

    # ── Summary ───────────────────────────────────────────────────
    total_tokens = sum(p["token_count"] for p in payloads)
    avg_tokens = total_tokens / len(payloads) if payloads else 0

    print(f"Sections found:    {len(sections)}")
    print(f"Chunks produced:   {len(payloads)}")
    print(f"Avg token count:   {avg_tokens:.1f}")
    print(f"Total token count: {total_tokens}")

    # ── Optional JSON output ──────────────────────────────────────
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payloads, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(payloads)} payloads to {output_path}")

    return payloads


def main():
    parser = argparse.ArgumentParser(
        description="MinerU PDF ingestion pipeline (proof-of-concept)",
    )
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--output", default=None, help="Path to write JSON output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_pipeline(args.pdf, args.output)


if __name__ == "__main__":
    main()
