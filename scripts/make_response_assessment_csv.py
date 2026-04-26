#!/usr/bin/env python3
"""Aggregate response assessment JSON files into a single CSV.

One row per assessment case.  Columns encode the judge's per-source scores
and the judge's own timing, which are the primary signals of interest.

Usage:
    python3 scripts/make_response_assessment_csv.py
    # outputs: bedrock_compare/response_assessment/results.csv
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSESS_DIR = ROOT / "bedrock_compare" / "response_assessment"

def parse_filename(name: str):
    """Extract (input_id, category, proficiency, topk) from a filename.

    Filename convention:  {id}_{category}_{prof}_{topk}.json
    Examples:
        21_lexical_vs_semantic_1_8.json  ->  id=21, prof=1, topk=8
        1_spatial_orientation_1_8.json   ->  id=1,  prof=1, topk=8
    """
    stem = name.replace(".json", "")
    parts = stem.rsplit("_", 2)
    # last part = topk, second-to-last = proficiency
    topk = int(parts[-1])
    proficiency = int(parts[-2])
    # Everything before the last two underscores, plus the first underscore-separated id
    prefix = parts[0]
    # prefix may itself contain underscores; the very first segment is the id
    tokens = prefix.split("_")
    input_id = int(tokens[0])
    category = "_".join(tokens[1:])
    return input_id, category, proficiency, topk


def row_for(record: dict, source: str) -> dict:
    """Build one CSV row for a single source within a record."""
    scores = record["scores"].get(source, {})
    response = record["responses"].get(source, {})
    return {
        "input_id":   record.get("input_id", ""),
        "category":   record.get("category", ""),
        "proficiency": record.get("proficiency", ""),
        "topk":       record.get("topk", ""),
        "prompt":     record.get("prompt", ""),
        "source":     source,
        "judge_model":    record.get("judge_model", ""),
        "judge_elapsed_seconds": record.get("judge_elapsed_seconds", ""),
        "response_elapsed_seconds": response.get("elapsed_seconds", ""),
        "response_error": response.get("error", ""),
        "retrieval_score":   scores.get("retrieval_score", ""),
        "retrieval_basis":   scores.get("retrieval_basis", ""),
        "response_score":    scores.get("response_score", ""),
        "response_basis":    scores.get("response_basis", ""),
    }


def main():
    if not ASSESS_DIR.is_dir():
        print(f"Error: {ASSESS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    files = sorted(f for f in ASSESS_DIR.iterdir() if f.name.endswith(".json"))
    if not files:
        print(f"No .json files in {ASSESS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Discover which sources exist across all files so we can build consistent headers
    all_sources = set()
    records = []
    for f in files:
        with open(f) as fh:
            rec = json.load(fh)
        # inject fields parsed from the filename (the JSON itself doesn't store all of them)
        inp_id, _, prof, topk = parse_filename(f.name)
        rec["proficiency"] = prof
        rec["topk"] = topk
        rec["input_id"] = inp_id  # override whatever was in the JSON
        records.append(rec)
        all_sources.update(rec.get("scores", {}).keys())

    sources = sorted(all_sources)

    fieldnames = [
        "input_id", "category", "proficiency", "topk",
        "prompt", "source",
        "judge_model", "judge_elapsed_seconds",
        "response_elapsed_seconds", "response_error",
        "retrieval_score", "retrieval_basis",
        "response_score", "response_basis",
    ]

    output = ASSESS_DIR / "results.csv"
    with open(output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            for src in sources:
                writer.writerow(row_for(rec, src))

    print(f"Wrote {len(records)} records × {len(sources)} sources = {len(records)*len(sources)} rows → {output}")


if __name__ == "__main__":
    main()