#!/usr/bin/env python3
"""Regenerate report.md + plots from a persisted run directory.

Usage:
    python scripts/regen_report.py <run_dir>

Reads critiques/, retrievals/, config.json, failures.jsonl and rewrites
report.md and report_assets/*. No LLM calls — pure post-processing.
"""

import argparse
import os
import sys

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

try:
    from eval_suite.report import build_report, render_pdf
except ImportError:
    from .eval_suite.report import build_report, render_pdf  # type: ignore


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Regenerate report from run dir")
    p.add_argument("run_dir", help="Path to a TEST_RUN/<timestamp>/ directory")
    p.add_argument("--no-pdf", action="store_true", help="Skip PDF render")
    p.add_argument("--pdf-engine", default=None,
                   help="Force pandoc --pdf-engine (typst/weasyprint/wkhtmltopdf/pdflatex)")
    args = p.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
        return 2
    if not os.path.isdir(os.path.join(run_dir, "critiques")):
        print(f"ERROR: missing critiques/ in {run_dir}", file=sys.stderr)
        return 2

    print(f"Regenerating report for: {run_dir}")
    md_path = build_report(run_dir)
    print(f"Wrote: {md_path}")

    if not args.no_pdf:
        pdf_path = render_pdf(run_dir, engine=args.pdf_engine)
        if pdf_path:
            print(f"Wrote: {pdf_path}")
        else:
            print("PDF render skipped (see warnings above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
