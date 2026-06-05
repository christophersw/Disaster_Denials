"""
Title: PDA Report Parser (CLI)
Description: Walk the downloaded PDA report PDFs, extract each with Claude into
    a validated PdaReport, flatten into normalized reports + report_counties
    rows (adding provenance from the manifest and run metadata), and append to
    data/reports.csv and data/report_counties.csv. Idempotent and resumable:
    PDFs already present in reports.csv are skipped. Failures are logged and the
    run continues.
Changelog:
    2026-06-05  Initial version.

Usage:
    .venv/bin/python parse_pda_reports.py            # all PDFs, resume
    .venv/bin/python parse_pda_reports.py --limit 5  # first 5 not-yet-done
    .venv/bin/python parse_pda_reports.py --glob 'data/pdfs/Denials/**/*.pdf'
"""

import argparse
import datetime
import glob as globlib
import sys

import anthropic
from dotenv import load_dotenv

from pda.columns import COUNTY_COLUMNS, REPORT_COLUMNS
from pda.extract import MODEL, extract_report
from pda.flatten import flatten
from pda.io import append_rows, done_source_pdfs
from pda.provenance import load_manifest, provenance_for

REPORTS_CSV = "data/reports.csv"
COUNTIES_CSV = "data/report_counties.csv"
MANIFEST_CSV = "data/manifest.csv"


def main(argv: list[str] | None = None) -> int:
    """Run the parser. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Extract FEMA PDA reports to CSV.")
    parser.add_argument("--glob", default="data/pdfs/**/*.pdf",
                        help="Glob of PDFs to process (default: all).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N not-yet-done PDFs.")
    args = parser.parse_args(argv)

    load_dotenv()
    client = anthropic.Anthropic()
    manifest = load_manifest(MANIFEST_CSV)
    done = done_source_pdfs(REPORTS_CSV)

    pdfs = sorted(p for p in globlib.glob(args.glob, recursive=True)
                  if p not in done)
    if args.limit is not None:
        pdfs = pdfs[:args.limit]

    print(f"{len(pdfs)} PDF(s) to process ({len(done)} already done).")

    ok = 0
    failed = 0
    for index, pdf_path in enumerate(pdfs, 1):
        try:
            with open(pdf_path, "rb") as handle:
                report = extract_report(client, handle.read())
            meta = {
                "parser_model": MODEL,
                "extracted_at": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
            }
            report_row, county_rows = flatten(
                report, pdf_path, provenance_for(pdf_path, manifest), meta)
            # Write the county rows first and the report row last: reports.csv is
            # the resume marker (done_source_pdfs reads it), so writing it last
            # guarantees that any PDF marked done already has its county rows on
            # disk — no silently-missing counties. A crash strictly between these
            # two writes can leave county rows for a not-yet-done PDF, which the
            # rerun re-appends; detect such rare duplicates by grouping on
            # source_pdf if needed.
            append_rows(COUNTIES_CSV, COUNTY_COLUMNS, county_rows)
            append_rows(REPORTS_CSV, REPORT_COLUMNS, [report_row])
            flag = " [needs_review]" if report.needs_review else ""
            print(f"[{index}/{len(pdfs)}] OK {pdf_path}{flag}")
            ok += 1
        except (anthropic.AuthenticationError,
                anthropic.PermissionDeniedError) as error:
            # These fail identically for every PDF — abort instead of burning
            # through the whole corpus logging the same credential error.
            print(f"Aborting: credential error on {pdf_path}: {error}",
                  file=sys.stderr)
            return 2
        except Exception as error:  # noqa: BLE001 — keep going on any one failure
            print(f"[{index}/{len(pdfs)}] FAIL {pdf_path}: {error}",
                  file=sys.stderr)
            failed += 1

    print(f"Done. {ok} ok, {failed} failed.")
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
