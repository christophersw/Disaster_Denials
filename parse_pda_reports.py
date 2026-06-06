"""
Title: PDA Report Parser (CLI)
Description: Walk the downloaded PDA report PDFs, extract each with Claude into
    a validated PdaReport, flatten into normalized reports + report_counties
    rows (adding provenance from the manifest and run metadata), and write each
    report with its counties to data/pda.db (SQLite) in a single transaction.
    Idempotent and resumable: PDFs already present in the reports table are
    skipped, and re-extracting a PDF replaces its rows. Failures are logged and
    the run continues.
Changelog:
    2026-06-05  Initial version.
    2026-06-05  Persist to SQLite (data/pda.db) instead of paired CSVs.

Usage:
    .venv/bin/python parse_pda_reports.py            # all PDFs, resume
    .venv/bin/python parse_pda_reports.py --limit 5  # first 5 not-yet-done
    .venv/bin/python parse_pda_reports.py --glob 'data/pdfs/Denials/**/*.pdf'
    .venv/bin/python parse_pda_reports.py --db data/pda.db
"""

import argparse
import datetime
import glob as globlib
import sys

import anthropic
from dotenv import load_dotenv

from pda.db import connect, done_source_pdfs, write_report
from pda.extract import MODEL, extract_report
from pda.flatten import flatten
from pda.provenance import load_manifest, provenance_for

DB_PATH = "data/pda.db"
MANIFEST_CSV = "data/manifest.csv"


def main(argv: list[str] | None = None) -> int:
    """Run the parser. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Extract FEMA PDA reports to CSV.")
    parser.add_argument("--glob", default="data/pdfs/**/*.pdf",
                        help="Glob of PDFs to process (default: all).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N not-yet-done PDFs.")
    parser.add_argument("--db", default=DB_PATH,
                        help="SQLite store path (default: data/pda.db).")
    args = parser.parse_args(argv)

    load_dotenv()
    client = anthropic.Anthropic()
    manifest = load_manifest(MANIFEST_CSV)
    conn = connect(args.db)
    done = done_source_pdfs(conn)

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
            # The report and its counties are written in one transaction, so a
            # crash or failure mid-write leaves nothing orphaned and a rerun
            # cleanly re-extracts (write_report replaces by source_pdf).
            write_report(conn, report_row, county_rows)
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
