"""
Title: PDA Batch Runner (CLI)
Description: Drive the Anthropic Message Batches API to extract the PDA corpus at
    half price, asynchronously. Three resumable subcommands:
      submit  — create batch(es) for not-yet-done, not-in-flight PDFs and record
                the custom_id↔source_pdf mapping in data/pda.db.
      collect — write results from any batch that has finished; safe to run
                repeatedly until everything is collected.
      status  — show how many reports are done, how many PDFs are in flight, and
                each open batch's processing state.
    The serial parse_pda_reports.py remains for spot-checks; this path is for the
    full corpus when you are not in a hurry.
Changelog:
    2026-06-05  Initial version.

Usage:
    .venv/bin/python batch_pda_reports.py submit              # submit all remaining
    .venv/bin/python batch_pda_reports.py submit --limit 50   # submit first 50 remaining
    .venv/bin/python batch_pda_reports.py submit --dry-run    # plan only, no API call
    .venv/bin/python batch_pda_reports.py collect             # write finished results
    .venv/bin/python batch_pda_reports.py status              # progress + batch states
"""

import argparse
import datetime
import glob as globlib
import sys

import anthropic
from dotenv import load_dotenv

from pda.batch import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_COUNT,
    build_batch_request,
    chunk_by_size,
    collect,
    custom_id_for,
    submit,
)
from pda.db import (
    connect,
    done_source_pdfs,
    open_batch_ids,
    pending_source_pdfs,
)
from pda.provenance import load_manifest, provenance_for

DB_PATH = "data/pda.db"
MANIFEST_CSV = "data/manifest.csv"
DEFAULT_GLOB = "data/pdfs/**/*.pdf"


def _candidate_pdfs(conn, glob_pattern: str, limit: int | None) -> list[str]:
    """Remaining PDFs (not yet in reports, not already in flight), capped by limit."""
    skip = done_source_pdfs(conn) | pending_source_pdfs(conn)
    pdfs = sorted(p for p in globlib.glob(glob_pattern, recursive=True)
                  if p not in skip)
    return pdfs[:limit] if limit is not None else pdfs


def _now() -> str:
    """UTC timestamp string for extracted_at."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cmd_submit(args) -> int:
    """Submit batches for remaining PDFs."""
    conn = connect(args.db)
    pdfs = _candidate_pdfs(conn, args.glob, args.limit)
    if not pdfs:
        print("Nothing to submit (all PDFs done or in flight).")
        return 0

    if args.dry_run:
        sized = []
        for path in pdfs:
            with open(path, "rb") as handle:
                request = build_batch_request(custom_id_for(path), handle.read())
            encoded = request["params"]["messages"][0]["content"][0]["source"]["data"]
            sized.append((path, len(encoded)))
        chunks = chunk_by_size(sized, args.max_bytes, DEFAULT_MAX_COUNT,
                               size_of=lambda item: item[1])
        total_mb = sum(size for _p, size in sized) / 1024 / 1024
        print(f"[dry-run] {len(pdfs)} PDF(s), ~{total_mb:.1f} MB encoded, "
              f"{len(chunks)} batch(es).")
        return 0

    load_dotenv()
    client = anthropic.Anthropic()
    batch_ids = submit(client, conn, pdfs, max_bytes=args.max_bytes)
    print(f"Submitted {len(pdfs)} PDF(s) across {len(batch_ids)} batch(es): "
          f"{', '.join(batch_ids)}")
    print("Run `collect` later to write results (batches usually finish within "
          "an hour; up to 24h).")
    return 0


def cmd_collect(args) -> int:
    """Write results from any finished batches."""
    load_dotenv()
    conn = connect(args.db)
    if not open_batch_ids(conn):
        print("No open batches. Nothing to collect.")
        return 0
    client = anthropic.Anthropic()
    manifest = load_manifest(MANIFEST_CSV)

    def on_result(source_pdf: str, status: str, detail: str = "") -> None:
        suffix = f": {detail}" if detail else ""
        print(f"  {status.upper()} {source_pdf}{suffix}",
              file=sys.stderr if status == "failed" else sys.stdout)

    ok, failed = collect(
        client, conn, lambda pdf: provenance_for(pdf, manifest),
        now=_now, on_result=on_result)
    still_open = open_batch_ids(conn)
    print(f"Collected: {ok} written, {failed} failed. "
          f"{len(still_open)} batch(es) still running.")
    return 1 if failed and not ok else 0


def cmd_status(args) -> int:
    """Show corpus progress and the state of any open batches."""
    load_dotenv()
    conn = connect(args.db)
    done = done_source_pdfs(conn)
    pending = pending_source_pdfs(conn)
    open_ids = open_batch_ids(conn)
    print(f"Reports written: {len(done)}")
    print(f"In flight (awaiting results): {len(pending)}")
    print(f"Open batches: {len(open_ids)}")
    if open_ids:
        client = anthropic.Anthropic()
        for batch_id in open_ids:
            try:
                batch = client.messages.batches.retrieve(batch_id)
                print(f"  {batch_id}: {batch.processing_status}")
            except Exception as error:  # noqa: BLE001 — status should never crash
                print(f"  {batch_id}: (could not retrieve: {error})",
                      file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch a subcommand."""
    parser = argparse.ArgumentParser(description="Extract FEMA PDA reports via the Batches API.")
    parser.add_argument("--db", default=DB_PATH,
                        help="SQLite store path (default: data/pda.db).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Submit batches for remaining PDFs.")
    p_submit.add_argument("--glob", default=DEFAULT_GLOB,
                          help="Glob of PDFs to consider (default: all).")
    p_submit.add_argument("--limit", type=int, default=None,
                          help="Submit at most N remaining PDFs.")
    p_submit.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                          help="Per-batch encoded-size budget (default ~180 MB).")
    p_submit.add_argument("--dry-run", action="store_true",
                          help="Plan batches without calling the API.")
    p_submit.set_defaults(func=cmd_submit)

    p_collect = sub.add_parser("collect", help="Write results from finished batches.")
    p_collect.set_defaults(func=cmd_collect)

    p_status = sub.add_parser("status", help="Show progress and batch states.")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
