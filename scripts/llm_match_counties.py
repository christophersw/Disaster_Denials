# scripts/llm_match_counties.py
"""
Title: LLM County → FIPS Matching CLI
Description: Resolves the residual unmatched PDA counties (those with a parsed
    name that exact + fuzzy matching could not place) using Claude Opus 4.8,
    constrained to a state-scoped candidate list. Applies confident, validated
    matches as fips_match_method='llm_match' with the model's confidence and
    rationale, and writes data/pda_county_fips_llm_matches.csv for review.
    Requires ANTHROPIC_API_KEY in the environment and that
    scripts/map_pda_counties_to_fips.py has already run.
Changelog:
    2026-06-09  Initial version.
"""

import argparse
import csv
import pathlib
import sys
import sqlite3

# Ensure the repo root is on sys.path so `pda` is importable when invoked as
# `.venv/bin/python scripts/llm_match_counties.py` from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import anthropic
from dotenv import load_dotenv

from pda.fips import add_fips_columns
from pda.llm_match import run, gather_unmatched, MODEL

DEFAULT_DB = "data/pda.db"
DEFAULT_REPORT = "data/pda_county_fips_llm_matches.csv"

# Leading characters a spreadsheet may interpret as the start of a formula.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value):
    """Neutralize spreadsheet formula injection in a CSV cell value."""
    if isinstance(value, str) and value.startswith(_FORMULA_TRIGGERS):
        return "'" + value
    return value


def main() -> None:
    """Run LLM matching over the residual unmatched counties and report."""
    parser = argparse.ArgumentParser(description="LLM-match PDA counties to FIPS")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dry-run", action="store_true",
                        help="List the rows that would be sent, without calling the API")
    args = parser.parse_args()
    load_dotenv()  # pick up ANTHROPIC_API_KEY from .env, matching the other CLIs

    conn = sqlite3.connect(args.db)
    try:
        add_fips_columns(conn)
        grouped = gather_unmatched(conn)
        pending = sum(len(items) for items in grouped.values())
        states = len(grouped)
        print(f"Residual unmatched named counties: {pending} across {states} states")

        if args.dry_run:
            for state in sorted(grouped):
                names = ", ".join(item["name"] for item in grouped[state])
                print(f"  {state}: {names}")
            print(f"Dry run — model {args.model} would be called once per state. "
                  "No API calls made.")
            return

        client = anthropic.Anthropic()
        summary = run(conn, client, model=args.model)
        details = summary["details"]
    finally:
        conn.close()

    # Highest-confidence last so the riskiest matches are easiest to review.
    details.sort(key=lambda r: (r.get("confidence") or 0.0))
    with open(DEFAULT_REPORT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "pda_name", "assigned_fips", "confidence", "reasoning"])
        for record in details:
            writer.writerow([_csv_safe(record.get("state")),
                             _csv_safe(record.get("name")),
                             record.get("fips"),
                             record.get("confidence"),
                             _csv_safe(record.get("reasoning"))])

    print(f"Applied (llm_match)    : {summary['applied']} rows "
          f"({len(details)} distinct names)")
    print(f"Review report          : {DEFAULT_REPORT}")


if __name__ == "__main__":
    main()
