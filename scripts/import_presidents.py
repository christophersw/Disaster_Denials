# scripts/import_presidents.py
"""
Title: Presidential Party-Match Import CLI
Description: Loads data/us_presidents_2025.csv into data/pda.db (presidents
    reference table), builds the state-year presidential rollup, adds the
    derived political columns, and materializes who-was-president and
    party-match for every report and county row. Idempotent: re-running
    refreshes all derived data. Downloads the CSV if it is missing. Prints a
    verification summary.
Changelog:
    2026-06-13  Initial version.
"""

import argparse
import pathlib
import sys
import sqlite3
import urllib.request

# Make `pda` importable when run as `.venv/bin/python scripts/import_presidents.py`.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pda.presidents_import import (
    create_presidents_table, iter_president_rows, load_presidents,
    create_state_summary_table, build_state_summary,
    add_political_columns, materialize_political_context,
)

DEFAULT_DB = "data/pda.db"
DEFAULT_CSV = "data/us_presidents_2025.csv"
DEFAULT_URL = ("https://raw.githubusercontent.com/jray-8/us-presidents-dataset/"
               "main/data/us_presidents_2025.csv")


def ensure_csv(csv_path, url):
    """Download the presidents CSV if it is not already present.

    Args:
        csv_path: local path to the CSV.
        url: raw URL to download from when the file is missing.
    """
    path = pathlib.Path(csv_path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, csv_path)


def main():
    """Parse arguments, run the idempotent import, and print a summary."""
    parser = argparse.ArgumentParser(
        description="Import US presidents and materialize party-match columns")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    ensure_csv(args.csv, args.url)

    conn = sqlite3.connect(args.db)
    try:
        create_presidents_table(conn)
        load_presidents(conn, list(iter_president_rows(args.csv)))
        create_state_summary_table(conn)
        build_state_summary(conn)
        add_political_columns(conn)
        counts = materialize_political_context(conn)

        presidents = conn.execute("SELECT COUNT(*) FROM presidents").fetchone()[0]
        states = conn.execute(
            "SELECT COUNT(*) FROM state_presidential_summary").fetchone()[0]
        distribution = conn.execute(
            "SELECT president_name, COUNT(*) FROM reports "
            "WHERE president_name IS NOT NULL "
            "GROUP BY president_name ORDER BY COUNT(*) DESC").fetchall()
    finally:
        conn.close()

    print(f"Presidents loaded      : {presidents}")
    print(f"State-year rollup rows : {states}")
    print(f"Reports labeled        : {counts['reports']}")
    print(f"  state match / differ / n/a : {counts['state_match_1']} / "
          f"{counts['state_match_0']} / {counts['state_match_none']}")
    print(f"County match / differ / n/a  : {counts['county_match_1']} / "
          f"{counts['county_match_0']} / {counts['county_match_none']}  "
          f"(no FIPS: {counts['county_no_fips']})")
    print("President distribution over reports:")
    for name, n in distribution:
        print(f"  {name:<20} {n}")


if __name__ == "__main__":
    main()
