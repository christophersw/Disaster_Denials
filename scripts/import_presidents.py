# scripts/import_presidents.py
"""
Title: Presidential Party-Match Import CLI
Description: Loads data/us_presidents_2025.csv into data/pda.db (presidents
    reference table), builds the state-year presidential rollup, adds the
    derived political columns, and materializes who-was-president and
    party-match for every report and county row. Idempotent: re-running
    refreshes all derived data. The CSV is the committed source of record
    (data/us_presidents_2025.csv); if it is missing, the script prints how to
    restore it rather than fetching from the network. Prints a verification
    summary.
Changelog:
    2026-06-13  Initial version.
"""

import argparse
import pathlib
import sys
import sqlite3

# Make `pda` importable when run as `.venv/bin/python scripts/import_presidents.py`.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pda.presidents_import import (
    create_presidents_table, iter_president_rows, load_presidents,
    create_state_summary_table, build_state_summary,
    add_political_columns, materialize_political_context,
)

DEFAULT_DB = "data/pda.db"
DEFAULT_CSV = "data/us_presidents_2025.csv"
# Source of the committed CSV, shown to the operator if the file is missing.
DATASET_URL = ("https://raw.githubusercontent.com/jray-8/us-presidents-dataset/"
               "main/data/us_presidents_2025.csv")


def resolve_csv(csv_path):
    """Resolve the CSV path within the project tree and confirm it exists.

    The committed data file is the source of record; this script does not fetch
    from the network. The path is constrained to the project tree so a stray
    ``--csv`` argument cannot read from outside the repo, and a missing file
    yields a clear restore instruction.

    Args:
        csv_path: local path to the CSV (resolved within the project tree).
    Returns:
        The resolved pathlib.Path to the CSV.
    Raises:
        ValueError: if the path escapes the project tree.
        FileNotFoundError: if the CSV is not present.
    """
    base = pathlib.Path.cwd().resolve()
    path = (base / csv_path).resolve()
    if base != path and base not in path.parents:
        raise ValueError(f"refusing to use a CSV path outside the project: {csv_path}")
    if not path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Restore the committed dataset with:\n"
            f"  curl -fsSL {DATASET_URL} -o {csv_path}")
    return path


def main():
    """Parse arguments, run the idempotent import, and print a summary."""
    parser = argparse.ArgumentParser(
        description="Import US presidents and materialize party-match columns")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()

    csv_path = resolve_csv(args.csv)

    conn = sqlite3.connect(args.db)
    try:
        create_presidents_table(conn)
        load_presidents(conn, list(iter_president_rows(str(csv_path))))
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
