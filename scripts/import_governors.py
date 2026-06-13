# scripts/import_governors.py
"""
Title: Governor Party-Match Import CLI
Description: Loads data/governors.csv into data/pda.db (governors table) and
    materializes the governor name/party and alignment flags on reports.
    Offline, deterministic, idempotent. The two flags read president_party /
    state_winner_party, so import_presidents.py must have run first; this script
    warns if those columns are not yet populated. The CSV is produced by
    scripts/scrape_governors.py and committed.
Changelog:
    2026-06-13  Initial version.
"""

import argparse
import pathlib
import sys
import sqlite3

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pda.governors_import import (
    create_governors_table, iter_governor_rows, load_governors,
    add_governor_columns, materialize_governor_context,
)

DEFAULT_DB = "data/pda.db"
DEFAULT_CSV = "data/governors.csv"


def resolve_csv(csv_path):
    """Resolve the CSV path within the project tree and confirm it exists.

    Args:
        csv_path: local path to data/governors.csv.
    Returns:
        The resolved pathlib.Path.
    Raises:
        ValueError: if the path escapes the project tree.
        FileNotFoundError: if the CSV is missing.
    """
    base = pathlib.Path.cwd().resolve()
    path = (base / csv_path).resolve()
    if base != path and base not in path.parents:
        raise ValueError(f"refusing to use a CSV path outside the project: {csv_path}")
    if not path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Generate it with:\n"
            f"  .venv/bin/python scripts/scrape_governors.py")
    return path


def presidents_ready(conn):
    """Return True if the presidential columns exist and are populated.

    Args:
        conn: open sqlite3 connection.
    Returns:
        bool — True when president_party exists and at least one row is set.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(reports)")}
    if "president_party" not in columns or "state_winner_party" not in columns:
        return False
    populated = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE president_party IS NOT NULL"
    ).fetchone()[0]
    return populated > 0


def main():
    """Parse arguments, run the idempotent import, and print a summary."""
    parser = argparse.ArgumentParser(
        description="Import governors and materialize alignment flags")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()

    csv_path = resolve_csv(args.csv)

    conn = sqlite3.connect(args.db)
    try:
        if not presidents_ready(conn):
            print("WARNING: president_party/state_winner_party not populated — "
                  "run scripts/import_presidents.py first, or the alignment "
                  "flags will be NULL.")
        create_governors_table(conn)
        load_governors(conn, list(iter_governor_rows(str(csv_path))))
        add_governor_columns(conn)
        counts = materialize_governor_context(conn)

        governors = conn.execute("SELECT COUNT(*) FROM governors").fetchone()[0]
    finally:
        conn.close()

    print(f"Governors loaded        : {governors}")
    print(f"Reports labeled         : {counts['reports']}")
    print(f"  with governor / none  : {counts['with_governor']} / {counts['no_governor']}")
    print(f"  gov vs president 1/0/na : {counts['vp_1']} / {counts['vp_0']} / {counts['vp_none']}")
    print(f"  gov vs state vote 1/0/na: {counts['vs_1']} / {counts['vs_0']} / {counts['vs_none']}")


if __name__ == "__main__":
    main()
