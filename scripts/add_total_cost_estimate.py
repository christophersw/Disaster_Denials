# scripts/add_total_cost_estimate.py
"""
Title: Total Cost Estimate Column
Description: Adds the derived total_cost_estimate column to the reports table in
    data/pda.db. It is a VIRTUAL generated column equal to the sum of
    ia_cost_estimate and pa_cost_estimate, treating a missing side as 0 but
    staying NULL when both sides are missing. Because it is generated (not
    stored) SQLite recomputes it on read, so it never goes stale and needs no
    re-run after re-extraction. Offline, deterministic, idempotent: a second run
    detects the column and does nothing. report_counties is intentionally left
    untouched — it carries no IA/PA cost estimates to sum.
Changelog:
    2026-06-14  Initial version.
"""

import argparse
import sqlite3

DEFAULT_DB = "data/pda.db"

COLUMN_NAME = "total_cost_estimate"

# A missing estimate counts as 0, but the total stays NULL (genuinely unknown)
# when neither estimate is present rather than reporting a misleading $0.
COLUMN_DEFINITION = (
    f"{COLUMN_NAME} REAL GENERATED ALWAYS AS ("
    "CASE WHEN ia_cost_estimate IS NULL AND pa_cost_estimate IS NULL THEN NULL "
    "ELSE COALESCE(ia_cost_estimate, 0) + COALESCE(pa_cost_estimate, 0) END"
    ") VIRTUAL"
)


def total_cost_estimate_exists(conn):
    """Return True if reports already has the total_cost_estimate column.

    Uses PRAGMA table_xinfo, not table_info: a VIRTUAL generated column is
    absent from table_info, so a table_info guard would loop forever re-adding
    a column that is already there.

    Args:
        conn: open sqlite3 connection.
    Returns:
        bool — True when the column is present.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_xinfo(reports)")}
    return COLUMN_NAME in columns


def add_total_cost_estimate(conn):
    """Add the total_cost_estimate generated column to reports if absent.

    Args:
        conn: open sqlite3 connection.
    Returns:
        bool — True if the column was added, False if it already existed.
    """
    if total_cost_estimate_exists(conn):
        return False
    conn.execute(f"ALTER TABLE reports ADD COLUMN {COLUMN_DEFINITION}")
    conn.commit()
    return True


def main():
    """Parse arguments, add the column idempotently, and print a summary."""
    parser = argparse.ArgumentParser(
        description="Add the derived total_cost_estimate column to reports")
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        added = add_total_cost_estimate(conn)
        total_rows, with_total = conn.execute(
            "SELECT COUNT(*), COUNT(total_cost_estimate) FROM reports"
        ).fetchone()
    finally:
        conn.close()

    print("Column total_cost_estimate: " + ("added" if added else "already present"))
    print(f"Reports                   : {total_rows}")
    print(f"  with a total / NULL     : {with_total} / {total_rows - with_total}")


if __name__ == "__main__":
    main()
