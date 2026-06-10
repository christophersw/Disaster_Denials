# scripts/migrate_mit_voter_data.py
"""
Title: MIT Voter Data Migration CLI
Description: Loads dataverse_files/countypres_2000-2024.csv into data/pda.db as
    the faithful county_presidential_returns table and the county-year
    county_presidential_summary rollup. Idempotent: both tables are replaced on
    each run. Prints a verification summary including any county-years whose
    bucketed votes disagree with the published total.
Changelog:
    2026-06-07  Initial version.
"""

import argparse
import pathlib
import sys
import sqlite3

# Ensure the repo root is on sys.path so `pda` is importable when the script
# is invoked as `.venv/bin/python scripts/migrate_mit_voter_data.py` from the
# repo root (Python sets sys.path[0] to the script's directory, not the CWD).
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pda.voter_import import (
    create_voter_tables, load_faithful, build_summary, iter_csv_rows,
)
from pda.fips import is_county_fips

DEFAULT_DB = "data/pda.db"
DEFAULT_CSV = "dataverse_files/countypres_2000-2024.csv"


def main() -> None:
    """Parse arguments, run the migration, and print a verification summary."""
    parser = argparse.ArgumentParser(description="Migrate MIT voter data into pda.db")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        create_voter_tables(conn)
        load_faithful(conn, list(iter_csv_rows(args.csv)))
        discrepancies = build_summary(conn)

        faithful_count = conn.execute(
            "SELECT COUNT(*) FROM county_presidential_returns").fetchone()[0]
        summary_count = conn.execute(
            "SELECT COUNT(*) FROM county_presidential_summary").fetchone()[0]
        years = conn.execute(
            "SELECT MIN(year), MAX(year) FROM county_presidential_returns").fetchone()
        # Report (don't hide) the non-county reporting units excluded from the
        # rollup: distinct (year, county_fips) groups whose FIPS is not a real
        # county code (e.g. 'NA', '2938000').
        groups = conn.execute(
            "SELECT DISTINCT year, county_fips FROM county_presidential_returns").fetchall()
        dropped = sorted({fips for _, fips in groups if not is_county_fips(fips)})
    finally:
        conn.close()

    print(f"Faithful rows loaded : {faithful_count:,}")
    print(f"Summary county-years : {summary_count:,}")
    print(f"Non-county FIPS dropped: {len(dropped)}  {dropped if dropped else ''}")
    print(f"Year range           : {years[0]}–{years[1]}")
    print(f"Vote-sum discrepancies: {len(discrepancies)}")
    for county_fips, year, bucket_sum, total in discrepancies[:10]:
        print(f"  {county_fips} {year}: buckets={bucket_sum} total={total}")
    if len(discrepancies) > 10:
        print(f"  …and {len(discrepancies) - 10} more")


if __name__ == "__main__":
    main()
