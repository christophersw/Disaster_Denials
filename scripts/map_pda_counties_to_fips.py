# scripts/map_pda_counties_to_fips.py
"""
Title: PDA County → FIPS Mapping CLI
Description: Adds county_fips / fips_match_method to report_counties, builds a
    (state, normalized-name) → FIPS crosswalk from the imported
    county_presidential_returns table, tags every county row, and writes the
    unmatched counties to data/pda_county_fips_unmatched.csv. Idempotent.
    Requires that scripts/migrate_mit_voter_data.py has already run.
Changelog:
    2026-06-07  Initial version.
"""

import argparse
import csv
import pathlib
import sys
import sqlite3

# Ensure the repo root is on sys.path so `pda` is importable when the script
# is invoked as `.venv/bin/python scripts/map_pda_counties_to_fips.py` from the
# repo root (Python sets sys.path[0] to the script's directory, not the CWD).
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pda.fips import add_fips_columns, build_state_index, map_report_counties

DEFAULT_DB = "data/pda.db"
DEFAULT_REPORT = "data/pda_county_fips_unmatched.csv"
DEFAULT_FUZZY_REPORT = "data/pda_county_fips_fuzzy_matches.csv"

# Leading characters a spreadsheet may interpret as the start of a formula.
# County names come from PDF extraction, so any value is possible.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value):
    """Neutralize spreadsheet formula injection in a CSV cell value.

    If a string value begins with a character a spreadsheet treats as a formula
    (= + - @ tab CR), prefix it with a single quote so it is rendered as text.
    Non-string values are returned unchanged.

    Args:
        value: a cell value of any type.
    Returns:
        The value, with a leading quote added when it would otherwise be parsed
        as a formula.
    """
    if isinstance(value, str) and value.startswith(_FORMULA_TRIGGERS):
        return "'" + value
    return value


def main() -> None:
    """Build the state index, map counties, and write the review reports."""
    parser = argparse.ArgumentParser(description="Map PDA counties to FIPS codes")
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        mit_rows = conn.execute(
            "SELECT DISTINCT state_po, county_name, county_fips "
            "FROM county_presidential_returns").fetchall()
        state_index = build_state_index(mit_rows)

        add_fips_columns(conn)
        counts = map_report_counties(conn, state_index)

        unmatched = conn.execute(
            "SELECT r.state_abbr, rc.county_name, rc.geo_type, COUNT(*) AS n "
            "FROM report_counties rc LEFT JOIN reports r "
            "ON rc.source_pdf = r.source_pdf "
            "WHERE rc.fips_match_method = 'unmatched' "
            "GROUP BY r.state_abbr, rc.county_name, rc.geo_type "
            "ORDER BY n DESC").fetchall()
        # Fuzzy matches were applied directly; list them (highest-confidence
        # first is least likely to be wrong) so they can be reviewed/reversed.
        fuzzy = conn.execute(
            "SELECT r.state_abbr, rc.county_name, rc.county_fips, "
            "ROUND(rc.fuzzy_score, 3) AS score, COUNT(*) AS n "
            "FROM report_counties rc LEFT JOIN reports r "
            "ON rc.source_pdf = r.source_pdf "
            "WHERE rc.fips_match_method = 'fuzzy_match' "
            "GROUP BY r.state_abbr, rc.county_name, rc.county_fips, rc.fuzzy_score "
            "ORDER BY score ASC, n DESC").fetchall()
    finally:
        conn.close()

    with open(DEFAULT_REPORT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state_abbr", "county_name", "geo_type", "occurrences"])
        writer.writerows(
            [_csv_safe(field) for field in row] for row in unmatched)

    with open(DEFAULT_FUZZY_REPORT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state_abbr", "county_name", "assigned_fips", "score", "occurrences"])
        writer.writerows(
            [_csv_safe(field) for field in row] for row in fuzzy)

    matchable = (counts["exact_normalized"] + counts["fuzzy_match"]
                 + counts["unmatched"])
    rate = (counts["exact_normalized"] + counts["fuzzy_match"]) / matchable if matchable else 0.0
    print(f"Matched (exact)        : {counts['exact_normalized']:,}")
    print(f"Matched (fuzzy)        : {counts['fuzzy_match']:,}")
    print(f"N/A (tribe/reservation): {counts['na_non_county']:,}")
    print(f"No source county (terr/AK): {counts['no_source_county']:,}")
    print(f"Unmatched              : {counts['unmatched']:,}")
    print(f"Match rate (matchable) : {rate:.1%}")
    print(f"Unmatched report       : {DEFAULT_REPORT} ({len(unmatched)} distinct)")
    print(f"Fuzzy-match report     : {DEFAULT_FUZZY_REPORT} ({len(fuzzy)} distinct)")


if __name__ == "__main__":
    main()
