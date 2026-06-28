# scripts/verify_political_coding.py
"""
Title: scripts/verify_political_coding.py — EDA gate for political flag semantics.
Description:
    Prints denial-rate crosstabs for the engineered partisan-alignment flags so
    the meaning of each flag's `=1` value is confirmed before any political
    coefficient sign is interpreted (spec §11). The raw governor_vs_president
    crosstab was counterintuitive in brainstorming; this gate exists to catch a
    reversed-coding mistake before it reaches the write-up.
Changelog:
    2026-06-28  Initial version.
"""

import argparse
import sqlite3

import pandas as pd

DEFAULT_DB = "data/pda.db"
FLAGS = ["state_party_match", "governor_vs_president", "county_party_match"]


def crosstab_denial_rate(df, flag_col):
    """Return n and denial_rate grouped by the values of flag_col.

    Args:
        df: frame with a binary 'denied' column and the flag column.
        flag_col: the flag to group by.
    Returns:
        DataFrame indexed by flag value with columns 'n' and 'denial_rate'.
    """
    g = df.dropna(subset=[flag_col]).groupby(flag_col)["denied"]
    return pd.DataFrame({"n": g.size(), "denial_rate": g.mean()})


def main():
    """Print the denial-rate crosstabs for each political flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        reports = pd.read_sql_query(
            "SELECT source_pdf, report_outcome, state_party_match, "
            "governor_vs_president FROM reports "
            "WHERE report_outcome IN ('Declared','Denied')",
            conn,
        )
        county = pd.read_sql_query(
            "SELECT r.report_outcome, c.county_party_match "
            "FROM report_counties c JOIN reports r USING (source_pdf) "
            "WHERE r.report_outcome IN ('Declared','Denied')",
            conn,
        )
    finally:
        conn.close()

    reports["denied"] = (reports["report_outcome"] == "Denied").astype(int)
    county["denied"] = (county["report_outcome"] == "Denied").astype(int)

    for flag in ["state_party_match", "governor_vs_president"]:
        print(f"\n=== {flag} ===")
        print(crosstab_denial_rate(reports, flag))
    print("\n=== county_party_match (county-row level) ===")
    print(crosstab_denial_rate(county, "county_party_match"))
    print(
        "\nREVIEW REQUIRED: confirm what =1 means for each flag "
        "(match vs opposite) before interpreting Model 1 signs."
    )


if __name__ == "__main__":
    main()
