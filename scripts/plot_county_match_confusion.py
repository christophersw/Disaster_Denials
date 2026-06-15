# scripts/plot_county_match_confusion.py
"""
Title: County-Match x Outcome Confusion Heatmaps
Description: Renders confusion-matrix-style heatmaps from data/pda.db relating
    whether a county's winning presidential party matched the sitting president's
    party to the PDA outcome (approved vs denied). Each panel is a 2x2 matrix:
    rows are "county matches president" (top) and "county differs" (bottom);
    columns are Approved (report_outcome 'Declared') and Denied ('Denied'). The
    unit of analysis is the report-county: a report covering many counties
    contributes one observation per county (report_counties joined to reports for
    the outcome). Appeals (report_outcome 'Denial of Appeal') and counties without
    presidential-vote data (county_party_match IS NULL) are excluded. Each cell is
    annotated with its raw count and its share of its row (each county-match row
    sums to 100%), and coloured by that row share on a shared 0-100% scale so
    panels are comparable. One figure holds five panels: all data on top, then
    Obama, Trump (2017-2021), Biden, and Trump (2025-) across the bottom (the two
    Trump terms are separate presidencies, distinguished by president_number and
    term years). Offline, deterministic: reads the DB and writes a static PNG.
Changelog:
    2026-06-14  Initial version.
"""

import argparse
import sqlite3

from confusion_heatmap import build_confusion_figure

DEFAULT_DB = "data/pda.db"
DEFAULT_OUTPUT = "docs/figures/county_match_confusion.png"

# Outcome groups key off report_outcome; 'Denial of Appeal' is excluded.
APPROVAL_OUTCOME = "Declared"
DENIAL_OUTCOME = "Denied"

# Presidents to break out, in chronological order. The two Trump terms are
# distinct president_numbers (45 = 2017-2021, 47 = 2025-).
PRESIDENT_NUMBERS = [44, 45, 46, 47]

# county_party_match flag values.
COUNTY_MATCHES = 1
COUNTY_DIFFERS = 0

COLORMAP = "Greens"


def _matrix_from_counts(counts):
    """Build the 2x2 count matrix for one panel from a (match, outcome) lookup.

    Args:
        counts (dict): maps (county_party_match, report_outcome) -> int count.
    Returns:
        list[list[int]] — rows [matches, differs], columns [approved, denied].
    """
    return [
        [counts.get((COUNTY_MATCHES, APPROVAL_OUTCOME), 0),
         counts.get((COUNTY_MATCHES, DENIAL_OUTCOME), 0)],
        [counts.get((COUNTY_DIFFERS, APPROVAL_OUTCOME), 0),
         counts.get((COUNTY_DIFFERS, DENIAL_OUTCOME), 0)],
    ]


def fetch_confusion_data(db_path):
    """Aggregate the county-match x outcome counts for every panel.

    Joins report_counties to reports for the outcome, restricts to Declared/Denied
    with a non-NULL county_party_match, then builds one all-data panel plus one
    panel per president in PRESIDENT_NUMBERS. The unit is the report-county row.
    Per-president labels (surname + term-year range) are read from the presidents
    table so the two Trump terms are disambiguated by years.

    Args:
        db_path (str): path to the pda.db SQLite database.
    Returns:
        list[dict] — ordered panels (all data first), each with:
            "label"  (str)              — panel title
            "matrix" (list[list[int]])  — 2x2 counts [match/differ][appr/deny]
            "n"      (int)              — total report-counties in the panel
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT rc.president_number, rc.county_party_match, r.report_outcome,
                   COUNT(*) AS n
            FROM report_counties rc
            JOIN reports r ON rc.source_pdf = r.source_pdf
            WHERE r.report_outcome IN (?, ?)
              AND rc.county_party_match IS NOT NULL
            GROUP BY rc.president_number, rc.county_party_match, r.report_outcome
            """,
            (APPROVAL_OUTCOME, DENIAL_OUTCOME),
        ).fetchall()
        president_rows = conn.execute(
            "SELECT number, name, term_start, term_end FROM presidents "
            "WHERE number IN (%s)" % ",".join("?" * len(PRESIDENT_NUMBERS)),
            PRESIDENT_NUMBERS,
        ).fetchall()
    finally:
        conn.close()

    # Tally counts per president and an all-data total, keyed by (match, outcome).
    all_counts = {}
    per_president_counts = {number: {} for number in PRESIDENT_NUMBERS}
    for president_number, match, outcome, count in rows:
        key = (match, outcome)
        all_counts[key] = all_counts.get(key, 0) + count
        if president_number in per_president_counts:
            bucket = per_president_counts[president_number]
            bucket[key] = bucket.get(key, 0) + count

    def make_panel(label, counts):
        matrix = _matrix_from_counts(counts)
        total = sum(value for row in matrix for value in row)
        return {"label": label, "matrix": matrix, "n": total}

    panels = [make_panel("All data", all_counts)]

    labels_by_number = {}
    for number, name, term_start, term_end in president_rows:
        surname = name.split()[-1]
        start_year = term_start[:4]
        end_year = term_end[:4] if term_end else ""
        labels_by_number[number] = f"{surname} ({start_year}–{end_year})"

    for number in PRESIDENT_NUMBERS:
        label = labels_by_number.get(number, f"President {number}")
        panels.append(make_panel(label, per_president_counts[number]))

    return panels


def main():
    """Parse arguments, aggregate the data, render the figure, and print a summary."""
    parser = argparse.ArgumentParser(
        description="Plot county-match x outcome confusion heatmaps"
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    panels = fetch_confusion_data(args.db)
    output_path = build_confusion_figure(
        panels, args.output,
        suptitle="FEMA PDA Outcomes by County–President Party Match",
        subtitle=(
            "Rows: county's winning party matches vs differs from the president's. "
            "Columns: approved (Declared) vs denied. Unit: report-county."
        ),
        footnote=(
            "Appeals and counties without presidential-vote data are excluded. "
            "Unit is report-county (a report spanning many counties contributes "
            "one observation per county). Cell colour = the cell's share of its "
            "county-match row (each row sums to 100%); annotations show the raw "
            "count and that row share."
        ),
        colorbar_label="Share of county-match row (%)",
        colormap=COLORMAP,
        row_labels=("County\nmatches", "County\ndiffers"),
    )

    print(f"Figure written to : {output_path}")
    for panel in panels:
        (match_appr, match_deny), (differ_appr, differ_deny) = panel["matrix"]
        print(
            f"  {panel['label']:<22} n={panel['n']:<6} "
            f"match[appr/deny]={match_appr}/{match_deny}  "
            f"differ[appr/deny]={differ_appr}/{differ_deny}"
        )


if __name__ == "__main__":
    main()
