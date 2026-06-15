# scripts/plot_governor_match_confusion.py
"""
Title: Governor-Match x Outcome Confusion Heatmaps
Description: Renders confusion-matrix-style heatmaps from data/pda.db relating
    whether the requesting governor's party matched the sitting president's party
    to the PDA outcome (approved vs denied). Each panel is a 2x2 matrix: rows are
    "governor matches president" (top) and "governor differs" (bottom); columns
    are Approved (report_outcome 'Declared') and Denied ('Denied'). Appeals
    (report_outcome 'Denial of Appeal') and reports without a governor
    (governor_vs_president IS NULL) are excluded. Each cell is annotated with its
    raw count and its share of its row (each governor-match row sums to 100%), and
    coloured by that row share on a shared 0-100% scale so panels are comparable.
    One figure holds five panels: all data on top, then Obama, Trump (2017-2021),
    Biden, and Trump (2025-) across the bottom (the two Trump terms are separate
    presidencies, distinguished by president_number and term years). Offline,
    deterministic: reads the DB and writes a static PNG.
Changelog:
    2026-06-14  Initial version.
"""

import argparse
import sqlite3

from confusion_heatmap import build_confusion_figure

DEFAULT_DB = "data/pda.db"
DEFAULT_OUTPUT = "docs/figures/governor_match_confusion.png"

# Outcome groups key off report_outcome; 'Denial of Appeal' is excluded.
APPROVAL_OUTCOME = "Declared"
DENIAL_OUTCOME = "Denied"

# Presidents to break out, in chronological order. The two Trump terms are
# distinct president_numbers (45 = 2017-2021, 47 = 2025-).
PRESIDENT_NUMBERS = [44, 45, 46, 47]

# governor_vs_president flag values.
GOVERNOR_MATCHES = 1
GOVERNOR_DIFFERS = 0

COLORMAP = "Greens"


def _matrix_from_counts(counts):
    """Build the 2x2 count matrix for one panel from a (match, outcome) lookup.

    Args:
        counts (dict): maps (governor_vs_president, report_outcome) -> int count.
    Returns:
        list[list[int]] — rows [matches, differs], columns [approved, denied].
    """
    return [
        [counts.get((GOVERNOR_MATCHES, APPROVAL_OUTCOME), 0),
         counts.get((GOVERNOR_MATCHES, DENIAL_OUTCOME), 0)],
        [counts.get((GOVERNOR_DIFFERS, APPROVAL_OUTCOME), 0),
         counts.get((GOVERNOR_DIFFERS, DENIAL_OUTCOME), 0)],
    ]


def fetch_confusion_data(db_path):
    """Aggregate the governor-match x outcome counts for every panel.

    Restricts to Declared/Denied outcomes with a non-NULL governor_vs_president,
    then builds one all-data panel plus one panel per president in
    PRESIDENT_NUMBERS. Per-president labels (surname + term-year range) are read
    from the presidents table so the two Trump terms are disambiguated by years.

    Args:
        db_path (str): path to the pda.db SQLite database.
    Returns:
        list[dict] — ordered panels (all data first), each with:
            "label"  (str)              — panel title
            "matrix" (list[list[int]])  — 2x2 counts [match/differ][appr/deny]
            "n"      (int)              — total reports in the panel
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT president_number, governor_vs_president, report_outcome,
                   COUNT(*) AS n
            FROM reports
            WHERE report_outcome IN (?, ?)
              AND governor_vs_president IS NOT NULL
            GROUP BY president_number, governor_vs_president, report_outcome
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
        description="Plot governor-match x outcome confusion heatmaps"
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    panels = fetch_confusion_data(args.db)
    output_path = build_confusion_figure(
        panels, args.output,
        suptitle="FEMA PDA Outcomes by Governor–President Party Match",
        subtitle=(
            "Rows: governor's party matches vs differs from the president's. "
            "Columns: approved (Declared) vs denied."
        ),
        footnote=(
            "Appeals and reports without a governor are excluded. Cell colour = "
            "the cell's share of its governor-match row (each row sums to 100%); "
            "annotations show the raw count and that row share."
        ),
        colorbar_label="Share of governor-match row (%)",
        colormap=COLORMAP,
        row_labels=("Governor\nmatches", "Governor\ndiffers"),
    )

    print(f"Figure written to : {output_path}")
    for panel in panels:
        (match_appr, match_deny), (differ_appr, differ_deny) = panel["matrix"]
        print(
            f"  {panel['label']:<22} n={panel['n']:<5} "
            f"match[appr/deny]={match_appr}/{match_deny}  "
            f"differ[appr/deny]={differ_appr}/{differ_deny}"
        )


if __name__ == "__main__":
    main()
