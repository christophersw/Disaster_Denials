# scripts/plot_disaster_bucketing.py
"""
Title: Disaster-Type Bucketing Bar Chart
Description: Shows how non-appeal PDA reports distribute across the primary
    disaster types assigned by scripts/disaster_type.classify (from the free-text
    incident_name). A horizontal matplotlib bar chart, one bar per type, sorted by
    report count and annotated with count and share, so the classification feeding
    the disaster-type Sankey is transparent (including how large the "Other"
    catch-all is). Appeals (report_outcome 'Denial of Appeal') are excluded to
    match the Sankey population. Offline, deterministic: reads the DB, writes a PNG.
Changelog:
    2026-06-14  Initial version.
"""

import argparse
import os
import sqlite3

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt

from disaster_type import classify

DEFAULT_DB = "data/pda.db"
DEFAULT_OUTPUT = "docs/figures/disaster_bucketing.png"

APPROVAL_OUTCOME = "Declared"
DENIAL_OUTCOME = "Denied"

BAR_COLOR = "#4c72b0"


def fetch_type_counts(db_path):
    """Count non-appeal reports per primary disaster type.

    Args:
        db_path (str): path to the pda.db SQLite database.
    Returns:
        list[tuple[str, int]] — (disaster_type, count), sorted by count desc.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT incident_name, COUNT(*) FROM reports "
            "WHERE report_outcome IN (?, ?) GROUP BY incident_name",
            (APPROVAL_OUTCOME, DENIAL_OUTCOME),
        ).fetchall()
    finally:
        conn.close()

    counts = {}
    for incident_name, count in rows:
        disaster_type = classify(incident_name)
        counts[disaster_type] = counts.get(disaster_type, 0) + count

    return sorted(counts.items(), key=lambda item: item[1], reverse=True)


def build_chart(type_counts, output_path):
    """Render the horizontal bar chart of report counts per disaster type.

    Args:
        type_counts (list[tuple[str, int]]): (type, count) sorted by count desc.
        output_path (str): file path for the rendered PNG.
    Returns:
        str — the output_path that was written.
    """
    total = sum(count for _, count in type_counts)
    # Plot largest bar at the top: reverse so the highest count is last (top) in
    # the bottom-up bar order.
    labels = [name for name, _ in reversed(type_counts)]
    counts = [count for _, count in reversed(type_counts)]

    fig, ax = plt.subplots(figsize=(11, 7))
    positions = range(len(labels))
    ax.barh(positions, counts, color=BAR_COLOR)
    ax.set_yticks(list(positions), labels)
    ax.set_xlabel("Number of PDA reports (appeals excluded)")
    ax.set_title("PDA Reports by Primary Disaster Type")

    max_count = max(counts) if counts else 0
    for position, count in zip(positions, counts):
        ax.text(
            count + max_count * 0.01, position,
            f"{count}  ({count / total * 100:.1f}%)",
            va="center", ha="left", fontsize=9,
        )
    ax.set_xlim(0, max_count * 1.12)
    ax.margins(y=0.01)

    fig.text(
        0.01, 0.01,
        "Primary type assigned from incident_name by priority "
        "(most specific hazard wins; 'Severe Storm' is the generic catch-all). "
        f"Total reports: {total}.",
        fontsize=8, color="gray",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    """Parse arguments, count by type, render the chart, and print a summary."""
    parser = argparse.ArgumentParser(
        description="Plot the disaster-type bucketing bar chart"
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    type_counts = fetch_type_counts(args.db)
    output_path = build_chart(type_counts, args.output)

    print(f"Chart written to : {output_path}")
    for name, count in type_counts:
        print(f"  {name:<20} {count}")


if __name__ == "__main__":
    main()
