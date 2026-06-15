# scripts/plot_disaster_type_table.py
"""
Title: Disaster-Type Classification Reference Table
Description: Renders a PNG reference table documenting how free-text incident_name
    values are mapped to a single primary disaster type by
    scripts/disaster_type.classify. One row per type in priority order (the order
    in which the first keyword hit wins), with the keywords that map to it, the
    number of non-appeal reports it captured, and example incident_name values
    that landed in it. Counts and examples are read live from data/pda.db so the
    table stays in sync with both the classifier and the data. Offline,
    deterministic: reads the DB and writes a static PNG.
Changelog:
    2026-06-14  Initial version.
"""

import argparse
import os
import sqlite3
import textwrap

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt

from disaster_type import DISASTER_TYPE_PRIORITY, OTHER_TYPE, classify

DEFAULT_DB = "data/pda.db"
DEFAULT_OUTPUT = "docs/figures/disaster_type_classification.png"

APPROVAL_OUTCOME = "Declared"
DENIAL_OUTCOME = "Denied"

EXAMPLES_PER_TYPE = 2

HEADER_COLOR = "#4c72b0"
ROW_SHADE = "#f1f3f7"
LINE_COLOR = "#cccccc"

# (header, x_left, wrap_width_chars, alignment). x in axes fraction [0, 1].
COLUMNS = [
    ("Priority", 0.012, None, "center"),
    ("Disaster type", 0.075, 18, "left"),
    ("Keywords matched", 0.235, 52, "left"),
    ("# Reports", 0.610, None, "center"),
    ("Example incident_name(s)", 0.700, 46, "left"),
]
# Center positions for the centered columns.
PRIORITY_CENTER = 0.04
REPORTS_CENTER = 0.645


def fetch_type_rows(db_path):
    """Collect per-type counts and example incident names from the DB.

    Args:
        db_path (str): path to the pda.db SQLite database.
    Returns:
        tuple(counts, examples, total):
            counts (dict[str, int]) — reports per disaster type
            examples (dict[str, str]) — up to EXAMPLES_PER_TYPE names, "; "-joined
            total (int) — total non-appeal reports classified
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
    names_by_type = {}
    for incident_name, count in rows:
        disaster_type = classify(incident_name)
        counts[disaster_type] = counts.get(disaster_type, 0) + count
        names_by_type.setdefault(disaster_type, []).append((incident_name, count))

    examples = {}
    for disaster_type, named_counts in names_by_type.items():
        top = sorted(named_counts, key=lambda item: item[1], reverse=True)
        examples[disaster_type] = "; ".join(
            name for name, _ in top[:EXAMPLES_PER_TYPE]
        )

    return counts, examples, sum(counts.values())


def build_table_rows(counts, examples):
    """Assemble the ordered table rows (priority types first, then Other).

    Args:
        counts (dict[str, int]): reports per disaster type.
        examples (dict[str, str]): example names per disaster type.
    Returns:
        list[dict] — each with priority, type, keywords, count, examples.
    """
    table_rows = []
    for priority, (disaster_type, keywords) in enumerate(DISASTER_TYPE_PRIORITY, 1):
        table_rows.append({
            "priority": str(priority),
            "type": disaster_type,
            "keywords": ", ".join(keywords),
            "count": counts.get(disaster_type, 0),
            "examples": examples.get(disaster_type, ""),
        })
    table_rows.append({
        "priority": "—",
        "type": OTHER_TYPE,
        "keywords": "— (no keyword match)",
        "count": counts.get(OTHER_TYPE, 0),
        "examples": examples.get(OTHER_TYPE, ""),
    })
    return table_rows


def _wrap(text, width):
    """Wrap text to a character width, returning the (possibly multi-line) string."""
    if width is None:
        return text
    return textwrap.fill(text, width=width) if text else ""


def build_table(table_rows, total, output_path):
    """Render the reference table to a PNG.

    Args:
        table_rows (list[dict]): rows from build_table_rows.
        total (int): total reports (for the caption).
        output_path (str): file path for the rendered PNG.
    Returns:
        str — the output_path that was written.
    """
    # Pre-wrap the text columns and compute each row's line count (= its height).
    rendered_rows = []
    for row in table_rows:
        cells = {
            "priority": row["priority"],
            "type": _wrap(row["type"], 18),
            "keywords": _wrap(row["keywords"], 52),
            "count": str(row["count"]),
            "examples": _wrap(row["examples"], 46),
        }
        line_count = max(text.count("\n") + 1 for text in cells.values())
        rendered_rows.append((cells, line_count))

    header_units = 1.6
    row_padding = 0.7  # vertical breathing room added to each row, in line units
    row_units = [line_count + row_padding for _, line_count in rendered_rows]
    total_units = header_units + sum(row_units)

    line_height_inches = 0.26
    fig_height = total_units * line_height_inches + 1.3  # + title/caption margin
    fig = plt.figure(figsize=(15, fig_height))
    ax = fig.add_axes((0.015, 0.06, 0.97, 0.86))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_units)
    ax.axis("off")

    # Header band.
    header_top = total_units
    header_bottom = total_units - header_units
    ax.axhspan(header_bottom, header_top, color=HEADER_COLOR)
    header_center = (header_top + header_bottom) / 2
    for header, x_left, _wrap_width, align in COLUMNS:
        x = (
            PRIORITY_CENTER if header == "Priority"
            else REPORTS_CENTER if header == "# Reports"
            else x_left
        )
        ax.text(
            x, header_center, header,
            ha=align, va="center", fontsize=10.5, color="white", weight="bold",
        )

    # Data rows, top to bottom.
    y_cursor = header_bottom
    for index, (cells, _line_count) in enumerate(rendered_rows):
        row_height = row_units[index]
        row_top = y_cursor
        row_bottom = y_cursor - row_height
        row_center = (row_top + row_bottom) / 2
        if index % 2 == 1:
            ax.axhspan(row_bottom, row_top, color=ROW_SHADE)
        ax.axhline(row_bottom, color=LINE_COLOR, linewidth=0.5)

        ax.text(PRIORITY_CENTER, row_center, cells["priority"],
                ha="center", va="center", fontsize=9)
        ax.text(COLUMNS[1][1], row_center, cells["type"],
                ha="left", va="center", fontsize=9, weight="bold")
        ax.text(COLUMNS[2][1], row_center, cells["keywords"],
                ha="left", va="center", fontsize=8.5, family="monospace")
        ax.text(REPORTS_CENTER, row_center, cells["count"],
                ha="center", va="center", fontsize=9)
        ax.text(COLUMNS[4][1], row_center, cells["examples"],
                ha="left", va="center", fontsize=8.5, style="italic")

        y_cursor = row_bottom

    # Vertical separators between columns.
    for boundary in (0.072, 0.232, 0.598, 0.697):
        ax.axvline(boundary, color=LINE_COLOR, linewidth=0.5)

    fig.suptitle(
        "Disaster-Type Classification: how incident_name maps to a primary type",
        fontsize=14, y=0.98,
    )
    fig.text(
        0.015, 0.015,
        "Each report is assigned the FIRST type (top to bottom) whose keyword "
        "appears in incident_name (case-insensitive substring match); 'Severe "
        "Storm' is the generic catch-all and 'Other' is the no-match fallback. "
        f"Appeals excluded; total reports classified: {total}.",
        fontsize=8.5, color="gray",
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    """Parse arguments, gather counts/examples, and render the reference table."""
    parser = argparse.ArgumentParser(
        description="Render the disaster-type classification reference table"
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    counts, examples, total = fetch_type_rows(args.db)
    table_rows = build_table_rows(counts, examples)
    output_path = build_table(table_rows, total, args.output)

    print(f"Table written to : {output_path}")
    for row in table_rows:
        print(f"  [{row['priority']:>2}] {row['type']:<20} {row['count']:>4}")


if __name__ == "__main__":
    main()
