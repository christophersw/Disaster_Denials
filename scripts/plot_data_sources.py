# scripts/plot_data_sources.py
"""
Title: Data Sources Contribution Graphic
Description: Renders a slide-ready (16:9) summary of the data sources behind
    data/pda.db and what each contributes to the SQLite database. Three primary
    sources flow left-to-right into the database: Parsed PDA reports (the core
    reports + report_counties tables), MIT Election Lab county presidential
    returns (FIPS-keyed vote lean), and scraped governor data (requesting
    governor + party). A supporting U.S. Presidents roster combines with the
    election and governor data to produce the political-alignment flags. The
    database panel shows the two core tables with source-coloured column chips
    plus the lookup tables and their row counts. Counts/columns are read from the
    live schema of data/pda.db at render time; layout is static. Writes a PNG.
Changelog:
    2026-06-15  Initial version.
"""

import argparse
import sqlite3

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

DEFAULT_DB = "data/pda.db"
DEFAULT_OUTPUT = "docs/figures/data_sources.png"

CANVAS_WIDTH = 16.0
CANVAS_HEIGHT = 9.0

INK = "#23252b"
MUTED = "#6b7280"
CARD_FILL = "#f7f8fb"
DB_FILL = "#eef1f6"

# One colour per source, reused on the database column chips so each source's
# contribution is visually traceable into the schema.
PDA_C = "#6a4c93"        # parsed PDA documents (violet)
MIT_C = "#3b6fb0"        # MIT election returns (blue)
GOV_C = "#e0922f"        # scraped governors (amber)
PRES_C = "#2a8c8c"       # presidents roster (teal)
DERIVED_C = "#7a8290"    # columns computed by combining sources (grey)


def fetch_counts(db_path):
    """Read row counts and enrichment coverage from the live database.

    Args:
        db_path (str): path to the pda.db SQLite database.
    Returns:
        dict[str, int] — counts keyed by a short name, plus coverage tallies.
    """
    conn = sqlite3.connect(db_path)
    try:
        def count(sql):
            return conn.execute(sql).fetchone()[0]
        return {
            "reports": count("SELECT COUNT(*) FROM reports"),
            "counties": count("SELECT COUNT(*) FROM report_counties"),
            "governors": count("SELECT COUNT(*) FROM governors"),
            "presidents": count("SELECT COUNT(*) FROM presidents"),
            "returns": count("SELECT COUNT(*) FROM county_presidential_returns"),
            "county_summary":
                count("SELECT COUNT(*) FROM county_presidential_summary"),
            "state_summary":
                count("SELECT COUNT(*) FROM state_presidential_summary"),
            "reports_with_gov":
                count("SELECT COUNT(governor_name) FROM reports"),
            "counties_with_fips":
                count("SELECT COUNT(county_fips) FROM report_counties"),
        }
    finally:
        conn.close()


def add_rounded_box(ax, cx, cy, w, h, facecolor, edgecolor="none",
                    linewidth=1.5, zorder=2, shadow=False, rounding=0.10):
    """Draw a rounded rectangle, optionally with a soft drop shadow.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        cx, cy (float): box centre in data units.
        w, h (float): box size in data units.
        facecolor (str): fill colour.
        edgecolor (str): border colour ("none" for none).
        linewidth (float): border width.
        zorder (int): draw order (shadow one below).
        shadow (bool): whether to draw a soft offset shadow.
        rounding (float): corner radius in data units.
    Returns:
        None. Adds patches to ax in place.
    """
    style = f"round,pad=0,rounding_size={rounding}"
    left, bottom = cx - w / 2, cy - h / 2
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (left + 0.05, bottom - 0.07), w, h, boxstyle=style,
            facecolor="black", edgecolor="none", alpha=0.10, zorder=zorder - 1))
    ax.add_patch(FancyBboxPatch(
        (left, bottom), w, h, boxstyle=style, facecolor=facecolor,
        edgecolor=edgecolor, linewidth=linewidth, zorder=zorder))


def add_source_card(ax, cx, cy, w, h, color, eyebrow, title, descriptor, stat,
                    compact=False):
    """Draw one data-source card on the left rail.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        cx, cy (float): card centre in data units.
        w, h (float): card size in data units.
        color (str): source accent colour (left bar + title + stat strip).
        eyebrow (str): small all-caps category label.
        title (str): bold source name.
        descriptor (str): one-line origin + contribution summary.
        stat (str): scale figures (own strip on tall cards; ignored if compact).
        compact (bool): use the shorter three-line layout (no stat strip),
            for the slim supporting card.
    Returns:
        None. Adds the card to ax in place.
    """
    add_rounded_box(ax, cx, cy, w, h, "white", edgecolor="#d9dee7",
                    linewidth=1.3, shadow=True, rounding=0.10)
    left = cx - w / 2
    # Colour accent bar down the left edge.
    ax.add_patch(Rectangle((left + 0.04, cy - h / 2 + 0.08), 0.12, h - 0.16,
                           facecolor=color, edgecolor="none", zorder=3))
    text_x = left + 0.36

    if compact:
        ax.text(text_x, cy + h / 2 - 0.18, eyebrow, ha="left", va="center",
                fontsize=7.5, fontweight="bold", color=color)
        ax.text(text_x, cy, title, ha="left", va="center", fontsize=12,
                fontweight="bold", color=INK)
        ax.text(text_x, cy - h / 2 + 0.19, descriptor, ha="left", va="center",
                fontsize=8.2, color="#4a4f57")
        return

    ax.text(text_x, cy + h / 2 - 0.26, eyebrow, ha="left", va="center",
            fontsize=7.8, fontweight="bold", color=color)
    ax.text(text_x, cy + h / 2 - 0.58, title, ha="left", va="center",
            fontsize=13.5, fontweight="bold", color=INK)
    ax.text(text_x, cy - 0.06, descriptor, ha="left", va="center",
            fontsize=8.8, color="#4a4f57")
    # Stat strip.
    ax.text(text_x, cy - h / 2 + 0.26, stat, ha="left", va="center",
            fontsize=8.6, fontweight="bold", color=color,
            family="monospace")


def add_chip(ax, cx, cy, w, h, text, color):
    """Draw one source-coloured column chip inside a table box.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        cx, cy (float): chip centre in data units.
        w, h (float): chip size in data units.
        text (str): chip label.
        color (str): source colour (tinted fill + border + text).
    Returns:
        None. Adds the chip to ax in place.
    """
    add_rounded_box(ax, cx, cy, w, h, to_rgba(color, 0.15), edgecolor=color,
                    linewidth=1.1, rounding=0.08, zorder=4)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.0,
            color=color, fontweight="bold", zorder=5)


def pack_chips(ax, chips, left, right, y_start, line_h=0.44, fontsize=8.0):
    """Lay chips left-to-right, wrapping to new rows within [left, right].

    Args:
        ax (matplotlib.axes.Axes): target axes.
        chips (list[tuple[str, str]]): (text, colour) chips.
        left, right (float): horizontal bounds in data units.
        y_start (float): centre y of the first chip row.
        line_h (float): vertical step between chip rows.
        fontsize (float): chip label size in points.
    Returns:
        None. Adds chips to ax in place.
    """
    gap = 0.16
    char_w = fontsize * 0.52 / 72.0  # 1 data unit == 1 inch == 72 pt
    x, y = left, y_start
    for text, color in chips:
        w = len(text) * char_w + 0.34
        if x + w > right and x > left:
            x, y = left, y - line_h
        add_chip(ax, x + w / 2, y, w, 0.34, text, color)
        x += w + gap


def add_table_box(ax, left, right, top, bottom, title, count_label, chips):
    """Draw a core-table box (title row + source-coloured column chips).

    Args:
        ax (matplotlib.axes.Axes): target axes.
        left, right, top, bottom (float): box bounds in data units.
        title (str): table name (monospace).
        count_label (str): row-count descriptor shown after the title.
        chips (list[tuple[str, str]]): (text, colour) column chips.
    Returns:
        None. Adds the box to ax in place.
    """
    cx, cy = (left + right) / 2, (top + bottom) / 2
    add_rounded_box(ax, cx, cy, right - left, top - bottom, "white",
                    edgecolor="#cfd5df", linewidth=1.3, rounding=0.09)
    ax.text(left + 0.20, top - 0.27, title, ha="left", va="center",
            fontsize=10.5, fontweight="bold", family="monospace", color=INK)
    ax.text(left + 0.20 + len(title) * 0.082 + 0.20, top - 0.27, count_label,
            ha="left", va="center", fontsize=8.6, color=MUTED)
    pack_chips(ax, chips, left + 0.20, right - 0.15, top - 0.66)


def add_lookup_card(ax, cx, cy, w, h, color, name, count_label):
    """Draw a small lookup/source-table card with its row count.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        cx, cy (float): card centre in data units.
        w, h (float): card size.
        color (str): source colour (top bar + count).
        name (str): table name (monospace, wrapped if long).
        count_label (str): row count.
    Returns:
        None. Adds the card to ax in place.
    """
    add_rounded_box(ax, cx, cy, w, h, "white", edgecolor=color, linewidth=1.2,
                    rounding=0.08)
    ax.add_patch(Rectangle((cx - w / 2 + 0.06, cy + h / 2 - 0.14),
                           w - 0.12, 0.07, facecolor=color, edgecolor="none",
                           zorder=3))
    ax.text(cx, cy + 0.10, name, ha="center", va="center", fontsize=7.6,
            family="monospace", color=INK)
    ax.text(cx, cy - 0.20, count_label, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=color)


def add_cylinder(ax, cx, cy, w, h, color):
    """Draw a small database-cylinder icon.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        cx, cy (float): icon centre in data units.
        w, h (float): icon width and height.
        color (str): fill colour.
    Returns:
        None. Adds the icon to ax in place.
    """
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + 0.04), w, h - 0.08,
                           facecolor=color, edgecolor="none", zorder=4))
    ax.add_patch(Ellipse((cx, cy + h / 2 - 0.02), w, 0.18, facecolor="white",
                         edgecolor=color, linewidth=1.4, zorder=5))
    ax.add_patch(Ellipse((cx, cy - h / 2 + 0.04), w, 0.18, facecolor=color,
                         edgecolor="none", zorder=4))


def build_figure(counts):
    """Build the full 16:9 data-sources figure.

    Args:
        counts (dict): row counts/coverage from fetch_counts.
    Returns:
        matplotlib.figure.Figure — the rendered figure, ready to save.
    """
    gov_pct = round(100 * counts["reports_with_gov"] / counts["reports"])
    fips_pct = round(100 * counts["counties_with_fips"] / counts["counties"])

    fig = plt.figure(figsize=(CANVAS_WIDTH, CANVAS_HEIGHT))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CANVAS_WIDTH)
    ax.set_ylim(0, CANVAS_HEIGHT)
    ax.axis("off")

    # ---- Title -----------------------------------------------------------
    ax.text(0.55, 8.52, "Three Data Sources, One Database", ha="left",
            va="center", fontsize=25, fontweight="bold", color=INK)
    ax.text(0.55, 7.96, "How data/pda.db is assembled — FEMA reports enriched "
            "with election and governor data", ha="left", va="center",
            fontsize=13, color=MUTED)

    # ---- Left rail: source cards -----------------------------------------
    card_w = 4.60
    card_cx = 0.55 + card_w / 2
    add_source_card(
        ax, card_cx, 6.45, card_w, 1.40, PDA_C, "SOURCE 1 · DOCUMENTS",
        "Parsed PDA Reports",
        "FEMA report PDFs parsed by Claude → the core tables",
        f"reports {counts['reports']:,}  ·  counties {counts['counties']:,}")
    add_source_card(
        ax, card_cx, 4.75, card_w, 1.40, MIT_C, "SOURCE 2 · ELECTIONS",
        "MIT Election Data",
        "County presidential returns, 2000–2024 (vote lean)",
        f"{counts['returns']:,} rows → {counts['county_summary']:,} county "
        f"summaries")
    add_source_card(
        ax, card_cx, 3.05, card_w, 1.40, GOV_C, "SOURCE 3 · OFFICEHOLDERS",
        "Scraped Governor Data",
        "Wikipedia lists + NGA roster → requesting governor",
        f"{counts['governors']} terms  ·  {gov_pct}% of reports matched")
    add_source_card(
        ax, card_cx, 1.62, card_w, 0.88, PRES_C, "SUPPORTING · PARTY BY TERM",
        f"U.S. Presidents  ({counts['presidents']})",
        "President at decision + party-alignment flags", "", compact=True)

    # ---- Flow arrows + join keys -----------------------------------------
    arrows = [
        (6.45, "writes the core tables"),
        (4.75, "join by county FIPS + nearest election year"),
        (3.05, "match by state + request date"),
        (1.62, "match by decision date"),
    ]
    for y, label in arrows:
        ax.add_patch(FancyArrowPatch(
            (5.20, y), (8.62, y), arrowstyle="-|>", mutation_scale=18,
            linewidth=2.2, color="#9aa3b0", shrinkA=0, shrinkB=0, zorder=1))
        ax.text(6.91, y + 0.20, label, ha="center", va="center", fontsize=7.6,
                color=MUTED, style="italic")

    # ---- Right: the database --------------------------------------------
    add_rounded_box(ax, 12.075, 4.25, 6.75, 5.80, DB_FILL, edgecolor="#aab2c0",
                    linewidth=1.6, rounding=0.12)
    add_cylinder(ax, 9.30, 6.78, 0.42, 0.50, "#5566aa")
    ax.text(9.65, 6.86, "data/pda.db", ha="left", va="center", fontsize=12.5,
            fontweight="bold", family="monospace", color=INK)
    ax.text(9.65, 6.55, "the assembled SQLite database", ha="left",
            va="center", fontsize=8.8, color=MUTED)

    add_table_box(
        ax, 8.95, 15.20, 6.30, 4.95, "reports",
        f"· {counts['reports']:,} rows (one per report)",
        [("PDA · ~45 extracted fields", PDA_C),
         ("MIT · state vote lean ×5", MIT_C),
         ("Gov · governor + party ×2", GOV_C),
         ("Pres · president ×3", PRES_C),
         ("derived · alignment flags ×3", DERIVED_C)])
    add_table_box(
        ax, 8.95, 15.20, 4.75, 3.55, "report_counties",
        f"· {counts['counties']:,} rows (one per county)",
        [("PDA · extracted county fields", PDA_C),
         ("MIT + Census · FIPS + county vote lean", MIT_C),
         ("derived · county party match", DERIVED_C)])

    # Lookup / source tables row.
    ax.text(8.95, 3.30, "Lookup tables loaded from the sources", ha="left",
            va="center", fontsize=8.4, fontweight="bold", color=MUTED)
    lookups = [
        (GOV_C, "governors", f"{counts['governors']}"),
        (PRES_C, "presidents", f"{counts['presidents']}"),
        (MIT_C, "county_pres_\nsummary", f"{counts['county_summary']:,}"),
        (MIT_C, "state_pres_\nsummary", f"{counts['state_summary']}"),
    ]
    lw = (15.20 - 8.95 - 3 * 0.18) / 4
    for i, (color, name, n) in enumerate(lookups):
        lcx = 8.95 + lw / 2 + i * (lw + 0.18)
        add_lookup_card(ax, lcx, 2.55, lw, 1.05, color, name, n)

    ax.text(12.075, 1.70, f"Raw MIT returns: {counts['returns']:,} rows   ·   "
            "+ U.S. Census FIPS crosswalk for county matching", ha="center",
            va="center", fontsize=8.0, color=MUTED, style="italic")

    return fig


def main():
    """Parse CLI args, read counts from the DB, render, and save the PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB,
                        help="path to pda.db (default: %(default)s)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="output PNG path (default: %(default)s)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="output resolution (default: %(default)s)")
    args = parser.parse_args()

    counts = fetch_counts(args.db)
    fig = build_figure(counts)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output} ({CANVAS_WIDTH:.0f}x{CANVAS_HEIGHT:.0f} @ "
          f"{args.dpi} dpi)")


if __name__ == "__main__":
    main()
