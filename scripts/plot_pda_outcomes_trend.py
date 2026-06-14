# scripts/plot_pda_outcomes_trend.py
"""
Title: PDA Outcomes & Average Damage Chart
Description: Renders a dual-axis chart of FEMA Preliminary Damage Assessment
    (PDA) trends from data/pda.db. The left axis is a per-year stacked bar of
    report counts — approvals (report_outcome 'Declared') on the bottom and
    denials (report_outcome 'Denied') stacked on top. The right axis is a line
    of the average total damage estimate (total_cost_estimate = ia + pa) per
    report each year, averaged over the approvals + denials that included a cost
    estimate (reports with no estimate are excluded from the average). The plot
    background is shaded into alternating light-gray/white swimlanes, one per
    presidential term running inauguration to inauguration, labelled with the
    president in office.
    Appeals (report_outcome 'Denial of Appeal') are excluded so an incident
    that was denied then appealed is not counted twice. Years are
    inauguration-aligned: each runs from Jan 20 to the following Jan 19, so a
    decision in early January counts under the outgoing administration. Term-years
    2008-2025 are complete (data starts October 2007, so term-year 2007 is partial
    and omitted; data runs into 2026, covering term-year 2025 through Jan 19 2026).
    Offline, deterministic: reads the DB and writes a static PNG.
Changelog:
    2026-06-14  Initial version.
"""

import argparse
import os
import sqlite3

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DEFAULT_DB = "data/pda.db"
DEFAULT_OUTPUT = "docs/figures/pda_outcomes_and_damage_2008_2025.png"

# Inauguration-aligned year window. A "year" runs from Jan 20 to the following
# Jan 19, so decisions in early January count under the outgoing administration.
# Term-years 2008-2025 are complete in the source data: the data starts Oct 2007
# (so term-year 2007 is partial and excluded) and runs into 2026 (covering all of
# term-year 2025 through Jan 19 2026).
START_YEAR = 2008
END_YEAR = 2025

# We realise the Jan 20 boundary by shifting each decision date back 19 days
# (Jan 20 -> Jan 1, Jan 19 -> Dec 31 of the prior year) before taking its year.
INAUGURATION_SHIFT = "-19 days"

# Report-outcome groups. We key off report_outcome (the actual decision) rather
# than report_type, which disagrees on a handful of records.
APPROVAL_OUTCOME = "Declared"
DENIAL_OUTCOME = "Denied"

# Visual palette.
APPROVAL_COLOR = "steelblue"
DENIAL_COLOR = "firebrick"
DAMAGE_COLOR = "darkgoldenrod"

# Background swimlane shading per presidential term, alternating between a light
# gray and "none" (the white figure background) so consecutive terms separate
# visually. Because the bands alternate, neighbouring presidents always differ.
BAND_COLORS = ("#e9e9e9", None)


def fetch_yearly_data(db_path):
    """Aggregate per-year approval/denial counts and average damage estimate.

    Years are inauguration-aligned: each runs from Jan 20 to the following Jan 19
    (see INAUGURATION_SHIFT), so a decision in early January counts under the
    outgoing administration rather than the incoming one. A single grouped query
    over reports — restricted to the Declared and Denied outcomes within
    term-years START_YEAR..END_YEAR — is pivoted into per-year series, with gap
    years filled with zeros so the x-axis is continuous. The average damage series
    is each year's mean total_cost_estimate across the approvals + denials that
    carried an estimate; reports with a NULL estimate are excluded from both the
    numerator and the denominator (SUM and COUNT of the column ignore NULLs).

    Args:
        db_path (str): path to the pda.db SQLite database.
    Returns:
        dict with four equal-length, year-ordered lists:
            "years"      (list[int])   — START_YEAR..END_YEAR (term-years)
            "approvals"  (list[int])   — count of Declared per year
            "denials"    (list[int])   — count of Denied per year
            "avg_damage" (list[float]) — mean estimate per year (0.0 if a year
                                         had no report with an estimate)
    """
    query = """
        SELECT strftime('%Y', date(decision_date, ?)) AS term_year,
               report_outcome,
               COUNT(*) AS n,
               SUM(total_cost_estimate) AS damage_sum,
               COUNT(total_cost_estimate) AS damage_count
        FROM reports
        WHERE report_outcome IN (?, ?)
          AND decision_date >= ? AND decision_date < ?
        GROUP BY term_year, report_outcome
    """
    # Inauguration-aligned window: term-year START_YEAR begins on its Jan 20, and
    # term-year END_YEAR runs through the following Jan 19 (exclusive upper bound
    # at the next Jan 20).
    window_start = f"{START_YEAR}-01-20"
    window_end = f"{END_YEAR + 1}-01-20"

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            query,
            (INAUGURATION_SHIFT, APPROVAL_OUTCOME, DENIAL_OUTCOME,
             window_start, window_end),
        ).fetchall()
    finally:
        conn.close()

    # Pivot rows into per-year lookups keyed by integer year. damage_sum and
    # damage_count tally only reports with a non-NULL estimate, accumulated
    # across both outcomes so the average spans approvals + denials together.
    approvals_by_year = {}
    denials_by_year = {}
    damage_sum_by_year = {}
    damage_count_by_year = {}
    for year_text, outcome, count, damage_sum, damage_count in rows:
        year = int(year_text)
        if outcome == APPROVAL_OUTCOME:
            approvals_by_year[year] = count
        elif outcome == DENIAL_OUTCOME:
            denials_by_year[year] = count
        damage_sum_by_year[year] = damage_sum_by_year.get(year, 0.0) + (damage_sum or 0.0)
        damage_count_by_year[year] = damage_count_by_year.get(year, 0) + (damage_count or 0)

    years = list(range(START_YEAR, END_YEAR + 1))
    approvals = [approvals_by_year.get(year, 0) for year in years]
    denials = [denials_by_year.get(year, 0) for year in years]

    avg_damage = []
    for year in years:
        count_with_estimate = damage_count_by_year.get(year, 0)
        if count_with_estimate:
            avg_damage.append(damage_sum_by_year[year] / count_with_estimate)
        else:
            avg_damage.append(0.0)

    return {
        "years": years,
        "approvals": approvals,
        "denials": denials,
        "avg_damage": avg_damage,
    }


def fetch_president_spans(db_path, years):
    """Group the (inauguration-aligned) chart years into spans by president.

    Because the year buckets are already inauguration-aligned, each term-year
    falls entirely within a single presidency. A year is mapped to its president
    by probing July 1 (any date inside the term-year works), and consecutive
    years under the same president are merged into one span. The span edges are
    therefore the bucket boundaries themselves: start_index..end_index are list
    indices of `years`, matching the integer bar positions in build_chart, and a
    band is drawn from start_index - 0.5 to end_index + 0.5.

    Args:
        db_path (str): path to the pda.db SQLite database.
        years (list[int]): the ordered chart years (x-axis).
    Returns:
        list[dict] — one entry per contiguous span, each with:
            "name"        (str) — president's full name
            "party"       (str) — party_bucket ('dem' | 'rep' | 'other')
            "start_index" (int) — x-position of the span's first year
            "end_index"   (int) — x-position of the span's last year
    """
    conn = sqlite3.connect(db_path)
    try:
        presidents = conn.execute(
            "SELECT number, name, term_start, term_end, party_bucket FROM presidents"
        ).fetchall()
    finally:
        conn.close()

    def president_on(probe_date):
        """Return (number, name, party) of the president serving on probe_date."""
        for number, name, term_start, term_end, party in presidents:
            # An empty term_end marks the sitting president (open-ended term).
            term_finish = term_end if term_end else "9999-12-31"
            if term_start <= probe_date < term_finish:
                return number, name, party
        return None

    spans = []
    for index, year in enumerate(years):
        match = president_on(f"{year}-07-01")
        if match is None:
            continue
        number, name, party = match
        if spans and spans[-1]["number"] == number:
            spans[-1]["end_index"] = index
        else:
            spans.append({
                "number": number, "name": name, "party": party,
                "start_index": index, "end_index": index,
            })
    return spans


def build_chart(data, president_spans, output_path):
    """Build the dual-axis stacked-bar + average-damage line figure and save a PNG.

    Left axis: stacked bars of approvals (bottom) and denials (top). Right axis:
    a per-year average-damage-estimate line, with y-ticks formatted as millions
    of dollars. The background is shaded into alternating swimlanes per
    presidential term. Both series share integer x-positions so the line aligns
    with the bars. The parent directory of output_path is created if it does not
    exist.

    Args:
        data (dict): the structure returned by fetch_yearly_data.
        president_spans (list[dict]): contiguous president spans from
            fetch_president_spans, used to draw and label the background bands.
        output_path (str): file path for the rendered PNG.
    Returns:
        str — the output_path that was written.
    """
    years = data["years"]
    positions = range(len(years))

    fig, ax_counts = plt.subplots(figsize=(13, 7.5))

    # Background swimlanes — one band per presidential term running from
    # inauguration to inauguration, drawn behind the bars (zorder 0) and
    # labelled with the president's surname along the top.
    for band_index, span in enumerate(president_spans):
        band_color = BAND_COLORS[band_index % len(BAND_COLORS)]
        if band_color is not None:
            ax_counts.axvspan(
                span["start_index"] - 0.5, span["end_index"] + 0.5,
                facecolor=band_color, edgecolor="none", zorder=0,
            )
        label_x = (span["start_index"] + span["end_index"]) / 2
        ax_counts.text(
            label_x, 0.985, span["name"].split()[-1],
            transform=ax_counts.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="dimgray", zorder=5,
        )

    # Left axis — stacked bars (zorder above the background bands).
    ax_counts.bar(
        positions, data["approvals"],
        color=APPROVAL_COLOR, label="Approvals (Declared)", zorder=2,
    )
    ax_counts.bar(
        positions, data["denials"], bottom=data["approvals"],
        color=DENIAL_COLOR, label="Denials", zorder=2,
    )
    ax_counts.set_ylabel("Number of PDA reports")
    ax_counts.set_xlabel("Year (inauguration-aligned: Jan 20 – Jan 19)")
    ax_counts.set_xticks(list(positions))
    ax_counts.set_xticklabels([str(year) for year in years], rotation=45)
    ax_counts.set_xlim(-0.5, len(years) - 0.5)

    # Right axis — average damage-estimate line, labelled in millions of dollars.
    ax_damage = ax_counts.twinx()
    ax_damage.plot(
        positions, data["avg_damage"],
        color=DAMAGE_COLOR, marker="o", linewidth=2,
        label="Average damage estimate",
    )
    ax_damage.set_ylabel("Average total damage estimate per report")
    ax_damage.set_ylim(bottom=0)
    ax_damage.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"${value / 1e6:.0f}M")
    )

    ax_counts.set_title(
        "FEMA PDA Outcomes and Average Damage Estimate, "
        f"{START_YEAR}–{END_YEAR}",
        pad=34,
    )

    # Combined legend spanning both axes, placed as a single row above the plot
    # so it never overlaps the bars or the swimlane labels.
    count_handles, count_labels = ax_counts.get_legend_handles_labels()
    damage_handles, damage_labels = ax_damage.get_legend_handles_labels()
    ax_counts.legend(
        count_handles + damage_handles,
        count_labels + damage_labels,
        loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
        frameon=False, fontsize=9,
    )

    fig.text(
        0.01, 0.06,
        "Years run inauguration to inauguration (Jan 20 – Jan 19), so early-January "
        "decisions count under the outgoing administration. Appeals excluded; "
        "bands mark the president.\n"
        "Average is over approvals + denials that carried a cost estimate; "
        "reports with no estimate (~154 approvals) are excluded.",
        fontsize=8, color="gray", va="top",
    )

    fig.subplots_adjust(left=0.07, right=0.92, top=0.85, bottom=0.22)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    """Parse arguments, aggregate the data, render the chart, and print a summary."""
    parser = argparse.ArgumentParser(
        description="Plot PDA outcomes and average damage estimate, 2008-2025"
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = fetch_yearly_data(args.db)
    president_spans = fetch_president_spans(args.db, data["years"])
    output_path = build_chart(data, president_spans, args.output)

    total_approvals = sum(data["approvals"])
    total_denials = sum(data["denials"])
    nonzero_avgs = [value for value in data["avg_damage"] if value]
    overall_avg = sum(nonzero_avgs) / len(nonzero_avgs) if nonzero_avgs else 0.0

    print(f"Chart written to          : {output_path}")
    print(f"Years                     : {START_YEAR}–{END_YEAR}")
    print(f"Approvals (Declared)      : {total_approvals}")
    print(f"Denials (Denied)          : {total_denials}")
    print(f"Mean of yearly averages   : ${overall_avg / 1e6:.1f}M")


if __name__ == "__main__":
    main()
