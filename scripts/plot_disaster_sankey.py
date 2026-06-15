# scripts/plot_disaster_sankey.py
"""
Title: Disaster-Type -> Outcome Sankey
Description: Renders a Sankey diagram of non-appeal PDA reports flowing from their
    primary disaster type (scripts/disaster_type.classify, derived from the
    free-text incident_name) to their outcome (Approved = report_outcome
    'Declared', Denied = 'Denied'). Each ribbon is one (type, outcome, president
    party) flow; its width is the report count and its colour is the sitting
    president's party — Democratic = blue, Republican = red — so each flow shows
    its partisan split. Appeals ('Denial of Appeal') are excluded. Built with
    Plotly and written both as an interactive HTML and a static PNG (via kaleido).
Changelog:
    2026-06-14  Initial version.
"""

import argparse
import os
import sqlite3

import plotly.graph_objects as go

from disaster_type import DISASTER_TYPE_ORDER, classify

DEFAULT_DB = "data/pda.db"
DEFAULT_OUTPUT_HTML = "docs/figures/disaster_sankey.html"
DEFAULT_OUTPUT_PNG = "docs/figures/disaster_sankey.png"

APPROVAL_OUTCOME = "Declared"
DENIAL_OUTCOME = "Denied"
OUTCOME_LABELS = {APPROVAL_OUTCOME: "Approved", DENIAL_OUTCOME: "Denied"}

# Ribbon colours by president party (semi-transparent so overlaps read).
PARTY_COLORS = {
    "Democratic": "rgba(31, 119, 180, 0.6)",   # blue
    "Republican": "rgba(214, 39, 40, 0.6)",     # red
    "Other": "rgba(140, 140, 140, 0.5)",        # gray (no/!2-party president)
}

NODE_COLOR_TYPE = "#d9d9d9"
NODE_COLOR_OUTCOME = "#9e9e9e"


def fetch_flows(db_path):
    """Aggregate report counts per (disaster type, outcome, president party).

    Args:
        db_path (str): path to the pda.db SQLite database.
    Returns:
        list[dict] — one flow per combination, each with "type" (str),
        "outcome" (str: 'Declared'/'Denied'), "party" (str), "count" (int).
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT incident_name, report_outcome, president_party, COUNT(*) "
            "FROM reports WHERE report_outcome IN (?, ?) "
            "GROUP BY incident_name, report_outcome, president_party",
            (APPROVAL_OUTCOME, DENIAL_OUTCOME),
        ).fetchall()
    finally:
        conn.close()

    aggregated = {}
    for incident_name, outcome, president_party, count in rows:
        disaster_type = classify(incident_name)
        party = president_party if president_party in PARTY_COLORS else "Other"
        key = (disaster_type, outcome, party)
        aggregated[key] = aggregated.get(key, 0) + count

    return [
        {"type": disaster_type, "outcome": outcome, "party": party, "count": count}
        for (disaster_type, outcome, party), count in aggregated.items()
    ]


def build_sankey(flows):
    """Build the Plotly Sankey figure from the aggregated flows.

    Nodes are the disaster types present (left, ordered by DISASTER_TYPE_ORDER)
    followed by Approved and Denied (right). Each flow becomes one coloured link.

    Args:
        flows (list[dict]): output of fetch_flows.
    Returns:
        plotly.graph_objects.Figure
    """
    # Disaster-type nodes in the canonical order, limited to those with data.
    present_types = {flow["type"] for flow in flows}
    type_nodes = [name for name in DISASTER_TYPE_ORDER if name in present_types]
    outcome_nodes = ["Approved", "Denied"]

    node_labels = type_nodes + outcome_nodes
    node_index = {label: index for index, label in enumerate(node_labels)}
    node_colors = (
        [NODE_COLOR_TYPE] * len(type_nodes)
        + [NODE_COLOR_OUTCOME] * len(outcome_nodes)
    )

    sources, targets, values, link_colors, customdata = [], [], [], [], []
    for flow in flows:
        sources.append(node_index[flow["type"]])
        targets.append(node_index[OUTCOME_LABELS[flow["outcome"]]])
        values.append(flow["count"])
        link_colors.append(PARTY_COLORS[flow["party"]])
        customdata.append(f"{flow['party']} president")

    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=node_labels,
                color=node_colors,
                pad=16,
                thickness=18,
                line=dict(color="#666666", width=0.5),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                customdata=customdata,
                hovertemplate=(
                    "%{source.label} → %{target.label}<br>"
                    "%{value} reports<br>%{customdata}<extra></extra>"
                ),
            ),
        )
    )

    figure.update_layout(
        title=dict(
            text="FEMA PDA Reports: Disaster Type → Outcome"
            "<br><sup>Ribbon colour = sitting president's party "
            "(blue = Democratic, red = Republican). Width = report count. "
            "Appeals excluded.</sup>",
            x=0.01,
        ),
        font=dict(size=13),
        width=1400,
        height=900,
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return figure


def main():
    """Parse arguments, build the Sankey, and write the HTML and PNG."""
    parser = argparse.ArgumentParser(
        description="Plot the disaster-type -> outcome Sankey"
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--html", default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--png", default=DEFAULT_OUTPUT_PNG)
    args = parser.parse_args()

    flows = fetch_flows(args.db)
    figure = build_sankey(flows)

    for path in (args.html, args.png):
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    figure.write_html(args.html)
    figure.write_image(args.png, scale=2)

    print(f"HTML written to : {args.html}")
    print(f"PNG written to  : {args.png}")
    print(f"Flows: {len(flows)}  |  total reports: {sum(f['count'] for f in flows)}")


if __name__ == "__main__":
    main()
