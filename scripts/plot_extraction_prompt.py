# scripts/plot_extraction_prompt.py
"""
Title: PDA Extraction Prompt & Data Spec Graphic
Description: Renders a slide-ready (16:9) graphic of the prompt Claude follows to
    parse each FEMA PDA report PDF. The left panel shows the system prompt — its
    opening role line plus six highlighted instructions, each annotated with the
    feature it demonstrates (no-hallucination guardrail, "not requested" vs zero,
    number normalization, legacy-vs-modern handling, requested-vs-granted, and
    self-review). The right panel shows the data spec the model must return: the
    report-level `reports` fields (grouped) and the per-county `report_counties`
    fields. A footer strip lists how the request is run (model, thinking, effort,
    caching, tool use, validation). Wording is drawn verbatim/condensed from
    pda/extract.py (SYSTEM_PROMPT, build_request) and pda/schema.py. Pure diagram:
    draws only matplotlib patches and text, reads no data, writes a PNG.
Changelog:
    2026-06-15  Initial version.
"""

import argparse
import textwrap

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DEFAULT_OUTPUT = "docs/figures/extraction_prompt_and_spec.png"

# Canvas is 16 wide x 9 tall, so one data unit equals one inch at figsize
# (16, 9). That 1-unit-per-inch fact is used to size the footer pills.
CANVAS_WIDTH = 16.0
CANVAS_HEIGHT = 9.0

INK = "#23252b"          # near-black body text
MUTED = "#6b7280"        # secondary grey text
HIGHLIGHT = "#ffe28a"    # highlighter-yellow span behind prompt quotes
CARD_FILL = "#f7f8fb"    # panel background
PILL_FILL = "#eef1f6"    # footer setting pills
PILL_EDGE = "#c3cad6"

PROMPT_ACCENT = "#6a4c93"    # violet — the prompt panel
REPORTS_ACCENT = "#3b6fb0"   # blue — the reports table
COUNTIES_ACCENT = "#2f8f5b"  # green — the report_counties table

# Six instructions lifted from SYSTEM_PROMPT, each paired with the feature it
# demonstrates (verbatim where short, lightly condensed where long).
PROMPT_FEATURES = [
    ("Never infer, estimate, or fabricate a value that is not present.",
     "Guardrail — an unknown field stays null, never a guess"),
    ('"... – (Not Requested)"  →  flag that program false, numbers null',
     'Records "not requested" as distinct from a genuine zero'),
    ('strip $ % separators;  "-" "N/A" "UNK" "Unknown"  →  null, never 0',
     "Normalizes numbers; a missing marker is never coerced to 0"),
    ("Populate whichever the report prints;  do NOT map legacy onto modern",
     "Handles ~18 years of shifting report layouts"),
    ("requested = asked for   ·   granted = actually made available",
     "Requested-vs-granted: the distinction the study turns on"),
    ('needs_review = true if anything is off  ·  "only raw facts"',
     "Self-flags ambiguity; derives nothing on its own"),
]

# The data spec (pda/schema.py): report-level field groups + the county fields.
REPORT_FIELD_GROUPS = [
    ("Identity & outcome",
     "outcome · dates · jurisdiction · state_abbr · requestor · "
     "disaster_number · type"),
    ("Requested programs",
     "ia_requested, pa_requested, hm_requested, pa_categories_requested"),
    ("Individual Assistance",
     "damage counts (destroyed/major/minor/affected), % insured, "
     "poverty/SSI/SNAP/ownership/unemployment/age/disability/ICC, "
     "legacy low-income & elderly, cost_estimate"),
    ("Public Assistance",
     "primary_impact, cost_estimate, statewide & countywide per-capita "
     "indicators"),
    ("Self-review",
     "needs_review, review_note"),
]
COUNTY_FIELDS = ("county_name, geo_type, per_capita_impact, "
                 "requested_ia / requested_pa, granted_ia / granted_pa, source")

# How the single request is run (pda/extract.py build_request).
RUN_SETTINGS = [
    "Claude Opus 4.8",
    "adaptive thinking",
    "effort: high",
    "cached system + schema prefix",
    "tool use → record_pda_report",
    "re-validated with Pydantic",
]


def add_rounded_box(ax, center_x, center_y, width, height, facecolor,
                    edgecolor="none", linewidth=1.5, zorder=2, shadow=False,
                    rounding=0.10):
    """Draw a rounded rectangle, optionally with a soft drop shadow.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        center_x, center_y (float): box centre in data units.
        width, height (float): box size in data units.
        facecolor (str): fill colour.
        edgecolor (str): border colour ("none" for no border).
        linewidth (float): border width.
        zorder (int): draw order (shadow sits one below).
        shadow (bool): whether to draw a soft offset shadow.
        rounding (float): corner radius in data units.
    Returns:
        None. Adds patches to ax in place.
    """
    style = f"round,pad=0,rounding_size={rounding}"
    left, bottom = center_x - width / 2, center_y - height / 2
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (left + 0.05, bottom - 0.07), width, height, boxstyle=style,
            facecolor="black", edgecolor="none", alpha=0.10,
            zorder=zorder - 1))
    ax.add_patch(FancyBboxPatch(
        (left, bottom), width, height, boxstyle=style, facecolor=facecolor,
        edgecolor=edgecolor, linewidth=linewidth, zorder=zorder))


def draw_prompt_panel(ax, left, right, top, bottom):
    """Draw the system-prompt panel with highlighted, annotated instructions.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        left, right, top, bottom (float): panel bounds in data units.
    Returns:
        None. Adds the panel to ax in place.
    """
    cx, cy = (left + right) / 2, (top + bottom) / 2
    add_rounded_box(ax, cx, cy, right - left, top - bottom, CARD_FILL,
                    edgecolor=PROMPT_ACCENT, linewidth=1.6, rounding=0.12)

    text_x = left + 0.40
    # "Editor window" motif so the panel reads as the prompt itself.
    for i, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        ax.add_patch(plt.Circle((text_x + 0.07 + i * 0.22, top - 0.34), 0.07,
                                facecolor=color, edgecolor="none", zorder=3))
    ax.text(text_x + 0.80, top - 0.34,
            "system prompt  ·  sent with every PDF", ha="left",
            va="center", fontsize=9, family="monospace", color=MUTED)

    # Opening role line (verbatim, condensed) shown as the prompt's header.
    role_line = ('“You are a meticulous data-extraction assistant given '
                 'one FEMA PDA report as a PDF. Return the fields defined by '
                 'the JSON schema.”')
    ax.text(text_x, top - 0.95, textwrap.fill(role_line, width=72),
            ha="left", va="top", fontsize=10.5, color=INK, linespacing=1.3)

    # Six highlighted instructions, each with a grey feature annotation.
    ys = [5.85, 5.10, 4.35, 3.60, 2.85, 2.10]
    for (quote, feature), y in zip(PROMPT_FEATURES, ys):
        ax.text(text_x, y, quote, ha="left", va="center", fontsize=8.2,
                family="monospace", color=INK, zorder=3,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=HIGHLIGHT,
                          edgecolor="none"))
        ax.text(text_x + 0.05, y - 0.32, "»  " + feature, ha="left",
                va="center", fontsize=8.4, color=MUTED, style="italic")


def draw_spec_table(ax, left, right, top, bottom, accent, title, subtitle,
                    rows, single_field_line=None):
    """Draw one data-spec card (a table's fields), grouped or flat.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        left, right, top, bottom (float): card bounds in data units.
        accent (str): card accent colour (title + rule).
        title (str): table name (monospace).
        subtitle (str): short descriptor after the title.
        rows (list[tuple[str, str]] | None): (group label, field list) rows.
        single_field_line (str | None): a single field list, when not grouped.
    Returns:
        None. Adds the card to ax in place.
    """
    cx, cy = (left + right) / 2, (top + bottom) / 2
    add_rounded_box(ax, cx, cy, right - left, top - bottom, "white",
                    edgecolor=accent, linewidth=1.5, rounding=0.10)

    text_x = left + 0.30
    ax.text(text_x, top - 0.30, title, ha="left", va="center", fontsize=11,
            family="monospace", fontweight="bold", color=accent)
    ax.text(text_x + len(title) * 0.085 + 0.25, top - 0.30, subtitle,
            ha="left", va="center", fontsize=8.8, color=MUTED)
    ax.plot([left + 0.30, right - 0.30], [top - 0.52, top - 0.52],
            color=accent, linewidth=0.8, alpha=0.4)

    wrap_chars = int((right - left - 0.7) / 0.066)  # ~chars per mono line
    if single_field_line is not None:
        ax.text(text_x, top - 0.82, textwrap.fill(single_field_line,
                width=wrap_chars), ha="left", va="top", fontsize=8.2,
                family="monospace", color="#444444", linespacing=1.35)
        return

    y = top - 0.80
    for label, fields in rows:
        ax.text(text_x, y, label, ha="left", va="top", fontsize=8.8,
                fontweight="bold", color=accent)
        wrapped = textwrap.fill(fields, width=wrap_chars)
        n_lines = wrapped.count("\n") + 1
        ax.text(text_x + 0.15, y - 0.24, wrapped, ha="left", va="top",
                fontsize=8.0, family="monospace", color="#444444",
                linespacing=1.3)
        y -= 0.30 + n_lines * 0.235 + 0.12


def draw_footer_pills(ax, items, y, fontsize=8.5):
    """Draw a centered single row of rounded setting pills.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        items (list[str]): pill labels.
        y (float): row centre in data units.
        fontsize (float): pill label size in points.
    Returns:
        None. Adds the pills to ax in place.
    """
    # 1 data unit == 1 inch == 72 pt; mono advance width ~0.6 em.
    char_w = fontsize * 0.6 / 72.0
    widths = [len(text) * char_w + 0.44 for text in items]
    gap = 0.30
    total = sum(widths) + gap * (len(items) - 1)
    x = (CANVAS_WIDTH - total) / 2.0
    for text, width in zip(items, widths):
        cx = x + width / 2
        add_rounded_box(ax, cx, y, width, 0.40, PILL_FILL,
                        edgecolor=PILL_EDGE, linewidth=1.0, rounding=0.16)
        ax.text(cx, y, text, ha="center", va="center", fontsize=fontsize,
                family="monospace", color=INK)
        x += width + gap


def build_figure():
    """Build the full 16:9 prompt-and-spec figure.

    Returns:
        matplotlib.figure.Figure — the rendered figure, ready to save.
    """
    fig = plt.figure(figsize=(CANVAS_WIDTH, CANVAS_HEIGHT))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CANVAS_WIDTH)
    ax.set_ylim(0, CANVAS_HEIGHT)
    ax.axis("off")

    # ---- Title block -----------------------------------------------------
    ax.text(0.55, 8.52, "Reading a PDF into Data", ha="left", va="center",
            fontsize=25, fontweight="bold", color=INK)
    ax.text(0.55, 7.96, "The prompt Claude follows for every FEMA report — "
            "and the data spec it must return", ha="left", va="center",
            fontsize=13, color=MUTED)

    # ---- Left: the prompt ------------------------------------------------
    draw_prompt_panel(ax, left=0.55, right=8.55, top=7.45, bottom=1.50)

    # ---- Right: the data spec --------------------------------------------
    ax.text(8.90, 7.50, "The data spec  —  what every report returns",
            ha="left", va="center", fontsize=12.5, fontweight="bold",
            color=INK)
    draw_spec_table(ax, left=8.85, right=15.45, top=7.18, bottom=2.85,
                    accent=REPORTS_ACCENT, title="reports",
                    subtitle="one row per report  (~45 fields)",
                    rows=REPORT_FIELD_GROUPS)
    draw_spec_table(ax, left=8.85, right=15.45, top=2.68, bottom=1.50,
                    accent=COUNTIES_ACCENT, title="report_counties",
                    subtitle="one row per county",
                    rows=None, single_field_line=COUNTY_FIELDS)

    # ---- Footer: how the request is run ----------------------------------
    ax.text(CANVAS_WIDTH / 2, 1.12, "HOW EACH PDF IS PROCESSED",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=MUTED)
    draw_footer_pills(ax, RUN_SETTINGS, y=0.55)

    return fig


def main():
    """Parse CLI args, render the graphic, and save it as a PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="output PNG path (default: %(default)s)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="output resolution (default: %(default)s)")
    args = parser.parse_args()

    fig = build_figure()
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output} ({CANVAS_WIDTH:.0f}x{CANVAS_HEIGHT:.0f} @ "
          f"{args.dpi} dpi)")


if __name__ == "__main__":
    main()
