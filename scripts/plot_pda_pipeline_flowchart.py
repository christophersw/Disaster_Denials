# scripts/plot_pda_pipeline_flowchart.py
"""
Title: PDA Parsing Pipeline Flowchart
Description: Renders a high-level, slide-ready (16:9) flow chart of the PDA
    parsing process, from downloading the FEMA Preliminary Damage Assessment
    report PDFs through LLM extraction to the normalized SQLite store
    (data/pda.db). Four numbered stages run left to right — Download, PDF
    Library, Parse to Schema (LLM), SQLite Database — and a "zoom-in" panel
    breaks open the parse stage (PDF document block -> Claude Opus 4.8 tool use
    -> Pydantic re-validation -> flatten -> transactional write) and notes the
    two run modes (serial vs Batches API at 50% cost). Pure diagram: it draws
    only matplotlib patches and text, reads no data, and writes a PNG. Offline,
    deterministic, idempotent.
Changelog:
    2026-06-15  Initial version.
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DEFAULT_OUTPUT = "docs/figures/pda_parsing_pipeline.png"

# Canvas is 16 wide x 9 tall so one data unit is square at figsize (16, 9).
CANVAS_WIDTH = 16.0
CANVAS_HEIGHT = 9.0

# Stage palette: semantic colours read on a white slide. Files/data in warm
# tones, the script/LLM work in cool tones, the database in green (the goal).
COLOR_DOWNLOAD = "#3b6fb0"   # acquire (script)
COLOR_PDFS = "#e0922f"       # files on disk (data artifact)
COLOR_PARSE = "#6a4c93"      # LLM extraction (the heart of the pipeline)
COLOR_DB = "#2f8f5b"         # SQLite store (the destination)
COLOR_SOURCE = "#5a6b7b"     # upstream web source (not a numbered stage)

ARROW_COLOR = "#444444"
INK = "#23252b"              # near-black body text
MUTED = "#6b7280"            # secondary grey text
PANEL_FILL = "#f4f6fa"       # detail-panel background
CHIP_FILL = "#ffffff"        # mini-step chips inside the detail panel

BOX_STYLE = "round,pad=0,rounding_size=0.16"


def add_rounded_box(ax, center_x, center_y, width, height, facecolor,
                    edgecolor="white", linewidth=1.6, zorder=2, shadow=True):
    """Draw a rounded rectangle (optionally with a soft drop shadow).

    Args:
        ax (matplotlib.axes.Axes): target axes.
        center_x (float), center_y (float): box centre in data units.
        width (float), height (float): box size in data units.
        facecolor (str): fill colour.
        edgecolor (str): border colour.
        linewidth (float): border width.
        zorder (int): draw order for the box (shadow sits just beneath).
        shadow (bool): whether to draw a soft offset shadow.
    Returns:
        None. Adds patches to ax in place.
    """
    left = center_x - width / 2
    bottom = center_y - height / 2
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (left + 0.05, bottom - 0.07), width, height,
            boxstyle=BOX_STYLE, facecolor="black", edgecolor="none",
            alpha=0.13, zorder=zorder - 1))
    ax.add_patch(FancyBboxPatch(
        (left, bottom), width, height, boxstyle=BOX_STYLE,
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        zorder=zorder))


def add_stage(ax, center_x, center_y, width, height, color, badge, eyebrow,
              title, description, code_chip):
    """Draw one numbered pipeline stage box with its labels.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        center_x (float), center_y (float): stage centre in data units.
        width (float), height (float): stage box size.
        color (str): stage fill colour.
        badge (str): the stage number shown in the corner circle.
        eyebrow (str): small all-caps category label.
        title (str): bold stage title.
        description (str): one- or two-line plain-language description.
        code_chip (str): monospace file/path shown in the bottom chip.
    Returns:
        None. Adds patches and text to ax in place.
    """
    add_rounded_box(ax, center_x, center_y, width, height, color)

    top = center_y + height / 2
    # Stage-number badge in the top-left corner (kept small so it clears the
    # eyebrow label).
    badge_x, badge_y = center_x - width / 2 + 0.30, top - 0.30
    ax.add_patch(plt.Circle((badge_x, badge_y), 0.17, facecolor="white",
                            edgecolor=color, linewidth=1.4, zorder=3))
    ax.text(badge_x, badge_y, badge, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=color, zorder=4)

    ax.text(center_x + 0.18, top - 0.30, eyebrow, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", alpha=0.85,
            zorder=4)
    ax.text(center_x, center_y + 0.34, title, ha="center", va="center",
            fontsize=15.5, fontweight="bold", color="white", zorder=4)
    ax.text(center_x, center_y - 0.16, description, ha="center", va="center",
            fontsize=10, color="white", alpha=0.95, linespacing=1.35,
            zorder=4)

    # Monospace file/path chip pinned to the bottom of the box.
    chip_w, chip_h = width - 0.30, 0.36
    chip_cy = center_y - height / 2 + 0.30
    ax.add_patch(FancyBboxPatch(
        (center_x - chip_w / 2, chip_cy - chip_h / 2), chip_w, chip_h,
        boxstyle="round,pad=0,rounding_size=0.08", facecolor="black",
        edgecolor="none", alpha=0.20, zorder=4))
    ax.text(center_x, chip_cy, code_chip, ha="center", va="center",
            fontsize=8.6, family="monospace", color="white", zorder=5)


def add_flow_arrow(ax, x_start, x_end, y, label=None):
    """Draw a left-to-right flow arrow between two stages, with a caption.

    Args:
        ax (matplotlib.axes.Axes): target axes.
        x_start (float), x_end (float): arrow endpoints (x) in data units.
        y (float): arrow height in data units.
        label (str | None): optional two-line caption above the arrow.
    Returns:
        None. Adds the arrow (and caption) to ax in place.
    """
    ax.add_patch(FancyArrowPatch(
        (x_start, y), (x_end, y), arrowstyle="-|>", mutation_scale=22,
        linewidth=2.6, color=ARROW_COLOR, shrinkA=0, shrinkB=0, zorder=2))
    if label:
        ax.text((x_start + x_end) / 2, y + 0.42, label, ha="center",
                va="center", fontsize=8.8, color=MUTED, style="italic",
                linespacing=1.25, zorder=2)


def add_mini_chip(ax, center_x, center_y, width, height, text, accent):
    """Draw a small light chip (a sub-step inside the detail panel).

    Args:
        ax (matplotlib.axes.Axes): target axes.
        center_x (float), center_y (float): chip centre in data units.
        width (float), height (float): chip size.
        text (str): chip label (may contain newlines).
        accent (str): colour for the chip's top accent bar and border.
    Returns:
        None. Adds patches and text to ax in place.
    """
    add_rounded_box(ax, center_x, center_y, width, height, CHIP_FILL,
                    edgecolor=accent, linewidth=1.4, shadow=False)
    # Thin accent bar along the top edge of the chip.
    ax.add_patch(FancyBboxPatch(
        (center_x - width / 2 + 0.08, center_y + height / 2 - 0.14),
        width - 0.16, 0.08, boxstyle="round,pad=0,rounding_size=0.03",
        facecolor=accent, edgecolor="none", zorder=3))
    ax.text(center_x, center_y - 0.04, text, ha="center", va="center",
            fontsize=8.9, color=INK, linespacing=1.3, zorder=3)


def build_figure():
    """Build the full 16:9 flowchart figure.

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
    ax.text(0.55, 8.50, "FEMA PDA Parsing Pipeline", ha="left", va="center",
            fontsize=25, fontweight="bold", color=INK)
    ax.text(0.55, 7.92, "From downloaded report PDFs to a clean, searchable "
            "database", ha="left", va="center", fontsize=13, color=MUTED)

    # ---- Main stage row --------------------------------------------------
    stage_y = 5.30
    stage_w, stage_h = 2.90, 1.85
    centers_x = [2.0, 6.0, 10.0, 14.0]

    # Upstream web source feeding the Download stage (not a numbered stage).
    add_rounded_box(ax, centers_x[0], 7.00, 2.70, 0.74, COLOR_SOURCE,
                    shadow=False)
    ax.text(centers_x[0], 7.16, "SOURCE", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white", alpha=0.85)
    ax.text(centers_x[0], 6.86, "FEMA report website", ha="center",
            va="center", fontsize=9.5, color="white")
    ax.add_patch(FancyArrowPatch(
        (centers_x[0], 7.00 - 0.37), (centers_x[0], stage_y + stage_h / 2),
        arrowstyle="-|>", mutation_scale=18, linewidth=2.2,
        color=ARROW_COLOR, shrinkA=0, shrinkB=0, zorder=2))

    add_stage(ax, centers_x[0], stage_y, stage_w, stage_h, COLOR_DOWNLOAD,
              "1", "COLLECT", "Download Reports",
              "Gather every FEMA report\nPDF from their website",
              "download_pda_reports.py")
    add_stage(ax, centers_x[1], stage_y, stage_w, stage_h, COLOR_PDFS,
              "2", "ORGANIZE", "PDF Library",
              "Every PDF, filed by\nreport type and year",
              "data/pdfs/<Type>/<Year>/*.pdf")
    add_stage(ax, centers_x[2], stage_y, stage_w, stage_h, COLOR_PARSE,
              "3", "READ WITH AI", "Pull Out the Facts",
              "A Claude AI model reads each\nPDF and pulls out the facts",
              "parse_pda_reports.py")
    add_stage(ax, centers_x[3], stage_y, stage_w, stage_h, COLOR_DB,
              "4", "STORE", "Final Database",
              "A clean, searchable record\nof every report and county",
              "data/pda.db")

    # Arrows between the four stages, captioned with what flows across.
    half = stage_w / 2
    add_flow_arrow(ax, centers_x[0] + half, centers_x[1] - half, stage_y,
                   "downloaded\nPDF files")
    add_flow_arrow(ax, centers_x[1] + half, centers_x[2] - half, stage_y,
                   "one PDF\nat a time")
    add_flow_arrow(ax, centers_x[2] + half, centers_x[3] - half, stage_y,
                   "checked\ndata rows")

    # ---- Zoom-in detail panel for stage 3 (Parse) ------------------------
    panel_left, panel_right = 0.55, 15.45
    panel_top, panel_bottom = 3.45, 1.05
    panel_cx = (panel_left + panel_right) / 2
    panel_cy = (panel_top + panel_bottom) / 2
    add_rounded_box(ax, panel_cx, panel_cy, panel_right - panel_left,
                    panel_top - panel_bottom, PANEL_FILL,
                    edgecolor=COLOR_PARSE, linewidth=1.6, shadow=False)

    # Dotted "zoom" connector from the Parse stage down into the panel.
    ax.add_patch(FancyArrowPatch(
        (centers_x[2], stage_y - stage_h / 2), (centers_x[2], panel_top),
        arrowstyle="-|>", mutation_scale=16, linewidth=1.8,
        linestyle=(0, (2, 2)), color=COLOR_PARSE, shrinkA=2, shrinkB=2,
        zorder=2))

    ax.text(panel_left + 0.35, panel_top - 0.30,
            "A closer look at step 3  —  turning each PDF into clean data",
            ha="left", va="center", fontsize=12, fontweight="bold",
            color=COLOR_PARSE)

    chip_y = 2.05
    chip_w, chip_h = 2.55, 0.96
    chip_centers = [2.05, 5.05, 8.05, 11.05, 14.05]
    chip_texts = [
        "Hand the whole PDF\nto the AI model",
        "The AI reads the\nreport's text and tables",
        "It fills out a fixed\nset of data fields",
        "Check the values &\nnote where they came from",
        "Save as rows in\nthe database",
    ]
    for cx, text in zip(chip_centers, chip_texts):
        add_mini_chip(ax, cx, chip_y, chip_w, chip_h, text, COLOR_PARSE)
    for left_cx, right_cx in zip(chip_centers[:-1], chip_centers[1:]):
        ax.add_patch(FancyArrowPatch(
            (left_cx + chip_w / 2, chip_y), (right_cx - chip_w / 2, chip_y),
            arrowstyle="-|>", mutation_scale=15, linewidth=1.8,
            color=COLOR_PARSE, shrinkA=0, shrinkB=0, zorder=2))

    # Run-mode note along the bottom of the panel.
    ax.text(panel_cx, panel_bottom + 0.28,
            "Two ways to run  —  one report at a time, or all at once for "
            "half-price overnight processing",
            ha="center", va="center", fontsize=9.5, color=MUTED)

    # ---- Footer ----------------------------------------------------------
    ax.text(panel_cx, 0.45,
            "Resumable & idempotent at every stage: already-downloaded PDFs "
            "and already-parsed reports are skipped, so re-runs never repeat "
            "work.", ha="center", va="center", fontsize=9.5, color=MUTED,
            style="italic")

    return fig


def main():
    """Parse CLI args, render the flowchart, and save it as a PNG."""
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
