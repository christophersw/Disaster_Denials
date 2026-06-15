# scripts/confusion_heatmap.py
"""
Title: Confusion-Matrix Heatmap Helpers
Description: Shared rendering for the PDA party-match confusion heatmaps
    (governor-vs-president and county-vs-president). Provides a single-panel 2x2
    heatmap drawer and a multi-panel figure layout (one panel centred on top, the
    rest across the bottom) with row-normalised colouring, count + row-% cell
    annotations, and one shared colorbar. Data fetching, panel labelling, and the
    chart-specific titles/footnotes live in the per-chart scripts; this module
    only renders. Each panel is a dict with "label" (str), "matrix"
    (2x2 list[list[int]] of [match/differ][approved/denied] counts), and "n"
    (int total).
Changelog:
    2026-06-14  Initial version (extracted from plot_governor_match_confusion.py).
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Row shares are percentages, so the colour scale is fixed 0-100 for every panel.
PERCENT_NORMALIZER = Normalize(vmin=0, vmax=100)

# Above this row share a cell is dark enough to need white annotation text.
_WHITE_TEXT_THRESHOLD = 55


def draw_confusion_panel(ax, panel, colormap, row_labels, col_labels):
    """Draw one 2x2 confusion heatmap onto the given axes.

    Cells are coloured by each cell's share of its row (so each row sums to
    100%) and annotated with the raw count and that row percentage.

    Args:
        ax: the matplotlib Axes to draw on.
        panel (dict): one panel with "label", "matrix" (2x2 counts), and "n".
        colormap (str): name of the matplotlib colormap for cell colours.
        row_labels (tuple[str, str]): top/bottom row tick labels.
        col_labels (tuple[str, str]): left/right column tick labels.
    Returns:
        None.
    """
    matrix = panel["matrix"]

    # Row-normalised percentages; a row with no reports stays at 0%.
    row_percentages = []
    for row in matrix:
        row_total = sum(row)
        if row_total:
            row_percentages.append([value / row_total * 100 for value in row])
        else:
            row_percentages.append([0.0, 0.0])

    ax.imshow(row_percentages, cmap=colormap, norm=PERCENT_NORMALIZER, aspect="auto")

    ax.set_xticks([0, 1], list(col_labels))
    ax.set_yticks([0, 1], list(row_labels))
    ax.tick_params(length=0)
    ax.set_title(f"{panel['label']}\nn = {panel['n']}", fontsize=11)

    for row_index, row in enumerate(matrix):
        for col_index, count in enumerate(row):
            percentage = row_percentages[row_index][col_index]
            text_color = "white" if percentage >= _WHITE_TEXT_THRESHOLD else "#222222"
            ax.text(
                col_index, row_index,
                f"{count}\n{percentage:.1f}%",
                ha="center", va="center", fontsize=11, color=text_color,
            )


def build_confusion_figure(panels, output_path, *, suptitle, subtitle, footnote,
                           colorbar_label, colormap, row_labels,
                           col_labels=("Approved", "Denied")):
    """Lay out all panels (panels[0] centred on top, the rest across the bottom).

    Args:
        panels (list[dict]): the all-data panel first, then up to four more.
        output_path (str): file path for the rendered PNG.
        suptitle (str): figure title.
        subtitle (str): one-line orientation note under the title.
        footnote (str): exclusion/colour note along the bottom.
        colorbar_label (str): label for the shared colorbar.
        colormap (str): matplotlib colormap name for all cells.
        row_labels (tuple[str, str]): top/bottom row tick labels.
        col_labels (tuple[str, str]): column tick labels (default Approved/Denied).
    Returns:
        str — the output_path that was written.
    """
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        2, 4, left=0.06, right=0.9, top=0.84, bottom=0.08,
        hspace=0.55, wspace=0.45,
    )

    all_panel, remaining_panels = panels[0], panels[1:]

    # All-data panel centred on the top row (spanning the middle two columns).
    ax_all = fig.add_subplot(grid[0, 1:3])
    draw_confusion_panel(ax_all, all_panel, colormap, row_labels, col_labels)

    # One panel per column across the bottom row.
    for column, panel in enumerate(remaining_panels):
        ax = fig.add_subplot(grid[1, column])
        draw_confusion_panel(ax, panel, colormap, row_labels, col_labels)

    # Shared colourbar for the row-share scale.
    scalar_mappable = ScalarMappable(norm=PERCENT_NORMALIZER, cmap=colormap)
    scalar_mappable.set_array([])
    colorbar_axis = fig.add_axes((0.93, 0.1, 0.015, 0.72))
    colorbar = fig.colorbar(scalar_mappable, cax=colorbar_axis)
    colorbar.set_label(colorbar_label)

    fig.suptitle(suptitle, fontsize=16, y=0.95)
    fig.text(0.06, 0.96, subtitle, fontsize=10, color="dimgray")
    fig.text(0.06, 0.02, footnote, fontsize=9, color="gray")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
