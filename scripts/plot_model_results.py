# scripts/plot_model_results.py
"""
Title: scripts/plot_model_results.py — Chart generation for PDA model KPIs and outcomes.
Description:
    Computes fresh KPIs for the three PDA decision models by calling their
    public API functions, then renders five publication-ready PNG charts to
    docs/models/figures/. Mirrors the computation pattern in
    scripts/run_all_models.py.

    Charts generated:
        fig1_performance.png        — grouped bar: ROC-AUC and PR-AUC for
                                      Models 2 (full) and 3.
        fig2_political_lift.png     — political-lift headline: M2 ablation
                                      bars + delta-with-CI panel for M2 & M3.
        fig3_m1_forest.png          — forest plot: M1 political odds ratios
                                      (Variational Bayes vs pooled Logit).
        fig4_feature_importance.png — top-12 M2 SHAP importances by category.
        fig5_stronghold.png         — stronghold signal comparison (two M3
                                      county-composition features).

    Run:
        .venv/bin/python -m scripts.plot_model_results [--db data/pda.db]

    Model 1's variational fit takes ~2-3 minutes; the full script takes
    approximately 4-6 minutes end-to-end.

Changelog:
    2026-06-28  Initial version.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pda.modeling import assemble, evaluation, model1_logit, model2_gbm, model3_county

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Output directory for all PNG files.
FIGURES_DIR = "docs/models/figures"

# Figure resolution in dots per inch.
DPI = 150

# Political feature names to show in the M1 forest plot.  These mirror the
# _POLITICAL_FEATURES list in scripts/run_all_models.py; election-year flags
# are excluded because they are collinear with the C(year) random intercept.
_POLITICAL_FEATURES = [
    "state_party_match",
    "governor_vs_president",
    "governor_vs_state_vote",
    "state_margin",
    "state_dem_share",
    "months_to_next_election",
    "swing_state",
    "share_affected_counties_pres_won",
    "max_pres_margin_affected",
    "dmg_weighted_mean_pres_margin",
    "pres_won_most_damaged_county",
    "pres_margin_dispersion",
]

# Feature-category sets for fig4 colour-coding.
_NEED_SET = set(assemble.NEED + assemble.IA_DEMOGRAPHIC_BLOCK)
_POLITICAL_SET = set(
    assemble.POLITICAL_STATE + assemble.POLITICAL_COUNTY + assemble.ENRICHMENT
)
# Anything not in the above two groups falls into Request/Jurisdiction/Other.

# Colour palette for the three feature categories (fig4).
_CAT_COLOURS = {
    "Need / Severity": "#E07B39",
    "Political / Partisan": "#C0392B",
    "Request / Jurisdiction / Other": "#2E86AB",
}

# Colour pair used across multiple charts (blue = model 2, orange = model 3 / reduced).
_BLUE = "#2E86AB"
_ORANGE = "#E07B39"
_RED = "#C0392B"


# ---------------------------------------------------------------------------
# Data-computation helpers
# ---------------------------------------------------------------------------

def _compute_m3_roc_auc(X, y, groups):
    """Compute Model 3 out-of-fold ROC-AUC on the valid (non-null-group) rows.

    Filters tribal-nation/territory rows with a null groups value exactly as
    county_ablation does, then runs one oof_predictions pass with the M3
    feature set and returns the ROC-AUC.

    Args:
        X (DataFrame): assembled feature matrix from assemble_features.
        y (Series|array): int denied target aligned to X.
        groups (Series): state_abbr Series aligned to X.
    Returns:
        float: out-of-fold ROC-AUC for the M3 feature set.
    """
    groups_s = groups if isinstance(groups, pd.Series) else pd.Series(groups)
    valid_mask = groups_s.notna().values
    X_valid = X.iloc[valid_mask]
    y_valid = np.asarray(y)[valid_mask]
    groups_valid = groups_s.values[valid_mask]
    Xs3 = model3_county._slice(X_valid, model3_county.feature_columns())
    m3_proba = evaluation.oof_predictions(
        model2_gbm.build_estimator(), Xs3, y_valid, groups_valid
    )
    return float(roc_auc_score(y_valid, m3_proba))


def _build_m1_combined(vb_table, pooled_table):
    """Align VB and pooled odds-ratio rows for the political features.

    Reindexes both tables to _POLITICAL_FEATURES and detects L1-zeroed pooled
    points: when the MLE Logit Hessian is singular the fallback uses L1
    regularisation, which collapses every standard error to zero. Features
    whose regularised coefficient is exactly zero get OR=1 with a collapsed
    CI; those are the L1-zeroed rows.

    Args:
        vb_table (DataFrame): from model1_logit.odds_ratio_table on the VB fit.
        pooled_table (DataFrame): from model1_logit.odds_ratio_table on the
            pooled Logit fit.
    Returns:
        DataFrame indexed by political feature name with columns vb_or,
        vb_ci_low, vb_ci_high, wald_or, wald_ci_low, wald_ci_high, l1_zeroed.
    """
    rows = [f for f in _POLITICAL_FEATURES if f in vb_table.index]
    pooled = pooled_table.reindex(rows)
    combined = pd.DataFrame({
        "vb_or":      vb_table.loc[rows, "odds_ratio"],
        "vb_ci_low":  vb_table.loc[rows, "ci_low"],
        "vb_ci_high": vb_table.loc[rows, "ci_high"],
        "wald_or":      pooled["odds_ratio"],
        "wald_ci_low":  pooled["ci_low"],
        "wald_ci_high": pooled["ci_high"],
    })
    # L1-zeroed: collapsed pooled CI (ci_low ≈ ci_high) AND OR ≈ 1.0.
    pooled_collapsed = np.isclose(
        combined["wald_ci_low"].to_numpy(dtype=float),
        combined["wald_ci_high"].to_numpy(dtype=float),
    )
    combined["l1_zeroed"] = pooled_collapsed & np.isclose(
        combined["wald_or"].to_numpy(dtype=float), 1.0
    )
    return combined


def _feature_category(name):
    """Return the display category label for a feature name (for fig4 colouring).

    Args:
        name (str): feature column name.
    Returns:
        str: one of the _CAT_COLOURS keys.
    """
    if name in _NEED_SET:
        return "Need / Severity"
    if name in _POLITICAL_SET:
        return "Political / Partisan"
    return "Request / Jurisdiction / Other"


# ---------------------------------------------------------------------------
# Chart-drawing functions
# ---------------------------------------------------------------------------

def plot_fig1(abl2, abl3, m3_roc_auc, out_dir):
    """Grouped bar chart: ROC-AUC and PR-AUC for Model 2 (full) and Model 3.

    Adds a dashed horizontal reference line at PR-AUC = 0.08 labelled
    "PR-AUC baseline (8% prevalence)". Annotates each bar's numeric value.

    Args:
        abl2 (dict): output of model2_gbm.political_ablation; keys
            full_roc_auc, full_pr_auc.
        abl3 (dict): output of model3_county.county_ablation; key m3_pr_auc.
        m3_roc_auc (float): out-of-fold ROC-AUC for Model 3.
        out_dir (str): directory for saving the PNG.
    Returns:
        None. Saves fig1_performance.png to out_dir.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Model 2\n(state features)", "Model 3\n(+ county composition)"]
    roc_vals = [abl2["full_roc_auc"], m3_roc_auc]
    pr_vals  = [abl2["full_pr_auc"],  abl3["m3_pr_auc"]]
    x = np.arange(len(labels))
    width = 0.32

    bars_roc = ax.bar(x - width / 2, roc_vals, width,
                      label="ROC-AUC", color=_BLUE, alpha=0.9, zorder=3)
    bars_pr  = ax.bar(x + width / 2, pr_vals,  width,
                      label="PR-AUC",  color=_ORANGE, alpha=0.9, zorder=3)

    for bar in list(bars_roc) + list(bars_pr):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.axhline(
        0.08, color="gray", linestyle="--", linewidth=1.3, zorder=2,
        label="PR-AUC baseline (8% prevalence)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Predictive performance — denial is highly predictable",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_performance.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_fig2(abl2, abl3, out_dir):
    """Two-panel headline political-lift chart.

    Panel (a): M2 full vs no-political PR-AUC as two bars, delta annotated.
    Panel (b): PR-AUC delta with 95% CI for M2 and M3 as points with
    horizontal error bars; vertical reference line at delta=0. Annotations
    make clear both CIs straddle zero.

    Args:
        abl2 (dict): output of model2_gbm.political_ablation; keys
            full_pr_auc, reduced_pr_auc, delta, delta_ci.
        abl3 (dict): output of model3_county.county_ablation; keys
            delta, delta_ci.
        out_dir (str): directory for saving the PNG.
    Returns:
        None. Saves fig2_political_lift.png to out_dir.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel (a): M2 ablation bars ----------------------------------------
    ax = axes[0]
    bar_labels = ["Full model\n(all features)", "No political\nfeatures"]
    vals    = [abl2["full_pr_auc"], abl2["reduced_pr_auc"]]
    colours = [_BLUE, _ORANGE]
    bars = ax.bar(bar_labels, vals, color=colours, alpha=0.9, width=0.45, zorder=3)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    delta2 = abl2["delta"]
    lo2, hi2 = abl2["delta_ci"]
    ax.annotate(
        f"Δ = {delta2:+.3f}\n95% CI [{lo2:.3f}, {hi2:.3f}]",
        xy=(0.5, max(vals) + 0.03),
        xycoords=("axes fraction", "data"),
        ha="center", fontsize=9, color="dimgray",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.85),
    )
    ax.set_ylim(0, min(1.0, max(vals) + 0.12))
    ax.set_ylabel("PR-AUC (out-of-fold)", fontsize=11)
    ax.set_title("Model 2: full vs no-political PR-AUC", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ---- Panel (b): Delta-with-CI for M2 and M3 -----------------------------
    ax = axes[1]
    delta3 = abl3["delta"]
    lo3, hi3 = abl3["delta_ci"]
    row_labels = ["M2 state-alignment\nlift", "M3 county-composition\nlift"]
    deltas = [delta2, delta3]
    cis    = [(lo2, hi2), (lo3, hi3)]
    y_pos  = [1.0, 0.0]
    pt_colours = [_BLUE, _ORANGE]

    for y, delta, (lo, hi), c, lbl in zip(
        y_pos, deltas, cis, pt_colours, row_labels
    ):
        ax.errorbar(
            delta, y,
            xerr=[[delta - lo], [hi - delta]],
            fmt="o", color=c, markersize=9,
            capsize=6, capthick=1.6, elinewidth=1.6, zorder=4,
        )
        side = 1 if (hi - lo) < 0.12 else -1
        xoff = max(abs(hi), abs(lo)) * 0.06
        ax.text(
            delta + xoff, y + 0.07,
            f"Δ = {delta:+.3f}\n[{lo:.3f}, {hi:.3f}]",
            va="bottom", ha="left", fontsize=8.5, color=c,
        )

    ax.axvline(0, color="black", linewidth=1.4, linestyle="-", zorder=3, label="Δ = 0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("PR-AUC lift (full minus reduced)", fontsize=10)
    ax.set_title(
        "Political Δ with 95% CI\n(both CIs straddle zero)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    all_extents = [abs(lo2), abs(hi2), abs(lo3), abs(hi3), abs(delta2), abs(delta3)]
    margin = max(all_extents) * 1.5 + 0.04
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-0.5, 1.8)

    fig.suptitle(
        "Political signal adds ~no predictive lift",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_political_lift.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_fig3(combined, out_dir):
    """Horizontal forest plot: M1 political odds ratios, VB vs pooled Logit.

    Sorted by VB odds ratio ascending (smallest at bottom). LOG x-axis.
    Reference line at OR=1. Pooled L1-zeroed points (OR shrunk to 1.0 by
    the L1 penalty) are shown as open diamond markers.

    Args:
        combined (DataFrame): from _build_m1_combined; indexed by feature
            name; columns vb_or, vb_ci_low, vb_ci_high, wald_or,
            wald_ci_low, wald_ci_high, l1_zeroed.
        out_dir (str): directory for saving the PNG.
    Returns:
        None. Saves fig3_m1_forest.png to out_dir.
    """
    combined_sorted = combined.sort_values("vb_or")
    n = len(combined_sorted)
    y_pos  = np.arange(n, dtype=float)
    offset = 0.20   # vertical spread between VB and pooled rows per feature

    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.60 + 1.8)))

    for i, (feat, row) in enumerate(combined_sorted.iterrows()):
        y_vb = y_pos[i] + offset
        y_wd = y_pos[i] - offset

        # Clip CIs to a sane visible range so log-scale doesn't explode.
        vb_lo  = max(row["vb_ci_low"],  0.05)
        vb_hi  = min(row["vb_ci_high"], 20.0)
        wd_lo  = max(row["wald_ci_low"],  0.05)
        wd_hi  = min(row["wald_ci_high"], 20.0)

        # Variational Bayes: filled circle.
        ax.plot(row["vb_or"], y_vb, "o", color=_BLUE, markersize=7, zorder=5)
        ax.plot(
            [vb_lo, vb_hi], [y_vb, y_vb],
            color=_BLUE, linewidth=1.3, zorder=4,
        )
        ax.plot([vb_lo, vb_lo], [y_vb - 0.06, y_vb + 0.06], color=_BLUE, lw=1.2)
        ax.plot([vb_hi, vb_hi], [y_vb - 0.06, y_vb + 0.06], color=_BLUE, lw=1.2)

        # Pooled Logit: filled or open diamond depending on L1-zeroed status.
        if row["l1_zeroed"]:
            ax.plot(row["wald_or"], y_wd, "D", color=_RED, markersize=7,
                    markerfacecolor="none", markeredgewidth=1.5, zorder=5)
        else:
            ax.plot(row["wald_or"], y_wd, "D", color=_RED, markersize=7, zorder=5)
        ax.plot(
            [wd_lo, wd_hi], [y_wd, y_wd],
            color=_RED, linewidth=1.3, zorder=4,
        )
        ax.plot([wd_lo, wd_lo], [y_wd - 0.06, y_wd + 0.06], color=_RED, lw=1.2)
        ax.plot([wd_hi, wd_hi], [y_wd - 0.06, y_wd + 0.06], color=_RED, lw=1.2)

    # Reference line at OR = 1.
    ax.axvline(1.0, color="black", linewidth=1.5, linestyle="-", zorder=2)

    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined_sorted.index, fontsize=9)
    ax.set_xlabel("Odds Ratio (log scale)", fontsize=10)
    ax.set_title(
        "Model 1 — political odds ratios are mostly not robust (CIs cross OR=1)",
        fontsize=11, fontweight="bold",
    )

    # Proxy-artist legend.
    vb_handle = mlines.Line2D(
        [], [], color=_BLUE, marker="o", linewidth=1.3, markersize=7,
        label="Variational Bayes (95% credible interval)",
    )
    wd_handle = mlines.Line2D(
        [], [], color=_RED, marker="D", linewidth=1.3, markersize=7,
        label="Pooled Logit / conservative (Wald CI)",
    )
    l1_handle = mlines.Line2D(
        [], [], color=_RED, marker="D", linewidth=0, markersize=7,
        markerfacecolor="none", markeredgewidth=1.5,
        label="L1-zeroed (pooled OR=1, penalty-shrunk)",
    )
    ax.legend(handles=[vb_handle, wd_handle, l1_handle], fontsize=8.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_m1_forest.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_fig4(imp_df, out_dir):
    """Horizontal bar chart: top-12 M2 features by mean |SHAP|, category-coloured.

    Features are colour-coded by category (Need/Severity, Political/Partisan,
    Request/Jurisdiction/Other). The largest-importance feature is at the top.

    Args:
        imp_df (DataFrame): from model2_gbm.shap_summary with an 'importance'
            column, sorted descending. Must have at least 12 rows.
        out_dir (str): directory for saving the PNG.
    Returns:
        None. Saves fig4_feature_importance.png to out_dir.
    """
    top12 = imp_df.head(12).copy()
    top12 = top12.iloc[::-1]   # reverse so largest is at top in barh

    categories = [_feature_category(f) for f in top12.index]
    colours    = [_CAT_COLOURS[c] for c in categories]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top12.index, top12["importance"],
                   color=colours, alpha=0.9, edgecolor="white", height=0.65)

    max_imp = top12["importance"].max()
    for bar, val in zip(bars, top12["importance"]):
        ax.text(
            val + max_imp * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=8,
        )

    # Category legend via proxy patches.
    legend_handles = [
        mpatches.Patch(facecolor=c, label=label, alpha=0.9)
        for label, c in _CAT_COLOURS.items()
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(
        "Model 2 — need & severity dominate; political features rank low",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_feature_importance.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_fig5(comp, out_dir):
    """Horizontal bar comparing the two stronghold feature importances.

    Annotates each bar's numeric value. The 'one strong county' bar is expected
    to be larger than the 'most counties' bar.

    Args:
        comp (dict): from model3_county.stronghold_comparison; keys
            one_county_max_margin and most_counties_share_won.
        out_dir (str): directory for saving the PNG.
    Returns:
        None. Saves fig5_stronghold.png to out_dir.
    """
    labels = [
        "max_pres_margin_affected\n(one strongly-favoring county)",
        "share_affected_counties_pres_won\n(most counties favored him)",
    ]
    values  = [comp["one_county_max_margin"], comp["most_counties_share_won"]]
    colours = [_BLUE, _ORANGE]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(labels, values, color=colours, alpha=0.9, height=0.45)

    max_val = max(values) if max(values) > 0 else 1.0
    for bar, val in zip(bars, values):
        ax.text(
            val + max_val * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=11, fontweight="bold",
        )

    ax.set_xlabel("Feature importance (mean |SHAP|)", fontsize=10)
    ax.set_title(
        "Model 3 — 'one strong county' > 'most counties' (but weak overall)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max_val * 1.35)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_stronghold.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Compute fresh model KPIs and render five PNG charts to docs/models/figures/.

    Assembles the feature matrix once then calls each model's public API
    functions in the same order as scripts/run_all_models.py. Prints key
    metrics to stdout as each step completes. Takes approximately 4-6
    minutes end-to-end (Model 1 VB fit dominates).

    Args: none (reads --db from CLI, default data/pda.db).
    Returns: None.
    Side effects: creates/overwrites five PNGs in docs/models/figures/.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default="data/pda.db",
        help="Path to the SQLite database (default: data/pda.db)",
    )
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---- Assemble feature matrix -------------------------------------------
    print("Assembling feature matrix from:", args.db)
    X, y, groups = assemble.assemble_features(args.db)
    y_arr = np.asarray(y)
    print(f"  {len(X)} rows x {X.shape[1]} columns | "
          f"denied={int(y_arr.sum())} ({100 * y_arr.mean():.1f}%)")

    # ---- Model 1: VB + pooled odds-ratio tables ----------------------------
    print("\nModel 1: prepare logit frame + VB fit (slow ~2-3 min) ...")
    frame = model1_logit.prepare_logit_frame(X, y, groups)
    vb_result = model1_logit.fit_mixed_logit(frame)
    vb_table  = model1_logit.odds_ratio_table(vb_result)
    print("  VB fit complete. Fitting pooled Logit ...")
    pooled_result = model1_logit.fit_pooled_logit(frame)
    pooled_table  = model1_logit.odds_ratio_table(pooled_result)
    combined_m1   = _build_m1_combined(vb_table, pooled_table)
    n_l1 = int(combined_m1["l1_zeroed"].sum())
    print(f"  Political features in table: {len(combined_m1)}  "
          f"L1-zeroed: {n_l1}")

    # ---- Model 2: political ablation + SHAP --------------------------------
    print("\nModel 2: political ablation (CV ~30 s) ...")
    abl2 = model2_gbm.political_ablation(X, y, groups)
    lo2, hi2 = abl2["delta_ci"]
    print(f"  full   PR-AUC={abl2['full_pr_auc']:.3f}  "
          f"ROC-AUC={abl2['full_roc_auc']:.3f}")
    print(f"  reduced PR-AUC={abl2['reduced_pr_auc']:.3f}")
    print(f"  M2 political lift D={abl2['delta']:+.3f}  "
          f"CI=[{lo2:.3f}, {hi2:.3f}]")

    print("  Computing SHAP importance on full M2 model ...")
    Xs2  = model2_gbm._slice(X, True)
    est2 = model2_gbm.build_estimator()
    est2.fit(Xs2, y_arr)
    imp2 = model2_gbm.shap_summary(est2, Xs2, y_arr)
    print(f"  Top 3 features: {list(imp2.head(3).index)}")
    print(f"  Top 3 importances: "
          f"{[round(v, 3) for v in imp2['importance'].head(3).tolist()]}")

    # ---- Model 3: county ablation + M3 ROC-AUC + stronghold ---------------
    print("\nModel 3: county ablation (CV ~30 s) ...")
    abl3 = model3_county.county_ablation(X, y, groups)
    lo3, hi3 = abl3["delta_ci"]
    print(f"  M2 PR-AUC={abl3['m2_pr_auc']:.3f}  "
          f"M3 PR-AUC={abl3['m3_pr_auc']:.3f}")
    print(f"  county lift D={abl3['delta']:+.3f}  "
          f"CI=[{lo3:.3f}, {hi3:.3f}]")

    print("  Computing M3 ROC-AUC (additional oof pass ~30 s) ...")
    m3_roc_auc = _compute_m3_roc_auc(X, y, groups)
    print(f"  M3 ROC-AUC={m3_roc_auc:.3f}")

    print("  Computing stronghold comparison ...")
    Xm3  = model3_county._slice(X, model3_county.feature_columns())
    est3 = model2_gbm.build_estimator()
    est3.fit(Xm3, y_arr)
    comp = model3_county.stronghold_comparison(est3, Xm3, y_arr)
    print(f"  one_county_max_margin={comp['one_county_max_margin']:.4f}  "
          f"most_counties_share_won={comp['most_counties_share_won']:.4f}")

    # ---- Generate charts ---------------------------------------------------
    print("\nGenerating charts ...")
    plot_fig1(abl2, abl3, m3_roc_auc, FIGURES_DIR)
    plot_fig2(abl2, abl3, FIGURES_DIR)
    plot_fig3(combined_m1, FIGURES_DIR)
    plot_fig4(imp2, FIGURES_DIR)
    plot_fig5(comp, FIGURES_DIR)

    print(f"\nAll 5 charts saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
