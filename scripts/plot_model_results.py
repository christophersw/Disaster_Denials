# scripts/plot_model_results.py
"""
Title: scripts/plot_model_results.py — Chart generation for PDA model KPIs and outcomes.
Description:
    Computes fresh KPIs for the PDA decision models by calling their
    public API functions, then renders nine publication-ready PNG charts to
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
        fig6_lay_effects.png        — lay-reader effects: Model 1 political &
                                      need factors as probability-point effects
                                      on the denial rate (estimate + range).
        fig7_m4_forest.png          — forest plot: Model 4 odds ratios ordered
                                      approval → denial, with per-row p-values,
                                      a 95% CI key/caption, and direction arrows.
        fig8_m4_curves.png          — Model 4 out-of-fold ROC curve and F1-vs-
                                      threshold curve (state-grouped CV); PR-AUC and
                                      prevalence baseline reported in the caption.
        fig9_m2_pr_auc.png          — standalone GBT (State Data Only) PR-AUC:
                                      full vs no-political bars with Δ ± 95% CI and
                                      the no-skill prevalence baseline.

    Run:
        .venv/bin/python -m scripts.plot_model_results [--db data/pda.db]

    Model 1's variational fit takes ~2-3 minutes; the full script takes
    approximately 4-6 minutes end-to-end.

Changelog:
    2026-06-28  Initial version.
    2026-06-29  Add fig6 lay-reader effects helper + chart.
    2026-06-30  fig7: order approval → denial, label rows with p-values + add
                direction arrows, drop the "__missing" indicator rows.
    2026-06-30  fig7: drop "simple logit" from the title; add a 95% CI legend key
                and caption clarifying what each band is.
    2026-06-30  Add fig8: Model 4 out-of-fold ROC + F1-vs-threshold curves
                (PR-AUC reported in the caption).
    2026-06-30  Rename Model 2 → "GBT — State Data Only" and Model 3 →
                "GBT — State + County Data" in figure titles/labels (fig1, fig2,
                fig4, fig5). Add fig9: standalone GBT state-data PR-AUC ablation.
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
from sklearn.metrics import average_precision_score, roc_auc_score

from pda.modeling import (
    assemble, evaluation, model1_logit, model2_gbm, model3_county,
    model4_simple_logit,
)

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
# fig6 — lay-reader effects (Model 1) constants
# ---------------------------------------------------------------------------

# Selected factors for the lay-reader chart. Each entry maps a Model 1 design
# feature to its plain-language label, the contrast its bar represents (+1 SD
# for continuous predictors, which M1 z-scores; "0→1" for binary), and the
# display category that drives colour. The three Political rows are always shown
# (their "no clear effect" is itself a finding); a feature missing from the
# fitted odds-ratio table is skipped with a warning rather than crashing.
_M1_LAY_FACTORS = [
    {"feature": "total_cost_estimate",
     "label": "Total damage (disaster size)", "contrast": "+1 SD", "category": "Need"},
    {"feature": "num_affected_counties",
     "label": "Number of counties hit", "contrast": "+1 SD", "category": "Need"},
    {"feature": "ia_residences_total",
     "label": "Homes damaged", "contrast": "+1 SD", "category": "Need"},
    {"feature": "pa_statewide_per_capita",
     "label": "Damage per person", "contrast": "+1 SD", "category": "Need"},
    {"feature": "state_party_match",
     "label": "State shares president's party", "contrast": "0→1",
     "category": "Political"},
    {"feature": "governor_vs_president",
     "label": "Governor's party vs. president's", "contrast": "0→1",
     "category": "Political"},
    {"feature": "dmg_weighted_mean_pres_margin",
     "label": "How much worst-hit areas favored the president", "contrast": "+1 SD",
     "category": "Political"},
]

# fig6 encodes certainty, not category: one colour for a clear effect and grey
# for "no clear effect" (range crosses zero). (The `category` field on each
# factor is retained as metadata but no longer drives colour.)
_M1_LAY_EFFECT = _ORANGE
_M1_LAY_UNCLEAR = "#9aa0a6"


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


def _logistic(x):
    """Standard logistic (sigmoid) function.

    Args:
        x (float | np.ndarray): log-odds value(s).
    Returns:
        float | np.ndarray: probabilities in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))


def _build_m1_lay_effects(vb_table):
    """Translate the M1 odds-ratio table into lay probability-point effects.

    For each selected factor, converts its odds ratio (per +1 SD for continuous
    predictors, 0->1 for binary) into the change in predicted denial probability
    at the model baseline:

        effect = logistic(b0 + log(OR)) - logistic(b0)

    where b0 is the fitted intercept log-odds (log of the table's Intercept-row
    odds ratio). The plausible range uses the OR confidence bounds; a factor
    whose range straddles zero is flagged ``unclear`` ("no clear effect").
    Factors absent from the table are skipped with a printed warning.

    Args:
        vb_table (DataFrame): output of model1_logit.odds_ratio_table on the VB
            fit; indexed by feature name with odds_ratio / ci_low / ci_high
            columns and an 'Intercept' (or 'const') row.
    Returns:
        tuple(DataFrame, float): (effects_df, baseline_p). effects_df is indexed
            by lay label with columns effect_pts, range_low_pts, range_high_pts,
            unclear (bool), category, contrast. baseline_p is logistic(b0).
    """
    intercept_row = "Intercept" if "Intercept" in vb_table.index else "const"
    if intercept_row not in vb_table.index:
        raise ValueError(
            "odds-ratio table has no Intercept/const row; cannot anchor baseline"
        )
    b0 = float(np.log(vb_table.loc[intercept_row, "odds_ratio"]))
    baseline_p = float(_logistic(b0))

    rows = []
    labels = []
    for spec in _M1_LAY_FACTORS:
        feat = spec["feature"]
        if feat not in vb_table.index:
            print(f"  [fig6] skipping '{feat}': not in M1 odds-ratio table")
            continue
        beta = np.log(float(vb_table.loc[feat, "odds_ratio"]))
        beta_lo = np.log(float(vb_table.loc[feat, "ci_low"]))
        beta_hi = np.log(float(vb_table.loc[feat, "ci_high"]))
        eff = (_logistic(b0 + beta) - baseline_p) * 100.0
        lo = (_logistic(b0 + beta_lo) - baseline_p) * 100.0
        hi = (_logistic(b0 + beta_hi) - baseline_p) * 100.0
        if lo > hi:                      # defensive: keep range ordered
            lo, hi = hi, lo
        rows.append({
            "effect_pts": eff,
            "range_low_pts": lo,
            "range_high_pts": hi,
            "unclear": bool(lo < 0.0 < hi),
            "category": spec["category"],
            "contrast": spec["contrast"],
        })
        labels.append(spec["label"])

    effects_df = pd.DataFrame(
        rows, index=labels,
        columns=["effect_pts", "range_low_pts", "range_high_pts",
                 "unclear", "category", "contrast"],
    )
    return effects_df, baseline_p


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

    labels = ["GBT\n(State Data Only)", "GBT\n(State + County Data)"]
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
    ax.set_title("GBT — State Data Only: full vs no-political PR-AUC",
                 fontsize=11, fontweight="bold")
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
        "GBT — State Data Only\n"
        "need & severity dominate; political features rank low",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_feature_importance.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
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
        "GBT — State + County Data\n"
        "'one strong county' > 'most counties' (but weak overall)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max_val * 1.35)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_stronghold.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_fig6_lay_effects(effects_df, baseline_p, out_dir):
    """Diverging horizontal effects chart for a lay reader (Approach A2).

    Each selected factor is one row: a dot at its probability-point effect on the
    denial rate and a line for the plausible range. A vertical zero-line marks
    "no change from baseline"; factors whose range crosses it render grey ("no
    clear effect") while factors with a clear effect share a single colour. The
    baseline denial rate is annotated. Continuous factors carry a small "+1 SD"
    note.

    Args:
        effects_df (DataFrame): from _build_m1_lay_effects; indexed by lay label
            with effect_pts / range_low_pts / range_high_pts / unclear /
            category / contrast columns.
        baseline_p (float): baseline denial probability, logistic(b0).
        out_dir (str): directory for saving the PNG.
    Returns:
        None. Saves fig6_lay_effects.png to out_dir.
    """
    n = len(effects_df)
    if n == 0:
        print("  [fig6] no factors to plot; skipping fig6_lay_effects.png")
        return
    # Sort ascending so the strongest "less likely denied" effects sit at the
    # bottom and the chart reads like a number line (negative left, positive
    # right).
    df = effects_df.sort_values("effect_pts", ascending=True)
    y_pos = np.arange(n, dtype=float)
    fig, ax = plt.subplots(figsize=(10, max(4.5, n * 0.62 + 1.8)))

    for y, (label, row) in zip(y_pos, df.iterrows()):
        colour = _M1_LAY_UNCLEAR if row["unclear"] else _M1_LAY_EFFECT
        # plausible-range line + end caps
        ax.plot([row["range_low_pts"], row["range_high_pts"]], [y, y],
                color=colour, linewidth=2.4, alpha=0.55, zorder=3,
                solid_capstyle="round")
        for x in (row["range_low_pts"], row["range_high_pts"]):
            ax.plot([x, x], [y - 0.10, y + 0.10], color=colour,
                    linewidth=1.4, alpha=0.7, zorder=3)
        # estimate dot
        ax.plot(row["effect_pts"], y, "o", color=colour, markersize=10,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        # "+1 SD" unit note for continuous contrasts
        if row["contrast"] == "+1 SD":
            ax.text(row["effect_pts"], y + 0.28, "+1 SD", ha="center",
                    va="bottom", fontsize=7.5, color="#888")

    ax.axvline(0.0, color="black", linewidth=1.6, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index, fontsize=10)
    ax.set_xlabel("Change in denial rate (percentage points)", fontsize=10.5)

    span = max(
        float(np.abs(df[["range_low_pts", "range_high_pts"]].to_numpy()).max()),
        1.0,
    )
    ax.set_xlim(-span * 1.28, span * 1.28)
    ax.set_ylim(-1.4, n - 0.4)

    # direction hints under the axis
    ax.annotate("◀ less likely denied", xy=(-span * 1.22, -1.0),
                fontsize=9, color=_BLUE, annotation_clip=False, va="center")
    ax.annotate("more likely denied ▶", xy=(span * 0.35, -1.0),
                fontsize=9, color=_RED, annotation_clip=False, va="center")

    # baseline callout — inside the top-left, clear of the title band above the
    # axes and of the (empty) left side of the top rows
    ax.text(0.015, 0.975,
            f"Baseline: about {round(baseline_p * 100)} in 100 requests denied",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f3f4f6", ec="#ccc"))

    # Build the legend from the colours ACTUALLY drawn so it never promises a
    # swatch with no member: an "Effect" key only if some factor has a clear
    # effect, and the grey key only if some factor is unclear.
    any_clear = bool((~df["unclear"]).any())
    any_unclear = bool(df["unclear"].any())
    legend_handles = []
    if any_clear:
        legend_handles.append(mpatches.Patch(
            facecolor=_M1_LAY_EFFECT, label="Effect"))
    if any_unclear:
        legend_handles.append(mpatches.Patch(
            facecolor=_M1_LAY_UNCLEAR, label="No clear effect (range crosses zero)"))
    # Legend outside the axes (right side), so it cannot cover the in-plot
    # direction hints at the bottom.
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8.5,
                  loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)

    # NOTE: this takeaway title encodes the CURRENT finding (political alignment
    # shows no clear effect); it is not recomputed from effects_df. Revisit it if
    # a future refit makes a political factor's effect clearly non-zero.
    ax.set_title(
        "What moves a FEMA denial — need and severity drive the decision;\n"
        "political alignment shows no clear effect",
        fontsize=12.5, fontweight="bold", pad=12,
    )
    ax.grid(axis="x", alpha=0.25)

    fig.text(
        0.5, -0.02,
        "Each bar is that factor's solo effect from the baseline (logistic link, so effects don't simply add).\n"
        "Ranges are Model 1 variational-Bayes credible intervals — indicative, narrower than a full posterior.",
        ha="center", fontsize=7.5, color="#888",
    )

    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_lay_effects.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Model 4 political features, highlighted in the fig7 forest plot.
_M4_POLITICAL = {"governor_vs_president", "share_affected_counties_pres_won"}
_M4_MUTED = "#888888"

# Coefficients with a Wald p-value below this get a filled marker and a bold
# label in fig7; weaker ones get an open marker and a greyed label.
_M4_SIG_LEVEL = 0.05

# Suffix of the median-imputation indicator columns excluded from fig7 (they are
# data-hygiene flags, not substantive predictors of the decision).
_M4_MISSING_SUFFIX = "__missing"

# Direction-arrow colours under the fig7 axis: the outcome is "denied", so an
# odds ratio below 1 pushes toward approval and above 1 pushes toward denial.
_DIR_APPROVE = "#2E8B57"   # green — OR < 1
_DIR_DENY = _RED           # red   — OR > 1


def plot_fig7_m4_forest(table, out_dir):
    """Horizontal forest plot: Model 4 odds ratios, ordered approval → denial.

    One row per retained feature, ordered by the odds ratio so the column reads
    top-to-bottom as a gradient from the most approval-leaning predictor (lowest
    OR, top) to the most denial-leaning predictor (highest OR, bottom). The
    outcome is "denied", so OR < 1 lowers the odds of denial (pushes toward
    approval) and OR > 1 raises them (pushes toward denial); paired arrows under
    the axis label that direction. Each row is annotated with its Wald p-value: a
    filled marker plus a bold label marks a significant coefficient (p < 0.05) and
    an open marker plus a greyed label a non-significant one. The median-imputation
    "__missing" indicator columns are dropped (data-hygiene flags, not substantive
    predictors). Political features use the accent colour, others muted grey. LOG
    x-axis with a reference line at OR = 1 (no effect); CI whiskers are clipped to a
    sane visible range so the log scale does not explode.

    Args:
        table (DataFrame): model4_simple_logit.odds_ratio_table output — indexed by
            feature, with columns odds_ratio, ci_low, ci_high, p_value, std_coef.
        out_dir (str): directory for saving the PNG.
    Returns:
        None. Saves fig7_m4_forest.png to out_dir.
    """
    # Drop the missingness indicators, then order by signed effect: sorting
    # std_coef descending puts the highest OR (most denial-leaning) at index 0,
    # which matplotlib draws at the bottom, so the column reads approval → denial
    # top-to-bottom and the markers fall on a top-left → bottom-right diagonal.
    substantive = table.loc[
        [f for f in table.index if not f.endswith(_M4_MISSING_SUFFIX)]
    ]
    ranked = substantive.sort_values("std_coef", ascending=False)
    n = len(ranked)
    y_pos = np.arange(n, dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.45 + 2.0)))
    for i, (feat, row) in enumerate(ranked.iterrows()):
        color = _BLUE if feat in _M4_POLITICAL else _M4_MUTED
        significant = bool(row["p_value"] < _M4_SIG_LEVEL)
        lo = max(row["ci_low"], 0.05)
        hi = min(row["ci_high"], 20.0)
        # Filled marker = significant; open marker = not significant.
        marker_kwargs = {"color": color, "markersize": 7, "zorder": 5}
        if not significant:
            marker_kwargs.update(markerfacecolor="none", markeredgewidth=1.5)
        ax.plot(row["odds_ratio"], y_pos[i], "o", **marker_kwargs)
        ax.plot([lo, hi], [y_pos[i], y_pos[i]], color=color, linewidth=1.4, zorder=4)
        ax.plot([lo, lo], [y_pos[i] - 0.08, y_pos[i] + 0.08], color=color, lw=1.2)
        ax.plot([hi, hi], [y_pos[i] - 0.08, y_pos[i] + 0.08], color=color, lw=1.2)

        # p-value label in a fixed column just outside the right spine.
        p = float(row["p_value"])
        p_text = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.annotate(
            p_text, xy=(1.015, y_pos[i]), xycoords=("axes fraction", "data"),
            va="center", ha="left", fontsize=7.5,
            color="#333" if significant else "#999",
            fontweight="bold" if significant else "normal",
        )

    ax.axvline(1.0, color="black", linewidth=1.5, linestyle="-", zorder=2)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranked.index, fontsize=9)
    ax.set_ylim(-1.4, n - 0.3)   # headroom below row 0 for the direction arrows
    ax.set_xlabel("Odds Ratio (log scale)", fontsize=10)
    ax.set_title(
        "Logistic Regression Model — what drives a PDA denial "
        "(ordered approval → denial)",
        fontsize=11, fontweight="bold",
    )

    # Column header for the p-value labels, aligned just outside the right spine.
    ax.annotate(
        "p-value", xy=(1.015, 1.005), xycoords="axes fraction",
        va="bottom", ha="left", fontsize=8, fontweight="bold", color="#333",
    )

    # Direction arrows along the bottom: which way each side of OR = 1 leans.
    ax.annotate(
        "◀ more likely APPROVED", xy=(0.02, -0.9),
        xycoords=("axes fraction", "data"), ha="left", va="center",
        fontsize=9, fontweight="bold", color=_DIR_APPROVE,
    )
    ax.annotate(
        "more likely DENIED ▶", xy=(0.98, -0.9),
        xycoords=("axes fraction", "data"), ha="right", va="center",
        fontsize=9, fontweight="bold", color=_DIR_DENY,
    )

    # Geometry key: what the dot and the band mean for every feature row.
    ci_handle = mlines.Line2D(
        [], [], color="#444", marker="o", linewidth=1.6, markersize=7,
        label="Odds ratio (dot) · 95% CI (band)",
    )
    political_handle = mlines.Line2D(
        [], [], color=_BLUE, marker="o", linewidth=1.4, markersize=7,
        label="Political feature",
    )
    other_handle = mlines.Line2D(
        [], [], color=_M4_MUTED, marker="o", linewidth=1.4, markersize=7,
        label="Need / request / structural",
    )
    sig_handle = mlines.Line2D(
        [], [], color="#555", marker="o", linewidth=0, markersize=7,
        label="Significant (p < 0.05)",
    )
    nonsig_handle = mlines.Line2D(
        [], [], color="#555", marker="o", linewidth=0, markersize=7,
        markerfacecolor="none", markeredgewidth=1.5,
        label="Not significant (p ≥ 0.05)",
    )
    ax.legend(
        handles=[ci_handle, political_handle, other_handle,
                 sig_handle, nonsig_handle],
        fontsize=8, loc="upper right", framealpha=0.9,
    )
    ax.grid(axis="x", alpha=0.25)

    # Caption spelling out the band for every feature and the OR = 1 nuance.
    fig.text(
        0.5, -0.03,
        "For each feature: the dot is the odds ratio and the horizontal band is its "
        "95% Wald confidence interval.\n"
        "A band that crosses the OR = 1 line (no effect) is not statistically "
        "distinguishable from no effect.",
        ha="center", fontsize=8, color="#666",
    )

    fig.tight_layout()
    path = os.path.join(out_dir, "fig7_m4_forest.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_fig8_m4_curves(y_true, y_proba, out_dir):
    """Two-panel discrimination chart for Model 4: ROC curve and F1-vs-threshold.

    Drawn from pooled grouped out-of-fold predictions (state-grouped CV), so it
    reflects held-out performance rather than in-sample fit. Left panel: ROC curve
    with the chance diagonal and ROC-AUC annotated. Right panel: F1 as a function
    of the decision threshold, marking the best achievable F1 (and its threshold)
    and the F1 at the default 0.50 cutoff — the imbalance-honest view, since F1 is
    precision/recall-based (unlike ROC-AUC) and the default threshold is usually
    suboptimal under a rare outcome. PR-AUC (average precision) and the no-skill
    prevalence baseline are reported in the caption. ROC-AUC, PR-AUC, and F1 are
    computed with the same metrics/folds as cv_fit_quality, so they match the
    reported numbers.

    Args:
        y_true (array): int 0/1 denial labels for the out-of-fold rows.
        y_proba (array): held-out P(denied) aligned to y_true.
        out_dir (str): directory for saving the PNG.
    Returns:
        None. Saves fig8_m4_curves.png to out_dir.
    """
    from sklearn.metrics import (
        average_precision_score, f1_score, precision_recall_curve, roc_curve,
    )

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    finite = np.isfinite(y_proba)            # defensive: drop any uncovered row
    y_true, y_proba = y_true[finite], y_proba[finite]
    prevalence = float(y_true.mean())

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    # F1 at every threshold breakpoint. precision_recall_curve drops the final
    # prec/rec pair (it has no threshold); guard the 0/0 where prec+rec == 0.
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    prec_t, rec_t = precision[:-1], recall[:-1]
    f1_curve = np.divide(
        2 * prec_t * rec_t, prec_t + rec_t,
        out=np.zeros_like(prec_t), where=(prec_t + rec_t) > 0,
    )
    best = int(f1_curve.argmax())
    f1_best, thr_best = float(f1_curve[best]), float(thresholds[best])
    f1_half = float(f1_score(y_true, (y_proba >= 0.5).astype(int)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel (a): ROC curve ----------------------------------------------
    ax = axes[0]
    ax.plot(fpr, tpr, color=_BLUE, linewidth=2.3, zorder=4,
            label=f"Logistic Regression (ROC-AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.3,
            zorder=2, label="Chance (0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.set_ylabel("True positive rate (recall)", fontsize=10)
    ax.set_title("ROC curve", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    # ---- Panel (b): F1 vs decision threshold -------------------------------
    ax = axes[1]
    ax.plot(thresholds, f1_curve, color=_ORANGE, linewidth=2.3, zorder=4,
            label="F1 vs threshold")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.1, zorder=2)
    ax.plot([thr_best], [f1_best], "o", color=_RED, markersize=9, zorder=6,
            label=f"Best F1 = {f1_best:.3f} @ t = {thr_best:.2f}")
    ax.plot([0.5], [f1_half], "s", color="#444", markersize=8, zorder=6,
            label=f"F1 @ default t = 0.50: {f1_half:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Decision threshold  (classify denied if P ≥ t)", fontsize=10)
    ax.set_ylabel("F1 score (denied class)", fontsize=10)
    ax.set_title("F1 vs decision threshold", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Logistic Regression Model — out-of-fold discrimination (state-grouped CV)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.02,
        f"PR-AUC (average precision) = {pr_auc:.3f}  vs.  {prevalence:.3f} no-skill "
        f"baseline (prevalence).\n"
        f"F1 is precision/recall-based, so — unlike ROC-AUC — it reflects the "
        f"{prevalence:.0%} class imbalance; the default t = 0.50 is usually "
        f"suboptimal under imbalance.",
        ha="center", fontsize=8, color="#666",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig8_m4_curves.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_fig9_m2_pr_auc(abl2, prevalence, out_dir):
    """Standalone bar chart: GBT (State Data Only) PR-AUC, full vs no-political.

    Isolates the GBT state-data political ablation — previously shown only beside
    the county model in fig2 — into a single panel: two out-of-fold PR-AUC bars
    (the full model with the state political block, and the reduced model with
    that block removed), annotated with the Δ and its bootstrap 95% CI. A dashed
    no-skill baseline at the class prevalence anchors how far both bars sit above
    chance, so the takeaway reads off the chart: the model predicts denial well,
    yet the two bars are indistinguishable — state political features add ~no
    predictive lift.

    Args:
        abl2 (dict): output of model2_gbm.political_ablation; keys
            full_pr_auc, reduced_pr_auc, delta, delta_ci.
        prevalence (float): denial rate on the ablation's valid (non-null-group)
            rows, drawn as the no-skill PR-AUC baseline.
        out_dir (str): directory for saving the PNG.
    Returns:
        None. Saves fig9_m2_pr_auc.png to out_dir.
    """
    full, reduced = abl2["full_pr_auc"], abl2["reduced_pr_auc"]
    delta = abl2["delta"]
    lo, hi = abl2["delta_ci"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = ["Full model\n(+ state political)", "No political\nfeatures"]
    vals = [full, reduced]
    bars = ax.bar(labels, vals, color=[_BLUE, _ORANGE], alpha=0.9,
                  width=0.5, zorder=3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    # No-skill baseline: both bars tower over chance yet match each other — that
    # contrast is the finding, not the (near-identical) bar heights themselves.
    ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(1.45, prevalence + 0.006, f"no-skill baseline ({prevalence:.0%})",
            ha="right", va="bottom", fontsize=8.5, color="gray")

    ax.annotate(
        f"Adding state political features:\nΔ PR-AUC = {delta:+.3f}   "
        f"95% CI [{lo:.3f}, {hi:.3f}]\n(interval straddles 0 — no reliable lift)",
        xy=(0.5, max(vals) + 0.05), xycoords=("axes fraction", "data"),
        ha="center", fontsize=9.5, color="dimgray",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9),
    )

    ax.set_ylim(0, max(vals) + 0.18)
    ax.set_ylabel("PR-AUC (out-of-fold, state-grouped CV)", fontsize=11)
    ax.set_title(
        "GBT — State Data Only: political features add ~no predictive lift",
        fontsize=12.5, fontweight="bold", pad=12,
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig9_m2_pr_auc.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    """Compute fresh model KPIs and render nine PNG charts to docs/models/figures/.

    Assembles the feature matrix once then calls each model's public API
    functions in the same order as scripts/run_all_models.py. Prints key
    metrics to stdout as each step completes. Takes approximately 4-6
    minutes end-to-end (Model 1 VB fit dominates).

    Args: none (reads --db from CLI, default data/pda.db).
    Returns: None.
    Side effects: creates/overwrites nine PNGs in docs/models/figures/.
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

    # ---- fig6 lay-reader effects (reuses the VB table above) ---------------
    effects_df, baseline_p = _build_m1_lay_effects(vb_table)
    estimand_rate = float(frame["denied"].mean())
    print(f"  fig6: baseline_p={baseline_p:.3f} "
          f"(M1 estimand denial rate={estimand_rate:.3f}); "
          f"{len(effects_df)} lay factors")
    if abs(baseline_p - estimand_rate) > 0.05:
        print(f"  [fig6 WARNING] model baseline {baseline_p:.3f} diverges from "
              f"observed estimand rate {estimand_rate:.3f} by >5 points")

    # ---- Model 2: political ablation + SHAP --------------------------------
    print("\nModel 2: political ablation (CV ~30 s) ...")
    abl2 = model2_gbm.political_ablation(X, y, groups)
    lo2, hi2 = abl2["delta_ci"]
    # Denial rate on the ablation's valid (non-null-group) rows — the no-skill
    # PR-AUC baseline drawn in fig9 (matches political_ablation's row filter).
    m2_groups = groups if isinstance(groups, pd.Series) else pd.Series(groups)
    m2_prevalence = float(y_arr[m2_groups.notna().values].mean())
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

    # ---- Model 4: simple single-level logit (full population) --------------
    print("\nModel 4: simple logit (full population) ...")
    m4_frame = model4_simple_logit.prepare_simple_logit_frame(X, y)
    m4_result = model4_simple_logit.fit_simple_logit(m4_frame)
    m4_table = model4_simple_logit.odds_ratio_table(m4_result)
    print(f"  pseudo-R^2={m4_result.pseudo_r2:.3f}  {len(m4_table)} features; "
          f"dropped (separation)={m4_result.dropped_features or 'none'}")

    print("  Computing grouped-CV out-of-fold predictions (fig8) ...")
    m4_y, m4_proba = model4_simple_logit.cv_oof_predictions(m4_frame, groups)
    print(f"  ROC-AUC={roc_auc_score(m4_y, m4_proba):.3f}  "
          f"PR-AUC={average_precision_score(m4_y, m4_proba):.3f}  "
          f"(baseline prevalence={m4_y.mean():.3f})")

    # ---- Generate charts ---------------------------------------------------
    print("\nGenerating charts ...")
    plot_fig1(abl2, abl3, m3_roc_auc, FIGURES_DIR)
    plot_fig2(abl2, abl3, FIGURES_DIR)
    plot_fig3(combined_m1, FIGURES_DIR)
    plot_fig4(imp2, FIGURES_DIR)
    plot_fig5(comp, FIGURES_DIR)
    plot_fig6_lay_effects(effects_df, baseline_p, FIGURES_DIR)
    plot_fig7_m4_forest(m4_table, FIGURES_DIR)
    plot_fig8_m4_curves(m4_y, m4_proba, FIGURES_DIR)
    plot_fig9_m2_pr_auc(abl2, m2_prevalence, FIGURES_DIR)

    print(f"\nAll 9 charts saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
