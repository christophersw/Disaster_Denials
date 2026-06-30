# scripts/run_all_models.py
"""
Title: scripts/run_all_models.py — Reproducible run-all entry point for PDA models.
Description:
    Single entry point that confirms political-flag coding via the EDA gate,
    assembles the feature matrix once, and fits/reports all three PDA decision-
    prediction models in sequence:

      * EDA gate  — denial-rate crosstabs confirming what =1 means for each flag.
      * Model 1   — hierarchical logit: political odds-ratio table (VB) plus
                    pooled-Logit Wald cross-check for the political features.
      * Model 2   — gradient boosting: predictability (ROC/PR-AUC) + political-
                    block ablation PR-AUC lift + top SHAP/permutation features.
      * Model 3   — county-composition gradient boosting: M2→M3 PR-AUC delta +
                    stronghold-feature importance comparison.

    Run:
        .venv/bin/python -m scripts.run_all_models [--db data/pda.db]

    Model 1's variational fit takes ~1–2 minutes; Models 2/3 take ~30 s each.

Changelog:
    2026-06-28  Initial version.
"""

import argparse

import numpy as np
import pandas as pd

from pda.modeling import (
    assemble, model1_logit, model2_gbm, model3_county, model4_simple_logit,
)
from scripts import verify_political_coding

# Political feature names highlighted in the Model 1 cross-check table.
# presidential_election_year / midterm_election_year are intentionally absent:
# they are constant within a calendar year, collinear with the C(year) random
# intercept, and dropped by prepare_logit_frame before fitting.
_POLITICAL_FEATURES = [
    "state_party_match", "governor_vs_president", "governor_vs_state_vote",
    "state_margin", "state_dem_share", "swing_state",
    "share_affected_counties_pres_won", "max_pres_margin_affected",
    "dmg_weighted_mean_pres_margin", "pres_won_most_damaged_county",
    "pres_margin_dispersion", "months_to_next_election",
]

# Caveat banner printed alongside the political cross-check table.
_VB_CAVEAT = (
    "NOTE: VB credible intervals are ANTI-CONSERVATIVE. Mean-field variational\n"
    "Bayes underestimates posterior variance, so vb_ci_* bounds are narrower\n"
    "than a proper posterior and must be read as INDICATIVE, not definitive\n"
    "significance bounds. The wald_ci_* columns are a pooled-Logit cross-check\n"
    "on the SAME design (Wald intervals when MLE is non-singular, otherwise the\n"
    "penalised fallback): where VB and the pooled fit AGREE that an interval\n"
    "excludes OR=1, the effect is more credibly bounded away from no-effect."
)


def _political_cross_check(frame, vb_table):
    """Join VB credible intervals with pooled-Logit Wald intervals for political rows.

    Fits the pooled Logit on the same M1 design and aligns its Wald-based odds-
    ratio intervals next to the VB credible intervals for the political features,
    so a reader can compare whether both methods agree on which effects are bounded
    away from OR=1. Detects the penalised fallback (when the MLE Hessian is
    singular) so the caller can warn that pooled CIs collapse to the point estimate.

    Args:
        frame (DataFrame): prepared M1 design, output of model1_logit.prepare_logit_frame.
        vb_table (DataFrame): output of model1_logit.odds_ratio_table on the VB/mixed fit.
    Returns:
        (combined, pooled_is_mle): combined is a DataFrame indexed by political
        feature name with columns vb_or, vb_ci_low, vb_ci_high, wald_or,
        wald_ci_low, wald_ci_high; pooled_is_mle is True when the pooled fit
        produced real Wald SEs rather than the penalised point-only fallback.
    """
    pooled = model1_logit.fit_pooled_logit(frame)
    wald_table = model1_logit.odds_ratio_table(pooled)
    rows = [f for f in _POLITICAL_FEATURES if f in vb_table.index]
    wald = wald_table.reindex(rows)
    combined = pd.DataFrame({
        "vb_or": vb_table.loc[rows, "odds_ratio"],
        "vb_ci_low": vb_table.loc[rows, "ci_low"],
        "vb_ci_high": vb_table.loc[rows, "ci_high"],
        "wald_or": wald["odds_ratio"],
        "wald_ci_low": wald["ci_low"],
        "wald_ci_high": wald["ci_high"],
    })
    bounds = wald_table[["ci_low", "ci_high"]].to_numpy()
    pooled_is_mle = bool(
        np.isfinite(bounds).all()
        and not np.isclose(wald_table["ci_low"], wald_table["ci_high"]).any()
    )
    return combined, pooled_is_mle


def main():
    """Run the EDA gate then fit and print results for Models 1, 2, and 3.

    Assembles the feature matrix once and passes it to each model. Prints
    all results to stdout. Takes 2–4 minutes end-to-end (Model 1 dominates).

    Args: none (reads --db from CLI, default data/pda.db).
    Returns: None.
    Side effects: prints gate crosstabs, three model summaries to stdout.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default="data/pda.db",
        help="Path to the SQLite database (default: data/pda.db)",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # EDA gate: confirm political-flag coding before interpreting any OR sign
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("### EDA GATE: political flag coding (confirm =1 meaning) ###")
    print("=" * 70)
    # verify_political_coding.main() reads --db from sys.argv; since both
    # parsers define --db, it sees the same value we parsed above.
    verify_political_coding.main()

    # -------------------------------------------------------------------------
    # Assemble features once; all three models share X, y, groups
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Assembling feature matrix from:", args.db)
    print("=" * 70)
    X, y, groups = assemble.assemble_features(args.db)
    n_denied = int(sum(y))
    print(f"Corpus: {len(X)} rows × {X.shape[1]} columns | "
          f"denied={n_denied} ({100 * n_denied / len(y):.1f}%)")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)

    # -------------------------------------------------------------------------
    # Model 1: hierarchical logit — effect size (spec §7 M1)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("### MODEL 1: hierarchical logit (effect size) ###")
    print("=" * 70)
    frame = model1_logit.prepare_logit_frame(X, y, groups)
    result = model1_logit.fit_mixed_logit(frame)
    table = model1_logit.odds_ratio_table(result)

    fit_path = (
        "mixed variational (BinomialBayesMixedGLM.fit_vb)"
        if hasattr(result, "fe_mean") else "pooled Logit fallback"
    )
    print(f"Fit on {len(frame)} state/DC rows "
          f"({int(frame['denied'].sum())} denials) via {fit_path}.")

    print("\n--- Full odds-ratio table (VB credible intervals) ---")
    print(table.round(3).to_string())

    combined, pooled_is_mle = _political_cross_check(frame, table)
    print("\n--- Political features: VB vs pooled-Wald cross-check ---")
    print(_VB_CAVEAT)
    if not pooled_is_mle:
        print(
            "\n(pooled MLE Logit was singular on this design; cross-check shows "
            "the L1-PENALISED pooled fit. Political features at OR=1.000 with "
            "identical CI were shrunk to zero by the penalty, not estimated as "
            "null with a tight interval.)"
        )
    print()
    print(combined.round(3).to_string())

    # -------------------------------------------------------------------------
    # Model 2: gradient boosting — predictability + political lift (spec §7 M2)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("### MODEL 2: gradient boosting (predictability + political lift) ###")
    print("=" * 70)
    ablation2 = model2_gbm.political_ablation(X, y, groups)
    lo2, hi2 = ablation2["delta_ci"]
    print(f"  full   PR-AUC={ablation2['full_pr_auc']:.3f}  "
          f"ROC-AUC={ablation2['full_roc_auc']:.3f}")
    print(f"  no-pol PR-AUC={ablation2['reduced_pr_auc']:.3f}  "
          f"ROC-AUC={ablation2['reduced_roc_auc']:.3f}")
    print(f"  political PR-AUC lift = {ablation2['delta']:.3f}  "
          f"(95% CI {lo2:.3f}..{hi2:.3f})")

    print("\n--- Top features by SHAP / permutation importance ---")
    est2 = model2_gbm.build_estimator()
    Xs2 = model2_gbm._slice(X, include_political=True)
    y_arr = np.asarray(y)
    est2.fit(Xs2, y_arr)
    print(model2_gbm.shap_summary(est2, Xs2, y_arr).head(15).round(4).to_string())

    # -------------------------------------------------------------------------
    # Model 3: county-composition lift (spec §7 M3)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("### MODEL 3: county-composition lift ###")
    print("=" * 70)
    ablation3 = model3_county.county_ablation(X, y, groups)
    lo3, hi3 = ablation3["delta_ci"]
    print(f"  M2 (state only)    PR-AUC={ablation3['m2_pr_auc']:.3f}")
    print(f"  M3 (state+county)  PR-AUC={ablation3['m3_pr_auc']:.3f}")
    print(f"  county lift = {ablation3['delta']:.3f}  "
          f"(95% CI {lo3:.3f}..{hi3:.3f})")

    print("\n--- Stronghold signal comparison ---")
    est3 = model2_gbm.build_estimator()
    Xs3 = model3_county._slice(X, model3_county.feature_columns())
    y_arr3 = np.asarray(y)
    est3.fit(Xs3, y_arr3)
    comp = model3_county.stronghold_comparison(est3, Xs3, y_arr3)
    print(f"  max_pres_margin_affected  (one strong county): "
          f"{comp['one_county_max_margin']:.4f}")
    print(f"  share_affected_counties_pres_won (most counties): "
          f"{comp['most_counties_share_won']:.4f}")

    # -------------------------------------------------------------------------
    # Model 4: simple single-level logit — full-population feature ranking
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("### MODEL 4: simple logit (full-population feature ranking) ###")
    print("=" * 70)
    m4_frame = model4_simple_logit.prepare_simple_logit_frame(X, y)
    m4_result = model4_simple_logit.fit_simple_logit(m4_frame)
    m4_table = model4_simple_logit.odds_ratio_table(m4_result)
    m4_scores = model4_simple_logit.cv_fit_quality(m4_frame, groups)
    print(f"Fit on {len(m4_frame)} reports "
          f"({int(m4_frame['denied'].sum())} denials) via plain MLE.")
    if m4_result.dropped_features:
        print(f"  Separation: dropped {m4_result.dropped_features} "
              f"(complete zero-cells, unidentifiable).")
    else:
        print("  Separation: none — all curated features retained.")
    print(f"  McFadden pseudo-R^2={m4_result.pseudo_r2:.3f}  "
          f"ROC-AUC={m4_scores['roc_auc']:.3f}  PR-AUC={m4_scores['pr_auc']:.3f}")
    m4_display = m4_table.copy()
    m4_display.insert(
        0, "pol",
        ["*" if name in {"governor_vs_president",
                         "share_affected_counties_pres_won"} else ""
         for name in m4_display.index],
    )
    print("\n--- Odds ratios ranked by |std coef| (* = political) ---")
    print(m4_display.round(3).to_string())

    print("\n" + "=" * 70)
    print("### ALL MODELS COMPLETE ###")
    print("=" * 70)


if __name__ == "__main__":
    main()
