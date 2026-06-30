# scripts/run_model4_simple_logit.py
"""
Title: scripts/run_model4_simple_logit.py — Fit and report Model 4.
Description: Assembles the full-population feature matrix, fits the simple
    single-level logit (MLE with the detect-and-drop separation policy), and prints
    the odds-ratio table ranked by |standardized coefficient| with the political
    rows marked, plus any features dropped for separation, the McFadden pseudo-R^2,
    and grouped-CV discrimination.
Changelog:
    2026-06-29  Initial version.
"""

import argparse

import pandas as pd

from pda.modeling import assemble, model4_simple_logit

# Marked with a '*' in the printed table (the study's political signal of interest).
_POLITICAL_FEATURES = {"governor_vs_president", "share_affected_counties_pres_won"}


def main():
    """Assemble features, fit Model 4, and print the ranked odds-ratio table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/pda.db")
    args = parser.parse_args()

    X, y, groups = assemble.assemble_features(args.db)
    frame = model4_simple_logit.prepare_simple_logit_frame(X, y)
    result = model4_simple_logit.fit_simple_logit(frame)
    table = model4_simple_logit.odds_ratio_table(result)
    scores = model4_simple_logit.cv_fit_quality(frame, groups)

    n = len(frame)
    n_denied = int(frame["denied"].sum())
    print(f"Model 4 (simple single-level logit) fit on {n} reports "
          f"({n_denied} denials, {100 * n_denied / n:.1f}%).")
    if result.dropped_features:
        print(f"Separation: dropped {len(result.dropped_features)} feature(s) as "
              f"complete zero-cells (unidentifiable): {result.dropped_features}")
    else:
        print("Separation: none — all curated features retained.")
    print(f"McFadden pseudo-R^2 = {result.pseudo_r2:.3f}   "
          f"LLR p = {result.llr_pvalue:.2e}")
    print(f"Grouped-CV discrimination: ROC-AUC={scores['roc_auc']:.3f}  "
          f"PR-AUC={scores['pr_auc']:.3f}  Brier={scores['brier']:.3f}")

    pd.set_option("display.width", 170)
    pd.set_option("display.max_rows", None)
    display = table.copy()
    display.insert(
        0, "pol",
        ["*" if name in _POLITICAL_FEATURES else "" for name in display.index],
    )
    print("\n=== Odds ratios, ranked by |standardized coef| "
          "(* marks political features) ===")
    print(display.round(3).to_string())


if __name__ == "__main__":
    main()
