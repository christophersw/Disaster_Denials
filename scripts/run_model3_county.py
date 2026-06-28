# scripts/run_model3_county.py
"""
Title: scripts/run_model3_county.py — Fit and report Model 3.
Description: Reports the M2->M3 ablation (does county composition add signal?)
    and the stronghold comparison (one county vs most counties), spec §7, M3.
Changelog:
    2026-06-28  Initial version.
"""

import argparse

import numpy as np

from pda.modeling import assemble, model2_gbm, model3_county


def main():
    """Entry point: run county ablation and stronghold comparison, then print.

    Args: none (reads --db from CLI, default data/pda.db).
    Returns: None.
    Side effects: prints M2 vs M3 PR-AUC, county lift delta + CI, and
        stronghold importance comparison to stdout.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/pda.db")
    args = parser.parse_args()

    X, y, groups = assemble.assemble_features(args.db)

    ablation = model3_county.county_ablation(X, y, groups)
    print("Model 3 — county-composition lift over state alignment")
    print(f"  M2 (state)        PR-AUC={ablation['m2_pr_auc']:.3f}")
    print(f"  M3 (state+county) PR-AUC={ablation['m3_pr_auc']:.3f}")
    lo, hi = ablation["delta_ci"]
    print(f"  county lift = {ablation['delta']:.3f} (95% CI {lo:.3f}..{hi:.3f})")

    # Fit a final estimator on the full feature set (no CV needed for importances).
    # NaN-group rows are fine to include here — we are not doing grouped CV.
    est = model2_gbm.build_estimator()
    Xs = model3_county._slice(X, model3_county.feature_columns())
    y_arr = np.asarray(y)
    est.fit(Xs, y_arr)

    comp = model3_county.stronghold_comparison(est, Xs, y_arr)
    print("\nStronghold signal:")
    print(f"  one county (max margin)   importance={comp['one_county_max_margin']:.4f}")
    print(f"  most counties (share won) importance={comp['most_counties_share_won']:.4f}")


if __name__ == "__main__":
    main()
