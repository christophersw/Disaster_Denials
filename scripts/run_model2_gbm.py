# scripts/run_model2_gbm.py
"""
Title: scripts/run_model2_gbm.py — Fit and report Model 2.
Description: Reports CV ROC-AUC/PR-AUC, the political ablation delta with its
    bootstrap CI, and the SHAP importance ranking (spec §7, M2).
Changelog:
    2026-06-28  Initial version.
"""

import argparse

from pda.modeling import assemble, model2_gbm


def main():
    """Parse args, run M2 political ablation and SHAP summary, print results.

    Args: none (reads --db from command line).
    Returns: None. Prints metrics to stdout.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/pda.db")
    args = parser.parse_args()

    X, y, groups = assemble.assemble_features(args.db)
    ablation = model2_gbm.political_ablation(X, y, groups)
    print("Model 2 — predictability and political lift")
    print(f"  full   PR-AUC={ablation['full_pr_auc']:.3f} "
          f"ROC-AUC={ablation['full_roc_auc']:.3f}")
    print(f"  no-pol PR-AUC={ablation['reduced_pr_auc']:.3f} "
          f"ROC-AUC={ablation['reduced_roc_auc']:.3f}")
    lo, hi = ablation["delta_ci"]
    print(f"  political PR-AUC lift = {ablation['delta']:.3f} "
          f"(95% CI {lo:.3f}..{hi:.3f})")

    est = model2_gbm.build_estimator()
    Xs = model2_gbm._slice(X, include_political=True)
    est.fit(Xs, y)
    print("\nTop features by importance:")
    print(model2_gbm.shap_summary(est, Xs, y).head(15).round(4).to_string())


if __name__ == "__main__":
    main()
