# scripts/run_model1_logit.py
"""
Title: scripts/run_model1_logit.py — Fit and report Model 1.
Description: Assembles features, fits the hierarchical logit on states + DC, and
    prints the odds-ratio table with the political rows highlighted (spec §7).
    Reports which fit path actually fired (mixed variational vs pooled fallback),
    and prints a pooled-Logit Wald cross-check alongside the VB credible intervals
    for the political features (VB intervals are anti-conservative — see below).
Changelog:
    2026-06-28  Initial version.
    2026-06-28  Drop year-collinear flags from the political block; add the
                pooled-Wald robustness cross-check and the VB caveat banner.
"""

import argparse

import numpy as np
import pandas as pd

from pda.modeling import assemble, model1_logit

# Feature rows highlighted as the political signal of interest (spec §7).
# presidential_election_year / midterm_election_year are intentionally NOT here:
# they are constant within a calendar year, perfectly collinear with the C(year)
# random intercept, and are dropped from the M1 fixed effects in
# prepare_logit_frame. months_to_next_election (varies within a year) is kept.
_POLITICAL_FEATURES = [
    "state_party_match", "governor_vs_president", "governor_vs_state_vote",
    "state_margin", "state_dem_share", "swing_state",
    "share_affected_counties_pres_won", "max_pres_margin_affected",
    "dmg_weighted_mean_pres_margin", "pres_won_most_damaged_county",
    "pres_margin_dispersion", "months_to_next_election",
]

# Caveat printed with the political table: VB credible intervals are too narrow.
_VB_CAVEAT = (
    "NOTE: VB credible intervals are ANTI-CONSERVATIVE. Mean-field variational\n"
    "Bayes underestimates posterior variance, so the vb_ci_* bounds are narrower\n"
    "than a proper posterior and should be read as INDICATIVE, not as definitive\n"
    "significance bounds. The wald_ci_* columns are a pooled-Logit cross-check on\n"
    "the SAME design (Wald intervals when the MLE is non-singular, otherwise the\n"
    "penalised fallback): where VB and the pooled fit AGREE that an interval\n"
    "excludes OR = 1, the effect is more credibly bounded away from no-effect."
)


def _political_cross_check(frame, vb_table):
    """Join VB odds-ratio intervals with pooled-Logit Wald intervals.

    Fits the pooled Logit on the same design and aligns its Wald-based odds-ratio
    intervals next to the VB credible intervals for the political features, so a
    reader can see whether the two interval types agree on which political effects
    are bounded away from OR = 1.

    Args:
        frame: the prepared M1 design (output of prepare_logit_frame).
        vb_table: odds_ratio_table(...) for the mixed/VB fit.
    Returns:
        (combined, pooled_is_mle): a DataFrame indexed by political feature with
        vb_or/vb_ci_low/vb_ci_high and wald_or/wald_ci_low/wald_ci_high columns,
        and a bool that is True when the pooled fit yielded real Wald standard
        errors (MLE Logit) rather than the penalised point-only fallback.
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
    # Detect the penalised fallback across the FULL pooled table: an MLE Logit
    # cannot return coefficients that are *exactly* zero, but the L1 fallback
    # (fired when the MLE Hessian is singular, as it is on this small, quasi-
    # separated design) shrinks weak coefficients to 0 -> OR == 1 with a CI that
    # collapses onto the point estimate. Any collapsed or non-finite CI bound
    # therefore signals the penalised path rather than clean Wald inference.
    bounds = wald_table[["ci_low", "ci_high"]].to_numpy()
    pooled_is_mle = bool(
        np.isfinite(bounds).all()
        and not np.isclose(wald_table["ci_low"], wald_table["ci_high"]).any()
    )
    return combined, pooled_is_mle


def main():
    """Assemble features, fit Model 1, and print the odds-ratio table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/pda.db")
    args = parser.parse_args()

    X, y, groups = assemble.assemble_features(args.db)
    frame = model1_logit.prepare_logit_frame(X, y, groups)
    result = model1_logit.fit_mixed_logit(frame)
    table = model1_logit.odds_ratio_table(result)

    path = "mixed variational (BinomialBayesMixedGLM.fit_vb)" \
        if hasattr(result, "fe_mean") else "pooled Logit fallback"
    print(f"Model 1 fit on {len(frame)} state/DC reports "
          f"({int(frame['denied'].sum())} denials) via {path}.")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    print("\n=== Full odds-ratio table (VB) ===")
    print(table.round(3).to_string())

    combined, pooled_is_mle = _political_cross_check(frame, table)
    print("\n=== Political feature odds ratios: VB vs pooled-Wald cross-check ===")
    print(_VB_CAVEAT)
    if not pooled_is_mle:
        print("(pooled MLE Logit was singular on this design, so the cross-check "
              "is the L1-PENALISED pooled fit; political features shown at "
              "OR = 1.000 with an identical CI were shrunk to exactly zero by the "
              "penalty, not estimated as null with a tight interval)")
    print()
    print(combined.round(3).to_string())


if __name__ == "__main__":
    main()
