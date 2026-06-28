# scripts/run_model1_logit.py
"""
Title: scripts/run_model1_logit.py — Fit and report Model 1.
Description: Assembles features, fits the hierarchical logit on states + DC, and
    prints the odds-ratio table with the political rows highlighted (spec §7).
    Reports which fit path actually fired (mixed variational vs pooled fallback).
Changelog:
    2026-06-28  Initial version.
"""

import argparse

import pandas as pd

from pda.modeling import assemble, model1_logit

# Feature rows highlighted as the political signal of interest (spec §7).
_POLITICAL_FEATURES = [
    "state_party_match", "governor_vs_president", "governor_vs_state_vote",
    "state_margin", "state_dem_share", "swing_state",
    "share_affected_counties_pres_won", "max_pres_margin_affected",
    "dmg_weighted_mean_pres_margin", "pres_won_most_damaged_county",
    "pres_margin_dispersion", "months_to_next_election",
    "presidential_election_year", "midterm_election_year",
]


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

    pd.set_option("display.width", 120)
    pd.set_option("display.max_rows", None)
    print("\n=== Full odds-ratio table ===")
    print(table.round(3).to_string())

    political = table.loc[[f for f in _POLITICAL_FEATURES if f in table.index]]
    print("\n=== Political feature odds ratios ===")
    print(political.round(3).to_string())


if __name__ == "__main__":
    main()
