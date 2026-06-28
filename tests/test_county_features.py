# tests/test_county_features.py
"""
Title: tests/test_county_features.py — Tests for county-composition features.
Description: Verifies the per-disaster aggregations that separate the "one
    stronghold county" vs "most counties favored him" hypotheses (spec §5.5).
Changelog:
    2026-06-28  Initial version.
"""

import pandas as pd

from pda.modeling.county_features import county_composition_features


def test_aggregations_for_one_disaster():
    counties = pd.DataFrame(
        {
            "source_pdf": ["a", "a", "a"],
            "county_margin": [30.0, -10.0, 5.0],   # pres won 2 of 3
            "per_capita_impact": [100.0, 0.0, 0.0],  # worst-hit is the +30 one
        }
    )
    out = county_composition_features(counties)
    row = out.loc["a"]
    assert row["num_affected_counties"] == 3
    assert abs(row["share_affected_counties_pres_won"] - 2 / 3) < 1e-9
    assert row["max_pres_margin_affected"] == 30.0
    assert row["pres_won_any_county_by_20plus"] == 1
    assert row["pres_won_most_damaged_county"] == 1   # highest-impact county is +30
    # Damage-weighted mean is dominated by the +30 county (all weight there).
    assert abs(row["dmg_weighted_mean_pres_margin"] - 30.0) < 1e-9


def test_dmg_weight_falls_back_to_unweighted_when_no_impact():
    counties = pd.DataFrame(
        {
            "source_pdf": ["b", "b"],
            "county_margin": [10.0, 20.0],
            "per_capita_impact": [0.0, 0.0],   # no weight → plain mean
        }
    )
    out = county_composition_features(counties)
    assert abs(out.loc["b", "dmg_weighted_mean_pres_margin"] - 15.0) < 1e-9
