# tests/test_features.py
"""
Title: tests/test_features.py — Tests for request-profile and enrichment.
Description: Verifies the request_profile categorical, the federal-election
    calendar features, and the swing-state indicator (spec §5.2, §5.6).
    Also verifies request_year (integer calendar year) added per controller
    instruction for Model 1 year random-effect grouping.
Changelog:
    2026-06-28  Initial version.
"""

import pandas as pd

from pda.modeling.features import (
    add_election_features,
    add_request_profile,
    add_swing_state,
)


def test_request_profile_categories():
    df = pd.DataFrame(
        {"ia_requested": [1, 0, 1, 0], "pa_requested": [0, 1, 1, 0]}
    )
    out = add_request_profile(df)
    assert out["request_profile"].tolist() == [
        "IA_only", "PA_only", "IA_and_PA", "neither",
    ]


def test_election_features_presidential_year():
    df = pd.DataFrame({"request_date": ["2020-09-01", "2022-06-15"]})
    out = add_election_features(df)
    assert out["presidential_election_year"].tolist() == [1, 0]
    assert out["midterm_election_year"].tolist() == [0, 1]
    # Sept 2020 → Nov 2020 general is ~2 months out.
    assert 1.0 <= out["months_to_next_election"].iloc[0] <= 3.0
    # request_year: integer calendar year (controller addition for Model 1)
    assert out["request_year"].tolist() == [2020, 2022]


def test_swing_state_threshold():
    df = pd.DataFrame({"state_margin": [3.0, -2.0, 25.0, None]})
    out = add_swing_state(df, threshold=8.0)
    assert out["swing_state"].tolist() == [1, 1, 0, 0]  # null → 0 (not swing)
