# tests/test_model1_logit.py
"""
Title: tests/test_model1_logit.py — Contract tests for Model 1.
Description: Verifies the M1 estimand filter (states + DC only), the
    request_year year-key guard, and that the odds-ratio table is well-formed.
    Heavy fitting is exercised against the real DB and skipped when absent.
Changelog:
    2026-06-28  Initial version.
"""

import os

import numpy as np
import pandas as pd
import pytest

from pda.modeling import model1_logit


def test_estimand_filter_keeps_states_and_dc_only():
    """The prepared frame must contain only state + federal_district rows."""
    X = pd.DataFrame(
        {
            "jurisdiction_type": ["state", "federal_district", "territory", "tribal"],
            "gubernatorial_alignment_applicable": [1, 0, 1, 0],
            "state_margin": [10.0, 90.0, np.nan, np.nan],
            "governor_party": ["R", "D", "D", None],
            "request_profile": ["IA_only", "PA_only", "IA_and_PA", "neither"],
        },
        index=["a", "b", "c", "d"],
    )
    y = pd.Series([0, 0, 1, 1], index=X.index, name="denied")
    groups = pd.Series(["TX", "DC", "PR", "AZ"], index=X.index)
    frame = model1_logit.prepare_logit_frame(X, y, groups)
    assert len(frame) == 2          # territory + tribal dropped
    assert set(frame.index) == {"a", "b"}


def test_prepare_drops_nan_request_year_rows():
    """Rows with an unparseable request_year are dropped from the M1 fit."""
    X = pd.DataFrame(
        {
            "jurisdiction_type": ["state", "state", "state"],
            "gubernatorial_alignment_applicable": [1, 1, 1],
            "state_party_match": [1.0, 0.0, 1.0],
            "request_year": [2014.0, np.nan, 2018.0],
        },
        index=["a", "b", "c"],
    )
    y = pd.Series([0, 1, 0], index=X.index, name="denied")
    groups = pd.Series(["TX", "FL", "OH"], index=X.index)
    frame = model1_logit.prepare_logit_frame(X, y, groups)
    assert set(frame.index) == {"a", "c"}        # NaN-year row 'b' dropped
    assert "year" in frame.columns
    assert frame["year"].dtype.kind in "iu"      # clean integer grouping key
    assert "request_year" not in frame.columns   # consumed into 'year'


@pytest.mark.skipif(not os.path.exists("data/pda.db"), reason="needs data/pda.db")
def test_fit_produces_odds_ratio_table():
    """End-to-end: fit on the real DB and emit a positive odds-ratio table."""
    from pda.modeling import assemble
    X, y, groups = assemble.assemble_features("data/pda.db")
    frame = model1_logit.prepare_logit_frame(X, y, groups)
    result = model1_logit.fit_mixed_logit(frame)
    table = model1_logit.odds_ratio_table(result)
    assert "odds_ratio" in table.columns
    assert "ci_low" in table.columns
    assert "ci_high" in table.columns
    assert (table["odds_ratio"] > 0).all()
    assert np.isfinite(table["odds_ratio"]).all()
    # The headline political feature must survive the pipeline into the table.
    assert "state_party_match" in table.index
