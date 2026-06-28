# tests/test_assemble.py
"""
Title: tests/test_assemble.py — Tests for feature assembly.
Description: Verifies the assembled matrix exposes only allow-listed feature
    groups (no leakage), carries the engineered columns, and aligns with the
    target. Uses the real data/pda.db when present, else skips.
Changelog:
    2026-06-28  Initial version.
"""

import os

import pytest

from pda.modeling import assemble
from pda.modeling.data import LEAKAGE_COLUMNS

DB = "data/pda.db"


@pytest.mark.skipif(not os.path.exists(DB), reason="needs data/pda.db")
def test_assembled_matrix_has_no_leakage_and_expected_columns():
    X, y, groups = assemble.assemble_features(DB)
    assert len(X) == len(y)
    assert len(X) == int((y == 0).sum() + (y == 1).sum())
    for col in LEAKAGE_COLUMNS:
        assert col not in X.columns
    for col in ["request_profile", "jurisdiction_type",
                "max_pres_margin_affected", "swing_state",
                "months_to_next_election"]:
        assert col in X.columns
    # Controller addition: request_year must be in X (ENRICHMENT allow-list)
    assert "request_year" in X.columns
    # Allow-list: every column belongs to a declared group.
    allowed = set(
        assemble.NEED + assemble.REQUEST + assemble.POLITICAL_STATE
        + assemble.POLITICAL_COUNTY + assemble.ENRICHMENT
        + assemble.JURISDICTION + assemble.IA_DEMOGRAPHIC_BLOCK
    )
    assert set(X.columns) <= allowed
    # Lower-bound: representative required columns must survive into X so a
    # future upstream rename can't silently drop a whole feature group.
    for col in ["total_cost_estimate", "ia_requested", "state_party_match",
                "request_year", "num_affected_counties"]:
        assert col in X.columns, f"Required column '{col}' missing from assembled X"


@pytest.mark.skipif(not os.path.exists(DB), reason="needs data/pda.db")
def test_target_base_rate_is_about_8_percent():
    _, y, _ = assemble.assemble_features(DB)
    rate = y.mean()
    assert 0.05 < rate < 0.12   # ~102 / 1279
