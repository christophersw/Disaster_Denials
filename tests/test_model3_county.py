# tests/test_model3_county.py
"""
Title: tests/test_model3_county.py — Contract tests for Model 3.
Description: Verifies M3's feature set adds the county-composition block on top
    of Model 2's, and the M2->M3 ablation returns the expected keys (spec §7, M3).
Changelog:
    2026-06-28  Initial version.
"""

import os

import pytest

from pda.modeling import assemble, model2_gbm, model3_county


def test_m3_features_add_county_block():
    m2 = set(model2_gbm.feature_columns(include_political=True))
    m3 = set(model3_county.feature_columns())
    assert set(assemble.POLITICAL_COUNTY) <= m3
    assert m2 <= m3                       # superset of Model 2
    assert m3 - m2 == set(assemble.POLITICAL_COUNTY)


@pytest.mark.skipif(not os.path.exists("data/pda.db"), reason="needs data/pda.db")
def test_county_ablation_keys():
    X, y, groups = assemble.assemble_features("data/pda.db")
    out = model3_county.county_ablation(X, y, groups)
    for key in ["m2_pr_auc", "m3_pr_auc", "delta", "delta_ci"]:
        assert key in out
