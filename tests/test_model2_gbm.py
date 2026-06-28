# tests/test_model2_gbm.py
"""
Title: tests/test_model2_gbm.py — Contract tests for Model 2.
Description: Verifies M2's feature set excludes county-composition columns,
    the estimator fits and predicts probabilities, and the ablation returns the
    expected keys. Full runs use the real DB and skip when absent.
Changelog:
    2026-06-28  Initial version.
"""

import os

import numpy as np
import pandas as pd
import pytest

from pda.modeling import assemble, model2_gbm


def test_feature_columns_exclude_county_block():
    cols = model2_gbm.feature_columns(include_political=True)
    assert not (set(cols) & set(assemble.POLITICAL_COUNTY))
    assert set(assemble.POLITICAL_STATE) <= set(cols)
    no_pol = model2_gbm.feature_columns(include_political=False)
    assert not (set(no_pol) & set(assemble.POLITICAL_STATE))


def test_estimator_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "total_cost_estimate": rng.normal(size=50),
        "request_profile": rng.choice(["IA_only", "PA_only"], size=50),
        "jurisdiction_type": ["state"] * 50,
    })
    y = (X["total_cost_estimate"] > 0).astype(int)
    est = model2_gbm.build_estimator()
    est.fit(X, y)
    proba = est.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.skipif(not os.path.exists("data/pda.db"), reason="needs data/pda.db")
def test_political_ablation_keys():
    X, y, groups = assemble.assemble_features("data/pda.db")
    out = model2_gbm.political_ablation(X, y, groups)
    for key in ["full_pr_auc", "reduced_pr_auc", "full_roc_auc", "reduced_roc_auc", "delta", "delta_ci"]:
        assert key in out
