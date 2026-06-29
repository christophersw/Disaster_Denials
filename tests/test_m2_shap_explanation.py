# tests/test_m2_shap_explanation.py
"""
Title: tests/test_m2_shap_explanation.py — Tests for the SHAP-Explanation helper.
Description: Verifies model2_gbm.shap_explanation returns a well-shaped
    shap.Explanation on the fitted M2 pipeline, and that the refactored
    shap_summary still ranks features by mean |SHAP|. DB-gated (skips when
    data/pda.db is absent), mirroring the other heavy model2 tests.
Changelog:
    2026-06-29  Initial version.
"""

import os

import numpy as np
import pytest

from pda.modeling import assemble, model2_gbm


@pytest.mark.skipif(not os.path.exists("data/pda.db"), reason="needs data/pda.db")
def test_shap_explanation_shape_matches_input():
    """The explanation has one row per input row and one column per feature."""
    X, y, _ = assemble.assemble_features("data/pda.db")
    y_arr = np.asarray(y)
    Xs2 = model2_gbm._slice(X, True)
    est = model2_gbm.build_estimator()
    est.fit(Xs2, y_arr)

    exp = model2_gbm.shap_explanation(est, Xs2)

    assert exp.values.shape[0] == len(Xs2)
    assert exp.values.shape[1] == len(exp.feature_names)
    assert np.isfinite(np.asarray(exp.base_values)).all()
    assert np.asarray(exp.base_values).shape[0] == len(Xs2)
    assert np.asarray(exp.data).shape == exp.values.shape


@pytest.mark.skipif(not os.path.exists("data/pda.db"), reason="needs data/pda.db")
def test_shap_summary_ranks_by_mean_abs_shap_after_refactor():
    """The refactored shap_summary still ranks by mean |SHAP| (regression guard)."""
    X, y, _ = assemble.assemble_features("data/pda.db")
    y_arr = np.asarray(y)
    Xs2 = model2_gbm._slice(X, True)
    est = model2_gbm.build_estimator()
    est.fit(Xs2, y_arr)

    summary = model2_gbm.shap_summary(est, Xs2, y_arr)
    exp = model2_gbm.shap_explanation(est, Xs2)
    expected_top = exp.feature_names[int(np.argmax(np.abs(exp.values).mean(axis=0)))]

    assert list(summary.columns) == ["importance"]
    assert summary.index[0] == expected_top
    assert summary["importance"].is_monotonic_decreasing
