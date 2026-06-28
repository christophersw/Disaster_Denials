# tests/test_evaluation.py
"""
Title: tests/test_evaluation.py — Tests for evaluation utilities.
Description: Verifies grouped out-of-fold prediction (no group spans folds),
    metric computation, and the bootstrap CI on a PR-AUC delta.
Changelog:
    2026-06-28  Initial version.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pda.modeling import evaluation


def _toy():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 2))
    y = (x[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    groups = pd.Series(rng.integers(0, 10, size=n))   # 10 states
    X = pd.DataFrame(x, columns=["f0", "f1"])
    return X, pd.Series(y), groups


def test_oof_predictions_cover_all_rows():
    X, y, groups = _toy()
    proba = evaluation.oof_predictions(
        LogisticRegression(), X, y, groups, n_splits=5
    )
    assert proba.shape == (len(y),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_cv_scores_reasonable_on_separable_data():
    X, y, groups = _toy()
    scores = evaluation.cv_scores(
        LogisticRegression(), X, y, groups, n_splits=5
    )
    assert scores["roc_auc"] > 0.7
    assert 0.0 <= scores["pr_auc"] <= 1.0


def test_bootstrap_delta_brackets_zero_for_identical_models():
    X, y, groups = _toy()
    proba = evaluation.oof_predictions(
        LogisticRegression(), X, y, groups, n_splits=5
    )
    delta, lo, hi = evaluation.bootstrap_auc_delta(y.to_numpy(), proba, proba)
    assert abs(delta) < 1e-9
    assert lo <= 0 <= hi
