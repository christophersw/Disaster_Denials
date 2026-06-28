# pda/modeling/evaluation.py
"""
Title: pda/modeling/evaluation.py — Grouped CV, metrics, and bootstrap CIs.
Description:
    Shared evaluation for the predictive models (spec §8). Uses
    StratifiedGroupKFold so a state never spans train/validation folds while the
    ~8% denial rate is preserved per fold. Reports ROC-AUC and PR-AUC on pooled
    out-of-fold predictions, and a paired bootstrap CI for the PR-AUC delta
    between two models (the ablation contrast).
Changelog:
    2026-06-28  Initial version.
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


def oof_predictions(estimator, X, y, groups, n_splits=5):
    """Return pooled out-of-fold predicted probabilities, grouped by `groups`.

    Args:
        estimator: an unfitted sklearn classifier with predict_proba.
        X: feature matrix. y: int target. groups: grouping key (e.g. state).
        n_splits: number of CV folds.
    Returns:
        np.ndarray of length len(y) with each row's held-out P(denied).
    """
    y = np.asarray(y)
    proba = np.full(len(y), np.nan)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    X_arr = X.reset_index(drop=True)
    for train_idx, test_idx in splitter.split(X_arr, y, groups=np.asarray(groups)):
        model = clone(estimator)
        model.fit(X_arr.iloc[train_idx], y[train_idx])
        proba[test_idx] = model.predict_proba(X_arr.iloc[test_idx])[:, 1]
    return proba


def cv_scores(estimator, X, y, groups, n_splits=5):
    """ROC-AUC, PR-AUC, and Brier score on pooled out-of-fold predictions.

    Brier is the calibration readout the spec asks for (§8): lower is better,
    and it is meaningful because the models output denial probabilities.

    Args:
        estimator: an unfitted sklearn classifier with predict_proba.
        X: feature matrix.
        y: int target array.
        groups: grouping key (e.g. state).
        n_splits: number of CV folds.
    Returns:
        dict with 'roc_auc', 'pr_auc', and 'brier'.
    """
    y = np.asarray(y)
    proba = oof_predictions(estimator, X, y, groups, n_splits=n_splits)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }


def bootstrap_auc_delta(y, proba_a, proba_b, n=1000, seed=0):
    """Paired bootstrap CI for PR-AUC(a) - PR-AUC(b).

    Args:
        y: int outcomes array.
        proba_a: predicted probabilities for model A.
        proba_b: predicted probabilities for model B.
        n: number of bootstrap resamples.
        seed: RNG seed for reproducibility.
    Returns:
        (delta, lo, hi): point estimate and 95% percentile interval.
    Side effects:
        Skips degenerate resamples where all labels are the same class.
    """
    y = np.asarray(y)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    point = average_precision_score(y, proba_a) - average_precision_score(y, proba_b)
    rng = np.random.default_rng(seed)
    deltas = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if y[s].sum() == 0 or y[s].sum() == len(s):
            continue   # skip degenerate resamples with one class
        deltas.append(
            average_precision_score(y[s], proba_a[s])
            - average_precision_score(y[s], proba_b[s])
        )
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(point), float(lo), float(hi)
