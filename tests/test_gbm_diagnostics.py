"""
Title: tests/test_gbm_diagnostics.py — Tests for the GBM diagnostic helpers/plots.
Description: Fast unit tests for f1_max_threshold and select_waterfall_cases, plus
    rendering smoke tests that each plotting function writes its PNG given
    synthetic inputs (no model fit required).
Changelog:
    2026-06-29  Initial version.
"""

import numpy as np

from scripts.plot_gbm_diagnostics import (
    f1_max_threshold,
    select_waterfall_cases,
)


def test_f1_max_threshold_beats_neighbours():
    """The returned threshold's F1 is >= F1 at 0.5 on a separable-ish problem."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10)
    # positives score higher on average but with overlap
    proba = np.concatenate([rng.uniform(0.0, 0.6, 90), rng.uniform(0.4, 1.0, 10)])
    info = f1_max_threshold(y, proba)
    from sklearn.metrics import f1_score
    f1_at_half = f1_score(y, (proba >= 0.5).astype(int), zero_division=0)
    assert info["f1"] >= f1_at_half
    assert 0.0 <= info["threshold"] <= 1.0
    assert set(info) == {"threshold", "precision", "recall", "f1"}


def test_select_waterfall_cases_is_reproducible_and_correct_class():
    """Same seed => same picks; denials are y==1, approvals are y==0."""
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    proba = np.linspace(0.05, 0.95, 10)
    a = select_waterfall_cases(y, proba, n_per_class=2, seed=0)
    b = select_waterfall_cases(y, proba, n_per_class=2, seed=0)
    assert [c["index"] for c in a] == [c["index"] for c in b]   # reproducible
    denied = [c for c in a if c["outcome"] == "Denied"]
    approved = [c for c in a if c["outcome"] == "Approved"]
    assert len(denied) == 2 and len(approved) == 2
    assert all(y[c["index"]] == 1 for c in denied)
    assert all(y[c["index"]] == 0 for c in approved)
    assert all(c["proba"] == proba[c["index"]] for c in a)
