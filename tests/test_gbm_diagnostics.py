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


def test_plot_pr_curve_writes_png(tmp_path):
    """PR-curve render writes its PNG from synthetic oof predictions."""
    from scripts.plot_gbm_diagnostics import f1_max_threshold, plot_pr_curve

    rng = np.random.default_rng(1)
    y = np.array([0] * 80 + [1] * 20)
    proba = np.concatenate([rng.uniform(0, 0.7, 80), rng.uniform(0.3, 1.0, 20)])
    plot_pr_curve(y, proba, f1_max_threshold(y, proba), str(tmp_path))
    assert (tmp_path / "m2_pr_curve.png").exists()


def test_plot_confusion_matrices_writes_png(tmp_path):
    """Confusion-matrix render writes its PNG from synthetic oof predictions."""
    from scripts.plot_gbm_diagnostics import (
        f1_max_threshold,
        plot_confusion_matrices,
    )

    rng = np.random.default_rng(2)
    y = np.array([0] * 80 + [1] * 20)
    proba = np.concatenate([rng.uniform(0, 0.7, 80), rng.uniform(0.3, 1.0, 20)])
    plot_confusion_matrices(y, proba, f1_max_threshold(y, proba), str(tmp_path))
    assert (tmp_path / "m2_confusion_matrices.png").exists()


def _synthetic_explanation(n_rows=40, n_feat=6):
    """Build a small valid shap.Explanation for rendering smoke tests."""
    import shap

    rng = np.random.default_rng(3)
    values = rng.normal(size=(n_rows, n_feat))
    data = rng.normal(size=(n_rows, n_feat))
    return shap.Explanation(
        values=values,
        base_values=np.full(n_rows, 0.1),
        data=data,
        feature_names=[f"feature_{i}" for i in range(n_feat)],
    )


def test_plot_shap_beeswarm_writes_png(tmp_path):
    from scripts.plot_gbm_diagnostics import plot_shap_beeswarm
    plot_shap_beeswarm(_synthetic_explanation(), str(tmp_path))
    assert (tmp_path / "m2_shap_beeswarm.png").exists()


def test_plot_shap_mean_bar_writes_png(tmp_path):
    from scripts.plot_gbm_diagnostics import plot_shap_mean_bar
    plot_shap_mean_bar(_synthetic_explanation(), str(tmp_path))
    assert (tmp_path / "m2_shap_mean_bar.png").exists()


def test_plot_shap_waterfalls_writes_png(tmp_path):
    from scripts.plot_gbm_diagnostics import (
        plot_shap_waterfalls,
        select_waterfall_cases,
    )
    exp = _synthetic_explanation()
    y = np.array([0, 1] * 20)
    proba = np.linspace(0.05, 0.95, 40)
    cases = select_waterfall_cases(y, proba, n_per_class=2, seed=0)
    plot_shap_waterfalls(exp, cases, str(tmp_path))
    assert (tmp_path / "m2_shap_waterfalls.png").exists()
