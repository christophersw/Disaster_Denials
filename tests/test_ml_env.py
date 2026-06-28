"""
Title: tests/test_ml_env.py — Modeling stack import smoke test
Description: Fails loudly if the modeling dependencies are missing or broken on
    this interpreter, so environment problems surface before any model runs.
Changelog:
    2026-06-28  Initial version.
"""

import importlib

import pytest

MODULES = ["pandas", "numpy", "sklearn", "statsmodels", "shap"]


@pytest.mark.parametrize("name", MODULES)
def test_modeling_dependency_imports(name):
    assert importlib.import_module(name) is not None


def test_hist_gradient_boosting_available():
    from sklearn.ensemble import HistGradientBoostingClassifier
    assert HistGradientBoostingClassifier is not None
