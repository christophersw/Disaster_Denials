# scripts/plot_gbm_diagnostics.py
"""
Title: scripts/plot_gbm_diagnostics.py — Model 2 GBM diagnostic figures.
Description:
    Renders standard diagnostic charts for Model 2 (the full
    HistGradientBoostingClassifier): a precision-recall curve and side-by-side
    confusion matrices (both on out-of-fold predictions), the SHAP global
    summary (beeswarm + mean-|SHAP| bar) and per-case SHAP waterfalls (both on a
    full-data fit). Complements the headline figures in plot_model_results.py.

    Charts generated (docs/models/figures/):
        m2_pr_curve.png            — out-of-fold precision-recall curve + AP.
        m2_confusion_matrices.png  — confusion matrices at 0.50 and F1-max.
        m2_shap_beeswarm.png       — SHAP beeswarm (top-12 features).
        m2_shap_mean_bar.png       — mean |SHAP| bar (top-12 features).
        m2_shap_waterfalls.png     — waterfalls for 2 denials + 2 approvals.

    Run:
        .venv/bin/python -m scripts.plot_gbm_diagnostics [--db data/pda.db]

Changelog:
    2026-06-29  Initial version.
"""

import argparse
import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from pda.modeling import assemble, evaluation, model2_gbm

# Output directory for all PNG files.
FIGURES_DIR = "docs/models/figures"
# Figure resolution in dots per inch.
DPI = 150
# Number of features to show in the SHAP summary plots.
TOP_N = 12
# Per-class count and RNG seed for the waterfall case selection.
N_PER_CLASS = 2
SEED = 0
# Palette (shared with plot_model_results.py).
_BLUE = "#2E86AB"
_RED = "#C0392B"


def f1_max_threshold(y, proba):
    """Return the probability threshold maximising F1 on (y, proba).

    Sweeps the thresholds from precision_recall_curve and picks the one with the
    highest F1 = 2PR/(P+R).

    Args:
        y: int array-like of true labels.
        proba: array-like of predicted P(positive).
    Returns:
        dict with float 'threshold', 'precision', 'recall', 'f1' at that threshold.
    """
    y = np.asarray(y)
    precision, recall, thresholds = precision_recall_curve(y, proba)
    # precision/recall have len == len(thresholds)+1; align to thresholds.
    p = precision[:-1]
    r = recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    best = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[best]),
        "precision": float(p[best]),
        "recall": float(r[best]),
        "f1": float(f1[best]),
    }


def select_waterfall_cases(y, proba, n_per_class=2, seed=0):
    """Pick reproducible denial and approval row indices for SHAP waterfalls.

    Uses a seeded RNG to choose n_per_class rows with y==1 (denied) and
    n_per_class rows with y==0 (approved).

    Args:
        y: int array-like of true labels (1=denied, 0=approved).
        proba: array-like of predicted P(denied) aligned to y.
        n_per_class: number of cases to pick per class.
        seed: RNG seed for reproducibility.
    Returns:
        list of dicts {"index": int, "outcome": "Denied"|"Approved",
        "proba": float}, denials first then approvals.
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    rng = np.random.default_rng(seed)
    denied = np.where(y == 1)[0]
    approved = np.where(y == 0)[0]
    pick_d = rng.choice(denied, size=min(n_per_class, len(denied)), replace=False)
    pick_a = rng.choice(approved, size=min(n_per_class, len(approved)), replace=False)
    cases = []
    for idx in list(pick_d) + list(pick_a):
        cases.append({
            "index": int(idx),
            "outcome": "Denied" if y[idx] == 1 else "Approved",
            "proba": float(proba[idx]),
        })
    return cases
