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


def plot_pr_curve(y, proba, thr_info, out_dir):
    """Plot the out-of-fold precision-recall curve with AP and the F1-max point.

    Args:
        y: int array-like of true labels.
        proba: array-like of out-of-fold P(denied).
        thr_info: dict from f1_max_threshold (threshold/precision/recall/f1).
        out_dir: directory for the PNG.
    Returns:
        None. Saves m2_pr_curve.png to out_dir.
    """
    y = np.asarray(y)
    precision, recall, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    prevalence = float(y.mean())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=_BLUE, lw=2, label=f"PR curve (AP = {ap:.3f})")
    ax.axhline(prevalence, ls="--", color="gray", lw=1.3,
               label=f"baseline (prevalence = {prevalence:.3f})")
    ax.plot(thr_info["recall"], thr_info["precision"], "o", color=_RED, ms=10,
            zorder=5,
            label=(f"F1-max (thr = {thr_info['threshold']:.2f}, "
                   f"F1 = {thr_info['f1']:.2f})"))
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Model 2 — precision–recall (out-of-fold)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "m2_pr_curve.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_confusion_matrices(y, proba, thr_info, out_dir):
    """Plot side-by-side confusion matrices at threshold 0.50 and F1-max.

    Args:
        y: int array-like of true labels.
        proba: array-like of out-of-fold P(denied).
        thr_info: dict from f1_max_threshold.
        out_dir: directory for the PNG.
    Returns:
        None. Saves m2_confusion_matrices.png to out_dir.
    """
    y = np.asarray(y)
    panels = [
        ("Threshold = 0.50", 0.5),
        (f"Threshold = {thr_info['threshold']:.2f} (F1-max)", thr_info["threshold"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (title, thr) in zip(axes, panels):
        pred = (np.asarray(proba) >= thr).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Approved", "Denied"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Approved", "Denied"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        cutoff = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                        color="white" if cm[i, j] > cutoff else "black",
                        fontsize=13, fontweight="bold")
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        ax.set_title(f"{title}\nprecision = {prec:.2f}   recall = {rec:.2f}   "
                     f"F1 = {f1:.2f}", fontsize=10)

    fig.suptitle("Model 2 — confusion matrices (out-of-fold)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "m2_confusion_matrices.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_shap_beeswarm(explanation, out_dir) -> None:
    """SHAP beeswarm summary (top-TOP_N features) for the full-data fit.

    Args:
        explanation: a shap.Explanation from model2_gbm.shap_explanation.
        out_dir: directory for the PNG.
    Returns:
        None. Saves m2_shap_beeswarm.png to out_dir.
    """
    import shap

    shap.plots.beeswarm(explanation, max_display=TOP_N, show=False)
    fig = plt.gcf()
    fig.suptitle("Model 2 — SHAP summary (beeswarm)", fontsize=12, fontweight="bold")
    path = os.path.join(out_dir, "m2_shap_beeswarm.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_shap_mean_bar(explanation, out_dir) -> None:
    """Mean |SHAP| bar (top-TOP_N features) for the full-data fit.

    Args:
        explanation: a shap.Explanation from model2_gbm.shap_explanation.
        out_dir: directory for the PNG.
    Returns:
        None. Saves m2_shap_mean_bar.png to out_dir.
    """
    import shap

    shap.plots.bar(explanation, max_display=TOP_N, show=False)
    fig = plt.gcf()
    fig.suptitle("Model 2 — mean |SHAP| importance", fontsize=12, fontweight="bold")
    path = os.path.join(out_dir, "m2_shap_mean_bar.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_shap_waterfalls(explanation, cases, out_dir) -> None:
    """Compose per-case SHAP waterfalls into a single 2x2 grid figure.

    Each waterfall is rendered to its own temporary PNG (SHAP's waterfall owns
    its figure and does not compose into shared subplots), then the panels are
    assembled into one 2x2 image grid via imshow — robust against SHAP's
    axes handling.

    Args:
        explanation: a shap.Explanation from model2_gbm.shap_explanation.
        cases: list of dicts from select_waterfall_cases (index/outcome/proba).
        out_dir: directory for the PNG.
    Returns:
        None. Saves m2_shap_waterfalls.png to out_dir.
    """
    import shap

    panels = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, case in enumerate(cases):
            shap.plots.waterfall(explanation[case["index"]], max_display=10,
                                 show=False)
            fig = plt.gcf()
            fig.suptitle(
                f"{case['outcome']} — predicted P(denied) = {case['proba']:.2f}",
                fontsize=11, fontweight="bold",
            )
            panel_path = os.path.join(tmp, f"w{i}.png")
            fig.savefig(panel_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            panels.append(plt.imread(panel_path))

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        flat = axes.ravel()
        for ax, img in zip(flat, panels):
            ax.imshow(img)
            ax.axis("off")
        for ax in flat[len(panels):]:
            ax.axis("off")
        fig.suptitle("Model 2 — SHAP waterfalls (individual cases)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, "m2_shap_waterfalls.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {path}")


def main():
    """Compute M2 diagnostics and render the five PNGs to docs/models/figures/.

    Performance figures use out-of-fold predictions; SHAP figures use a
    full-data fit. Prints each saved path. Takes ~1-2 minutes.

    Args: none (reads --db from CLI, default data/pda.db).
    Returns: None.
    Side effects: creates/overwrites five PNGs in docs/models/figures/.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default="data/pda.db",
        help="Path to the SQLite database (default: data/pda.db)",
    )
    args = parser.parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Assembling feature matrix from:", args.db)
    X, y, groups = assemble.assemble_features(args.db)
    y_arr = np.asarray(y)
    Xs2 = model2_gbm._slice(X, True)
    print(f"  {len(Xs2)} rows x {Xs2.shape[1]} cols | "
          f"denied={int(y_arr.sum())} ({100 * y_arr.mean():.1f}%)")

    print("Out-of-fold predictions (grouped CV) ...")
    # Filter to rows with a state grouping key — tribal nations / territories
    # have no state_abbr; StratifiedGroupKFold cannot sort a mixed float-NaN /
    # str groups array on numpy >= 2.x. Mirrors the filter in political_ablation.
    valid_mask = groups.notna().values
    Xs2_valid = Xs2.iloc[valid_mask]
    y_valid = y_arr[valid_mask]
    groups_valid = groups.values[valid_mask]
    oof = evaluation.oof_predictions(
        model2_gbm.build_estimator(), Xs2_valid, y_valid, groups_valid)
    thr_info = f1_max_threshold(y_valid, oof)
    print(f"  F1-max threshold={thr_info['threshold']:.3f} "
          f"(P={thr_info['precision']:.2f} R={thr_info['recall']:.2f} "
          f"F1={thr_info['f1']:.2f})")

    plot_pr_curve(y_valid, oof, thr_info, FIGURES_DIR)
    plot_confusion_matrices(y_valid, oof, thr_info, FIGURES_DIR)

    print("Full-data fit + SHAP explanation ...")
    est = model2_gbm.build_estimator()
    est.fit(Xs2, y_arr)
    explanation = model2_gbm.shap_explanation(est, Xs2)
    plot_shap_beeswarm(explanation, FIGURES_DIR)
    plot_shap_mean_bar(explanation, FIGURES_DIR)

    full_proba = est.predict_proba(Xs2)[:, 1]
    cases = select_waterfall_cases(y_arr, full_proba,
                                   n_per_class=N_PER_CLASS, seed=SEED)
    plot_shap_waterfalls(explanation, cases, FIGURES_DIR)

    print(f"All 5 diagnostic charts saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
