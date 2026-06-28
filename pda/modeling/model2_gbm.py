# pda/modeling/model2_gbm.py
"""
Title: pda/modeling/model2_gbm.py — Gradient boosting + ablation + SHAP (Model 2).
Description:
    The flexible predictive model (spec §7, M2). Uses scikit-learn
    HistGradientBoostingClassifier (native NaN handling — satisfies the spec's
    "gradient-boosted trees with native missing-value handling", §6) over the
    state-level political + need + request + enrichment features (NOT the
    county-composition block — that is Model 3). Two readouts of the one model:
    (a) ablation — fit with vs without the political block and compare PR-AUC
    with a bootstrap CI; (b) SHAP — mean |SHAP| attribution, with a permutation
    -importance fallback if shap is unavailable on this interpreter.

    Pipeline: _DropDegenerateCols → ColumnTransformer(OrdinalEncoder | passthrough)
    → HistGradientBoostingClassifier.

    The _DropDegenerateCols step is required because sklearn 1.9 HistGBM's
    binning stage calls sliding_window_view(distinct_values, 2) and raises
    ValueError when a training-fold column has fewer than 2 distinct non-NaN
    values. pa_primary_impact is 100% null in the current corpus; several
    IA-demographic columns are 80-93% null and can become all-null in a
    training fold when the states with demographic data land in the test fold.
Changelog:
    2026-06-28  Initial version.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from pda.modeling import assemble, evaluation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _DropDegenerateCols(BaseEstimator, TransformerMixin):
    """Drop columns with fewer than 2 distinct non-NaN values in training data.

    HistGradientBoostingClassifier's binning stage requires at least 2
    distinct non-NaN values per feature column (it calls
    sliding_window_view(distinct_values, 2)). Columns that are all-NaN or
    constant in the training fold are removed during fit; the same mask is
    applied at transform time so train/test column sets stay aligned.

    Args: none.
    Fit sets: keep_ (list of column names or int indices to retain).
    Returns: X restricted to keep_.
    """

    def fit(self, X, y=None):
        """Identify columns with ≥ 2 distinct non-NaN values.

        Args:
            X: DataFrame or ndarray.
            y: ignored.
        Returns: self.
        """
        if hasattr(X, "columns"):
            self.keep_ = [c for c in X.columns if X[c].nunique(dropna=True) >= 2]
        else:
            self.keep_ = [i for i in range(X.shape[1])
                          if pd.Series(X[:, i]).nunique(dropna=True) >= 2]
        return self

    def transform(self, X):
        """Apply the column mask learned at fit time.

        Args:
            X: DataFrame or ndarray with the same column set as fit input.
        Returns: X restricted to keep_ columns.
        """
        if hasattr(X, "columns"):
            return X[[c for c in self.keep_ if c in X.columns]]
        return X[:, self.keep_]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def feature_columns(include_political=True):
    """Return M2's feature set: need + request + enrichment + jurisdiction (+ political).

    Excludes POLITICAL_COUNTY (Model 3 owns that block). With
    include_political=False, drops POLITICAL_STATE too — the reduced model used
    in the ablation comparison.

    Args:
        include_political: bool — whether to include the POLITICAL_STATE block.
    Returns:
        list[str] of column names.
    """
    cols = (assemble.NEED + assemble.REQUEST + assemble.ENRICHMENT
            + assemble.JURISDICTION + assemble.IA_DEMOGRAPHIC_BLOCK)
    if include_political:
        cols = cols + assemble.POLITICAL_STATE
    return cols


def build_estimator():
    """Build the M2 pipeline: degenerate-column drop → ordinal encoding → HistGBM.

    Step 1 (_DropDegenerateCols): removes any column with < 2 distinct non-NaN
    values from the training fold so the HGB binning step never encounters an
    empty or single-value column (which would raise ValueError in sklearn 1.9).

    Step 2 (ColumnTransformer/OrdinalEncoder): make_column_selector picks the
    categorical (object-dtype) columns at fit time — assemble.assemble_features
    has already coerced every other column to numeric, so exactly the genuine
    categoricals get ordinal-encoded. OrdinalEncoder maps unknowns to -1 and
    preserves NaN so the booster routes missing values natively.

    Step 3 (HistGradientBoostingClassifier): class_weight='balanced' corrects
    for the ~8% positive rate; early_stopping avoids over-fitting.

    Returns:
        sklearn.pipeline.Pipeline with steps 'drop_deg', 'pre', and 'clf'.
    """
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1,
        encoded_missing_value=np.nan,
    )
    pre = ColumnTransformer(
        transformers=[
            ("cat", encoder, make_column_selector(dtype_include="object")),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    clf = HistGradientBoostingClassifier(
        class_weight="balanced", random_state=0, max_iter=300,
        learning_rate=0.05, early_stopping=True,
    )
    return Pipeline([("drop_deg", _DropDegenerateCols()), ("pre", pre), ("clf", clf)])


def _slice(X, include_political):
    """Return X restricted to M2's feature columns that are present in X.

    Args:
        X: the full assembled feature DataFrame.
        include_political: bool — whether to include POLITICAL_STATE columns.
    Returns:
        DataFrame with only the M2-relevant columns present in X.
    """
    cols = [c for c in feature_columns(include_political) if c in X.columns]
    return X[cols]


def political_ablation(X, y, groups, n_splits=5):
    """Compare full vs no-political-block models on out-of-fold PR-AUC.

    Fits two estimators via oof_predictions: one with the POLITICAL_STATE block
    (full model) and one without (reduced model). Computes a bootstrap CI for
    the PR-AUC delta to quantify the predictive lift that politics adds.

    Rows with a null groups value (tribal nations and territories that lack a
    state abbreviation) are excluded — StratifiedGroupKFold cannot sort a mixed
    float-NaN / str groups array on numpy ≥ 2.x.

    Args:
        X: assembled feature matrix from assemble_features.
        y: int denied target array.
        groups: state_abbr Series for grouped CV (StratifiedGroupKFold).
        n_splits: number of CV folds (default 5).
    Returns:
        dict with keys full_pr_auc, reduced_pr_auc, full_roc_auc,
        reduced_roc_auc, delta (float), delta_ci (tuple lo, hi).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    # Filter rows without a state grouping key — tribal nations / territories
    # that have no state abbreviation cannot be assigned to a CV fold by
    # StratifiedGroupKFold (numpy sort fails on mixed float-NaN / str arrays).
    groups_s = groups if isinstance(groups, pd.Series) else pd.Series(groups)
    valid_mask = groups_s.notna().values
    X_valid = X.iloc[valid_mask] if hasattr(X, "iloc") else X[valid_mask]
    y_valid = np.asarray(y)[valid_mask]
    groups_valid = groups_s.values[valid_mask]

    y_arr = y_valid
    full = evaluation.oof_predictions(
        build_estimator(), _slice(X_valid, True), y_arr, groups_valid, n_splits)
    reduced = evaluation.oof_predictions(
        build_estimator(), _slice(X_valid, False), y_arr, groups_valid, n_splits)
    delta, lo, hi = evaluation.bootstrap_auc_delta(y_arr, full, reduced)
    return {
        "full_pr_auc": float(average_precision_score(y_arr, full)),
        "reduced_pr_auc": float(average_precision_score(y_arr, reduced)),
        "full_roc_auc": float(roc_auc_score(y_arr, full)),
        "reduced_roc_auc": float(roc_auc_score(y_arr, reduced)),
        "delta": delta,
        "delta_ci": (lo, hi),
    }


def shap_summary(estimator, X):
    """Return mean |SHAP| per feature, or permutation importance as a fallback.

    Attempts shap.TreeExplainer on the fitted HistGradientBoostingClassifier
    (extracted from the pipeline via named_steps). Falls back to sklearn
    permutation_importance if shap is unavailable or raises for this
    model/data combination.

    The SHAP path chains through all preprocessing steps (drop_deg → pre) to
    produce the encoded matrix the clf was trained on, so feature indices
    in shap_values align with X_enc.columns.

    Args:
        estimator: a fitted Pipeline from build_estimator.
        X: the feature slice the estimator was fitted on (same columns).
    Returns:
        DataFrame indexed by feature name with an 'importance' column,
        sorted descending by importance.
    """
    try:
        import shap
        drop_deg = estimator.named_steps["drop_deg"]
        pre = estimator.named_steps["pre"]
        clf = estimator.named_steps["clf"]
        X_dropped = drop_deg.transform(X)
        X_enc = pre.transform(X_dropped)
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(X_enc)
        vals = values[1] if isinstance(values, list) else values
        imp = np.abs(vals).mean(axis=0)
        return (pd.DataFrame({"importance": imp}, index=X_enc.columns)
                .sort_values("importance", ascending=False))
    except Exception:
        result = permutation_importance(
            estimator, X, estimator.predict(X), n_repeats=10, random_state=0)
        return (pd.DataFrame({"importance": result.importances_mean}, index=X.columns)
                .sort_values("importance", ascending=False))
