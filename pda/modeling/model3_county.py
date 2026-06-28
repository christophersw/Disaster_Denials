# pda/modeling/model3_county.py
"""
Title: pda/modeling/model3_county.py — County-composition model (Model 3).
Description:
    Reuses Model 2's gradient-boosting engine and adds the county-composition
    political block (spec §7, M3). Because the only structural difference from
    Model 2 is that block, the M2->M3 comparison is itself the ablation that
    isolates the incremental value of fine-grained county political composition
    beyond state-level alignment. stronghold_comparison reads the importances of
    the "one stronghold county" vs "most counties" features to say which form of
    favoritism (if any) the data supports.

    NaN-group filtering: StratifiedGroupKFold cannot sort mixed float-NaN / str
    groups arrays on numpy >= 2.x. county_ablation mirrors model2_gbm.political_
    ablation exactly — it drops the ~43 tribal / territory rows whose state_abbr
    is null before building folds. The SAME valid_mask is applied to X, y, and
    groups for BOTH the M2-feature model and the M3-feature model so the PR-AUC
    delta is a fair paired comparison on identical rows and identical folds.
Changelog:
    2026-06-28  Initial version.
"""

import numpy as np
import pandas as pd

from pda.modeling import assemble, evaluation, model2_gbm


def feature_columns():
    """Return M3's feature set: Model 2's political feature set plus POLITICAL_COUNTY.

    Model 2 provides NEED + REQUEST + ENRICHMENT + JURISDICTION +
    IA_DEMOGRAPHIC_BLOCK + POLITICAL_STATE. Model 3 adds POLITICAL_COUNTY on
    top, and that addition is the entire structural change being ablated.

    Returns:
        list[str] of column names (M2 set followed by county-composition columns).
    """
    return model2_gbm.feature_columns(include_political=True) + assemble.POLITICAL_COUNTY


def _slice(X, columns):
    """Restrict X to columns that are present in X.

    Args:
        X: the full assembled feature DataFrame.
        columns: list of column names desired.
    Returns:
        DataFrame with only the columns from `columns` that exist in X.
    """
    return X[[c for c in columns if c in X.columns]]


def county_ablation(X, y, groups, n_splits=5):
    """M2 (state political only) vs M3 (state + county) on out-of-fold PR-AUC.

    Filters rows with null group keys before CV, mirroring model2_gbm.political_
    ablation exactly. The same valid_mask applies to both the M2-feature model
    and the M3-feature model so the delta is a fair paired comparison.

    Args:
        X: assembled feature matrix from assemble.assemble_features.
        y: int denied target array aligned to X.
        groups: state_abbr Series for grouped CV (StratifiedGroupKFold).
        n_splits: number of CV folds (default 5).
    Returns:
        dict with keys m2_pr_auc (float), m3_pr_auc (float), delta (float),
        and delta_ci (tuple lo, hi).
    """
    from sklearn.metrics import average_precision_score

    # Filter rows without a state grouping key — tribal nations / territories
    # that have no state abbreviation cannot be assigned to a CV fold by
    # StratifiedGroupKFold (numpy sort fails on mixed float-NaN / str arrays).
    groups_s = groups if isinstance(groups, pd.Series) else pd.Series(groups)
    valid_mask = groups_s.notna().values
    X_valid = X.iloc[valid_mask] if hasattr(X, "iloc") else X[valid_mask]
    y_valid = np.asarray(y)[valid_mask]
    groups_valid = groups_s.values[valid_mask]

    y_arr = y_valid
    m2_proba = evaluation.oof_predictions(
        model2_gbm.build_estimator(),
        _slice(X_valid, model2_gbm.feature_columns(include_political=True)),
        y_arr, groups_valid, n_splits,
    )
    m3_proba = evaluation.oof_predictions(
        model2_gbm.build_estimator(),
        _slice(X_valid, feature_columns()),
        y_arr, groups_valid, n_splits,
    )
    delta, lo, hi = evaluation.bootstrap_auc_delta(y_arr, m3_proba, m2_proba)
    return {
        "m2_pr_auc": float(average_precision_score(y_arr, m2_proba)),
        "m3_pr_auc": float(average_precision_score(y_arr, m3_proba)),
        "delta": delta,
        "delta_ci": (lo, hi),
    }


def stronghold_comparison(estimator, X, y_true):
    """Importance of 'one stronghold county' vs 'most counties favored him'.

    Computes feature importances via model2_gbm.shap_summary (SHAP with
    permutation-importance fallback) and extracts the two county-composition
    features that capture the two qualitatively different political hypotheses:
    - max_pres_margin_affected: did ANY single high-margin county get hit?
    - share_affected_counties_pres_won: did MOST affected counties vote for him?

    Args:
        estimator: a fitted M3 pipeline (from model2_gbm.build_estimator).
        X: the M3 feature slice the estimator was fitted on (same columns).
        y_true: array-like of true binary labels aligned to X's rows.
    Returns:
        dict with keys one_county_max_margin (float) and
        most_counties_share_won (float); 0.0 when the feature was absent.
    """
    importance = model2_gbm.shap_summary(estimator, X, y_true)["importance"]
    return {
        "one_county_max_margin": float(
            importance.get("max_pres_margin_affected", 0.0)
        ),
        "most_counties_share_won": float(
            importance.get("share_affected_counties_pres_won", 0.0)
        ),
    }
