# pda/modeling/model4_simple_logit.py
"""
Title: pda/modeling/model4_simple_logit.py — Simple single-level logit (Model 4).
Description:
    A flat, interpretable logistic regression over the full PDA population
    (spec docs/superpowers/specs/2026-06-29-model4-simple-logit-design.md). Unlike
    Model 1 it has NO state/year hierarchy: it competes a curated ~14-column design
    and reports an odds-ratio table ranked by |standardized coefficient|, answering
    "which features matter most to the Declared-vs-Denied decision." Continuous
    predictors are log/z-scaled so coefficients are per-standard-deviation; binary
    flags and one-hot dummies stay on their 0/1 scale as category contrasts.

    Political features undefined on some jurisdictions (governor_vs_president for
    DC/tribes; county vote share for territories/tribes) are neutral-filled; the
    jurisdiction_type dummies mark and absorb those rows, so a separate
    gubernatorial_alignment_applicable flag would be collinear and is omitted.

    Fit: plain statsmodels Logit (MLE, Wald CIs/p-values) on the natural class
    balance — NO class weighting (this is an inferential model). Separation policy
    is detect-and-drop, instrumented: with a rare outcome a feature can perfectly
    separate denials and send the MLE coefficient to +/-inf; when that happens the
    most-offending predictor is identified, dropped, and the model refit, with the
    dropped feature(s) recorded so their frequency is visible across runs. Firth's
    penalized likelihood is the documented upgrade if separation proves frequent
    (firthlogist does not support this project's Python 3.14, so a future Firth would
    be a from-scratch penalized IRLS swapped into the fallback branch).
Changelog:
    2026-06-29  Initial version (MLE + detect-and-drop separation policy).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

# The only non-predictor column carried through the design frame.
_RESERVED = {"denied"}

# Continuous need/severity features: log1p (heavy right-skew) then z-score.
_LOG_FEATURES = ["total_cost_estimate", "pa_statewide_per_capita"]
# County-stronghold political feature (continuous, [0, 1]); z-scored.
_COUNTY_FEATURE = "share_affected_counties_pres_won"
# Governor-alignment political feature (binary 0/1); kept on natural scale.
_GOV_FEATURE = "governor_vs_president"
# Election-timing feature (continuous); z-scored.
_TIMING_FEATURE = "months_to_next_election"

# Genuine categoricals. The FIRST listed level is the reference (dropped by
# get_dummies drop_first), chosen as a well-populated, interpretable baseline:
# 'state' for jurisdiction (1,166 reports) and 'PA_only' for request profile (the
# most common request type, 657 reports). 'neither' is NOT used as the reference:
# it is a single pathological report (requested neither IA nor PA, and was denied),
# so as a reference it would leave the whole request-profile block unidentified
# (huge standard errors); kept as a dummy instead, it is a complete zero-cell that
# the separation policy drops cleanly. Every observed value must appear in the list.
_CATEGORICALS = {
    "jurisdiction_type": ["state", "territory", "federal_district", "tribal"],
    "request_profile": ["PA_only", "IA_only", "IA_and_PA", "neither"],
}

# A fitted coefficient larger than this on the log-odds scale (OR > 7e10) is a
# (quasi-)separation artifact, not a real effect — it triggers the drop policy.
_SEPARATION_COEF_THRESHOLD = 25.0


def _predictor_columns(frame):
    """Return the predictor column names of a prepared frame (excludes 'denied').

    Args:
        frame: a frame produced by prepare_simple_logit_frame.
    Returns:
        List of column names excluding the reserved 'denied' key.
    """
    return [c for c in frame.columns if c not in _RESERVED]


def prepare_simple_logit_frame(X, y):
    """Build Model 4's curated, finite, single-level logit design frame.

    Pipeline: select the curated predictors; add a missingness indicator and
    median-impute then log1p the two cost features; missingness-indicate and
    median-impute the county feature; neutral-fill (0) the governor flag where
    undefined; median-impute election timing; one-hot encode the two categoricals
    with explicit reference categories; cast predictors to float and fill any
    residual NaN; drop zero-variance columns; and z-score the continuous predictors
    (binary flags / dummies stay on their 0/1 scale).

    Args:
        X: assembled feature matrix (index = source_pdf), from assemble_features.
        y: integer 'denied' target Series aligned to X.
    Returns:
        A DataFrame indexed like X carrying an int 'denied' column and finite float
        predictor columns. No 'gubernatorial_alignment_applicable' column (excluded
        as collinear with the jurisdiction_type dummies).
    """
    frame = pd.DataFrame(index=X.index)
    frame["denied"] = pd.Series(y).reindex(X.index).astype(int)

    # Cost / per-capita need features: indicator + median-impute + log1p.
    for col in _LOG_FEATURES:
        numeric = pd.to_numeric(X[col], errors="coerce")
        frame[f"{col}__missing"] = numeric.isna().astype(int)
        filled = numeric.fillna(numeric.median())
        frame[col] = np.log1p(filled.clip(lower=0))

    # County stronghold share: indicator + median-impute.
    county = pd.to_numeric(X[_COUNTY_FEATURE], errors="coerce")
    frame[f"{_COUNTY_FEATURE}__missing"] = county.isna().astype(int)
    frame[_COUNTY_FEATURE] = county.fillna(county.median())

    # Governor alignment: neutral-fill 0 where undefined (DC/tribes). The
    # jurisdiction_type dummies mark and absorb those rows (see module docstring).
    frame[_GOV_FEATURE] = pd.to_numeric(X[_GOV_FEATURE], errors="coerce").fillna(0.0)

    # Election timing: median-impute any unparseable-date rows.
    timing = pd.to_numeric(X[_TIMING_FEATURE], errors="coerce")
    frame[_TIMING_FEATURE] = timing.fillna(timing.median())

    # Categoricals with explicit reference (first listed level dropped).
    for col, categories in _CATEGORICALS.items():
        frame[col] = pd.Categorical(X[col].astype("object"), categories=categories)
    frame = pd.get_dummies(frame, columns=list(_CATEGORICALS), drop_first=True)

    # Cast predictors to float and fill any residual NaN (e.g. all-null median).
    for col in _predictor_columns(frame):
        series = pd.to_numeric(frame[col], errors="coerce").astype(float)
        frame[col] = series.fillna(series.median()).fillna(0.0)

    # Drop zero-variance predictors (e.g. an all-zero missingness indicator).
    constant = [c for c in _predictor_columns(frame)
                if frame[c].nunique(dropna=True) <= 1]
    frame = frame.drop(columns=constant)

    # Standardize continuous predictors; binary flags / dummies keep 0/1 scale.
    for col in _predictor_columns(frame):
        values = frame[col].astype(float)
        if set(pd.unique(values.dropna())).issubset({0.0, 1.0}):
            continue
        std = values.std()
        if std > 0:
            frame[col] = (values - values.mean()) / std

    return frame


@dataclass
class SimpleLogitResult:
    """Container for a fitted Model 4 logit and its separation bookkeeping.

    Attributes:
        feature_names: retained predictor names (intercept excluded), aligned to
            the arrays below.
        params: fitted coefficients on the standardized design.
        bse: coefficient standard errors.
        pvalues: two-sided Wald p-values per coefficient.
        ci_low / ci_high: 95% Wald interval bounds on the COEFFICIENT (log-odds) scale.
        pseudo_r2: McFadden pseudo-R^2 of the retained-design fit.
        llr_pvalue: likelihood-ratio-test p-value of the retained-design fit.
        dropped_features: predictors removed by the detect-and-drop separation
            policy, in the order they were dropped (empty when the fit was clean).
    """
    feature_names: List[str]
    params: np.ndarray
    bse: np.ndarray
    pvalues: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    pseudo_r2: float
    llr_pvalue: Optional[float]
    dropped_features: List[str]


def _try_mle(frame, predictors, target):
    """Attempt an MLE Logit fit on the given predictors with a robust optimizer.

    Uses BFGS rather than statsmodels' default Newton step: Newton inverts the
    Hessian each iteration and raises LinAlgError when a separated feature makes it
    singular, whereas BFGS converges to the same MLE on well-posed designs without
    that failure. Wald standard errors still come from the analytic Hessian at the
    optimum (optimizer-independent), so inference is unchanged.

    Args:
        frame: prepared Model 4 design frame.
        predictors: predictor column names to include this attempt.
        target: int 'denied' Series.
    Returns:
        A fitted statsmodels Logit results object, or None if the fit raised.
    """
    design = sm.add_constant(frame[predictors].astype(float), has_constant="add")
    try:
        return sm.Logit(target, design).fit(method="bfgs", disp=False, maxiter=1000)
    except Exception:
        return None


def _find_separating_dummy(frame, predictors, target):
    """Return a binary predictor that perfectly separates the outcome, or None.

    A 0/1 predictor causes complete separation when every row in one of its levels
    shares a single outcome (e.g. DC: 8 reports, 0 denials → the federal_district
    dummy perfectly predicts "not denied"). Such a coefficient is unidentified (the
    MLE runs to ±∞), so the feature must be dropped. Detection is from the DATA, not
    from optimizer fallout, so it pinpoints the genuine separator without disturbing
    strong-but-legitimate predictors. Only complete zero-cells qualify; quasi-
    separation (a rare but nonzero minority count) is left to the fit.

    Args:
        frame: prepared Model 4 design frame.
        predictors: the predictor names currently in the design.
        target: int 'denied' Series.
    Returns:
        The name of a perfectly-separating binary predictor, or None if there is
        none. When several separate, returns the one with the smallest minority
        level (the most extreme zero-cell) for a deterministic drop order.
    """
    outcome = target.to_numpy()
    candidates = []
    for col in predictors:
        values = frame[col].to_numpy(dtype=float)
        if not set(np.unique(values)).issubset({0.0, 1.0}):
            continue  # only binary predictors can form a 2x2 zero-cell
        on = outcome[values == 1.0]
        off = outcome[values == 0.0]
        if on.size == 0 or off.size == 0:
            continue  # constant column (should not survive the zero-variance drop)
        if on.min() == on.max() or off.min() == off.max():
            candidates.append((min(on.size, off.size), col))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _is_separated(result):
    """Return True when a fit shows the residual signature of separation.

    Flags a fit that failed outright, returned a non-finite standard error, or
    produced a pathologically large coefficient (|coef| > threshold on the log-odds
    scale). Mere non-convergence with finite, sane estimates is NOT flagged — that
    is an optimizer artifact, not separation, and dropping a feature for it would
    remove a legitimate predictor.

    Args:
        result: a statsmodels Logit results object, or None.
    Returns:
        bool — True if the fit is unusable for an honest odds-ratio table.
    """
    if result is None:
        return True
    if not np.all(np.isfinite(np.asarray(result.bse, dtype=float))):
        return True
    params = result.params.drop("const", errors="ignore")
    if len(params):
        worst = float(np.max(np.abs(params.to_numpy(dtype=float))))
        if worst > _SEPARATION_COEF_THRESHOLD:
            return True
    return False


def _most_separating_predictor(frame, predictors, target, result):
    """Identify the predictor most responsible for residual (continuous) separation.

    Reached only after complete zero-cell dummies are already removed, so this
    handles the rarer case of a continuous feature driving a coefficient past the
    pathological threshold. Prefers the largest finite |coefficient| from the fit;
    if the fit raised entirely, ranks candidates with a one-shot L2-penalized logit
    used ONLY to choose which column to drop (never reported).

    Args:
        frame: prepared Model 4 design frame.
        predictors: the predictor names currently in the design.
        target: int 'denied' Series.
        result: the offending statsmodels result, or None.
    Returns:
        The name of the predictor to drop.
    """
    if result is not None:
        params = result.params.reindex(predictors)
        finite = params[np.isfinite(params.to_numpy(dtype=float))]
        if len(finite):
            return str(finite.abs().idxmax())

    from sklearn.linear_model import LogisticRegression

    design = frame[predictors].astype(float).to_numpy()
    ranker = LogisticRegression(C=1.0, max_iter=1000)
    ranker.fit(design, target.to_numpy())
    weights = pd.Series(np.abs(ranker.coef_[0]), index=predictors)
    return str(weights.idxmax())


def _result_from_mle(result, predictors, dropped):
    """Build a SimpleLogitResult from a converged statsmodels Logit (MLE) result.

    Args:
        result: a converged statsmodels Logit results object.
        predictors: retained predictor column names, in design order.
        dropped: predictors removed by the separation policy (in drop order).
    Returns:
        A SimpleLogitResult.
    """
    conf = result.conf_int()
    return SimpleLogitResult(
        feature_names=list(predictors),
        params=result.params.reindex(predictors).to_numpy(dtype=float),
        bse=result.bse.reindex(predictors).to_numpy(dtype=float),
        pvalues=result.pvalues.reindex(predictors).to_numpy(dtype=float),
        ci_low=conf.reindex(predictors)[0].to_numpy(dtype=float),
        ci_high=conf.reindex(predictors)[1].to_numpy(dtype=float),
        pseudo_r2=float(result.prsquared),
        llr_pvalue=float(result.llr_pvalue),
        dropped_features=list(dropped),
    )


def fit_simple_logit(frame):
    """Fit Model 4: plain MLE Logit with the detect-and-drop separation policy.

    Two-stage separation policy. First, any binary predictor that perfectly
    separates the outcome (a complete zero-cell, e.g. DC's 0 denials in 8 reports)
    is removed up front — detected from the data so the genuine separator is
    pinpointed without disturbing strong legitimate predictors. Then the MLE is fit
    (robust BFGS); if a residual continuous feature still drives a coefficient past
    the pathological threshold, the largest-|coef| predictor is dropped and the
    model refit, repeating until the fit is clean. Dropped predictors are recorded
    on the result so their frequency is visible across runs.

    Args:
        frame: output of prepare_simple_logit_frame.
    Returns:
        A SimpleLogitResult; read it via odds_ratio_table.
    Raises:
        RuntimeError: if every predictor is dropped (a degenerate design).
    """
    target = frame["denied"].astype(int)
    remaining = _predictor_columns(frame)
    dropped = []

    while remaining:
        # Stage 1: remove a complete zero-cell separator (data-based, exact).
        separating_dummy = _find_separating_dummy(frame, remaining, target)
        if separating_dummy is not None:
            remaining = [c for c in remaining if c != separating_dummy]
            dropped.append(separating_dummy)
            continue
        # Stage 2: fit; accept unless a residual coefficient is still pathological.
        result = _try_mle(frame, remaining, target)
        if not _is_separated(result):
            return _result_from_mle(result, remaining, dropped)
        offender = _most_separating_predictor(frame, remaining, target, result)
        remaining = [c for c in remaining if c != offender]
        dropped.append(offender)

    raise RuntimeError(
        "Model 4: every predictor was dropped for separation; design is degenerate."
    )


def odds_ratio_table(result):
    """Return odds ratios, 95% Wald intervals, p-values, and the ranking key.

    Args:
        result: a SimpleLogitResult from fit_simple_logit.
    Returns:
        DataFrame indexed by retained feature name with columns 'odds_ratio',
        'ci_low', 'ci_high', 'p_value', and 'std_coef' (the standardized
        coefficient), sorted by |std_coef| descending — the "which features matter
        most" order. Dropped features are NOT rows here; read result.dropped_features.
    """
    table = pd.DataFrame(
        {
            "odds_ratio": np.exp(result.params),
            "ci_low": np.exp(result.ci_low),
            "ci_high": np.exp(result.ci_high),
            "p_value": result.pvalues,
            "std_coef": result.params,
        },
        index=result.feature_names,
    )
    order = table["std_coef"].abs().sort_values(ascending=False).index
    return table.loc[order]


def cv_fit_quality(frame, groups, n_splits=5):
    """Grouped-CV discrimination for the Model 4 design (a fit-quality check).

    Mirrors the prepared design into a plain sklearn LogisticRegression (its
    default mild L2) and scores pooled out-of-fold predictions via the shared
    StratifiedGroupKFold harness, grouped by state. The mild penalty keeps every
    held-out refit stable even when a fold is near-separated; it is used ONLY to
    measure discrimination and never feeds the reported coefficients. Rows lacking
    a state grouping key are dropped from this CV pass (as in Models 2/3); the
    odds-ratio table itself uses the full design.

    Args:
        frame: output of prepare_simple_logit_frame.
        groups: state_abbr Series aligned to the frame's index.
        n_splits: number of CV folds.
    Returns:
        dict with float 'roc_auc', 'pr_auc', and 'brier'.
    """
    from sklearn.linear_model import LogisticRegression

    from pda.modeling import evaluation

    predictors = _predictor_columns(frame)
    design = frame[predictors].astype(float)
    target = frame["denied"].astype(int).to_numpy()
    grp = pd.Series(groups).reindex(frame.index)
    keep = grp.notna().to_numpy()

    estimator = LogisticRegression(max_iter=1000)
    return evaluation.cv_scores(
        estimator,
        design.loc[keep].reset_index(drop=True),
        target[keep],
        grp[keep].to_numpy(),
        n_splits=n_splits,
    )
