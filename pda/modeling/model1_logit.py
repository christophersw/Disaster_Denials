# pda/modeling/model1_logit.py
"""
Title: pda/modeling/model1_logit.py — Hierarchical logistic regression (Model 1).
Description:
    The interpretable effect-size model (spec §7, M1). Estimates the odds of
    denial as a function of need, request, state-level political, and enrichment
    features, with random intercepts for state and year so partisan-alignment
    effects are identified within-state/within-year rather than across geography
    or era. Fit on states + DC only (the M1 estimand, §5.7/§7); tribes and
    territories are excluded. Uses statsmodels BinomialBayesMixedGLM (variational
    fit) for the random effects, with a pooled penalized Logit fallback if the
    mixed fit fails to converge given the small (~88) positive class.

    Deviations from the Task-9 brief (documented in the Task-9 report):
      * Categorical detection uses ``not is_numeric_dtype`` rather than
        ``dtype == "object"``: under pandas 3.0 the genuine categoricals
        (requestor_type / governor_party / president_party) carry the new
        ``str`` dtype, which ``== "object"`` would silently miss, leaving
        non-numeric columns to crash the ``astype(float)`` design step.
      * ``request_year`` is coerced to a clean integer ``year`` key and rows
        with an unparseable (NaN) year are DROPPED, never fed as float NaN into
        the C(year) grouping. ``request_year`` itself is removed from the fixed
        effects (the year random effect already carries era variation).
      * Full missingness handling: the brief only median-imputes four cost
        columns, but the real design carries many partially-null numerics. We
        add missingness indicators for the moderately-null cost + county-
        political blocks, neutral-fill (0) the gubernatorial-alignment numerics
        for non-applicable jurisdictions (DC), median-impute every remaining
        numeric, and drop zero-variance / all-null columns so the design matrix
        is fully finite and rank-light enough to fit.
      * ``odds_ratio_table`` reads the mixed result via ``fe_mean`` / ``fe_sd``
        and ``result.model.fep_names`` (the brief's ``result.fe_names`` does not
        exist on BayesMixedGLMResults), and cleans the patsy ``Q('col')`` term
        names back to bare feature names so political rows are directly
        addressable.
      * Continuous predictors are z-scored before fitting. Cost/severity columns
        reach 1e8 unscaled, overflowing the logit link so the variational fit
        collapsed onto the prior (every odds ratio == 1); standardising fixes
        convergence and makes continuous effects per-standard-deviation, while
        binary flags stay on their 0/1 scale (odds ratios as category contrasts).
      * Rows whose state_abbr is null are dropped: a few tribal nations classify
        as 'state' yet carry no state key, and they misalign the variance-
        component design (patsy drops them only from C(state)) — the actual cause
        of the failed mixed fit. Linearly-dependent predictors are then pruned so
        the design is full rank (the county-political missingness indicators are
        identical; DC induces exact collinearity), which both stabilises the
        variational fit and lets the pooled fallback invert its Hessian.
Changelog:
    2026-06-28  Initial version.
"""

import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from pda.modeling import assemble

# Jurisdictions retained for the M1 estimand: the 50 states + DC (§7).
_STATE_ESTIMAND = {"state", "federal_district"}

# Gubernatorial-alignment numerics. For non-applicable jurisdictions (DC has no
# governor) these are neutralised to 0 so DC contributes no fabricated alignment;
# the gubernatorial_alignment_applicable flag carries DC's structural offset.
_GOV_NUMERIC = ["governor_vs_president", "governor_vs_state_vote"]
# Gubernatorial categorical neutralised the same way (set NaN -> all-zero dummies).
_GOV_CATEGORICAL = ["governor_party"]

# Moderately-null numerics: get a missingness indicator before median-impute (§6).
# Cost block + the county-political enrichment block (~11% null where no county
# FIPS match exists).
_INDICATOR_COLS = [
    "pa_statewide_per_capita", "total_cost_estimate",
    "pa_cost_estimate", "ia_cost_estimate",
    "share_affected_counties_pres_won", "max_pres_margin_affected",
    "dmg_weighted_mean_pres_margin", "pres_won_most_damaged_county",
    "pres_margin_dispersion",
]

# Columns dropped from the design after they have served their purpose: the
# estimand selector (jurisdiction_type) and the IA-demographic block (§5.3/§6).
_DROP_FROM_DESIGN = ["jurisdiction_type"]

# Reserved frame columns that are never predictors.
_RESERVED = {"denied", "state", "year"}

# Matches a patsy Q('name') term so we can recover the bare feature name.
_Q_TERM = re.compile(r"^Q\('(.+)'\)$")


def _predictor_columns(frame):
    """Return the fixed-effect predictor column names of a prepared frame.

    Args:
        frame: a frame produced by prepare_logit_frame.
    Returns:
        List of column names excluding the reserved denied/state/year keys.
    """
    return [c for c in frame.columns if c not in _RESERVED]


def _drop_linearly_dependent(df, tol=1e-8):
    """Greedily drop predictors that are linearly dependent on earlier ones.

    Keeps an implicit intercept in the basis and retains a predictor only when
    it raises the rank of the kept set, so the resulting design (with intercept)
    is full rank. Predictors are processed in column order, so the engineered
    political/severity numerics — which appear before the one-hot dummies and
    missingness indicators — are preferentially retained over redundant encodings.

    Args:
        df: a frame whose non-reserved columns are float predictors.
        tol: numerical rank tolerance.
    Returns:
        A copy of df with linearly-dependent predictor columns removed.
    """
    predictors = _predictor_columns(df)
    if not predictors:
        return df
    basis = np.ones((len(df), 1))
    rank = 1
    kept = []
    for col in predictors:
        candidate = np.column_stack([basis, df[col].to_numpy(dtype=float)])
        new_rank = np.linalg.matrix_rank(candidate, tol=tol)
        if new_rank > rank:
            basis = candidate
            rank = new_rank
            kept.append(col)
    redundant = [c for c in predictors if c not in kept]
    return df.drop(columns=redundant)


def prepare_logit_frame(X, y, groups):
    """Build the M1 design frame: restrict estimand, encode, handle missingness.

    Pipeline: filter to states + DC; neutralise DC's gubernatorial features;
    drop the estimand selector and the IA-demographic block; derive a clean
    integer ``year`` key from request_year (dropping unparseable-date rows);
    add missingness indicators and median-impute moderately-null numerics;
    one-hot encode the genuine categoricals; neutral-fill gubernatorial numerics;
    median-impute any residual NaN; and drop zero-variance columns so the design
    matrix is finite and well-posed.

    Args:
        X: assembled feature matrix (index = source_pdf), from assemble_features.
        y: integer 'denied' target Series aligned to X.
        groups: state_abbr Series aligned to X (the state random-effect key).
    Returns:
        A model-ready DataFrame carrying a 'denied' column, a 'state' column, an
        integer 'year' column, and finite float predictor columns (numeric +
        one-hot dummies + missingness indicators), with the IA-demographic block
        removed.
    """
    df = X.copy()
    df["denied"] = np.asarray(y)
    df["state"] = np.asarray(groups)

    # --- M1 estimand: states + DC only (§7) -------------------------------
    df = df[df["jurisdiction_type"].isin(_STATE_ESTIMAND)].copy()

    # Drop rows with no state grouping key: a handful of tribal nations (e.g.
    # Navajo Nation, Oglala Sioux, the Pueblos) fall through jurisdiction
    # classification to 'state' but carry a null state_abbr. They cannot anchor
    # a state random intercept, and leaving them in misaligns the variance-
    # component design (patsy drops them from C(state)), corrupting the fit.
    df = df[df["state"].notna()].copy()

    # --- Neutralise non-applicable gubernatorial features (DC; §5.7) -------
    if "gubernatorial_alignment_applicable" in df.columns:
        not_applicable = df["gubernatorial_alignment_applicable"] == 0
    else:
        not_applicable = pd.Series(False, index=df.index)
    for col in _GOV_NUMERIC + _GOV_CATEGORICAL:
        if col in df.columns:
            df.loc[not_applicable, col] = np.nan

    # --- Drop the estimand selector and the ~84%-null IA block (§5.3/§6) ---
    drop_cols = list(_DROP_FROM_DESIGN) + [
        c for c in assemble.IA_DEMOGRAPHIC_BLOCK if c in df.columns
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --- Year random-effect key from request_year (REQUIRED guard) --------
    # request_year is float64 with NaN for unparseable dates. Coerce to a clean
    # integer label and DROP NaN-year rows (states+DC have ~3); never feed float
    # NaN into the C(year) grouping. Remove request_year from the fixed effects:
    # the year random effect already absorbs era variation.
    if "request_year" in df.columns:
        df["request_year"] = pd.to_numeric(df["request_year"], errors="coerce")
        df = df[df["request_year"].notna()].copy()
        df["year"] = df["request_year"].astype(int)
        df = df.drop(columns=["request_year"])
    else:
        proxy = df.get("presidential_election_year",
                       pd.Series(0, index=df.index))
        df["year"] = pd.to_numeric(proxy, errors="coerce").fillna(0).astype(int)

    # --- Missingness indicators + median-impute moderately-null numerics (§6)
    for col in _INDICATOR_COLS:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}__missing"] = numeric.isna().astype(int)
            df[col] = numeric.fillna(numeric.median())

    # --- Neutral-fill gubernatorial numerics (0 = no alignment signal) -----
    for col in _GOV_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # --- One-hot encode the genuine categoricals --------------------------
    # pandas 3.0 gives string columns the 'str' dtype, so select by "not numeric"
    # rather than "== object" (which would miss them).
    cat_cols = [
        c for c in df.columns
        if c not in _RESERVED and not pd.api.types.is_numeric_dtype(df[c])
    ]
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False, drop_first=True)

    # --- Finalise: cast predictors to float, impute residual NaN, prune ----
    predictors = _predictor_columns(df)
    for col in predictors:
        series = pd.to_numeric(df[col], errors="coerce").astype(float)
        df[col] = series.fillna(series.median())

    # Drop zero-variance / all-null predictor columns (e.g. constant
    # presidential_alignment_applicable in this estimand, all-null
    # pa_primary_impact, all-zero missingness indicators).
    constant = [
        c for c in _predictor_columns(df)
        if df[c].nunique(dropna=True) <= 1
    ]
    df = df.drop(columns=constant)

    # Final safety net: no NaN may reach the fitter.
    remaining = _predictor_columns(df)
    df[remaining] = df[remaining].fillna(0.0)

    # Drop linearly-dependent predictors so the design (with intercept) is
    # full rank. The county-political block shares one missing-data pattern, so
    # its five missingness indicators are identical; DC also induces exact
    # collinearity (requestor_type_Mayor == 1 - gubernatorial_alignment_
    # applicable). A rank-deficient design yields a singular Hessian in the
    # pooled fallback and degrades the variational fit.
    df = _drop_linearly_dependent(df)

    # --- Standardise continuous predictors (z-score) ----------------------
    # Cost/severity/margin features span many orders of magnitude (dollars in
    # the 1e8 range vs 0/1 flags). Left unscaled the linear predictor overflows
    # and the variational fit collapses onto the prior (every odds ratio == 1).
    # Binary flags and one-hot dummies keep their natural 0/1 scale so their
    # odds ratios read as direct category contrasts (e.g. aligned vs not);
    # continuous predictors become per-standard-deviation effects.
    for col in _predictor_columns(df):
        values = df[col].astype(float)
        if set(pd.unique(values.dropna())).issubset({0.0, 1.0}):
            continue  # binary: leave on natural scale
        std = values.std()
        if std > 0:
            df[col] = (values - values.mean()) / std

    return df


def fit_mixed_logit(frame):
    """Fit the mixed-effects logit (random intercepts: state, year).

    Tries a Bayesian variational fit (BinomialBayesMixedGLM.fit_vb) with random
    intercepts for state and year. Falls back to a pooled Logit (MLE, with an
    L2/L1-penalised backstop) if the variational fit raises.

    Args:
        frame: output of prepare_logit_frame.
    Returns:
        A fitted statsmodels results object (mixed BayesMixedGLMResults or pooled
        Logit results); read it via odds_ratio_table, which handles both.
    """
    predictors = _predictor_columns(frame)
    formula = "denied ~ " + " + ".join("Q('%s')" % c for c in predictors)
    vc_formulas = {"state": "0 + C(state)", "year": "0 + C(year)"}
    try:
        model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas, frame)
        result = model.fit_vb(minim_opts={"maxiter": 1000})
    except Exception:
        return _fit_pooled(frame)

    # Guard against a degenerate variational fit that never left the prior
    # (every fixed-effect posterior mean == 0 -> every odds ratio == 1). That is
    # not a usable effect-size table, so prefer the pooled fit in that case.
    if np.allclose(np.asarray(result.fe_mean, dtype=float), 0.0, atol=1e-6):
        return _fit_pooled(frame)
    return result


def _fit_pooled(frame):
    """Pooled-Logit fallback used when the mixed variational fit fails.

    Attempts an ordinary MLE Logit (which yields Wald standard errors and so
    real confidence intervals); if that raises (e.g. perfect separation given
    the small positive class), retries with a penalised fit that returns stable
    point estimates (intervals then collapse to the point estimate).

    Args:
        frame: output of prepare_logit_frame.
    Returns:
        A fitted statsmodels Logit results object.
    """
    predictors = _predictor_columns(frame)
    design = sm.add_constant(frame[predictors].astype(float), has_constant="add")
    target = frame["denied"].astype(int)
    try:
        return sm.Logit(target, design).fit(disp=False, maxiter=200)
    except Exception:
        return sm.Logit(target, design).fit_regularized(alpha=1.0, disp=False)


def _clean_name(name):
    """Strip a patsy Q('col') wrapper back to the bare feature name.

    Args:
        name: a parameter/term name, possibly wrapped as Q('feature').
    Returns:
        The unwrapped feature name, or the input unchanged if not wrapped.
    """
    match = _Q_TERM.match(str(name))
    return match.group(1) if match else str(name)


def odds_ratio_table(result):
    """Return odds ratios and 95% intervals for each fixed-effect coefficient.

    Handles both result types:
      * Bayesian mixed result — posterior mean ± 1.96·posterior sd of the fixed
        effects (fe_mean / fe_sd, names from result.model.fep_names).
      * Pooled Logit fallback — MLE coefficient ± 1.96·bse, or the point estimate
        alone when standard errors are unavailable (penalised fit).

    Args:
        result: a fitted result from fit_mixed_logit.
    Returns:
        DataFrame indexed by (cleaned) feature name with 'odds_ratio',
        'ci_low', and 'ci_high' columns.
    """
    if hasattr(result, "fe_mean"):
        names = [_clean_name(n) for n in result.model.fep_names]
        params = pd.Series(np.asarray(result.fe_mean, dtype=float), index=names)
        sd = pd.Series(np.asarray(result.fe_sd, dtype=float), index=names)
    else:
        params = pd.Series(result.params)
        params.index = [_clean_name(n) for n in params.index]
        try:
            sd = pd.Series(result.bse)
            sd.index = [_clean_name(n) for n in sd.index]
            sd = sd.reindex(params.index).fillna(0.0)
        except Exception:
            sd = pd.Series(0.0, index=params.index)

    table = pd.DataFrame({
        "odds_ratio": np.exp(params),
        "ci_low": np.exp(params - 1.96 * sd),
        "ci_high": np.exp(params + 1.96 * sd),
    })
    return table
