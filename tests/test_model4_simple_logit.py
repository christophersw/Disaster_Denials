# tests/test_model4_simple_logit.py
"""
Title: tests/test_model4_simple_logit.py — Contract tests for Model 4.
Description: Verifies the simple-logit design frame (curated columns, explicit
    reference categories, neutral-fill + missingness flags, standardization),
    the odds-ratio table shape and ranking, the detect-and-drop separation policy,
    and grouped-CV fit quality. Heavy real-DB fits are skipped when data/pda.db is
    absent.
Changelog:
    2026-06-29  Initial version.
"""

import os

import numpy as np
import pandas as pd
import pytest

from pda.modeling import model4_simple_logit


def _toy_X():
    """A 4-row feature matrix spanning all four jurisdiction classes."""
    return pd.DataFrame(
        {
            "total_cost_estimate": [1_000_000.0, 5_000_000.0, np.nan, 250_000.0],
            "pa_statewide_per_capita": [3.5, 8.0, 1.0, np.nan],
            "request_profile": ["IA_only", "PA_only", "IA_and_PA", "neither"],
            "jurisdiction_type": ["state", "territory", "federal_district", "tribal"],
            "share_affected_counties_pres_won": [0.6, np.nan, 0.4, np.nan],
            "governor_vs_president": [1.0, 0.0, np.nan, np.nan],
            "months_to_next_election": [12.0, 3.0, 20.0, 7.0],
            "gubernatorial_alignment_applicable": [1, 1, 0, 0],
        },
        index=["a", "b", "c", "d"],
    )


def test_prepare_selects_curated_columns_and_drops_helpers():
    """Curated predictors are present; the collinear applicability flag is gone."""
    X = _toy_X()
    y = pd.Series([0, 1, 0, 1], index=X.index, name="denied")
    frame = model4_simple_logit.prepare_simple_logit_frame(X, y)

    assert "denied" in frame.columns
    assert "total_cost_estimate" in frame.columns
    assert "governor_vs_president" in frame.columns
    # Explicit reference categories dropped (state, PA_only); other levels kept.
    assert "jurisdiction_type_territory" in frame.columns
    assert "jurisdiction_type_state" not in frame.columns
    assert "request_profile_IA_only" in frame.columns
    assert "request_profile_PA_only" not in frame.columns
    # The applicability flag is excluded as collinear with the jurisdiction dummies.
    assert "gubernatorial_alignment_applicable" not in frame.columns
    # No NaN may reach the fitter.
    predictors = [c for c in frame.columns if c != "denied"]
    assert not frame[predictors].isna().any().any()


def test_prepare_neutral_fills_undefined_governor():
    """Governor flag is 0 for DC/tribal rows; county-missing flag set where null."""
    X = _toy_X()
    y = pd.Series([0, 1, 0, 1], index=X.index, name="denied")
    frame = model4_simple_logit.prepare_simple_logit_frame(X, y)
    # Rows c (federal_district) and d (tribal) had NaN governor -> filled to 0.
    # governor_vs_president is binary, so it is NOT z-scored and stays on 0/1 scale.
    assert frame.loc["c", "governor_vs_president"] == 0.0
    assert frame.loc["d", "governor_vs_president"] == 0.0
    # County share was null for rows b and d -> indicator 1 there, 0 elsewhere.
    miss = "share_affected_counties_pres_won__missing"
    assert frame.loc["b", miss] == 1
    assert frame.loc["d", miss] == 1
    assert frame.loc["a", miss] == 0


def test_prepare_standardizes_continuous_not_binary():
    """Continuous predictors are z-scored (mean~0, sd~1); 0/1 flags untouched."""
    X = _toy_X()
    y = pd.Series([0, 1, 0, 1], index=X.index, name="denied")
    frame = model4_simple_logit.prepare_simple_logit_frame(X, y)
    timing = frame["months_to_next_election"]
    assert abs(float(timing.mean())) < 1e-9
    assert abs(float(timing.std()) - 1.0) < 1e-9
    # Binary governor flag keeps its 0/1 values.
    assert set(pd.unique(frame["governor_vs_president"])).issubset({0.0, 1.0})


def _separable_frame():
    """A frame where sep_feature perfectly equals the label (complete separation).

    noise_feature is unrelated to the label, so after sep_feature is dropped the
    refit converges cleanly — the policy should drop exactly one feature.
    """
    rng = np.random.default_rng(0)
    n = 60
    denied = np.array([0] * 45 + [1] * 15)
    return pd.DataFrame(
        {
            "denied": denied,
            "sep_feature": denied.astype(float),   # perfectly separates
            "noise_feature": rng.normal(size=n),   # unrelated
        }
    )


def test_separation_drops_offending_feature():
    """A separating design drops that feature and returns finite ORs for the rest."""
    frame = _separable_frame()
    result = model4_simple_logit.fit_simple_logit(frame)
    assert "sep_feature" in result.dropped_features
    table = model4_simple_logit.odds_ratio_table(result)
    assert "sep_feature" not in table.index
    assert "noise_feature" in table.index
    assert np.isfinite(table["odds_ratio"].to_numpy()).all()
    assert np.isfinite(table[["ci_low", "ci_high"]].to_numpy()).all()


def test_clean_fit_drops_nothing_and_exp_consistent():
    """A non-separating design drops nothing; ORs are exp-consistent with coefs."""
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    logits = -2.0 + 0.9 * x
    denied = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    frame = pd.DataFrame({"denied": denied, "x": (x - x.mean()) / x.std()})
    result = model4_simple_logit.fit_simple_logit(frame)
    assert result.dropped_features == []
    table = model4_simple_logit.odds_ratio_table(result)
    assert list(table.columns) == ["odds_ratio", "ci_low", "ci_high",
                                   "p_value", "std_coef"]
    row = table.loc["x"]
    assert np.isclose(row["odds_ratio"], np.exp(row["std_coef"]))
    assert row["ci_low"] < row["odds_ratio"] < row["ci_high"]


@pytest.mark.skipif(not os.path.exists("data/pda.db"), reason="needs data/pda.db")
def test_fit_and_cv_real_db():
    """End-to-end: fit on the real DB, emit a finite ranked table + CV metrics."""
    from pda.modeling import assemble
    X, y, groups = assemble.assemble_features("data/pda.db")
    frame = model4_simple_logit.prepare_simple_logit_frame(X, y)
    result = model4_simple_logit.fit_simple_logit(frame)
    table = model4_simple_logit.odds_ratio_table(result)
    assert (table["odds_ratio"] > 0).all()
    assert np.isfinite(table["odds_ratio"].to_numpy()).all()
    assert np.isfinite(table[["ci_low", "ci_high"]].to_numpy()).all()
    # Political features survive into the table unless separation dropped them.
    survivors = set(table.index) | set(result.dropped_features)
    assert "governor_vs_president" in survivors
    assert "share_affected_counties_pres_won" in survivors

    scores = model4_simple_logit.cv_fit_quality(frame, groups)
    assert set(scores) == {"roc_auc", "pr_auc", "brier"}
    assert 0.0 <= scores["roc_auc"] <= 1.0
    assert 0.0 <= scores["pr_auc"] <= 1.0
    assert np.isfinite(scores["brier"])
