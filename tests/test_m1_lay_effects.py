"""
Title: tests/test_m1_lay_effects.py — Unit tests for the fig6 lay-effects helper.
Description: Verifies _build_m1_lay_effects converts an odds-ratio table into
    probability-point effects with correct signs and magnitudes, flags ranges
    that cross zero as 'unclear', anchors the baseline on the Intercept row, and
    skips selected features absent from the table without crashing.
Changelog:
    2026-06-29  Initial version.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.plot_model_results import (
    _build_m1_lay_effects,
    _logistic,
    _M1_LAY_FACTORS,
)


def _table(rows):
    """Build a synthetic odds-ratio table indexed by feature name.

    Args:
        rows (dict): feature_name -> (odds_ratio, ci_low, ci_high).
    Returns:
        DataFrame with odds_ratio / ci_low / ci_high columns.
    """
    return pd.DataFrame(
        [
            {"odds_ratio": orr, "ci_low": lo, "ci_high": hi}
            for orr, lo, hi in rows.values()
        ],
        index=list(rows.keys()),
    )


def _full_table(baseline_odds=0.1111111):
    """A table with an Intercept row plus every selected factor (all clear)."""
    rows = {"Intercept": (baseline_odds, baseline_odds, baseline_odds)}
    for spec in _M1_LAY_FACTORS:
        rows[spec["feature"]] = (0.7, 0.55, 0.85)  # clearly below 1
    return _table(rows)


def test_baseline_anchored_on_intercept():
    """baseline_p is logistic(log(intercept_odds)) = p implied by the odds."""
    table = _full_table(baseline_odds=0.1 / 0.9)  # odds for p = 0.10
    _, baseline_p = _build_m1_lay_effects(table)
    assert baseline_p == pytest.approx(0.10, abs=1e-6)


def test_protective_factor_is_negative_and_clear():
    """OR < 1 with a CI bounded below 1 => negative effect, unclear False."""
    table = _table({
        "Intercept": (0.1 / 0.9, 0.1 / 0.9, 0.1 / 0.9),  # p = 0.10
        "total_cost_estimate": (0.5, 0.4, 0.6),
    })
    effects, _ = _build_m1_lay_effects(table)
    row = effects.loc["Total damage (disaster size)"]
    assert row["effect_pts"] < 0
    assert row["range_high_pts"] < 0          # whole range below zero
    assert bool(row["unclear"]) is False


def test_magnitude_matches_logistic_delta():
    """A known OR maps to the exact probability-point delta at the baseline."""
    table = _table({
        "Intercept": (0.1 / 0.9, 0.1 / 0.9, 0.1 / 0.9),  # p = 0.10
        "total_cost_estimate": (2.0, 1.5, 2.7),
    })
    effects, baseline_p = _build_m1_lay_effects(table)
    b0 = np.log(0.1 / 0.9)
    expected = (_logistic(b0 + np.log(2.0)) - baseline_p) * 100.0
    assert effects.loc["Total damage (disaster size)", "effect_pts"] == pytest.approx(
        expected, abs=1e-6
    )
    assert effects.loc["Total damage (disaster size)", "effect_pts"] > 0


def test_unclear_when_ci_crosses_one():
    """An OR interval straddling 1 => range straddles 0 => unclear True."""
    table = _table({
        "Intercept": (0.1 / 0.9, 0.1 / 0.9, 0.1 / 0.9),
        "state_party_match": (1.1, 0.7, 1.6),
    })
    effects, _ = _build_m1_lay_effects(table)
    row = effects.loc["State shares president's party"]
    assert bool(row["unclear"]) is True
    assert row["range_low_pts"] < 0 < row["range_high_pts"]


def test_missing_feature_is_skipped_not_fatal(capsys):
    """A selected feature absent from the table is dropped, others survive.

    Also verifies that the skipped feature name appears in the printed warning.
    """
    rows = {"Intercept": (0.1 / 0.9, 0.1 / 0.9, 0.1 / 0.9)}
    # include all but the first selected factor
    for spec in _M1_LAY_FACTORS[1:]:
        rows[spec["feature"]] = (0.8, 0.6, 0.95)
    effects, _ = _build_m1_lay_effects(_table(rows))
    assert _M1_LAY_FACTORS[0]["label"] not in effects.index
    assert len(effects) == len(_M1_LAY_FACTORS) - 1
    captured = capsys.readouterr()
    assert _M1_LAY_FACTORS[0]["feature"] in captured.out


def test_full_table_yields_all_factors_with_columns():
    """A complete table yields one row per selected factor with all columns."""
    effects, _ = _build_m1_lay_effects(_full_table())
    assert list(effects.index) == [s["label"] for s in _M1_LAY_FACTORS]
    for col in ("effect_pts", "range_low_pts", "range_high_pts",
                "unclear", "category", "contrast"):
        assert col in effects.columns


def test_plot_fig6_writes_png(tmp_path):
    """Rendering a small synthetic effects frame produces the PNG file."""
    from scripts.plot_model_results import plot_fig6_lay_effects

    effects_df = pd.DataFrame(
        {
            "effect_pts":     [-5.0, 0.3],
            "range_low_pts":  [-7.0, -1.2],
            "range_high_pts": [-3.0, 1.8],
            "unclear":        [False, True],
            "category":       ["Need", "Political"],
            "contrast":       ["+1 SD", "0→1"],
        },
        index=["Total damage (disaster size)", "State shares president's party"],
    )
    plot_fig6_lay_effects(effects_df, 0.08, str(tmp_path))
    assert (tmp_path / "fig6_lay_effects.png").exists()


def test_const_intercept_fallback():
    """baseline_p is correct when intercept row is keyed 'const', not 'Intercept'."""
    const_odds = 0.25 / 0.75   # odds for p = 0.25
    table = _table({
        "const": (const_odds, const_odds, const_odds),
        "total_cost_estimate": (0.8, 0.6, 0.95),
    })
    _, baseline_p = _build_m1_lay_effects(table)
    expected = _logistic(np.log(const_odds))
    assert baseline_p == pytest.approx(expected, abs=1e-6)


def test_plot_fig6_empty_effects_does_not_crash(tmp_path):
    """The real all-missing producer output is skipped without raising/writing."""
    from scripts.plot_model_results import (
        _build_m1_lay_effects,
        plot_fig6_lay_effects,
    )

    # Intercept-only table => every selected factor is skipped => empty frame.
    table = pd.DataFrame(
        [{"odds_ratio": 0.1 / 0.9, "ci_low": 0.1 / 0.9, "ci_high": 0.1 / 0.9}],
        index=["Intercept"],
    )
    effects_df, baseline_p = _build_m1_lay_effects(table)
    assert len(effects_df) == 0
    plot_fig6_lay_effects(effects_df, baseline_p, str(tmp_path))
    assert not (tmp_path / "fig6_lay_effects.png").exists()
