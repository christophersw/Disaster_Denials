# pda/modeling/county_features.py
"""
Title: pda/modeling/county_features.py — Per-disaster county-composition features.
Description:
    Aggregates report_counties up to one row per disaster (source_pdf) to test
    whether the president favors disasters that hit his strongholds (spec §5.5).
    Separates "one county heavily favored him" (max margin, any-county-by-20)
    from "most counties favored him" (share won, damage-weighted mean margin).
    county_margin is the president's vote margin in that county; positive means
    his party carried it (verify sign in the Task 8 EDA gate).
Changelog:
    2026-06-28  Initial version.
"""

import sqlite3

import numpy as np
import pandas as pd

DEFAULT_DB = "data/pda.db"


def load_county_frame(db_path=DEFAULT_DB):
    """Load the county rows needed for composition features.

    Args:
        db_path (str): Path to the SQLite database.

    Returns:
        pandas.DataFrame: Rows from report_counties with columns
            source_pdf, county_margin, per_capita_impact.
    """
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(
            "SELECT source_pdf, county_margin, per_capita_impact "
            "FROM report_counties",
            conn,
        )
    finally:
        conn.close()


def _weighted_mean(margins, weights):
    """Damage-weighted mean margin; falls back to the plain mean if all weights
    are zero/missing so a disaster with no per-capita data still gets a value.

    Args:
        margins (array-like): Per-county president margin values.
        weights (array-like): Per-county per_capita_impact values (non-negative).

    Returns:
        float: Weighted mean of margins, or plain mean if total weight is zero,
            or np.nan if no valid margin values exist.
    """
    margins = pd.to_numeric(margins, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    valid = margins.notna()
    margins, weights = margins[valid], weights[valid]
    if len(margins) == 0:
        return np.nan
    if weights.sum() <= 0:
        return float(margins.mean())
    return float(np.average(margins, weights=weights))


def county_composition_features(county_df):
    """Aggregate county rows to one composition row per disaster.

    Produces seven features per disaster that separate the "one stronghold
    county" hypothesis from the "most counties favored him" hypothesis.

    Note: county_margin is stored as a fraction (e.g. 0.20 = 20 percentage
    points). The pres_won_any_county_by_20plus field uses a threshold of 0.20
    (not 20) to match the fraction scale.

    Args:
        county_df (pandas.DataFrame): Rows with columns source_pdf,
            county_margin (fraction, range roughly −1 to 1), and
            per_capita_impact.

    Returns:
        pandas.DataFrame: One row per disaster indexed by source_pdf with
            columns: num_affected_counties, share_affected_counties_pres_won,
            max_pres_margin_affected, pres_won_any_county_by_20plus,
            dmg_weighted_mean_pres_margin, pres_won_most_damaged_county,
            pres_margin_dispersion.

    Side effects:
        None. Input DataFrame is not modified.
    """
    rows = {}
    for source_pdf, grp in county_df.groupby("source_pdf"):
        margins = pd.to_numeric(grp["county_margin"], errors="coerce")
        valid = margins.dropna()
        impacts = pd.to_numeric(grp["per_capita_impact"], errors="coerce").fillna(0.0)
        if len(valid) == 0:
            won_share = np.nan
            max_margin = np.nan
            any_20 = np.nan
            dispersion = np.nan
            won_worst = np.nan
        else:
            won_share = float((valid > 0).mean())
            max_margin = float(valid.max())
            any_20 = int((valid >= 0.20).any())  # threshold is a fraction (0.20 ≈ 20 pts)
            dispersion = float(valid.std(ddof=0))
            pos = int(impacts.to_numpy().argmax())
            worst_margin = margins.iloc[pos]
            won_worst = int(pd.notna(worst_margin) and worst_margin > 0)
        rows[source_pdf] = {
            "num_affected_counties": int(len(grp)),
            "share_affected_counties_pres_won": won_share,
            "max_pres_margin_affected": max_margin,
            "pres_won_any_county_by_20plus": any_20,
            "dmg_weighted_mean_pres_margin": _weighted_mean(
                grp["county_margin"], grp["per_capita_impact"]
            ),
            "pres_won_most_damaged_county": won_worst,
            "pres_margin_dispersion": dispersion,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "source_pdf"
    return out
