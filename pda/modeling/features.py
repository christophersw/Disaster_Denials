# pda/modeling/features.py
"""
Title: pda/modeling/features.py — Request-profile and political enrichment.
Description:
    Derived report-level features. request_profile collapses the IA/PA request
    flags into one categorical so the additive logit (Model 1) can see the
    combination it would otherwise miss (spec §5.2). The election-calendar and
    swing-state features add political context not stored in the DB (spec §5.6):
    months to the next federal election, presidential/midterm-year flags, and a
    competitive-state indicator derived from |state_margin|. Also produces
    request_year (integer calendar year) as a grouping key for Model 1 year
    random effects — added per controller instruction, not in original brief.
Changelog:
    2026-06-28  Initial version.
"""

import pandas as pd


def add_request_profile(df):
    """Add request_profile ∈ {IA_only, PA_only, IA_and_PA, neither}.

    Args:
        df: frame with integer/boolean 'ia_requested' and 'pa_requested'.
    Returns:
        Copy of df with a string 'request_profile' column.
    """
    out = df.copy()
    ia = out["ia_requested"].fillna(0).astype(int) == 1
    pa = out["pa_requested"].fillna(0).astype(int) == 1
    profile = pd.Series("neither", index=out.index, dtype="object")
    profile[ia & ~pa] = "IA_only"
    profile[~ia & pa] = "PA_only"
    profile[ia & pa] = "IA_and_PA"
    out["request_profile"] = profile
    return out


def _next_federal_election(date):
    """Return the date of the next federal general election (first Tue Nov).

    Federal general elections fall in even years on the Tuesday after the first
    Monday in November. We approximate the day as Nov 5 (always within the
    valid first-Tuesday-after-first-Monday window for month-scale features).

    Args:
        date: a pandas Timestamp.
    Returns:
        pandas Timestamp for the next federal general election on/after date.
    """
    year = date.year if date.year % 2 == 0 else date.year + 1
    election = pd.Timestamp(year=year, month=11, day=5)
    if date > election:
        election = pd.Timestamp(year=year + 2, month=11, day=5)
    return election


def add_election_features(df, date_col="request_date"):
    """Add months_to_next_election, presidential/midterm-year flags, and request_year.

    Args:
        df: frame with a parseable date column (default 'request_date').
        date_col: name of the date column to read.
    Returns:
        Copy of df with 'months_to_next_election' (float),
        'presidential_election_year' (0/1), 'midterm_election_year' (0/1),
        and 'request_year' (int year, NaN for unparseable dates).
        request_year is used as a year random-effect grouping key in Model 1.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    next_elec = dates.apply(
        lambda d: _next_federal_election(d) if pd.notna(d) else pd.NaT
    )
    out["months_to_next_election"] = (
        (next_elec - dates).dt.days / 30.4375
    )
    years = dates.dt.year
    out["presidential_election_year"] = (
        (years % 4 == 0).fillna(False).astype(int)
    )
    out["midterm_election_year"] = (
        ((years % 2 == 0) & (years % 4 != 0)).fillna(False).astype(int)
    )
    # request_year: integer calendar year for Model 1 year random-effect grouping
    # (controller addition — not in the original task brief)
    out["request_year"] = dates.dt.year
    return out


def add_swing_state(df, threshold=0.08):
    """Add swing_state = 1 when |state_margin| < threshold, else 0 (null → 0).

    Note: state_margin is stored as a fraction (e.g. 0.08 = 8 percentage
    points), not as percentage points.  The threshold is likewise in fraction
    units; the default 0.08 corresponds to an 8-point competitive margin.

    Args:
        df: frame with numeric 'state_margin' (fraction, range roughly −1 to 1).
        threshold: competitiveness cutoff as a fraction (default 0.08 ≈ 8 pts;
            see §5.6).
    Returns:
        Copy of df with an int 'swing_state' column.
    """
    out = df.copy()
    margin = pd.to_numeric(out["state_margin"], errors="coerce")
    out["swing_state"] = (margin.abs() < threshold).fillna(False).astype(int)
    return out


def add_enrichment_features(df, swing_threshold=0.08):
    """Apply request_profile + election + swing-state features in one call.

    Args:
        df: frame with 'ia_requested', 'pa_requested', 'request_date',
            and 'state_margin' columns.
        swing_threshold: competitiveness cutoff passed to add_swing_state,
            expressed as a fraction (default 0.08 ≈ 8 percentage points;
            state_margin is stored as a fraction, not percentage points).
    Returns:
        Copy of df with all enrichment columns applied.
    """
    out = add_request_profile(df)
    out = add_election_features(out)
    out = add_swing_state(out, threshold=swing_threshold)
    return out
