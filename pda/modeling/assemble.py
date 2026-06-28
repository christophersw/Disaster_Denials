# pda/modeling/assemble.py
"""
Title: pda/modeling/assemble.py — Feature assembly and column groups.
Description:
    Builds the single per-report feature matrix every model consumes. Loads the
    modeling frame, joins county-composition features, and adds jurisdiction and
    enrichment features, then selects ONLY allow-listed column groups so no
    leakage column (spec §4) can reach a model. Column groups are exported so
    each model can choose its own subset (e.g. Model 2 omits POLITICAL_COUNTY;
    the logit omits IA_DEMOGRAPHIC_BLOCK).
Changelog:
    2026-06-28  Initial version. request_year added to ENRICHMENT per controller
                instruction (needed by Model 1 as a year random-effect key).
"""

import pandas as pd

from pda.modeling import county_features, data, features, jurisdiction

NEED = [
    "total_cost_estimate", "ia_cost_estimate", "pa_cost_estimate",
    "pa_statewide_per_capita", "pa_statewide_per_capita_indicator",
    "pa_countywide_per_capita_indicator", "pa_primary_impact",
    "ia_residences_total", "ia_destroyed", "ia_major", "ia_minor", "ia_affected",
]
REQUEST = [
    "ia_requested", "pa_requested", "hm_requested", "request_profile",
    "requestor_type",
]
POLITICAL_STATE = [
    "state_party_match", "governor_party", "governor_vs_president",
    "governor_vs_state_vote", "state_margin", "state_dem_share",
    "president_party",
]
POLITICAL_COUNTY = [
    "num_affected_counties", "share_affected_counties_pres_won",
    "max_pres_margin_affected", "pres_won_any_county_by_20plus",
    "dmg_weighted_mean_pres_margin", "pres_won_most_damaged_county",
    "pres_margin_dispersion",
]
# request_year added per controller instruction: needed by Model 1 as a year
# random-effect grouping key. It is produced by features.add_election_features.
ENRICHMENT = [
    "months_to_next_election", "presidential_election_year",
    "midterm_election_year", "swing_state", "request_year",
]
JURISDICTION = [
    "jurisdiction_type", "presidential_alignment_applicable",
    "gubernatorial_alignment_applicable",
]
# Kept available for the trees only (native NaN handling); dropped by the logit.
IA_DEMOGRAPHIC_BLOCK = [
    "ia_pct_poverty", "ia_pct_snap", "ia_pct_ssi", "ia_unemployment",
    "ia_pct_age_65_plus", "ia_pct_age_18_under", "ia_pct_disability",
    "ia_pct_insured", "ia_pct_flood_insured", "ia_pct_ownership",
    "ia_icc_ratio", "ia_pct_low_income", "ia_pct_elderly",
]

# Genuine categoricals; everything else in a group is coerced to numeric so the
# tree encoder (make_column_selector on object dtype) picks exactly these.
CATEGORICAL = [
    "request_profile", "requestor_type", "governor_party", "president_party",
    "jurisdiction_type",
]
GROUP_KEY = "state_abbr"


def assemble_features(db_path=data.DEFAULT_DB):
    """Assemble the per-report feature matrix, target, and CV grouping key.

    Args:
        db_path: path to the SQLite database.
    Returns:
        (X, y, groups): X is the allow-listed feature matrix (index=source_pdf,
        no identifier columns); y is the int 'denied' target; groups is the
        state_abbr Series used only for grouped CV.
    """
    frame, target = data.load_modeling_frame(db_path)
    frame = jurisdiction.add_jurisdiction_features(frame)
    frame = features.add_enrichment_features(frame)

    counties = county_features.load_county_frame(db_path)
    comp = county_features.county_composition_features(counties)
    frame = frame.join(comp, how="left")

    keep = (
        NEED + REQUEST + POLITICAL_STATE + POLITICAL_COUNTY
        + ENRICHMENT + JURISDICTION + IA_DEMOGRAPHIC_BLOCK
    )
    keep = [c for c in keep if c in frame.columns]
    X = frame[keep].copy()

    # Coerce non-categorical columns to numeric ('' / text-nulls → NaN) so only
    # the genuine categoricals remain object dtype for the encoder.
    for col in X.columns:
        if col not in CATEGORICAL:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    groups = frame[GROUP_KEY].loc[X.index]
    return X, target.loc[X.index], groups
