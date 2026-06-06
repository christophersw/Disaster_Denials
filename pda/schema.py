"""
Title: PDA Extraction Schema
Description: Pydantic models defining the exact data contract the LLM must
    return for one FEMA PDA report, plus the JSON Schema used for structured
    outputs. Single source of truth for the report-level and county-level
    fields. `extra="forbid"` makes the generated schema emit
    additionalProperties:false, which structured outputs require.
Changelog:
    2026-06-05  Initial version (normalized two-table design).
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class County(BaseModel):
    """One geographic unit named in a report.

    Attributes:
        county_name: Raw name as printed (null only when no unit is named).
        geo_type: The kind of unit, for filtering non-county jurisdictions.
        per_capita_impact: Dollar figure from the PA countywide per-capita list.
        requested_ia/requested_pa: Governor requested this unit for IA/PA.
        granted_ia/granted_pa: Assistance actually made available here
            (always false for denials/appeals).
        source: Where in the document this unit was found.
    """

    model_config = ConfigDict(extra="forbid")

    county_name: Optional[str]
    geo_type: Literal[
        "county", "parish", "borough", "tribe", "reservation",
        "city-county", "municipality", "unknown",
    ]
    per_capita_impact: Optional[float]
    requested_ia: bool
    requested_pa: bool
    granted_ia: bool
    granted_pa: bool
    source: Literal["per_capita", "narrative", "both", "none"]


class PdaReport(BaseModel):
    """All report-level (state-level) fields for one PDA report PDF."""

    model_config = ConfigDict(extra="forbid")

    # Identity & outcome
    report_outcome: Optional[Literal["Declared", "Denied", "Denial of Appeal"]]
    decision_date: Optional[str]            # YYYY-MM-DD
    jurisdiction_name: Optional[str]        # state or tribe
    state_abbr: Optional[str]               # two-letter USPS; null for tribes
    requestor_type: Optional[str]           # Governor / Tribal Chairman / ...
    requestor_name: Optional[str]
    incident_name: Optional[str]
    incident_begin: Optional[str]           # YYYY-MM-DD
    incident_end: Optional[str]             # YYYY-MM-DD
    request_date: Optional[str]             # YYYY-MM-DD
    disaster_number: Optional[int]          # numeric part only; null for denials
    declaration_type: Optional[Literal["DR", "EM"]]
    denial_reason: Optional[str]
    original_denial_date: Optional[str]     # appeals only
    appeal_date: Optional[str]              # appeals only

    # Requested programs
    ia_requested: bool
    pa_requested: bool
    hm_requested: bool
    pa_categories_requested: Optional[str]  # e.g. "A,B" or "A-F"

    # Individual Assistance (state-level)
    ia_residences_total: Optional[float]
    ia_destroyed: Optional[float]
    ia_major: Optional[float]
    ia_minor: Optional[float]
    ia_affected: Optional[float]
    ia_pct_insured: Optional[float]
    ia_pct_flood_insured: Optional[float]
    ia_pct_poverty: Optional[float]
    ia_pct_ssi: Optional[float]
    ia_pct_snap: Optional[float]
    ia_pct_ownership: Optional[float]
    ia_unemployment: Optional[float]
    ia_pct_age_65_plus: Optional[float]
    ia_pct_age_18_under: Optional[float]
    ia_pct_disability: Optional[float]
    ia_icc_ratio: Optional[float]
    ia_pct_low_income: Optional[float]      # legacy (older reports)
    ia_pct_elderly: Optional[float]         # legacy (older reports)
    ia_cost_estimate: Optional[float]

    # Public Assistance (state-level)
    pa_primary_impact: Optional[str]
    pa_cost_estimate: Optional[float]
    pa_statewide_per_capita: Optional[float]
    pa_statewide_per_capita_indicator: Optional[float]
    pa_countywide_per_capita_indicator: Optional[float]

    # Counties
    counties: list[County]

    # Review
    needs_review: bool
    review_note: Optional[str]


def json_schema() -> dict:
    """Return the JSON Schema for structured outputs (additionalProperties:false)."""
    return PdaReport.model_json_schema()
