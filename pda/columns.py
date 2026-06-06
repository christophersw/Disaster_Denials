"""
Title: PDA CSV Column Order
Description: The exact, ordered column lists for the two output CSVs. Defined
    once so the writers, the flattener, and the tests agree.
Changelog:
    2026-06-05  Initial version.
"""

# Provenance/meta columns are added by the pipeline; the rest mirror schema.PdaReport.
REPORT_COLUMNS = [
    "source_pdf", "report_type", "url", "posted_date",
    "report_outcome", "decision_date", "jurisdiction_name", "state_abbr",
    "requestor_type", "requestor_name", "incident_name",
    "incident_begin", "incident_end", "request_date",
    "disaster_number", "declaration_type", "denial_reason",
    "original_denial_date", "appeal_date",
    "ia_requested", "pa_requested", "hm_requested", "pa_categories_requested",
    "ia_residences_total", "ia_destroyed", "ia_major", "ia_minor", "ia_affected",
    "ia_pct_insured", "ia_pct_flood_insured", "ia_pct_poverty", "ia_pct_ssi",
    "ia_pct_snap", "ia_pct_ownership", "ia_unemployment", "ia_pct_age_65_plus",
    "ia_pct_age_18_under", "ia_pct_disability", "ia_icc_ratio",
    "ia_pct_low_income", "ia_pct_elderly", "ia_cost_estimate",
    "pa_primary_impact", "pa_cost_estimate", "pa_statewide_per_capita",
    "pa_statewide_per_capita_indicator", "pa_countywide_per_capita_indicator",
    "needs_review", "review_note", "parser_model", "extracted_at",
]

COUNTY_COLUMNS = [
    "source_pdf", "county_name", "geo_type", "per_capita_impact",
    "requested_ia", "requested_pa", "granted_ia", "granted_pa", "source",
]
