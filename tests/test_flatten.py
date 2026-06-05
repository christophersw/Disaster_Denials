"""Tests for flattening a PdaReport into reports + report_counties rows."""

from pda.columns import COUNTY_COLUMNS, REPORT_COLUMNS
from pda.flatten import flatten
from pda.schema import County, PdaReport


def _denial() -> PdaReport:
    return PdaReport(
        report_outcome="Denied", decision_date="2024-11-20",
        jurisdiction_name="Idaho", state_abbr="ID",
        requestor_type="Governor", requestor_name="Brad Little",
        incident_name="Gwen Fire", incident_begin="2024-07-24",
        incident_end="2024-08-09", request_date="2024-10-07",
        disaster_number=None, declaration_type=None,
        denial_reason="Not of such severity and magnitude.",
        original_denial_date=None, appeal_date=None,
        ia_requested=True, pa_requested=False, hm_requested=True,
        pa_categories_requested=None,
        ia_residences_total=70, ia_destroyed=42, ia_major=0, ia_minor=1,
        ia_affected=27, ia_pct_insured=66.5, ia_pct_flood_insured=None,
        ia_pct_poverty=14.1, ia_pct_ssi=7.8, ia_pct_snap=9.2,
        ia_pct_ownership=99.0, ia_unemployment=3.2, ia_pct_age_65_plus=20.3,
        ia_pct_age_18_under=21.4, ia_pct_disability=17.7, ia_icc_ratio=8.57,
        ia_pct_low_income=None, ia_pct_elderly=None, ia_cost_estimate=1066144,
        pa_primary_impact=None, pa_cost_estimate=None,
        pa_statewide_per_capita=None, pa_statewide_per_capita_indicator=1.84,
        pa_countywide_per_capita_indicator=4.60,
        counties=[County(
            county_name="Nez Perce", geo_type="county", per_capita_impact=None,
            requested_ia=True, requested_pa=False, granted_ia=True, granted_pa=False,
            source="narrative",
        )],
        needs_review=False, review_note=None,
    )


PROV = {"report_type": "Denials", "url": "u", "posted_date": "2024-11-20T12:00:00Z"}
META = {"parser_model": "claude-opus-4-8", "extracted_at": "2026-06-05T00:00:00Z"}
PDF = "data/pdfs/Denials/2024/PDAReport_Denial-ID.pdf"


def test_one_report_row_with_provenance_and_meta():
    report_row, _ = flatten(_denial(), PDF, PROV, META)
    assert report_row["source_pdf"] == PDF
    assert report_row["report_type"] == "Denials"
    assert report_row["parser_model"] == "claude-opus-4-8"
    assert set(report_row.keys()) == set(REPORT_COLUMNS)


def test_county_rows_carry_fk_and_columns():
    _, county_rows = flatten(_denial(), PDF, PROV, META)
    assert len(county_rows) == 1
    assert county_rows[0]["source_pdf"] == PDF
    assert county_rows[0]["county_name"] == "Nez Perce"
    assert set(county_rows[0].keys()) == set(COUNTY_COLUMNS)


def test_denial_forces_granted_false():
    """A denial must never record granted assistance, even if the model slips."""
    report = _denial()
    report.counties[0].granted_ia = True   # model error
    report.counties[0].granted_pa = True
    _, county_rows = flatten(report, PDF, PROV, META)
    assert county_rows[0]["granted_ia"] is False
    assert county_rows[0]["granted_pa"] is False


def test_report_with_no_counties_emits_zero_county_rows():
    report = _denial()
    report.counties = []
    _, county_rows = flatten(report, PDF, PROV, META)
    assert county_rows == []
