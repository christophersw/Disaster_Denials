"""End-to-end test on one real PDF. Skipped without an API key."""

import os

import pytest

PDF = "data/pdfs/Denials/2024/PDAReport_Denial-ID.pdf"


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") or not os.path.exists(PDF),
    reason="needs ANTHROPIC_API_KEY and the sample PDF",
)
def test_extract_one_real_denial():
    import anthropic
    from pda.extract import extract_report

    with open(PDF, "rb") as handle:
        report = extract_report(anthropic.Anthropic(), handle.read())

    assert report.report_outcome == "Denied"
    assert report.state_abbr == "ID"
    assert report.ia_requested is True
    assert report.pa_requested is False
    # Idaho/Gwen Fire denial: PA not requested, so every county granted_pa false
    assert all(c.granted_pa is False for c in report.counties)
