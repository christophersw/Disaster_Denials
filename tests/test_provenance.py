"""Tests for deriving provenance from a PDF path and the manifest."""

from pda.provenance import provenance_for, report_type_from_path


def test_report_type_from_path():
    p = "data/pdfs/AppealDenials/2026/FY26PDAReport_AppealDenial-CO.pdf"
    assert report_type_from_path(p) == "AppealDenials"


def test_provenance_joins_manifest_on_local_path():
    manifest = {
        "data/pdfs/Denials/2024/PDAReport_Denial-ID.pdf": {
            "url": "https://example.gov/ID.pdf",
            "posted_date": "2024-11-20T12:00:00Z",
        }
    }
    prov = provenance_for("data/pdfs/Denials/2024/PDAReport_Denial-ID.pdf", manifest)
    assert prov["report_type"] == "Denials"
    assert prov["url"] == "https://example.gov/ID.pdf"
    assert prov["posted_date"] == "2024-11-20T12:00:00Z"


def test_provenance_tolerates_missing_manifest_row():
    prov = provenance_for("data/pdfs/Other/2024/x.pdf", {})
    assert prov["report_type"] == "Other"
    assert prov["url"] is None
    assert prov["posted_date"] is None
