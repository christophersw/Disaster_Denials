"""Tests for append-writers and resume set."""

from pda.columns import COUNTY_COLUMNS, REPORT_COLUMNS
from pda.io import append_rows, done_source_pdfs


def test_append_writes_header_once_then_rows(tmp_path):
    path = tmp_path / "reports.csv"
    append_rows(str(path), REPORT_COLUMNS, [{c: None for c in REPORT_COLUMNS}])
    row = {c: None for c in REPORT_COLUMNS}
    row["source_pdf"] = "a.pdf"
    append_rows(str(path), REPORT_COLUMNS, [row])
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("source_pdf,")
    assert len(lines) == 3  # header + 2 rows


def test_done_source_pdfs_reads_existing(tmp_path):
    path = tmp_path / "reports.csv"
    row = {c: None for c in REPORT_COLUMNS}
    row["source_pdf"] = "data/pdfs/Denials/2024/x.pdf"
    append_rows(str(path), REPORT_COLUMNS, [row])
    assert done_source_pdfs(str(path)) == {"data/pdfs/Denials/2024/x.pdf"}


def test_done_source_pdfs_missing_file_is_empty(tmp_path):
    assert done_source_pdfs(str(tmp_path / "nope.csv")) == set()


def test_county_columns_round_trip(tmp_path):
    path = tmp_path / "counties.csv"
    row = {c: None for c in COUNTY_COLUMNS}
    row["source_pdf"] = "a.pdf"
    row["county_name"] = "Nez Perce"
    append_rows(str(path), COUNTY_COLUMNS, [row])
    assert "Nez Perce" in path.read_text(encoding="utf-8")
