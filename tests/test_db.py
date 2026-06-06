"""Tests for the SQLite store: transactional writes, resume, idempotency."""

import sqlite3

import pytest

from pda.columns import COUNTY_COLUMNS, REPORT_COLUMNS
from pda.db import (
    connect,
    done_source_pdfs,
    mark_batch_item,
    open_batch_ids,
    pending_source_pdfs,
    record_batch_items,
    source_pdf_for,
    write_report,
)


def _report_row(source_pdf: str, **overrides) -> dict:
    """A reports row with every column null except source_pdf and overrides."""
    row = {column: None for column in REPORT_COLUMNS}
    row["source_pdf"] = source_pdf
    row.update(overrides)
    return row


def _county_row(source_pdf: str, county_name: str, **overrides) -> dict:
    """A report_counties row keyed to source_pdf with a county name."""
    row = {column: None for column in COUNTY_COLUMNS}
    row["source_pdf"] = source_pdf
    row["county_name"] = county_name
    row.update(overrides)
    return row


def test_write_report_persists_report_and_counties(tmp_path):
    conn = connect(str(tmp_path / "pda.db"))
    write_report(
        conn,
        _report_row("a.pdf", jurisdiction_name="Idaho"),
        [_county_row("a.pdf", "Nez Perce"), _county_row("a.pdf", "Ada")],
    )
    assert done_source_pdfs(conn) == {"a.pdf"}
    names = conn.execute(
        "SELECT county_name FROM report_counties "
        "WHERE source_pdf = 'a.pdf' ORDER BY county_name"
    ).fetchall()
    assert [row[0] for row in names] == ["Ada", "Nez Perce"]


def test_done_source_pdfs_empty_on_fresh_db(tmp_path):
    conn = connect(str(tmp_path / "pda.db"))
    assert done_source_pdfs(conn) == set()


def test_rewrite_replaces_counties_without_duplicating(tmp_path):
    conn = connect(str(tmp_path / "pda.db"))
    write_report(
        conn,
        _report_row("a.pdf"),
        [_county_row("a.pdf", "Ada"), _county_row("a.pdf", "Boise")],
    )
    # Re-extracting the same PDF must replace, not append.
    write_report(
        conn,
        _report_row("a.pdf", jurisdiction_name="Idaho"),
        [_county_row("a.pdf", "Ada")],
    )
    assert conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM report_counties").fetchone()[0] == 1
    assert conn.execute(
        "SELECT jurisdiction_name FROM reports WHERE source_pdf = 'a.pdf'"
    ).fetchone()[0] == "Idaho"


def test_write_is_atomic_report_rolls_back_if_a_county_fails(tmp_path):
    conn = connect(str(tmp_path / "pda.db"))
    # The second county points at a PDF with no reports row, so its foreign key
    # fails. The whole write — including the report itself — must roll back.
    failing_counties = [
        _county_row("a.pdf", "Ada"),
        _county_row("ORPHAN.pdf", "Ghost"),
    ]
    with pytest.raises(sqlite3.IntegrityError):
        write_report(conn, _report_row("a.pdf"), failing_counties)
    assert done_source_pdfs(conn) == set()
    assert conn.execute("SELECT COUNT(*) FROM report_counties").fetchone()[0] == 0


def test_record_and_resolve_batch_items(tmp_path):
    conn = connect(str(tmp_path / "pda.db"))
    record_batch_items(conn, "batch_1", [("cid_a", "a.pdf"), ("cid_b", "b.pdf")])
    assert pending_source_pdfs(conn) == {"a.pdf", "b.pdf"}
    assert open_batch_ids(conn) == ["batch_1"]
    assert source_pdf_for(conn, "cid_a") == "a.pdf"
    assert source_pdf_for(conn, "missing") is None


def test_written_items_drop_out_of_pending_and_open(tmp_path):
    conn = connect(str(tmp_path / "pda.db"))
    record_batch_items(conn, "batch_1", [("cid_a", "a.pdf"), ("cid_b", "b.pdf")])
    mark_batch_item(conn, "cid_a", "written")
    # a.pdf is no longer pending; the batch stays open while b.pdf is unresolved.
    assert pending_source_pdfs(conn) == {"b.pdf"}
    assert open_batch_ids(conn) == ["batch_1"]
    mark_batch_item(conn, "cid_b", "failed")
    # No submitted items remain, so the batch is closed and nothing is pending.
    assert pending_source_pdfs(conn) == set()
    assert open_batch_ids(conn) == []


def test_record_batch_items_reassigns_failed_pdf_to_new_batch(tmp_path):
    conn = connect(str(tmp_path / "pda.db"))
    record_batch_items(conn, "batch_1", [("cid_a", "a.pdf")])
    mark_batch_item(conn, "cid_a", "failed")
    # Resubmitting the same PDF (same custom_id) overwrites the failed row.
    record_batch_items(conn, "batch_2", [("cid_a", "a.pdf")])
    assert pending_source_pdfs(conn) == {"a.pdf"}
    assert open_batch_ids(conn) == ["batch_2"]
