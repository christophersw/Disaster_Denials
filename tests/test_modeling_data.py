# tests/test_modeling_data.py
"""
Title: tests/test_modeling_data.py — Tests for the modeling-frame loader.
Description: Verifies target definition, appeal exclusion, and leakage removal
    against a tiny in-memory reports table.
Changelog:
    2026-06-28  Initial version.
"""

import sqlite3

import pandas as pd

from pda.modeling.data import LEAKAGE_COLUMNS, load_modeling_frame


def _tiny_db(path):
    """Build a minimal SQLite reports table that covers all 14 leakage columns.

    All 14 LEAKAGE_COLUMNS are present so the drop logic in load_modeling_frame
    is genuinely exercised for every one of them, not just the 3 that were
    originally included.  Non-leakage predictor columns (state_abbr,
    pa_cost_estimate) are retained to verify they are NOT dropped.
    """
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE reports ("
        "source_pdf TEXT PRIMARY KEY, "
        "report_outcome, "
        "state_abbr, "
        "disaster_number, "
        "denial_reason, "
        "pa_cost_estimate, "
        "declaration_type, "
        "original_denial_date, "
        "appeal_date, "
        "decision_date, "
        "posted_date, "
        "report_type, "
        "needs_review, "
        "review_note, "
        "parser_model, "
        "extracted_at, "
        "url"
        ")"
    )
    conn.executemany(
        "INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                "a.pdf", "Declared", "TX", "DR-1", None, 100.0,
                "DR", "2020-01-01", None, "2020-03-01", "2020-04-01",
                "Standard", 0, None, "gpt-4", "2024-01-01", "http://example.com/a",
            ),
            (
                "b.pdf", "Denied", "CA", None, "insufficient", 5.0,
                "DR", "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01",
                "Standard", 0, None, "gpt-4", "2024-01-01", "http://example.com/b",
            ),
            (
                "c.pdf", "Denial of Appeal", "CA", None, "insufficient", 5.0,
                "DR", "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01",
                "Standard", 0, None, "gpt-4", "2024-01-01", "http://example.com/c",
            ),
        ],
    )
    conn.commit()
    conn.close()


def test_excludes_appeals_and_builds_target(tmp_path):
    db = str(tmp_path / "t.db")
    _tiny_db(db)
    features, target = load_modeling_frame(db)
    assert len(features) == 2                      # appeal row dropped
    assert target.tolist() == [0, 1]               # Declared=0, Denied=1
    assert target.name == "denied"


def test_leakage_columns_removed(tmp_path):
    db = str(tmp_path / "t.db")
    _tiny_db(db)
    features, _ = load_modeling_frame(db)
    for col in LEAKAGE_COLUMNS:
        assert col not in features.columns
    assert features.index.name == "source_pdf"     # kept as index for joins
    assert "state_abbr" in features.columns        # grouping key retained
    assert "pa_cost_estimate" in features.columns  # legit predictor retained
