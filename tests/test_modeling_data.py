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
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE reports (source_pdf TEXT PRIMARY KEY, report_outcome, "
        "state_abbr, disaster_number, denial_reason, pa_cost_estimate)"
    )
    conn.executemany(
        "INSERT INTO reports VALUES (?,?,?,?,?,?)",
        [
            ("a.pdf", "Declared", "TX", "DR-1", None, 100.0),
            ("b.pdf", "Denied", "CA", None, "insufficient", 5.0),
            ("c.pdf", "Denial of Appeal", "CA", None, "insufficient", 5.0),
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
