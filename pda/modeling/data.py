# pda/modeling/data.py
"""
Title: pda/modeling/data.py — Modeling-frame loader and target definition.
Description:
    Loads the report-level Declared-vs-Denied modeling frame from data/pda.db.
    Keeps only rows whose report_outcome is 'Declared' or 'Denied' (the 99
    'Denial of Appeal' rows are excluded, spec §3), builds the integer `denied`
    target (1 for Denied), and removes the leakage columns (spec §4). The frame
    is indexed by source_pdf so county-composition features can be joined later
    without source_pdf ever becoming a model feature.
Changelog:
    2026-06-28  Initial version.
"""

import sqlite3

import pandas as pd

DEFAULT_DB = "data/pda.db"
TARGET = "denied"

# Reveal or postdate the decision — never predictors. source_pdf becomes the
# index (kept for joins, not a feature); the rest are dropped outright.
LEAKAGE_COLUMNS = [
    "disaster_number", "declaration_type", "denial_reason",
    "original_denial_date", "appeal_date", "decision_date", "posted_date",
    "report_outcome", "report_type", "needs_review", "review_note",
    "parser_model", "extracted_at", "url",
]


def load_modeling_frame(db_path=DEFAULT_DB):
    """Load the Declared-vs-Denied modeling frame.

    Args:
        db_path: path to the SQLite database.
    Returns:
        (features, target): a DataFrame indexed by source_pdf with leakage
        columns removed, and an int Series named 'denied' aligned to it.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM reports", conn)
    finally:
        conn.close()

    df = df[df["report_outcome"].isin(["Declared", "Denied"])].copy()
    df = df.set_index("source_pdf")
    target = (df["report_outcome"] == "Denied").astype(int)
    target.name = TARGET

    drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    features = df.drop(columns=drop)

    return features, target
