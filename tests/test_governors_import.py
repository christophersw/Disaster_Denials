# tests/test_governors_import.py
"""Tests for governor party-match import logic."""

import sqlite3

from pda.governors_import import (
    GOVERNOR_CSV_COLUMNS, create_governors_table, iter_governor_rows,
    load_governors,
)


def test_iter_governor_rows_derives_bucket_and_nulls_blank_term_end(tmp_path):
    csv_path = tmp_path / "g.csv"
    csv_path.write_text(
        "state_abbr,name,party,term_start,term_end,source_url\n"
        "TX,Greg Abbott,Republican,2015-01-20,,https://en.wikipedia.org/wiki/x\n"
        "CA,Jerry Brown,Democratic,2011-01-03,2019-01-07,https://en.wikipedia.org/wiki/y\n"
        "AK,Bill Walker,Independent,2014-12-01,2018-12-03,https://en.wikipedia.org/wiki/z\n",
        encoding="utf-8")
    rows = list(iter_governor_rows(str(csv_path)))
    assert rows[0]["state_abbr"] == "TX"
    assert rows[0]["term_end"] is None          # blank -> None (sitting)
    assert rows[0]["party_bucket"] == "rep"
    assert rows[1]["party_bucket"] == "dem"
    assert rows[2]["party_bucket"] == "other"   # Independent
    assert rows[1]["term_end"] == "2019-01-07"


def test_load_governors_round_trips(tmp_path):
    csv_path = tmp_path / "g.csv"
    csv_path.write_text(
        "state_abbr,name,party,term_start,term_end,source_url\n"
        "TX,Greg Abbott,Republican,2015-01-20,,https://en.wikipedia.org/wiki/x\n",
        encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    create_governors_table(conn)
    load_governors(conn, list(iter_governor_rows(str(csv_path))))
    row = conn.execute(
        "SELECT state_abbr, name, party_bucket, term_start, term_end "
        "FROM governors").fetchone()
    assert row == ("TX", "Greg Abbott", "rep", "2015-01-20", None)
    assert GOVERNOR_CSV_COLUMNS[0] == "state_abbr"


from pda.governors_import import governor_for_date, governor_rows_from_extraction

_GOVS = [
    {"name": "Rick Perry", "party": "Republican", "party_bucket": "rep",
     "term_start": "2000-12-21", "term_end": "2015-01-20"},
    {"name": "Greg Abbott", "party": "Republican", "party_bucket": "rep",
     "term_start": "2015-01-20", "term_end": None},
]


def test_governor_for_date_half_open_boundary():
    assert governor_for_date(_GOVS, "2014-06-01")["name"] == "Rick Perry"
    # Inauguration day belongs to the incoming governor (half-open).
    assert governor_for_date(_GOVS, "2015-01-20")["name"] == "Greg Abbott"
    assert governor_for_date(_GOVS, "2025-01-01")["name"] == "Greg Abbott"  # sitting
    assert governor_for_date(_GOVS, None) is None
    assert governor_for_date([], "2014-06-01") is None


def test_governor_rows_from_extraction_filters_pre_window():
    extracted = [
        {"name": "Old Gov", "party": "Democratic",
         "term_start": "1999-01-01", "term_end": "2003-01-01"},   # dropped
        {"name": "Mid Gov", "party": "Republican",
         "term_start": "2003-01-01", "term_end": "2011-01-01"},   # kept (ends in window)
        {"name": "Now Gov", "party": "Democratic",
         "term_start": "2011-01-01", "term_end": ""},             # kept (sitting)
    ]
    rows = governor_rows_from_extraction("XX", "https://w/x", extracted)
    assert [r["name"] for r in rows] == ["Mid Gov", "Now Gov"]
    assert rows[0]["state_abbr"] == "XX"
    assert rows[1]["term_end"] == ""
    assert all(r["source_url"] == "https://w/x" for r in rows)


def test_governor_rows_from_extraction_drops_degenerate_tenures():
    # Bad extraction sometimes emits placeholder rows with term_end == term_start
    # (or no start). These are not real tenures and must be dropped.
    extracted = [
        {"name": "Placeholder", "party": "Democratic",
         "term_start": "2007-01-01", "term_end": "2007-01-01"},   # degenerate
        {"name": "Inverted", "party": "Republican",
         "term_start": "2015-01-01", "term_end": "2011-01-01"},   # inverted
        {"name": "No Start", "party": "Republican",
         "term_start": "", "term_end": "2018-01-01"},             # missing start
        {"name": "Real Gov", "party": "Democratic",
         "term_start": "2015-01-06", "term_end": "2021-03-02"},   # kept
    ]
    rows = governor_rows_from_extraction("RI", "https://w/ri", extracted)
    assert [r["name"] for r in rows] == ["Real Gov"]


def test_governor_rows_from_extraction_drops_contained_tenure():
    # A hallucinated historical governor clamped into the window sits inside the
    # real governor's span; one governor serves at a time, so it must be dropped.
    extracted = [
        {"name": "Bogus Old", "party": "Republican",
         "term_start": "2007-01-01", "term_end": "2007-01-15"},   # inside Rendell
        {"name": "Ed Rendell", "party": "Democratic",
         "term_start": "2003-01-21", "term_end": "2011-01-18"},    # real, kept
        {"name": "Tom Corbett", "party": "Republican",
         "term_start": "2011-01-18", "term_end": "2015-01-20"},    # real, kept
        {"name": "Josh Shapiro", "party": "Democratic",
         "term_start": "2023-01-17", "term_end": ""},              # sitting, kept
    ]
    rows = governor_rows_from_extraction("PA", "https://w/pa", extracted)
    assert [r["name"] for r in rows] == ["Ed Rendell", "Tom Corbett", "Josh Shapiro"]


from pda.db import connect as db_connect
from pda.presidents_import import add_political_columns
from pda.governors_import import (
    add_governor_columns, materialize_governor_context,
)


def _seed_governors(conn):
    create_governors_table(conn)
    conn.executemany(
        "INSERT INTO governors (state_abbr, name, party, party_bucket, "
        "term_start, term_end) VALUES (?,?,?,?,?,?)",
        [("TX", "Greg Abbott", "Republican", "rep", "2015-01-20", None),
         ("CA", "Gavin Newsom", "Democratic", "dem", "2019-01-07", None),
         ("AK", "Bill Walker", "Independent", "other", "2014-12-01", "2018-12-03"),
         ("PR", "Pedro Pierluisi", "Democratic", "dem", "2021-01-02", None)])
    conn.commit()


def test_materialize_governor_flags():
    conn = db_connect(":memory:")
    add_political_columns(conn)      # president_party / state_winner_party columns
    add_governor_columns(conn)
    _seed_governors(conn)

    # source_pdf, request_date, decision_date, state_abbr, president_party, state_winner_party
    conn.executemany(
        "INSERT INTO reports (source_pdf, request_date, decision_date, "
        "state_abbr, president_party, state_winner_party) VALUES (?,?,?,?,?,?)",
        [("r_tx", "2018-03-01", "2018-04-01", "TX", "Republican", "Republican"),
         ("r_ca", "2020-03-01", "2020-04-01", "CA", "Republican", "Democratic"),
         ("r_ak", "2016-06-01", "2016-07-01", "AK", "Democratic", "Republican"),
         ("r_pr", "2022-03-01", "2022-04-01", "PR", "Democratic", None),
         ("r_tribe", "2018-03-01", "2018-04-01", None, "Republican", None)])
    conn.commit()

    counts = materialize_governor_context(conn)

    def row(pdf):
        return conn.execute(
            "SELECT governor_name, governor_party, governor_vs_president, "
            "governor_vs_state_vote FROM reports WHERE source_pdf=?",
            (pdf,)).fetchone()

    assert row("r_tx") == ("Greg Abbott", "Republican", 1, 1)   # R gov, R pres, R state
    assert row("r_ca") == ("Gavin Newsom", "Democratic", 0, 1)  # D gov, R pres, D state
    # Independent governor -> both flags NULL
    assert row("r_ak") == ("Bill Walker", "Independent", None, None)
    # Territory: vs_president resolves (national, D gov + D pres -> 1),
    # vs_state_vote NULL (no MIT vote for territories)
    assert row("r_pr") == ("Pedro Pierluisi", "Democratic", 1, None)
    # Tribe (no state_abbr) -> no governor, both flags NULL
    assert row("r_tribe") == (None, None, None, None)

    assert counts["reports"] == 5
    assert counts["no_governor"] == 1
