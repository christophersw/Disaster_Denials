# tests/test_presidents_import.py
"""Tests for presidential party-match import logic."""

import sqlite3

import pytest

from pda.presidents_import import (
    parse_term_date, bucket_president_party,
)


def test_parse_term_date_converts_to_iso():
    assert parse_term_date("April 30, 1789") == "1789-04-30"
    assert parse_term_date("January 20, 2021") == "2021-01-20"


def test_parse_term_date_blank_is_none():
    assert parse_term_date("") is None
    assert parse_term_date("   ") is None
    assert parse_term_date("NA") is None      # CSV null token
    assert parse_term_date(None) is None


def test_parse_term_date_rejects_bad_format():
    with pytest.raises(ValueError):
        parse_term_date("2021-01-20")


def test_bucket_president_party():
    assert bucket_president_party("Democratic") == "dem"
    assert bucket_president_party("Republican") == "rep"
    assert bucket_president_party("Whig") == "other"
    assert bucket_president_party("Unaffiliated") == "other"
    # Exact-token match: the historical party must NOT collapse to 'dem'.
    assert bucket_president_party("Democratic-Republican") == "other"
    # Pipe-delimited list (historical multi-party presidents).
    assert bucket_president_party("Whig | Republican") == "rep"
    assert bucket_president_party("") == "other"


from pda.presidents_import import (
    create_presidents_table, iter_president_rows, load_presidents,
)


def test_load_presidents_round_trips(tmp_path):
    csv_path = tmp_path / "p.csv"
    csv_path.write_text(
        "number,name,birth,death,term_start,term_end,party,election,vice_president\n"
        '45,Donald Trump,1946,,"January 20, 2017","January 20, 2021",Republican,2016,Mike Pence\n'
        '46,Joseph Biden,1942,,"January 20, 2021",,Democratic,2020,Kamala Harris\n',
        encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    create_presidents_table(conn)
    load_presidents(conn, list(iter_president_rows(str(csv_path))))
    rows = conn.execute(
        "SELECT number, term_start, term_end, party_bucket FROM presidents "
        "ORDER BY number").fetchall()
    assert rows[0] == (45, "2017-01-20", "2021-01-20", "rep")
    assert rows[1] == (46, "2021-01-20", None, "dem")  # sitting president


from pda.presidents_import import election_year_for_date, president_for_date

_PRES = [
    {"number": 44, "name": "Barack Obama", "party": "Democratic",
     "party_bucket": "dem", "term_start": "2009-01-20", "term_end": "2017-01-20"},
    {"number": 45, "name": "Donald Trump", "party": "Republican",
     "party_bucket": "rep", "term_start": "2017-01-20", "term_end": "2021-01-20"},
    {"number": 46, "name": "Joseph Biden", "party": "Democratic",
     "party_bucket": "dem", "term_start": "2021-01-20", "term_end": None},
]


def test_election_year_for_date():
    assert election_year_for_date("2018-03-01") == 2016
    assert election_year_for_date("2020-11-02") == 2016   # day before election
    assert election_year_for_date("2020-12-15") == 2020   # after election
    assert election_year_for_date("2008-01-01") == 2004
    assert election_year_for_date("2025-02-01") == 2024   # capped to data
    assert election_year_for_date(None) is None


def test_president_for_date_half_open_boundary():
    assert president_for_date(_PRES, "2018-03-01")["number"] == 45
    assert president_for_date(_PRES, "2020-12-15")["number"] == 45
    # Inauguration day belongs to the incoming president (half-open).
    assert president_for_date(_PRES, "2021-01-20")["number"] == 46
    assert president_for_date(_PRES, "2025-06-01")["number"] == 46  # sitting
    assert president_for_date(_PRES, None) is None


from pda.voter_import import create_voter_tables
from pda.presidents_import import create_state_summary_table, build_state_summary


def _insert_county_summary(conn, rows):
    conn.executemany(
        "INSERT INTO county_presidential_summary (county_fips, year, state_po, "
        "county_name, total_votes, dem_votes, rep_votes, other_votes, "
        "dem_share, rep_share, dem_margin) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def test_build_state_summary_sums_counties_and_picks_winner():
    conn = sqlite3.connect(":memory:")
    create_voter_tables(conn)
    create_state_summary_table(conn)
    # TX 2016: two counties; statewide rep wins. CA 2016: dem wins.
    _insert_county_summary(conn, [
        ("48201", 2016, "TX", "HARRIS", 1000, 600, 400, 0, 0.6, 0.4, 0.2),
        ("48001", 2016, "TX", "ANDERSON", 1000, 100, 900, 0, 0.1, 0.9, -0.8),
        ("06037", 2016, "CA", "LOS ANGELES", 1000, 700, 300, 0, 0.7, 0.3, 0.4),
    ])
    build_state_summary(conn)
    tx = conn.execute(
        "SELECT dem_votes, rep_votes, winner_party FROM state_presidential_summary "
        "WHERE state_po='TX' AND year=2016").fetchone()
    assert tx == (700, 1300, "Republican")
    ca = conn.execute(
        "SELECT winner_party FROM state_presidential_summary "
        "WHERE state_po='CA' AND year=2016").fetchone()
    assert ca == ("Democratic",)


from pda.presidents_import import party_match


def test_party_match():
    assert party_match("rep", "Republican") == 1
    assert party_match("dem", "Democratic") == 1
    assert party_match("rep", "Democratic") == 0
    assert party_match("dem", "Republican") == 0
    assert party_match(None, "Republican") is None       # no president
    assert party_match("rep", None) is None               # no winner
    assert party_match("other", "Republican") is None     # third-party president


from pda.db import connect as db_connect
from pda.fips import add_fips_columns
from pda.presidents_import import (
    add_political_columns, materialize_political_context,
)


def _seed_presidents(conn):
    create_presidents_table(conn)
    conn.executemany(
        "INSERT INTO presidents (number, name, party, party_bucket, "
        "term_start, term_end) VALUES (?,?,?,?,?,?)",
        [(45, "Donald Trump", "Republican", "rep", "2017-01-20", "2021-01-20"),
         (46, "Joseph Biden", "Democratic", "dem", "2021-01-20", None)])
    conn.commit()


def test_materialize_sets_match_flags_and_handles_missing_fips():
    conn = db_connect(":memory:")          # creates reports + report_counties
    add_fips_columns(conn)                  # adds county_fips to report_counties
    create_voter_tables(conn)
    create_state_summary_table(conn)
    add_political_columns(conn)
    _seed_presidents(conn)

    # State rollup: TX 2016 -> Republican, CA 2016 -> Democratic.
    conn.executemany(
        "INSERT INTO state_presidential_summary (state_po, year, total_votes, "
        "dem_votes, rep_votes, other_votes, dem_share, rep_share, dem_margin, "
        "winner_party) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [("TX", 2016, 1000, 400, 600, 0, 0.4, 0.6, -0.2, "Republican"),
         ("CA", 2016, 1000, 700, 300, 0, 0.7, 0.3, 0.4, "Democratic")])
    # County rollup: Harris TX 2016 leaned Democratic (margin > 0).
    conn.execute(
        "INSERT INTO county_presidential_summary (county_fips, year, state_po, "
        "county_name, total_votes, dem_votes, rep_votes, other_votes, "
        "dem_share, rep_share, dem_margin) VALUES "
        "('48201',2016,'TX','HARRIS',1000,600,400,0,0.6,0.4,0.2)")

    # Two reports under Trump (2018): one in TX (R state), one in CA (D state).
    conn.executemany(
        "INSERT INTO reports (source_pdf, decision_date, state_abbr) "
        "VALUES (?,?,?)",
        [("r1", "2018-03-01", "TX"), ("r2", "2018-03-01", "CA")])
    conn.executemany(
        "INSERT INTO report_counties (source_pdf, county_name, county_fips) "
        "VALUES (?,?,?)",
        [("r1", "Harris County", "48201"),   # Dem county under R president -> 0
         ("r1", "Nowhere County", None)])     # no FIPS -> NULL match
    conn.commit()

    counts = materialize_political_context(conn)

    r1 = conn.execute(
        "SELECT president_number, president_party, match_election_year, "
        "state_winner_party, state_party_match FROM reports "
        "WHERE source_pdf='r1'").fetchone()
    assert r1 == (45, "Republican", 2016, "Republican", 1)      # R pres, R state
    r2_match = conn.execute(
        "SELECT state_party_match FROM reports WHERE source_pdf='r2'").fetchone()[0]
    assert r2_match == 0                                         # R pres, D state

    harris = conn.execute(
        "SELECT county_winner_party, county_party_match, president_number "
        "FROM report_counties WHERE county_fips='48201'").fetchone()
    assert harris == ("Democratic", 0, 45)                      # R pres, D county
    no_fips = conn.execute(
        "SELECT county_party_match FROM report_counties "
        "WHERE county_fips IS NULL").fetchone()[0]
    assert no_fips is None
    assert counts["county_no_fips"] == 1
    assert counts["reports"] == 2
