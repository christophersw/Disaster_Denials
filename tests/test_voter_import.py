# tests/test_voter_import.py
"""Tests for MIT voter-data import logic (bucketing, mode dedup, rollup)."""

from pda.voter_import import bucket_party, summarize_county_year


def test_bucket_party():
    assert bucket_party("DEMOCRAT") == "dem"
    assert bucket_party("REPUBLICAN") == "rep"
    assert bucket_party("LIBERTARIAN") == "other"
    assert bucket_party("") == "other"


def test_summarize_excludes_ballot_accounting_pseudo_candidates():
    # MIT ships administrative rows (TOTAL VOTES CAST, UNDERVOTES, OVERVOTES)
    # that are not candidates; they must not inflate other_votes. After
    # excluding them, dem+rep+other reconciles to the published total.
    rows = [
        {"candidate": "HARRIS", "party": "DEMOCRAT", "candidatevotes": 100,
         "totalvotes": 250, "mode": "TOTAL"},
        {"candidate": "TRUMP", "party": "REPUBLICAN", "candidatevotes": 150,
         "totalvotes": 250, "mode": "TOTAL"},
        {"candidate": "TOTAL VOTES CAST", "party": "", "candidatevotes": 250,
         "totalvotes": 250, "mode": "TOTAL"},
        {"candidate": "UNDERVOTES", "party": "OTHER", "candidatevotes": 7,
         "totalvotes": 250, "mode": "TOTAL"},
        {"candidate": "OVERVOTES", "party": "OTHER", "candidatevotes": 3,
         "totalvotes": 250, "mode": "TOTAL"},
    ]
    summary = summarize_county_year(rows)
    assert summary["dem_votes"] == 100
    assert summary["rep_votes"] == 150
    assert summary["other_votes"] == 0
    assert summary["dem_votes"] + summary["rep_votes"] + summary["other_votes"] == 250


def test_summarize_prefers_total_mode_over_breakdowns():
    rows = [
        {"candidate": "BIDEN", "party": "DEMOCRAT", "candidatevotes": 100,
         "totalvotes": 250, "mode": "TOTAL"},
        {"candidate": "BIDEN", "party": "DEMOCRAT", "candidatevotes": 60,
         "totalvotes": 250, "mode": "ELECTION DAY"},
        {"candidate": "BIDEN", "party": "DEMOCRAT", "candidatevotes": 40,
         "totalvotes": 250, "mode": "ABSENTEE"},
        {"candidate": "TRUMP", "party": "REPUBLICAN", "candidatevotes": 150,
         "totalvotes": 250, "mode": "TOTAL"},
    ]
    summary = summarize_county_year(rows)
    assert summary["dem_votes"] == 100   # not 200
    assert summary["rep_votes"] == 150
    assert summary["other_votes"] == 0
    assert summary["total_votes"] == 250


def test_summarize_sums_breakdowns_when_no_total():
    rows = [
        {"candidate": "BIDEN", "party": "DEMOCRAT", "candidatevotes": 60,
         "totalvotes": 250, "mode": "ELECTION DAY"},
        {"candidate": "BIDEN", "party": "DEMOCRAT", "candidatevotes": 40,
         "totalvotes": 250, "mode": "ABSENTEE"},
        {"candidate": "TRUMP", "party": "REPUBLICAN", "candidatevotes": 90,
         "totalvotes": 250, "mode": "ELECTION DAY"},
        {"candidate": "TRUMP", "party": "REPUBLICAN", "candidatevotes": 60,
         "totalvotes": 250, "mode": "ABSENTEE"},
    ]
    summary = summarize_county_year(rows)
    assert summary["dem_votes"] == 100
    assert summary["rep_votes"] == 150
    assert summary["other_votes"] == 0


def test_summarize_buckets_third_parties_as_other():
    rows = [
        {"candidate": "BIDEN", "party": "DEMOCRAT", "candidatevotes": 100,
         "totalvotes": 250, "mode": "TOTAL"},
        {"candidate": "TRUMP", "party": "REPUBLICAN", "candidatevotes": 130,
         "totalvotes": 250, "mode": "TOTAL"},
        {"candidate": "JORGENSEN", "party": "LIBERTARIAN", "candidatevotes": 20,
         "totalvotes": 250, "mode": "TOTAL"},
    ]
    summary = summarize_county_year(rows)
    assert summary["other_votes"] == 20


# tests/test_voter_import.py  (append)
import sqlite3

from pda.voter_import import (
    create_voter_tables, load_faithful, build_summary, iter_csv_rows,
)


def _fixture_rows():
    common = {"year": 2020, "state": "ALABAMA", "office": "US PRESIDENT",
              "version": "x"}
    return [
        {**common, "state_po": "AL", "county_name": "AUTAUGA",
         "county_fips": "1001", "candidate": "BIDEN", "party": "DEMOCRAT",
         "candidatevotes": 100, "totalvotes": 250, "mode": "TOTAL"},
        {**common, "state_po": "AL", "county_name": "AUTAUGA",
         "county_fips": "1001", "candidate": "BIDEN", "party": "DEMOCRAT",
         "candidatevotes": 100, "totalvotes": 250, "mode": "ELECTION DAY"},
        {**common, "state_po": "AL", "county_name": "AUTAUGA",
         "county_fips": "1001", "candidate": "TRUMP", "party": "REPUBLICAN",
         "candidatevotes": 150, "totalvotes": 250, "mode": "TOTAL"},
        {**common, "state_po": "AL", "county_name": "BALDWIN",
         "county_fips": "1003", "candidate": "BIDEN", "party": "DEMOCRAT",
         "candidatevotes": 40, "totalvotes": 100, "mode": "ABSENTEE"},
        {**common, "state_po": "AL", "county_name": "BALDWIN",
         "county_fips": "1003", "candidate": "TRUMP", "party": "REPUBLICAN",
         "candidatevotes": 60, "totalvotes": 100, "mode": "ABSENTEE"},
    ]


def test_load_faithful_inserts_every_row_verbatim():
    conn = sqlite3.connect(":memory:")
    create_voter_tables(conn)
    load_faithful(conn, _fixture_rows())
    count = conn.execute("SELECT COUNT(*) FROM county_presidential_returns").fetchone()[0]
    assert count == 5
    raw = conn.execute(
        "SELECT DISTINCT county_fips FROM county_presidential_returns "
        "ORDER BY county_fips").fetchall()
    assert raw == [("1001",), ("1003",)]


def test_build_summary_excludes_non_county_fips_rows():
    # Non-county reporting units must not enter the county-keyed rollup. CT/ME/RI
    # overseas & write-in units all share county_fips 'NA' (which the old code
    # collapsed into one bogus '000NA' county-year with an impossible margin);
    # MO Kansas City uses a 7-digit '2938000'. Only the real county survives.
    conn = sqlite3.connect(":memory:")
    create_voter_tables(conn)
    common = {"year": 2020, "state": "X", "office": "US PRESIDENT", "version": "x"}
    rows = [
        {**common, "state_po": "AL", "county_name": "AUTAUGA", "county_fips": "1001",
         "candidate": "BIDEN", "party": "DEMOCRAT", "candidatevotes": 100,
         "totalvotes": 250, "mode": "TOTAL"},
        {**common, "state_po": "AL", "county_name": "AUTAUGA", "county_fips": "1001",
         "candidate": "TRUMP", "party": "REPUBLICAN", "candidatevotes": 150,
         "totalvotes": 250, "mode": "TOTAL"},
        {**common, "state_po": "RI", "county_name": "FEDERAL PRECINCT",
         "county_fips": "NA", "candidate": "BIDEN", "party": "DEMOCRAT",
         "candidatevotes": 600, "totalvotes": 700, "mode": "TOTAL"},
        {**common, "state_po": "MO", "county_name": "KANSAS CITY",
         "county_fips": "2938000", "candidate": "TRUMP", "party": "REPUBLICAN",
         "candidatevotes": 300, "totalvotes": 500, "mode": "TOTAL"},
    ]
    load_faithful(conn, rows)
    build_summary(conn)
    fips = [r[0] for r in conn.execute(
        "SELECT county_fips FROM county_presidential_summary ORDER BY county_fips")]
    assert fips == ["01001"]   # 'NA'/'000NA' and '2938000' dropped


def test_build_summary_one_row_per_county_year_with_dedup():
    conn = sqlite3.connect(":memory:")
    create_voter_tables(conn)
    load_faithful(conn, _fixture_rows())
    build_summary(conn)
    rows = conn.execute(
        "SELECT county_fips, year, dem_votes, rep_votes, other_votes, "
        "total_votes, dem_share, rep_share, dem_margin "
        "FROM county_presidential_summary ORDER BY county_fips").fetchall()
    assert len(rows) == 2
    autauga = rows[0]
    assert autauga[0] == "01001"          # padded
    assert autauga[2] == 100              # dem, TOTAL not double-counted
    assert autauga[3] == 150              # rep
    assert autauga[5] == 250              # total
    assert abs(autauga[6] - 0.4) < 1e-9   # dem_share
    assert abs(autauga[8] - (-0.2)) < 1e-9  # dem_margin = (100-150)/250
