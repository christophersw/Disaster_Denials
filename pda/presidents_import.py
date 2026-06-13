# pda/presidents_import.py
"""
Title: Presidential Party-Match Import
Description: Ingests the us-presidents-dataset CSV into a `presidents` reference
    table, builds a state-year presidential rollup from the existing county
    rollup, and materializes derived columns on `reports` and `report_counties`
    answering: who was president at a denial's decision_date, and did the
    president's party match the state/county presidential vote. The derived
    columns are managed here (guarded ALTER TABLE) and kept out of the
    extraction column contract; re-run the import to refresh them after any
    re-extraction.
Changelog:
    2026-06-13  Initial version.
"""

import csv
import datetime

from pda.fips import pad_fips

# Columns of the presidents table, in insert order.
PRESIDENT_COLUMNS = [
    "number", "name", "birth", "death", "term_start", "term_end",
    "party", "party_bucket", "election_years", "vice_president",
]

# CSV header fields the loader requires; a missing field fails loudly.
_REQUIRED_CSV_FIELDS = {
    "number", "name", "birth", "death", "term_start", "term_end",
    "party", "election", "vice_president",
}


def parse_term_date(value):
    """Convert a "Month Day, YYYY" date to ISO YYYY-MM-DD.

    Args:
        value: Raw date string from the CSV (e.g. "April 30, 1789"), or an
            empty/"NA"/None value for a sitting president's open term_end.
    Returns:
        ISO date string "YYYY-MM-DD", or None when the input is blank/"NA".
    Raises:
        ValueError: if a non-blank value does not parse as "%B %d, %Y".
    """
    if value is None:
        return None
    text = value.strip()
    if not text or text.upper() == "NA":   # CSV uses "NA" as the null token
        return None
    return datetime.datetime.strptime(text, "%B %d, %Y").date().isoformat()


def bucket_president_party(party):
    """Bucket a president's party label into 'dem' / 'rep' / 'other'.

    The CSV may carry a pipe-delimited list for presidents who changed party;
    an exact token match is required so the historical 'Democratic-Republican'
    party is NOT treated as 'Democratic'.

    Args:
        party: Raw party string (e.g. "Republican", "Whig | Republican").
    Returns:
        'rep' if any token is exactly 'Republican', 'dem' if any token is
        exactly 'Democratic', else 'other'.
    """
    tokens = [token.strip() for token in (party or "").split("|")]
    if "Republican" in tokens:
        return "rep"
    if "Democratic" in tokens:
        return "dem"
    return "other"


def _to_int_or_none(value):
    """Parse an integer that may be blank or the CSV's "NA" null token.

    Args:
        value: Raw CSV string.
    Returns:
        int, or None when blank/"NA".
    """
    text = "" if value is None else str(value).strip()
    if not text or text.upper() == "NA":   # CSV uses "NA" as the null token
        return None
    return int(text)


def create_presidents_table(conn):
    """Create the presidents reference table if it does not exist.

    Args:
        conn: open sqlite3 connection.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS presidents ("
        "number INTEGER PRIMARY KEY, name TEXT, birth INTEGER, death INTEGER, "
        "term_start TEXT, term_end TEXT, party TEXT, party_bucket TEXT, "
        "election_years TEXT, vice_president TEXT)")
    conn.commit()


def iter_president_rows(csv_path):
    """Yield president rows from the dataset CSV, keyed by PRESIDENT_COLUMNS.

    Args:
        csv_path: path to us_presidents_2025.csv.
    Yields:
        dicts keyed by PRESIDENT_COLUMNS, dates converted to ISO and a derived
        party_bucket.
    Raises:
        ValueError: if the CSV header is missing an expected field.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:  # strip BOM
        reader = csv.DictReader(handle)
        missing = _REQUIRED_CSV_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"presidents CSV missing expected columns: {sorted(missing)}")
        for record in reader:
            yield {
                "number": _to_int_or_none(record["number"]),
                "name": (record["name"] or "").strip(),
                "birth": _to_int_or_none(record["birth"]),
                "death": _to_int_or_none(record["death"]),
                "term_start": parse_term_date(record["term_start"]),
                "term_end": parse_term_date(record["term_end"]),
                "party": (record["party"] or "").strip(),
                "party_bucket": bucket_president_party(record["party"]),
                "election_years": (record["election"] or "").strip(),
                "vice_president": (record["vice_president"] or "").strip(),
            }


def load_presidents(conn, rows):
    """Replace and repopulate the presidents table from rows.

    Args:
        conn: open sqlite3 connection (table already created).
        rows: iterable of dicts keyed by PRESIDENT_COLUMNS.
    """
    conn.execute("DELETE FROM presidents")
    placeholders = ", ".join("?" for _ in PRESIDENT_COLUMNS)
    columns = ", ".join(PRESIDENT_COLUMNS)
    with conn:
        conn.executemany(
            f"INSERT INTO presidents ({columns}) VALUES ({placeholders})",
            ([row[column] for column in PRESIDENT_COLUMNS] for row in rows),
        )


# Real US presidential election dates (Tuesday after the first Monday of
# November), capped to the years the MIT data covers (2000-2024). Used to pick
# the most recent election on or before a denial's decision_date.
ELECTION_DATES = [
    (2000, "2000-11-07"),
    (2004, "2004-11-02"),
    (2008, "2008-11-04"),
    (2012, "2012-11-06"),
    (2016, "2016-11-08"),
    (2020, "2020-11-03"),
    (2024, "2024-11-05"),
]


def election_year_for_date(decision_date):
    """Return the most recent presidential election year on or before a date.

    Compares against the actual election dates, so a denial in the Nov-Dec of
    an election year maps to that year's (already-held) election. ISO date
    strings compare correctly as plain strings.

    Args:
        decision_date: ISO date string "YYYY-MM-DD", or None.
    Returns:
        Election year (int) from ELECTION_DATES, or None if the date is None or
        precedes the earliest known election.
    """
    if not decision_date:
        return None
    for year, election_date in reversed(ELECTION_DATES):
        if election_date <= decision_date:
            return year
    return None


def president_for_date(presidents, decision_date):
    """Return the president holding office on a given date.

    The interval is half-open [term_start, term_end): an inauguration-day date
    belongs to the incoming president (outgoing term_end == incoming
    term_start). A None term_end is the sitting president.

    Args:
        presidents: iterable of dicts with 'term_start', 'term_end' (ISO
            strings; term_end may be None) plus identity fields.
        decision_date: ISO date string "YYYY-MM-DD", or None.
    Returns:
        The matching president dict, or None if the date is None or uncovered.
    """
    if not decision_date:
        return None
    for president in presidents:
        start = president["term_start"]
        end = president["term_end"]
        if start and start <= decision_date and (end is None or decision_date < end):
            return president
    return None


def create_state_summary_table(conn):
    """Create the state-year presidential rollup table if it does not exist.

    Args:
        conn: open sqlite3 connection.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS state_presidential_summary ("
        "state_po TEXT, year INTEGER, total_votes INTEGER, dem_votes INTEGER, "
        "rep_votes INTEGER, other_votes INTEGER, dem_share REAL, "
        "rep_share REAL, dem_margin REAL, winner_party TEXT, "
        "PRIMARY KEY (state_po, year))")
    conn.commit()


def build_state_summary(conn):
    """Rebuild the state-year rollup by summing the county rollup.

    Sums dem/rep/other/total votes across all counties in
    county_presidential_summary (non-county units were already excluded when
    that table was built) per (state_po, year), then computes shares, margin,
    and the popular-vote winner party.

    Args:
        conn: open sqlite3 connection with a populated
            county_presidential_summary table.
    """
    conn.execute("DELETE FROM state_presidential_summary")
    grouped = conn.execute(
        "SELECT state_po, year, SUM(total_votes), SUM(dem_votes), "
        "SUM(rep_votes), SUM(other_votes) "
        "FROM county_presidential_summary GROUP BY state_po, year").fetchall()
    rows = []
    for state_po, year, total, dem, rep, other in grouped:
        dem, rep, other, total = dem or 0, rep or 0, other or 0, total or 0
        dem_share = dem / total if total else None
        rep_share = rep / total if total else None
        margin = (dem - rep) / total if total else None
        winner = "Democratic" if dem > rep else "Republican" if rep > dem else None
        rows.append((state_po, year, total, dem, rep, other,
                     dem_share, rep_share, margin, winner))
    with conn:
        conn.executemany(
            "INSERT INTO state_presidential_summary (state_po, year, "
            "total_votes, dem_votes, rep_votes, other_votes, dem_share, "
            "rep_share, dem_margin, winner_party) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


# Derived political columns. Kept OUT of REPORT_COLUMNS/COUNTY_COLUMNS so the
# extraction pipeline is untouched; managed here via guarded ALTER TABLE.
_REPORT_POLITICAL_COLUMNS = {
    "president_number": "INTEGER",
    "president_name": "TEXT",
    "president_party": "TEXT",
    "match_election_year": "INTEGER",
    "state_winner_party": "TEXT",
    "state_dem_share": "REAL",
    "state_rep_share": "REAL",
    "state_margin": "REAL",
    "state_party_match": "INTEGER",
}

_COUNTY_POLITICAL_COLUMNS = {
    "president_number": "INTEGER",
    "president_name": "TEXT",
    "president_party": "TEXT",
    "match_election_year": "INTEGER",
    "county_winner_party": "TEXT",
    "county_dem_share": "REAL",
    "county_rep_share": "REAL",
    "county_margin": "REAL",
    "county_party_match": "INTEGER",
}


def _add_columns(conn, table, columns):
    """Add any missing columns to a table (guarded by PRAGMA table_info).

    Args:
        conn: open sqlite3 connection.
        table: target table name.
        columns: {column_name: sql_type} to ensure exist.
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, sql_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    conn.commit()


def add_political_columns(conn):
    """Add derived political columns to reports and report_counties (idempotent).

    Args:
        conn: open sqlite3 connection.
    """
    _add_columns(conn, "reports", _REPORT_POLITICAL_COLUMNS)
    _add_columns(conn, "report_counties", _COUNTY_POLITICAL_COLUMNS)


def _winner_bucket(winner_party):
    """Bucket a winner party label to 'dem' / 'rep' / 'other'."""
    if winner_party == "Democratic":
        return "dem"
    if winner_party == "Republican":
        return "rep"
    return "other"


def party_match(president_bucket, winner_party):
    """Return 1 / 0 / None for a president-vs-geography party comparison.

    Args:
        president_bucket: the president's 'dem'/'rep'/'other' bucket, or None.
        winner_party: the geography's winning party label, or None.
    Returns:
        1 if both sides are major parties and equal, 0 if both are major and
        differ, None if either side is missing or third-party.
    """
    winner = _winner_bucket(winner_party)
    if president_bucket in ("dem", "rep") and winner in ("dem", "rep"):
        return 1 if president_bucket == winner else 0
    return None


def materialize_political_context(conn):
    """Populate the derived political columns on reports and report_counties.

    Per report: looks up the president on decision_date, the most-recent
    election cycle, and that state's winner; writes president fields, state
    shares/margin, and state_party_match. Per county: writes president fields
    (denormalized from the parent report) plus the county's winner/shares/margin
    and county_party_match.

    Args:
        conn: open sqlite3 connection with presidents, both summaries, and the
            political columns already created.
    Returns:
        dict of summary counts (reports, state_match_1/0/none,
        county_match_1/0/none, county_no_fips).
    """
    presidents = [
        dict(zip(("number", "name", "party", "party_bucket",
                  "term_start", "term_end"), row))
        for row in conn.execute(
            "SELECT number, name, party, party_bucket, term_start, term_end "
            "FROM presidents")
    ]
    state_lookup = {
        (state_po, year): (winner, dem_share, rep_share, margin)
        for state_po, year, winner, dem_share, rep_share, margin in conn.execute(
            "SELECT state_po, year, winner_party, dem_share, rep_share, "
            "dem_margin FROM state_presidential_summary")
    }
    county_lookup = {
        (fips, year): (dem_share, rep_share, margin)
        for fips, year, dem_share, rep_share, margin in conn.execute(
            "SELECT county_fips, year, dem_share, rep_share, dem_margin "
            "FROM county_presidential_summary")
    }

    counts = {"reports": 0, "state_match_1": 0, "state_match_0": 0,
              "state_match_none": 0, "county_match_1": 0, "county_match_0": 0,
              "county_match_none": 0, "county_no_fips": 0}

    report_updates = []
    per_report = {}  # source_pdf -> (number, name, party, bucket, year)
    for source_pdf, decision_date, state_abbr in conn.execute(
            "SELECT source_pdf, decision_date, state_abbr FROM reports"):
        president = president_for_date(presidents, decision_date)
        year = election_year_for_date(decision_date)
        number = president["number"] if president else None
        name = president["name"] if president else None
        party = president["party"] if president else None
        bucket = president["party_bucket"] if president else None

        winner = dem_share = rep_share = margin = None
        if state_abbr is not None and year is not None:
            state_row = state_lookup.get((state_abbr, year))
            if state_row:
                winner, dem_share, rep_share, margin = state_row
        match = party_match(bucket, winner)

        report_updates.append((number, name, party, year, winner,
                               dem_share, rep_share, margin, match, source_pdf))
        per_report[source_pdf] = (number, name, party, bucket, year)
        counts["reports"] += 1
        counts["state_match_" + ("none" if match is None else str(match))] += 1

    with conn:
        conn.executemany(
            "UPDATE reports SET president_number=?, president_name=?, "
            "president_party=?, match_election_year=?, state_winner_party=?, "
            "state_dem_share=?, state_rep_share=?, state_margin=?, "
            "state_party_match=? WHERE source_pdf=?", report_updates)

    county_updates = []
    for county_id, source_pdf, county_fips in conn.execute(
            "SELECT county_id, source_pdf, county_fips FROM report_counties"):
        number, name, party, bucket, year = per_report.get(
            source_pdf, (None, None, None, None, None))
        winner = dem_share = rep_share = margin = None
        if county_fips is None:
            counts["county_no_fips"] += 1
        elif year is not None:
            county_row = county_lookup.get((pad_fips(county_fips), year))
            if county_row:
                dem_share, rep_share, margin = county_row
                winner = ("Democratic" if (margin or 0) > 0
                          else "Republican" if (margin or 0) < 0 else None)
        match = party_match(bucket, winner)

        county_updates.append((number, name, party, year, winner,
                               dem_share, rep_share, margin, match, county_id))
        counts["county_match_" + ("none" if match is None else str(match))] += 1

    with conn:
        conn.executemany(
            "UPDATE report_counties SET president_number=?, president_name=?, "
            "president_party=?, match_election_year=?, county_winner_party=?, "
            "county_dem_share=?, county_rep_share=?, county_margin=?, "
            "county_party_match=? WHERE county_id=?", county_updates)

    return counts
