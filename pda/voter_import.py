# pda/voter_import.py
"""
Title: MIT Voter Data Import
Description: Loads the MIT Election Lab county presidential returns into a
    faithful SQLite table and builds a county-year rollup. The rollup applies
    MIT's mode convention: where a candidate has a 'TOTAL' mode row it is used
    alone, otherwise the per-mode breakdown rows are summed — preventing the
    double counting that naive aggregation would cause in 2020 and 2024.
Changelog:
    2026-06-07  Initial version.
"""

import csv

# Faithful table columns, in insert order. Keyed off the CSV's own field names.
FAITHFUL_COLUMNS = [
    "year", "state", "state_po", "county_name", "county_fips", "office",
    "candidate", "party", "candidatevotes", "totalvotes", "version", "mode",
]

# Administrative / ballot-accounting rows the MIT data carries as if they were
# candidates (e.g. SC/TX/WI/WV 2024 ship a "TOTAL VOTES CAST" row equal to the
# whole total; DC ships UNDERVOTES/OVERVOTES). They are not real candidates and
# must be excluded from the rollup or they inflate other_votes. The faithful
# table still keeps them verbatim.
NON_CANDIDATE_LABELS = {"TOTAL VOTES CAST", "UNDERVOTES", "OVERVOTES", "SPOILED"}


def bucket_party(party: str) -> str:
    """Map a party label to a 'dem' / 'rep' / 'other' bucket.

    Args:
        party: MIT party string (e.g. 'DEMOCRAT', 'LIBERTARIAN', '').
    Returns:
        'dem', 'rep', or 'other'.
    """
    normalized = (party or "").strip().upper()
    if normalized == "DEMOCRAT":
        return "dem"
    if normalized == "REPUBLICAN":
        return "rep"
    return "other"


def _candidate_votes(rows: list) -> int:
    """Total votes for one candidate within one county-year, with mode dedup.

    If any row carries mode 'TOTAL', only the TOTAL rows are summed; otherwise
    all (breakdown) rows are summed.

    Args:
        rows: candidate-level rows (dicts with 'candidatevotes' and 'mode').
    Returns:
        Integer vote count for the candidate.
    """
    total_rows = [r for r in rows if (r["mode"] or "").strip().upper() == "TOTAL"]
    chosen = total_rows if total_rows else rows
    return sum(int(r["candidatevotes"] or 0) for r in chosen)


def summarize_county_year(rows: list) -> dict:
    """Aggregate one county-year's rows into party-bucketed vote totals.

    Args:
        rows: all faithful rows for a single (year, county_fips). Each dict has
            'candidate', 'party', 'candidatevotes', 'totalvotes', 'mode'.
    Returns:
        dict with dem_votes, rep_votes, other_votes, total_votes (ints).
    """
    by_candidate: dict = {}
    party_of: dict = {}
    for row in rows:
        if (row["candidate"] or "").strip().upper() in NON_CANDIDATE_LABELS:
            continue  # ballot-accounting artifact, not a real candidate
        by_candidate.setdefault(row["candidate"], []).append(row)
        party_of[row["candidate"]] = row["party"]

    buckets = {"dem": 0, "rep": 0, "other": 0}
    for candidate, candidate_rows in by_candidate.items():
        buckets[bucket_party(party_of[candidate])] += _candidate_votes(candidate_rows)

    # totalvotes is constant per county-year; prefer a TOTAL row if present.
    total_rows = [r for r in rows if (r["mode"] or "").strip().upper() == "TOTAL"]
    source = total_rows[0] if total_rows else rows[0]
    return {
        "dem_votes": buckets["dem"],
        "rep_votes": buckets["rep"],
        "other_votes": buckets["other"],
        "total_votes": int(source["totalvotes"] or 0),
    }


# pda/voter_import.py  (append)
from pda.fips import pad_fips, is_county_fips


def create_voter_tables(conn) -> None:
    """Create the faithful and rollup voter tables if they do not exist.

    Args:
        conn: open sqlite3 connection.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS county_presidential_returns ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "year INTEGER, state TEXT, state_po TEXT, county_name TEXT, "
        "county_fips TEXT, office TEXT, candidate TEXT, party TEXT, "
        "candidatevotes INTEGER, totalvotes INTEGER, version TEXT, mode TEXT)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_returns_fips_year "
        "ON county_presidential_returns (county_fips, year)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS county_presidential_summary ("
        "county_fips TEXT, year INTEGER, state_po TEXT, county_name TEXT, "
        "total_votes INTEGER, dem_votes INTEGER, rep_votes INTEGER, "
        "other_votes INTEGER, dem_share REAL, rep_share REAL, dem_margin REAL, "
        "PRIMARY KEY (county_fips, year))")
    conn.commit()


def iter_csv_rows(csv_path: str):
    """Yield dict rows from the MIT CSV, casting numeric fields to int.

    Args:
        csv_path: path to countypres_2000-2024.csv.
    Yields:
        dicts keyed by FAITHFUL_COLUMNS.
    """
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            yield {
                "year": int(record["year"]),
                "state": record["state"],
                "state_po": record["state_po"],
                "county_name": record["county_name"],
                "county_fips": record["county_fips"],
                "office": record["office"],
                "candidate": record["candidate"],
                "party": record["party"],
                "candidatevotes": _to_int(record["candidatevotes"]),
                "totalvotes": _to_int(record["totalvotes"]),
                "version": record["version"],
                "mode": record["mode"],
            }


def _to_int(value):
    """Parse an integer that may be blank or expressed with a decimal point.

    Args:
        value: raw CSV string.
    Returns:
        int, or None when blank.
    """
    if value is None or value == "" or value.strip().upper() in ("NA", "N/A", "NULL", "NONE"):
        return None
    try:
        return int(value)
    except ValueError:
        return int(round(float(value)))


def load_faithful(conn, rows) -> None:
    """Replace and repopulate the faithful returns table from rows.

    Args:
        conn: open sqlite3 connection (tables already created).
        rows: iterable of dicts keyed by FAITHFUL_COLUMNS.
    """
    conn.execute("DELETE FROM county_presidential_returns")
    placeholders = ", ".join("?" for _ in FAITHFUL_COLUMNS)
    columns = ", ".join(FAITHFUL_COLUMNS)
    with conn:
        conn.executemany(
            f"INSERT INTO county_presidential_returns ({columns}) "
            f"VALUES ({placeholders})",
            ([row[column] for column in FAITHFUL_COLUMNS] for row in rows),
        )


def build_summary(conn) -> list:
    """Rebuild the rollup table from the faithful table; return discrepancies.

    Reads every (year, county_fips) group from the faithful table, applies the
    mode-dedup + party-bucket rules, computes shares and margin, and writes one
    summary row per county-year (FIPS zero-padded).

    Non-county reporting units (CT/ME/RI overseas & write-in precincts sharing
    the placeholder FIPS 'NA', MO Kansas City's 7-digit '2938000') are skipped:
    they are not counties, would otherwise collapse into bogus shared
    county-years with impossible margins, and have no place in a county-keyed
    rollup used for FIPS-linked analysis.

    Args:
        conn: open sqlite3 connection with a populated faithful table.
    Returns:
        list of (county_fips, year, bucket_sum, total_votes) tuples where the
        bucketed votes do not equal the published total_votes (for reporting).
    """
    conn.execute("DELETE FROM county_presidential_summary")
    cursor = conn.execute(
        "SELECT year, county_fips, state_po, county_name, candidate, party, "
        "candidatevotes, totalvotes, mode FROM county_presidential_returns")
    column_names = [d[0] for d in cursor.description]
    groups: dict = {}
    meta: dict = {}
    for raw in cursor.fetchall():
        row = dict(zip(column_names, raw))
        key = (row["year"], row["county_fips"])
        groups.setdefault(key, []).append(row)
        meta[key] = (row["state_po"], row["county_name"])

    discrepancies = []
    summary_rows = []
    for (year, county_fips), group in groups.items():
        if not is_county_fips(county_fips):
            continue  # non-county reporting unit — excluded from the rollup
        summary = summarize_county_year(group)
        total = summary["total_votes"]
        bucket_sum = summary["dem_votes"] + summary["rep_votes"] + summary["other_votes"]
        if total and bucket_sum != total:
            discrepancies.append((county_fips, year, bucket_sum, total))
        dem_share = summary["dem_votes"] / total if total else None
        rep_share = summary["rep_votes"] / total if total else None
        margin = (summary["dem_votes"] - summary["rep_votes"]) / total if total else None
        state_po, county_name = meta[(year, county_fips)]
        summary_rows.append((
            pad_fips(county_fips), year, state_po, county_name, total,
            summary["dem_votes"], summary["rep_votes"], summary["other_votes"],
            dem_share, rep_share, margin,
        ))
    with conn:
        conn.executemany(
            "INSERT INTO county_presidential_summary (county_fips, year, "
            "state_po, county_name, total_votes, dem_votes, rep_votes, "
            "other_votes, dem_share, rep_share, dem_margin) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", summary_rows)
    return discrepancies
