# pda/governors_import.py
"""
Title: Governor Party-Match Import
Description: Loads governors (one row per continuous gubernatorial tenure) from
    the committed data/governors.csv into a `governors` table, and materializes
    derived columns on `reports`: the requesting governor's name/party and two
    alignment flags — governor vs. sitting President and governor vs. state
    presidential vote. Reuses pda.presidents_import for party bucketing and the
    match rule. Governor columns are kept out of the extraction column contract
    and managed here via guarded ALTER TABLE; re-run import_governors.py to
    refresh. Requires import_presidents.py to have run first (the flags read the
    president/state-vote columns).
Changelog:
    2026-06-13  Initial version.
"""

import csv

from pda.presidents_import import bucket_president_party, party_match

# Columns of data/governors.csv (the committed source of record).
GOVERNOR_CSV_COLUMNS = [
    "state_abbr", "name", "party", "term_start", "term_end", "source_url",
]

# Earliest date that matters (the denial window opens in 2007).
WINDOW_START = "2007-01-01"


def create_governors_table(conn):
    """Create the governors table and its lookup index if absent.

    Args:
        conn: open sqlite3 connection.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS governors ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, state_abbr TEXT, name TEXT, "
        "party TEXT, party_bucket TEXT, term_start TEXT, term_end TEXT, "
        "source_url TEXT)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_governors_state_start "
        "ON governors (state_abbr, term_start)")
    conn.commit()


def iter_governor_rows(csv_path):
    """Yield governor dicts from data/governors.csv.

    Derives party_bucket from party; a blank term_end becomes None (sitting).

    Args:
        csv_path: path to governors.csv.
    Yields:
        dicts with keys state_abbr, name, party, party_bucket, term_start,
        term_end, source_url.
    Raises:
        ValueError: if the CSV header is missing an expected column.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:  # strip BOM
        reader = csv.DictReader(handle)
        missing = set(GOVERNOR_CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"governors CSV missing expected columns: {sorted(missing)}")
        for record in reader:
            term_end = (record["term_end"] or "").strip()
            yield {
                "state_abbr": (record["state_abbr"] or "").strip(),
                "name": (record["name"] or "").strip(),
                "party": (record["party"] or "").strip(),
                "party_bucket": bucket_president_party(record["party"]),
                "term_start": (record["term_start"] or "").strip(),
                "term_end": term_end or None,
                "source_url": (record["source_url"] or "").strip(),
            }


def load_governors(conn, rows):
    """Replace and repopulate the governors table from rows.

    Args:
        conn: open sqlite3 connection (table already created).
        rows: iterable of dicts from iter_governor_rows.
    """
    conn.execute("DELETE FROM governors")
    columns = ["state_abbr", "name", "party", "party_bucket", "term_start",
               "term_end", "source_url"]
    placeholders = ", ".join("?" for _ in columns)
    with conn:
        conn.executemany(
            f"INSERT INTO governors ({', '.join(columns)}) "
            f"VALUES ({placeholders})",
            ([row[column] for column in columns] for row in rows))


def governor_for_date(governors_for_state, anchor_date):
    """Return the governor serving on anchor_date for one state, or None.

    Half-open interval [term_start, term_end): an inauguration-day date belongs
    to the incoming governor; a None term_end is the sitting governor.

    Args:
        governors_for_state: list of governor dicts (term_start, term_end, …)
            already filtered to a single state.
        anchor_date: ISO date string "YYYY-MM-DD", or None.
    Returns:
        The matching governor dict, or None.
    """
    if not anchor_date:
        return None
    for governor in governors_for_state:
        start = governor["term_start"]
        end = governor["term_end"]
        if start and start <= anchor_date and (end is None or anchor_date < end):
            return governor
    return None


def governor_rows_from_extraction(state_abbr, source_url, governors):
    """Build governors.csv row dicts from one state's extracted governor list.

    Drops tenures that ended before the 2007 window; keeps sitting governors
    (empty term_end).

    Args:
        state_abbr: 2-letter state/territory code to stamp on each row.
        source_url: Wikipedia article URL the rows came from.
        governors: list of dicts with name, party, term_start, term_end.
    Returns:
        list of dicts keyed by GOVERNOR_CSV_COLUMNS.
    """
    rows = []
    for governor in governors:
        term_end = (governor.get("term_end") or "").strip()
        if term_end and term_end < WINDOW_START:
            continue
        rows.append({
            "state_abbr": state_abbr,
            "name": (governor.get("name") or "").strip(),
            "party": (governor.get("party") or "").strip(),
            "term_start": (governor.get("term_start") or "").strip(),
            "term_end": term_end,
            "source_url": source_url,
        })
    return rows


# Derived governor columns on reports (report/state grain). Kept out of
# REPORT_COLUMNS; managed here via guarded ALTER TABLE.
_GOVERNOR_COLUMNS = {
    "governor_name": "TEXT",
    "governor_party": "TEXT",
    "governor_vs_president": "INTEGER",
    "governor_vs_state_vote": "INTEGER",
}


def add_governor_columns(conn):
    """Add the governor columns to reports if absent (idempotent).

    Args:
        conn: open sqlite3 connection.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(reports)")}
    for name, sql_type in _GOVERNOR_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE reports ADD COLUMN {name} {sql_type}")
    conn.commit()


def materialize_governor_context(conn):
    """Populate the governor columns on reports.

    Anchor = COALESCE(request_date, decision_date). The governor is the
    `governors` row for the report's state covering the anchor date. The two
    flags reuse party_match against the already-materialized president_party and
    state_winner_party columns.

    Args:
        conn: open sqlite3 connection with governors + governor columns created
            and the presidential columns already populated.
    Returns:
        dict of summary counts (reports, with_governor, no_governor, and
        vs_president / vs_state_vote 1/0/none tallies).
    """
    by_state = {}
    for row in conn.execute(
            "SELECT state_abbr, name, party, party_bucket, term_start, term_end "
            "FROM governors"):
        governor = dict(zip(
            ("state_abbr", "name", "party", "party_bucket", "term_start",
             "term_end"), row))
        by_state.setdefault(governor["state_abbr"], []).append(governor)

    counts = {"reports": 0, "with_governor": 0, "no_governor": 0,
              "vp_1": 0, "vp_0": 0, "vp_none": 0,
              "vs_1": 0, "vs_0": 0, "vs_none": 0}
    updates = []
    for source_pdf, request_date, decision_date, state_abbr, president_party, \
            state_winner_party in conn.execute(
            "SELECT source_pdf, request_date, decision_date, state_abbr, "
            "president_party, state_winner_party FROM reports"):
        anchor = request_date or decision_date
        governor = governor_for_date(by_state.get(state_abbr, []), anchor) \
            if state_abbr else None
        if governor:
            name, party, bucket = (
                governor["name"], governor["party"], governor["party_bucket"])
            counts["with_governor"] += 1
        else:
            name = party = bucket = None
            counts["no_governor"] += 1
        vs_president = party_match(bucket, president_party)
        vs_state = party_match(bucket, state_winner_party)
        updates.append((name, party, vs_president, vs_state, source_pdf))
        counts["reports"] += 1
        counts["vp_" + ("none" if vs_president is None else str(vs_president))] += 1
        counts["vs_" + ("none" if vs_state is None else str(vs_state))] += 1

    with conn:
        conn.executemany(
            "UPDATE reports SET governor_name=?, governor_party=?, "
            "governor_vs_president=?, governor_vs_state_vote=? "
            "WHERE source_pdf=?", updates)
    return counts
