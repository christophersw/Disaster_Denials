"""
Title: PDA SQLite Store
Description: A single-file SQLite store for the extracted dataset, replacing the
    pair of append-only CSVs. One report and all its county rows are written in
    a single transaction, so a crash or failure mid-write leaves nothing
    orphaned (the flaw the CSV path could only work around). source_pdf is the
    primary key on `reports` and a cascading foreign key on `report_counties`,
    so re-extracting a PDF replaces its rows cleanly and a resume run skips PDFs
    already recorded. Column sets are reused verbatim from pda.columns, and
    columns are declared without a type so values keep their Python storage
    class (int / float / str / None) for faithful SQL querying.
Changelog:
    2026-06-05  Initial version.
"""

import sqlite3

from pda.columns import COUNTY_COLUMNS, REPORT_COLUMNS


def _quote(columns: list[str]) -> str:
    """Return the columns as a comma-separated, double-quoted list."""
    return ", ".join(f'"{column}"' for column in columns)


def _reports_ddl() -> str:
    """CREATE TABLE for reports: source_pdf is the primary key."""
    others = [c for c in REPORT_COLUMNS if c != "source_pdf"]
    body = ", ".join([
        '"source_pdf" TEXT PRIMARY KEY',
        *(f'"{column}"' for column in others),
    ])
    return f"CREATE TABLE IF NOT EXISTS reports ({body})"


def _counties_ddl() -> str:
    """CREATE TABLE for report_counties: cascading FK back to reports."""
    others = [c for c in COUNTY_COLUMNS if c != "source_pdf"]
    body = ", ".join([
        "county_id INTEGER PRIMARY KEY AUTOINCREMENT",
        '"source_pdf" TEXT NOT NULL',
        *(f'"{column}"' for column in others),
        'FOREIGN KEY ("source_pdf") REFERENCES reports ("source_pdf") '
        "ON DELETE CASCADE",
    ])
    return f"CREATE TABLE IF NOT EXISTS report_counties ({body})"


def _batch_items_ddl() -> str:
    """CREATE TABLE for batch_items: custom_id ↔ source_pdf ↔ batch_id.

    Tracks Batches API requests in flight so a submit/collect run is resumable
    across process restarts. status is 'submitted' until the result is written
    ('written') or gives up ('failed').
    """
    body = ", ".join([
        "custom_id TEXT PRIMARY KEY",
        "batch_id TEXT NOT NULL",
        "source_pdf TEXT NOT NULL",
        "status TEXT NOT NULL DEFAULT 'submitted'",
    ])
    return f"CREATE TABLE IF NOT EXISTS batch_items ({body})"


def connect(db_path: str) -> sqlite3.Connection:
    """Open the store, enabling foreign keys and creating the schema if absent.

    Args:
        db_path: Path to the SQLite file (created if it does not exist).
    Returns:
        An open connection with foreign-key enforcement on.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(_reports_ddl())
    conn.execute(_counties_ddl())
    conn.execute(_batch_items_ddl())
    conn.commit()
    return conn


def write_report(conn: sqlite3.Connection, report_row: dict,
                 county_rows: list[dict]) -> None:
    """Write one report and its county rows atomically (upsert + replace).

    Re-writing an existing source_pdf replaces the report row and all of its
    county rows. The whole operation runs in one transaction: if any county
    insert fails, the report write rolls back with it.

    Args:
        conn: An open connection from connect().
        report_row: A dict keyed by REPORT_COLUMNS (source_pdf required).
        county_rows: Dicts keyed by COUNTY_COLUMNS, each with the same
            source_pdf as report_row.
    Raises:
        sqlite3.IntegrityError: if a county row violates the foreign key; the
            transaction is rolled back before the exception propagates.
    """
    source_pdf = report_row["source_pdf"]
    report_values = [report_row.get(column) for column in REPORT_COLUMNS]
    county_values = [
        [row.get(column) for column in COUNTY_COLUMNS] for row in county_rows
    ]
    with conn:  # commits on success, rolls back on any exception
        conn.execute(
            f"INSERT OR REPLACE INTO reports ({_quote(REPORT_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(REPORT_COLUMNS))})",
            report_values,
        )
        conn.execute(
            "DELETE FROM report_counties WHERE source_pdf = ?", (source_pdf,)
        )
        if county_values:
            conn.executemany(
                f"INSERT INTO report_counties ({_quote(COUNTY_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(COUNTY_COLUMNS))})",
                county_values,
            )


def done_source_pdfs(conn: sqlite3.Connection) -> set[str]:
    """Return the set of source_pdf values already recorded in reports.

    Args:
        conn: An open connection from connect().
    Returns:
        Set of source_pdf strings (empty if no reports written yet).
    """
    return {row[0] for row in conn.execute("SELECT source_pdf FROM reports")}


def record_batch_items(conn: sqlite3.Connection, batch_id: str,
                       pairs: list[tuple[str, str]]) -> None:
    """Record (custom_id, source_pdf) pairs for a submitted batch.

    Upserts on custom_id so resubmitting a previously failed PDF (same
    deterministic custom_id) overwrites its row with the new batch and resets
    its status to 'submitted'.

    Args:
        conn: An open connection from connect().
        batch_id: The Batches API batch id these items belong to.
        pairs: (custom_id, source_pdf) tuples.
    """
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO batch_items "
            "(custom_id, batch_id, source_pdf, status) "
            "VALUES (?, ?, ?, 'submitted')",
            [(custom_id, batch_id, source_pdf) for custom_id, source_pdf in pairs],
        )


def pending_source_pdfs(conn: sqlite3.Connection) -> set[str]:
    """Return source_pdfs with a batch item still awaiting a result.

    These are in flight (status 'submitted') and must not be resubmitted.

    Args:
        conn: An open connection from connect().
    Returns:
        Set of source_pdf strings.
    """
    return {
        row[0] for row in conn.execute(
            "SELECT source_pdf FROM batch_items WHERE status = 'submitted'")
    }


def open_batch_ids(conn: sqlite3.Connection) -> list[str]:
    """Return batch ids that still have unresolved (submitted) items.

    Args:
        conn: An open connection from connect().
    Returns:
        Batch ids in insertion order.
    """
    return [
        row[0] for row in conn.execute(
            "SELECT DISTINCT batch_id FROM batch_items "
            "WHERE status = 'submitted' ORDER BY rowid")
    ]


def source_pdf_for(conn: sqlite3.Connection, custom_id: str) -> str | None:
    """Return the source_pdf for a custom_id, or None if unknown.

    Args:
        conn: An open connection from connect().
        custom_id: The batch request custom_id.
    Returns:
        The source_pdf string, or None.
    """
    row = conn.execute(
        "SELECT source_pdf FROM batch_items WHERE custom_id = ?", (custom_id,)
    ).fetchone()
    return row[0] if row else None


def mark_batch_item(conn: sqlite3.Connection, custom_id: str,
                    status: str) -> None:
    """Set the status of a batch item ('written' or 'failed').

    Args:
        conn: An open connection from connect().
        custom_id: The batch request custom_id.
        status: New status.
    """
    with conn:
        conn.execute(
            "UPDATE batch_items SET status = ? WHERE custom_id = ?",
            (status, custom_id),
        )
