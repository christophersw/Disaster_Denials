"""
Title: PDA CSV I/O
Description: Append-only CSV writers (write the header on first touch) and a
    resume helper that reads which source_pdfs are already recorded, so a run
    can be stopped and restarted without re-billing finished reports.
Changelog:
    2026-06-05  Initial version.
"""

import csv
import os


def append_rows(path: str, columns: list[str], rows: list[dict]) -> None:
    """Append rows to a CSV, writing the header if the file is new/empty.

    Args:
        path: Destination CSV path.
        columns: Ordered column names (also the DictWriter fieldnames).
        rows: Row dicts keyed by `columns`.
    """
    if not rows:
        return
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def done_source_pdfs(reports_path: str) -> set[str]:
    """Return the set of source_pdf values already written to reports.csv.

    Args:
        reports_path: Path to the reports CSV.
    Returns:
        Set of source_pdf strings (empty if the file does not exist).
    """
    if not os.path.exists(reports_path):
        return set()
    done: set[str] = set()
    with open(reports_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("source_pdf")
            if value:
                done.add(value)
    return done
