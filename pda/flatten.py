"""
Title: PDA Flattener
Description: Turn one validated PdaReport (plus provenance and run metadata)
    into a single reports row and a list of report_counties rows. Pure: no I/O.
    Enforces the invariant that denials/appeals never record granted assistance.
Changelog:
    2026-06-05  Initial version.
"""

from pda.columns import COUNTY_COLUMNS, REPORT_COLUMNS
from pda.schema import PdaReport


def flatten(report: PdaReport, source_pdf: str, provenance: dict, meta: dict
            ) -> tuple[dict, list[dict]]:
    """Split a report into a reports row and report_counties rows.

    Args:
        report: The validated model output for one PDF.
        source_pdf: The PDF path; primary key + county foreign key.
        provenance: {"report_type", "url", "posted_date"}.
        meta: {"parser_model", "extracted_at"}.
    Returns:
        (report_row, county_rows) — dicts keyed exactly by the column lists.
    """
    data = report.model_dump()
    is_denial = report.report_outcome in ("Denied", "Denial of Appeal")

    report_row = {
        "source_pdf": source_pdf,
        "report_type": provenance["report_type"],
        "url": provenance["url"],
        "posted_date": provenance["posted_date"],
        "parser_model": meta["parser_model"],
        "extracted_at": meta["extracted_at"],
    }
    for column in REPORT_COLUMNS:
        if column not in report_row:
            report_row[column] = data.get(column)

    county_rows = []
    for county in report.counties:
        county_data = county.model_dump()
        row = {"source_pdf": source_pdf}
        for column in COUNTY_COLUMNS:
            if column == "source_pdf":
                continue
            row[column] = county_data.get(column)
        if is_denial:
            row["granted_ia"] = False
            row["granted_pa"] = False
        county_rows.append(row)

    return report_row, county_rows
