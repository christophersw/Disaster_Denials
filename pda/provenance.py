"""
Title: PDA Provenance Derivation
Description: Derive provenance columns the model does not see — report_type
    (from the folder), and url + posted_date (from data/manifest.csv, joined
    on the manifest's local_path == the PDF path). Pure functions over plain
    dicts so they are trivially testable.
Changelog:
    2026-06-05  Initial version.
"""

import csv
from pathlib import Path

KNOWN_TYPES = {
    "MajorDisaster", "Emergency", "Expedited",
    "Denials", "AppealDenials", "Other",
}


def report_type_from_path(pdf_path: str) -> str:
    """Return the report type from `data/pdfs/<Type>/<Year>/file.pdf`.

    Args:
        pdf_path: Path to a PDF under data/pdfs.
    Returns:
        The <Type> path segment, or "Other" if it is not a known type.
    """
    parts = Path(pdf_path).parts
    for part in parts:
        if part in KNOWN_TYPES:
            return part
    return "Other"


def load_manifest(manifest_path: str) -> dict[str, dict]:
    """Index data/manifest.csv by local_path.

    Args:
        manifest_path: Path to data/manifest.csv.
    Returns:
        Mapping of local_path -> {"url", "posted_date"}.
    """
    index: dict[str, dict] = {}
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            local_path = row.get("local_path")
            if local_path:
                index[local_path] = {
                    "url": row.get("url") or None,
                    "posted_date": row.get("posted_date") or None,
                }
    return index


def provenance_for(pdf_path: str, manifest: dict[str, dict]) -> dict:
    """Build the provenance columns for one PDF.

    Args:
        pdf_path: Path to the PDF (must match the manifest's local_path).
        manifest: Output of load_manifest().
    Returns:
        Dict with report_type, url, posted_date.
    """
    row = manifest.get(pdf_path, {})
    return {
        "report_type": report_type_from_path(pdf_path),
        "url": row.get("url"),
        "posted_date": row.get("posted_date"),
    }
