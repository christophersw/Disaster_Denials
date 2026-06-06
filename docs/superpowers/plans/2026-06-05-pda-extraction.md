# PDA Report Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `parse_pda_reports.py`, an LLM extractor that turns the 1,378 FEMA PDA report PDFs into two normalized CSVs (`data/reports.csv` + `data/report_counties.csv`).

**Architecture:** A thin CLI orchestrator walks the downloaded PDFs, and for each one sends the native PDF to Claude (`claude-opus-4-8`, adaptive thinking, `effort: high`) as a `document` block, constrained by a JSON-Schema structured output. The validated JSON is split into one `reports` row plus N `report_counties` rows, provenance columns are added from the file path and `data/manifest.csv`, and rows are appended idempotently so the run is resumable. Pure logic (provenance, flattening, resumability) is unit-tested; extraction quality is gated by a model-set `needs_review` flag.

**Tech Stack:** Python 3.14 (`.venv`), `anthropic` SDK, `pydantic` v2 (schema + validation), `pytest` (tests). No `pdfplumber` — the native PDF is sent to the model, so no local text extraction is needed.

**Spec:** `docs/superpowers/specs/2026-06-05-pda-extraction-design.md`

---

## Open item to confirm before execution

- **Model id.** Plan uses `claude-opus-4-8`. The saved memory and `README.md` reference `claude-opus-4-7`. The model id is a single constant (`MODEL` in `pda_extract.py`); confirm 4.8 vs 4.7 before running Task 7 at scale. Everything else is identical between the two.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `pda/__init__.py` | Marks the package. |
| `pda/schema.py` | Pydantic models (`County`, `PdaReport`) — the extraction contract + the JSON Schema sent to the API. Single source of truth for fields. |
| `pda/provenance.py` | Derive provenance (`report_type`, `state_abbr`-fallback, `url`, `posted_date`) from a PDF path + the manifest. Pure. |
| `pda/flatten.py` | Turn a validated `PdaReport` + provenance into one `reports` dict and a list of `report_counties` dicts, with fixed column order. Pure. |
| `pda/io.py` | CSV column definitions, append-writers, and "which `source_pdf`s are already done" resume logic. |
| `pda/extract.py` | Build and send the Claude request for one PDF; return a validated `PdaReport`. The only file that touches the network. |
| `parse_pda_reports.py` | CLI: iterate PDFs, skip done ones, call extract → flatten → write, show progress. |
| `tests/test_schema.py` | Structural checks on the generated JSON Schema. |
| `tests/test_provenance.py` | Path/manifest → provenance. |
| `tests/test_flatten.py` | Flattening rules (denial grants false, null handling, no-county row, column order). |
| `tests/test_io.py` | Resume set + append round-trip. |
| `tests/test_extract_request.py` | Request is shaped correctly (no network). |
| `tests/test_integration.py` | One real PDF end-to-end, skipped without `ANTHROPIC_API_KEY`. |

---

## Task 1: Dependencies and test scaffolding

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `pda/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Pin runtime deps**

Replace `requirements.txt` with (drops `pdfplumber`, which the native-PDF approach doesn't need; keeps the downloader deps):

```
# Downloader
curl_cffi>=0.7
tqdm>=4.66

# Parser
anthropic>=0.69
pydantic>=2.7
python-dotenv>=1.0
```

- [ ] **Step 2: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest>=8.0
```

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

- [ ] **Step 4: Create the package markers**

`pda/__init__.py`:

```python
"""PDA report extraction package."""
```

`tests/__init__.py`:

```python
```

- [ ] **Step 5: Install into the venv**

Run: `.venv/bin/pip install -r requirements-dev.txt`
Expected: installs `anthropic`, `pydantic`, `pytest` (and deps) without error.

- [ ] **Step 6: Verify pytest collects nothing yet**

Run: `.venv/bin/python -m pytest -q`
Expected: `no tests ran` (exit code 5) — confirms pytest is wired up.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt requirements-dev.txt pytest.ini pda/__init__.py tests/__init__.py
git commit --no-gpg-sign -m "chore: parser deps and test scaffolding"
```

---

## Task 2: Extraction schema (Pydantic models)

**Files:**
- Create: `pda/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

`tests/test_schema.py`:

```python
"""Structural tests for the extraction JSON Schema."""

from pda.schema import PdaReport, json_schema


def test_schema_forbids_extra_properties_on_all_objects():
    """Structured outputs require additionalProperties:false on every object."""
    schema = json_schema()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)


def test_report_has_counties_list_and_core_fields():
    fields = PdaReport.model_fields
    for name in ("report_outcome", "decision_date", "ia_requested",
                 "pa_requested", "counties", "needs_review"):
        assert name in fields


def test_legacy_and_modern_ia_fields_coexist():
    fields = PdaReport.model_fields
    for name in ("ia_pct_poverty", "ia_pct_ssi", "ia_pct_low_income", "ia_pct_elderly"):
        assert name in fields
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_schema.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pda.schema'`

- [ ] **Step 3: Write `pda/schema.py`**

```python
"""
Title: PDA Extraction Schema
Description: Pydantic models defining the exact data contract the LLM must
    return for one FEMA PDA report, plus the JSON Schema used for structured
    outputs. Single source of truth for the report-level and county-level
    fields. `extra="forbid"` makes the generated schema emit
    additionalProperties:false, which structured outputs require.
Changelog:
    2026-06-05  Initial version (normalized two-table design).
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class County(BaseModel):
    """One geographic unit named in a report.

    Attributes:
        county_name: Raw name as printed (null only when no unit is named).
        geo_type: The kind of unit, for filtering non-county jurisdictions.
        per_capita_impact: Dollar figure from the PA countywide per-capita list.
        requested_ia/requested_pa: Governor requested this unit for IA/PA.
        granted_ia/granted_pa: Assistance actually made available here
            (always false for denials/appeals).
        source: Where in the document this unit was found.
    """

    model_config = ConfigDict(extra="forbid")

    county_name: Optional[str]
    geo_type: Literal[
        "county", "parish", "borough", "tribe", "reservation",
        "city-county", "municipality", "unknown",
    ]
    per_capita_impact: Optional[float]
    requested_ia: bool
    requested_pa: bool
    granted_ia: bool
    granted_pa: bool
    source: Literal["per_capita", "narrative", "both", "none"]


class PdaReport(BaseModel):
    """All report-level (state-level) fields for one PDA report PDF."""

    model_config = ConfigDict(extra="forbid")

    # Identity & outcome
    report_outcome: Optional[Literal["Declared", "Denied", "Denial of Appeal"]]
    decision_date: Optional[str]            # YYYY-MM-DD
    jurisdiction_name: Optional[str]        # state or tribe
    state_abbr: Optional[str]               # two-letter USPS; null for tribes
    requestor_type: Optional[str]           # Governor / Tribal Chairman / ...
    requestor_name: Optional[str]
    incident_name: Optional[str]
    incident_begin: Optional[str]           # YYYY-MM-DD
    incident_end: Optional[str]             # YYYY-MM-DD
    request_date: Optional[str]             # YYYY-MM-DD
    disaster_number: Optional[int]          # numeric part only; null for denials
    declaration_type: Optional[Literal["DR", "EM"]]
    denial_reason: Optional[str]
    original_denial_date: Optional[str]     # appeals only
    appeal_date: Optional[str]              # appeals only

    # Requested programs
    ia_requested: bool
    pa_requested: bool
    hm_requested: bool
    pa_categories_requested: Optional[str]  # e.g. "A,B" or "A-F"

    # Individual Assistance (state-level)
    ia_residences_total: Optional[float]
    ia_destroyed: Optional[float]
    ia_major: Optional[float]
    ia_minor: Optional[float]
    ia_affected: Optional[float]
    ia_pct_insured: Optional[float]
    ia_pct_flood_insured: Optional[float]
    ia_pct_poverty: Optional[float]
    ia_pct_ssi: Optional[float]
    ia_pct_snap: Optional[float]
    ia_pct_ownership: Optional[float]
    ia_unemployment: Optional[float]
    ia_pct_age_65_plus: Optional[float]
    ia_pct_age_18_under: Optional[float]
    ia_pct_disability: Optional[float]
    ia_icc_ratio: Optional[float]
    ia_pct_low_income: Optional[float]      # legacy (older reports)
    ia_pct_elderly: Optional[float]         # legacy (older reports)
    ia_cost_estimate: Optional[float]

    # Public Assistance (state-level)
    pa_primary_impact: Optional[str]
    pa_cost_estimate: Optional[float]
    pa_statewide_per_capita: Optional[float]
    pa_statewide_per_capita_indicator: Optional[float]
    pa_countywide_per_capita_indicator: Optional[float]

    # Counties
    counties: list[County]

    # Review
    needs_review: bool
    review_note: Optional[str]


def json_schema() -> dict:
    """Return the JSON Schema for structured outputs (additionalProperties:false)."""
    return PdaReport.model_json_schema()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_schema.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pda/schema.py tests/test_schema.py
git commit --no-gpg-sign -m "feat: PDA extraction schema (pydantic + json schema)"
```

---

## Task 3: Provenance from path + manifest

**Files:**
- Create: `pda/provenance.py`
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write the failing test**

`tests/test_provenance.py`:

```python
"""Tests for deriving provenance from a PDF path and the manifest."""

from pda.provenance import provenance_for, report_type_from_path


def test_report_type_from_path():
    p = "data/pdfs/AppealDenials/2026/FY26PDAReport_AppealDenial-CO.pdf"
    assert report_type_from_path(p) == "AppealDenials"


def test_provenance_joins_manifest_on_local_path():
    manifest = {
        "data/pdfs/Denials/2024/PDAReport_Denial-ID.pdf": {
            "url": "https://example.gov/ID.pdf",
            "posted_date": "2024-11-20T12:00:00Z",
        }
    }
    prov = provenance_for("data/pdfs/Denials/2024/PDAReport_Denial-ID.pdf", manifest)
    assert prov["report_type"] == "Denials"
    assert prov["url"] == "https://example.gov/ID.pdf"
    assert prov["posted_date"] == "2024-11-20T12:00:00Z"


def test_provenance_tolerates_missing_manifest_row():
    prov = provenance_for("data/pdfs/Other/2024/x.pdf", {})
    assert prov["report_type"] == "Other"
    assert prov["url"] is None
    assert prov["posted_date"] is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_provenance.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pda.provenance'`

- [ ] **Step 3: Write `pda/provenance.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_provenance.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pda/provenance.py tests/test_provenance.py
git commit --no-gpg-sign -m "feat: provenance derivation from path + manifest"
```

---

## Task 4: Flatten model output into two tables

**Files:**
- Create: `pda/columns.py`
- Create: `pda/flatten.py`
- Test: `tests/test_flatten.py`

- [ ] **Step 1: Write the failing test**

`tests/test_flatten.py`:

```python
"""Tests for flattening a PdaReport into reports + report_counties rows."""

from pda.columns import COUNTY_COLUMNS, REPORT_COLUMNS
from pda.flatten import flatten
from pda.schema import County, PdaReport


def _denial() -> PdaReport:
    return PdaReport(
        report_outcome="Denied", decision_date="2024-11-20",
        jurisdiction_name="Idaho", state_abbr="ID",
        requestor_type="Governor", requestor_name="Brad Little",
        incident_name="Gwen Fire", incident_begin="2024-07-24",
        incident_end="2024-08-09", request_date="2024-10-07",
        disaster_number=None, declaration_type=None,
        denial_reason="Not of such severity and magnitude.",
        original_denial_date=None, appeal_date=None,
        ia_requested=True, pa_requested=False, hm_requested=True,
        pa_categories_requested=None,
        ia_residences_total=70, ia_destroyed=42, ia_major=0, ia_minor=1,
        ia_affected=27, ia_pct_insured=66.5, ia_pct_flood_insured=None,
        ia_pct_poverty=14.1, ia_pct_ssi=7.8, ia_pct_snap=9.2,
        ia_pct_ownership=99.0, ia_unemployment=3.2, ia_pct_age_65_plus=20.3,
        ia_pct_age_18_under=21.4, ia_pct_disability=17.7, ia_icc_ratio=8.57,
        ia_pct_low_income=None, ia_pct_elderly=None, ia_cost_estimate=1066144,
        pa_primary_impact=None, pa_cost_estimate=None,
        pa_statewide_per_capita=None, pa_statewide_per_capita_indicator=1.84,
        pa_countywide_per_capita_indicator=4.60,
        counties=[County(
            county_name="Nez Perce", geo_type="county", per_capita_impact=None,
            requested_ia=True, requested_pa=False, granted_ia=True, granted_pa=False,
            source="narrative",
        )],
        needs_review=False, review_note=None,
    )


PROV = {"report_type": "Denials", "url": "u", "posted_date": "2024-11-20T12:00:00Z"}
META = {"parser_model": "claude-opus-4-8", "extracted_at": "2026-06-05T00:00:00Z"}
PDF = "data/pdfs/Denials/2024/PDAReport_Denial-ID.pdf"


def test_one_report_row_with_provenance_and_meta():
    report_row, _ = flatten(_denial(), PDF, PROV, META)
    assert report_row["source_pdf"] == PDF
    assert report_row["report_type"] == "Denials"
    assert report_row["parser_model"] == "claude-opus-4-8"
    assert set(report_row.keys()) == set(REPORT_COLUMNS)


def test_county_rows_carry_fk_and_columns():
    _, county_rows = flatten(_denial(), PDF, PROV, META)
    assert len(county_rows) == 1
    assert county_rows[0]["source_pdf"] == PDF
    assert county_rows[0]["county_name"] == "Nez Perce"
    assert set(county_rows[0].keys()) == set(COUNTY_COLUMNS)


def test_denial_forces_granted_false():
    """A denial must never record granted assistance, even if the model slips."""
    report = _denial()
    report.counties[0].granted_ia = True   # model error
    report.counties[0].granted_pa = True
    _, county_rows = flatten(report, PDF, PROV, META)
    assert county_rows[0]["granted_ia"] is False
    assert county_rows[0]["granted_pa"] is False


def test_report_with_no_counties_emits_zero_county_rows():
    report = _denial()
    report.counties = []
    _, county_rows = flatten(report, PDF, PROV, META)
    assert county_rows == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_flatten.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pda.columns'`

- [ ] **Step 3: Write `pda/columns.py`**

```python
"""
Title: PDA CSV Column Order
Description: The exact, ordered column lists for the two output CSVs. Defined
    once so the writers, the flattener, and the tests agree.
Changelog:
    2026-06-05  Initial version.
"""

# Provenance/meta columns are added by the pipeline; the rest mirror schema.PdaReport.
REPORT_COLUMNS = [
    "source_pdf", "report_type", "url", "posted_date",
    "report_outcome", "decision_date", "jurisdiction_name", "state_abbr",
    "requestor_type", "requestor_name", "incident_name",
    "incident_begin", "incident_end", "request_date",
    "disaster_number", "declaration_type", "denial_reason",
    "original_denial_date", "appeal_date",
    "ia_requested", "pa_requested", "hm_requested", "pa_categories_requested",
    "ia_residences_total", "ia_destroyed", "ia_major", "ia_minor", "ia_affected",
    "ia_pct_insured", "ia_pct_flood_insured", "ia_pct_poverty", "ia_pct_ssi",
    "ia_pct_snap", "ia_pct_ownership", "ia_unemployment", "ia_pct_age_65_plus",
    "ia_pct_age_18_under", "ia_pct_disability", "ia_icc_ratio",
    "ia_pct_low_income", "ia_pct_elderly", "ia_cost_estimate",
    "pa_primary_impact", "pa_cost_estimate", "pa_statewide_per_capita",
    "pa_statewide_per_capita_indicator", "pa_countywide_per_capita_indicator",
    "needs_review", "review_note", "parser_model", "extracted_at",
]

COUNTY_COLUMNS = [
    "source_pdf", "county_name", "geo_type", "per_capita_impact",
    "requested_ia", "requested_pa", "granted_ia", "granted_pa", "source",
]
```

- [ ] **Step 4: Write `pda/flatten.py`**

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_flatten.py -q`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add pda/columns.py pda/flatten.py tests/test_flatten.py
git commit --no-gpg-sign -m "feat: flatten report into normalized reports + county rows"
```

---

## Task 5: CSV writers and resume logic

**Files:**
- Create: `pda/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing test**

`tests/test_io.py`:

```python
"""Tests for append-writers and resume set."""

from pda.columns import COUNTY_COLUMNS, REPORT_COLUMNS
from pda.io import append_rows, done_source_pdfs


def test_append_writes_header_once_then_rows(tmp_path):
    path = tmp_path / "reports.csv"
    append_rows(str(path), REPORT_COLUMNS, [{c: None for c in REPORT_COLUMNS}])
    row = {c: None for c in REPORT_COLUMNS}
    row["source_pdf"] = "a.pdf"
    append_rows(str(path), REPORT_COLUMNS, [row])
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("source_pdf,")
    assert len(lines) == 3  # header + 2 rows


def test_done_source_pdfs_reads_existing(tmp_path):
    path = tmp_path / "reports.csv"
    row = {c: None for c in REPORT_COLUMNS}
    row["source_pdf"] = "data/pdfs/Denials/2024/x.pdf"
    append_rows(str(path), REPORT_COLUMNS, [row])
    assert done_source_pdfs(str(path)) == {"data/pdfs/Denials/2024/x.pdf"}


def test_done_source_pdfs_missing_file_is_empty(tmp_path):
    assert done_source_pdfs(str(tmp_path / "nope.csv")) == set()


def test_county_columns_round_trip(tmp_path):
    path = tmp_path / "counties.csv"
    row = {c: None for c in COUNTY_COLUMNS}
    row["source_pdf"] = "a.pdf"
    row["county_name"] = "Nez Perce"
    append_rows(str(path), COUNTY_COLUMNS, [row])
    assert "Nez Perce" in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_io.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pda.io'`

- [ ] **Step 3: Write `pda/io.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_io.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add pda/io.py tests/test_io.py
git commit --no-gpg-sign -m "feat: append-only CSV writers + resume set"
```

---

## Task 6: Extraction request (the Claude call)

**Files:**
- Create: `pda/extract.py`
- Test: `tests/test_extract_request.py`

- [ ] **Step 1: Write the failing test (no network)**

`tests/test_extract_request.py`:

```python
"""Tests that the request is shaped correctly, without calling the API."""

import base64

from pda.extract import MODEL, build_request


def test_build_request_has_pdf_document_and_schema():
    req = build_request(b"%PDF-1.4 fake bytes")
    assert req["model"] == MODEL
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"]["effort"] == "high"
    assert req["output_config"]["format"]["type"] == "json_schema"

    content = req["messages"][0]["content"]
    doc = next(b for b in content if b["type"] == "document")
    assert doc["source"]["media_type"] == "application/pdf"
    # data is base64 of the input bytes
    assert base64.b64decode(doc["source"]["data"]) == b"%PDF-1.4 fake bytes"


def test_system_prompt_is_cacheable():
    req = build_request(b"x")
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_extract_request.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pda.extract'`

- [ ] **Step 3: Write `pda/extract.py`**

```python
"""
Title: PDA Extraction (Claude call)
Description: Build and send the structured-output request for one PDA report
    PDF and return a validated PdaReport. The native PDF is sent as a document
    block so the model reads the two-column layout and county lists directly;
    a JSON Schema (from pda.schema) constrains the output. Adaptive thinking +
    effort:high prioritize accuracy. The system prompt + schema form a stable
    cached prefix; the PDF is the per-request tail.
Changelog:
    2026-06-05  Initial version.
"""

import base64

import anthropic

from pda.schema import PdaReport, json_schema

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """\
You are a meticulous data-extraction assistant. You are given one FEMA \
Preliminary Damage Assessment (PDA) report as a PDF. Return the fields defined \
by the JSON schema. Never infer, estimate, or fabricate a value that is not \
present in the document.

Document structure:
- The report opens with a "<Jurisdiction> – <incident name>" line and an \
outcome line: "Declared <date>", "Denied on <date>", or "Denial of Appeal" \
followed by "Denied on <date>".
- One or more narrative paragraphs describe who requested what (Individual \
Assistance, Public Assistance with categories, Hazard Mitigation) and for which \
counties. For approvals, a later sentence states what the President made \
available and where — this can be a SUBSET of what was requested.
- Two data blocks follow — Individual Assistance and Public Assistance — each a \
bulleted list. A block headed "... – (Not Requested)" means that program was \
not requested; set its *_requested flag false and all its numbers null.
- Numbered footnotes at the bottom are legal boilerplate. Ignore them.

Extraction rules:
- Numbers: strip "$", "%", and thousands separators and return a number. A dash \
"-" or "N/A" means there is no value: return null. Never coerce "-" to 0.
- Dates: return as YYYY-MM-DD.
- report_outcome: "Declared", "Denied", or "Denial of Appeal" from the outcome \
line. decision_date: the date on that line.
- disaster_number: numeric part of e.g. "FEMA-4807-DR" -> 4807; \
declaration_type: "DR" or "EM"; both null for denials/appeals with no number.
- requestor_type: "Governor", "Tribal Chairman", "Acting Chairman", etc.; \
requestor_name: the named person.
- For appeals, set original_denial_date (the first denial) and appeal_date.
- Insured split: "62.4% Flood 6.0%" -> ia_pct_insured=62.4, \
ia_pct_flood_insured=6.0. "8.1% SSI / 16.3% SNAP" -> ia_pct_ssi, ia_pct_snap.
- Legacy vs modern IA fields: older reports print "Percentage of low income \
households" and "Percentage of elderly households" -> ia_pct_low_income, \
ia_pct_elderly. Modern reports print poverty/SSI/SNAP/ownership/unemployment/\
age/disability/ICC. Populate whichever the report actually prints; leave the \
others null. Do NOT map legacy fields onto modern ones.
- pa_categories_requested: the PA categories named, e.g. "A,B" or "A-F".

Counties (one entry per geographic unit):
- Primary source is the PA "Countywide per capita impact" list: each \
"Name ($amount)" -> per_capita_impact set, source "per_capita".
- Also include units named in the narrative. requested_ia/requested_pa: the \
unit was REQUESTED for that program. granted_ia/granted_pa: assistance was \
actually MADE AVAILABLE there (approvals only). For denials and appeal denials, \
set granted_ia and granted_pa false for every unit.
- A unit in both the per-capita list and the narrative -> source "both", keep \
the per-capita amount.
- geo_type: county / parish / borough / tribe / reservation / city-county / \
municipality / unknown, based on how the unit is described.
- If the narrative gives only a count ("18 counties") with no names, do not \
invent names. If no unit is named anywhere, return exactly one entry with \
county_name null, source "none", all flags false.

needs_review: true if the document departs from this structure or any required \
field is ambiguous; put a short reason in review_note. No derived fields — \
return only raw facts.\
"""

USER_INSTRUCTION = (
    "Extract this FEMA PDA report. Return JSON matching the schema exactly."
)


def build_request(pdf_bytes: bytes) -> dict:
    """Build the kwargs for one client.messages.create call.

    Args:
        pdf_bytes: Raw bytes of the PDF.
    Returns:
        A dict of keyword arguments for the Messages API.
    """
    encoded = base64.standard_b64encode(pdf_bytes).decode("ascii")
    return {
        "model": MODEL,
        "max_tokens": 16000,
        "thinking": {"type": "adaptive"},
        "system": [{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": encoded,
                    },
                },
                {"type": "text", "text": USER_INSTRUCTION},
            ],
        }],
        "output_config": {
            "format": {"type": "json_schema", "schema": json_schema()},
            "effort": "high",
        },
    }


def extract_report(client: anthropic.Anthropic, pdf_bytes: bytes) -> PdaReport:
    """Extract one PDF into a validated PdaReport.

    Args:
        client: An Anthropic client.
        pdf_bytes: Raw PDF bytes.
    Returns:
        The validated PdaReport.
    Raises:
        pydantic.ValidationError: if the response does not match the schema.
    """
    response = client.messages.create(**build_request(pdf_bytes))
    text = next(block.text for block in response.content if block.type == "text")
    return PdaReport.model_validate_json(text)
```

- [ ] **Step 4: Run the request-shape tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_extract_request.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add pda/extract.py tests/test_extract_request.py
git commit --no-gpg-sign -m "feat: structured-output extraction request for one PDF"
```

---

## Task 7: CLI orchestrator

**Files:**
- Create: `parse_pda_reports.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the guarded integration test**

`tests/test_integration.py`:

```python
"""End-to-end test on one real PDF. Skipped without an API key."""

import os

import pytest

PDF = "data/pdfs/Denials/2024/PDAReport_Denial-ID.pdf"


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") or not os.path.exists(PDF),
    reason="needs ANTHROPIC_API_KEY and the sample PDF",
)
def test_extract_one_real_denial():
    import anthropic
    from pda.extract import extract_report

    with open(PDF, "rb") as handle:
        report = extract_report(anthropic.Anthropic(), handle.read())

    assert report.report_outcome == "Denied"
    assert report.state_abbr == "ID"
    assert report.ia_requested is True
    assert report.pa_requested is False
    # Idaho/Gwen Fire denial: PA not requested, so every county granted_pa false
    assert all(c.granted_pa is False for c in report.counties)
```

- [ ] **Step 2: Write `parse_pda_reports.py`**

```python
"""
Title: PDA Report Parser (CLI)
Description: Walk the downloaded PDA report PDFs, extract each with Claude into
    a validated PdaReport, flatten into normalized reports + report_counties
    rows (adding provenance from the manifest and run metadata), and append to
    data/reports.csv and data/report_counties.csv. Idempotent and resumable:
    PDFs already present in reports.csv are skipped. Failures are logged and the
    run continues.
Changelog:
    2026-06-05  Initial version.

Usage:
    .venv/bin/python parse_pda_reports.py            # all PDFs, resume
    .venv/bin/python parse_pda_reports.py --limit 5  # first 5 not-yet-done
    .venv/bin/python parse_pda_reports.py --glob 'data/pdfs/Denials/**/*.pdf'
"""

import argparse
import datetime
import glob as globlib
import sys

import anthropic
from dotenv import load_dotenv

from pda.columns import COUNTY_COLUMNS, REPORT_COLUMNS
from pda.extract import MODEL, extract_report
from pda.flatten import flatten
from pda.io import append_rows, done_source_pdfs
from pda.provenance import load_manifest, provenance_for

REPORTS_CSV = "data/reports.csv"
COUNTIES_CSV = "data/report_counties.csv"
MANIFEST_CSV = "data/manifest.csv"


def main(argv: list[str] | None = None) -> int:
    """Run the parser. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Extract FEMA PDA reports to CSV.")
    parser.add_argument("--glob", default="data/pdfs/**/*.pdf",
                        help="Glob of PDFs to process (default: all).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N not-yet-done PDFs.")
    args = parser.parse_args(argv)

    load_dotenv()
    client = anthropic.Anthropic()
    manifest = load_manifest(MANIFEST_CSV)
    done = done_source_pdfs(REPORTS_CSV)

    pdfs = sorted(p for p in globlib.glob(args.glob, recursive=True)
                  if p not in done)
    if args.limit is not None:
        pdfs = pdfs[:args.limit]

    print(f"{len(pdfs)} PDF(s) to process ({len(done)} already done).")

    ok = 0
    failed = 0
    for index, pdf_path in enumerate(pdfs, 1):
        try:
            with open(pdf_path, "rb") as handle:
                report = extract_report(client, handle.read())
            meta = {
                "parser_model": MODEL,
                "extracted_at": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
            }
            report_row, county_rows = flatten(
                report, pdf_path, provenance_for(pdf_path, manifest), meta)
            append_rows(REPORTS_CSV, REPORT_COLUMNS, [report_row])
            append_rows(COUNTIES_CSV, COUNTY_COLUMNS, county_rows)
            flag = " [needs_review]" if report.needs_review else ""
            print(f"[{index}/{len(pdfs)}] OK {pdf_path}{flag}")
            ok += 1
        except Exception as error:  # noqa: BLE001 — keep going on any one failure
            print(f"[{index}/{len(pdfs)}] FAIL {pdf_path}: {error}",
                  file=sys.stderr)
            failed += 1

    print(f"Done. {ok} ok, {failed} failed.")
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the full unit suite (no network)**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS for all schema/provenance/flatten/io/extract-request tests; `test_integration.py` shows `s` (skipped) if no key is set.

- [ ] **Step 4: Smoke-test on ONE real PDF (uses the API — small cost)**

Run: `.venv/bin/python parse_pda_reports.py --glob 'data/pdfs/Denials/2024/PDAReport_Denial-ID.pdf' --limit 1`
Expected: prints `1/1 OK ...`; `data/reports.csv` has a header + 1 row; `data/report_counties.csv` has a header + ≥1 row.

- [ ] **Step 5: Confirm resume skips the done PDF**

Run: `.venv/bin/python parse_pda_reports.py --glob 'data/pdfs/Denials/2024/PDAReport_Denial-ID.pdf' --limit 1`
Expected: prints `0 PDF(s) to process (1 already done).`

- [ ] **Step 6: Run the guarded integration test against the API**

Run: `ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY .venv/bin/python -m pytest tests/test_integration.py -q`
Expected: PASS (1 passed) — confirms the Idaho denial extracts as expected.

- [ ] **Step 7: Commit**

```bash
git add parse_pda_reports.py tests/test_integration.py data/reports.csv data/report_counties.csv
git commit --no-gpg-sign -m "feat: PDA parser CLI (resumable extraction to two CSVs)"
```

> **Scaling note (not a code step):** after the smoke test passes and the model id is confirmed, run a batch (e.g. `--glob 'data/pdfs/Denials/**/*.pdf'`, then `--glob 'data/pdfs/AppealDenials/**/*.pdf'`, then the rest) and audit the rows where `needs_review` is true. Full-corpus extraction across 1,378 PDFs is a billed run — confirm the per-report cost on the first batch before processing everything.

---

## Task 8: Update the README to the two-table design

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the Parse + Output sections**

In `README.md`, update the pipeline diagram and the "2. Parse" / "Output" sections so they describe the normalized two-table output (`data/reports.csv` + `data/report_counties.csv`) instead of the single flat `data/denial_counties.csv`. Specifically:
- Change the pipeline diagram's parse output to `data/reports.csv` + `data/report_counties.csv`.
- State the model as `claude-opus-4-8` (adaptive thinking, `effort: high`) — or `claude-opus-4-7` if that is what was confirmed in the Open Item above.
- Replace the single `record_pda_report` tool-schema description with: structured outputs (`output_config.format`) from `pda/schema.py`, the request built by `pda/extract.py`, and the two-table flatten by `pda/flatten.py`.
- List the `reports` columns (REPORT_COLUMNS) and `report_counties` columns (COUNTY_COLUMNS).
- Note the request/granted-per-county distinction, the legacy vs modern IA fields, `geo_type`, `declaration_type`, and that FIPS resolution is a Phase-2 step (not done by the LLM).
- Update the join section: denials/appeals join OpenFEMA `DeclarationDenials` on `state_abbr` + `decision_date`; approvals join `DisasterDeclarationsSummaries` on `disaster_number`; `report_counties.county_name` resolves to FIPS in Phase 2 for the presidential-lean join.

- [ ] **Step 2: Sanity-check the README renders**

Run: `.venv/bin/python -c "print(open('README.md').read()[:1])"`
Expected: prints `#` (file readable; no broken edit).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit --no-gpg-sign -m "docs: README to normalized two-table extraction design"
```

---

## Self-Review

**Spec coverage:**
- Two normalized tables → Tasks 2 (schema), 4 (flatten), 5 (io). ✅
- Request/granted per county → `schema.County` + `flatten` denial-forces-false test. ✅
- Legacy + modern IA fields → `schema.PdaReport` + `test_schema`. ✅
- geo_type / declaration_type / requestor_type / appeal dates / pa_categories → `schema.PdaReport`. ✅
- No derived fields; raw facts only → system prompt in `extract.py`. ✅
- Native PDF document block, adaptive thinking, effort high, caching → `extract.build_request` + `test_extract_request`. ✅
- FIPS kept out of the LLM → no FIPS field in schema; README Phase-2 note (Task 8). ✅
- Resumable / idempotent → `io.done_source_pdfs` + CLI skip + `test_io`. ✅
- All report types (approvals as control) → CLI default glob `data/pdfs/**/*.pdf`. ✅
- Provenance from manifest → `provenance.py` + `test_provenance`. ✅
- README rewrite deliverable → Task 8. ✅

**Placeholder scan:** No TBDs; every code step has complete code and exact run commands. The one deferred decision (model id) is a single named constant, flagged in the Open Item.

**Type consistency:** `REPORT_COLUMNS`/`COUNTY_COLUMNS` are defined once in `pda/columns.py` and referenced by `flatten`, `io`, the CLI, and tests. `MODEL`, `build_request`, `extract_report`, `flatten`, `append_rows`, `done_source_pdfs`, `load_manifest`, `provenance_for` names match across their definitions and call sites.
