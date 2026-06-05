# Disaster_Denials

Research on FEMA disaster declaration outcomes — **major disaster declarations,
emergency declarations, denials, and appeal denials** — built from FEMA's
[Preliminary Damage Assessment (PDA) reports](https://www.fema.gov/disaster/how-declared/preliminary-damage-assessments/reports).

The goal is a **machine-readable, normalized dataset** of the Individual
Assistance (IA) and Public Assistance (PA) information in each report, suitable
for ML and data visualization, joinable to FEMA's OpenFEMA datasets, and — the
driving research question — to county political lean (via FIPS) to study whether
denial outcomes correlate with partisan lean.

## Pipeline

```
FEMA PDA reports (PDF)  ──►  1. download  ──►  data/pdfs/<Type>/<Year>/*.pdf
                                                      │
                                                      ▼
                            2. parse (LLM)  ──►  data/reports.csv          (one row per report)
                                                 data/report_counties.csv  (one row per report×county)
                                                      │
                                                      ▼
                            3. resolve + join  ─►  county → FIPS, then OpenFEMA +
                               (Phase 2)            county presidential returns
```

1. **Download** every PDA report PDF, organized by type and year.
2. **Parse** each PDF into a **normalized two-table** dataset using Claude with
   structured outputs: report-level (state-level) fields in `data/reports.csv`,
   and per-county fields in `data/report_counties.csv`.
3. **Resolve & join** (Phase 2) — resolve each `county_name` to a FIPS code via
   a Census crosswalk, then join to OpenFEMA and to county presidential returns.

All five report types are parsed (`MajorDisaster` / `Expedited` / `Denials` /
`AppealDenials` / `Other`): denials are the analysis treatment, approvals the
control group.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then paste your Anthropic API key into .env
```

`.env` holds `ANTHROPIC_API_KEY` and is gitignored.

## 1. Download — `download_pda_reports.py`

Crawls the paginated FEMA reports index one page at a time and downloads the
PDFs linked on each page. FEMA sits behind Akamai bot protection that blocks
ordinary HTTP clients, so requests use `curl_cffi` with a Chrome TLS
fingerprint.

```bash
.venv/bin/python download_pda_reports.py            # all report types (default)
.venv/bin/python download_pda_reports.py --verbose  # + per-page console logging
.venv/bin/python download_pda_reports.py --denials-only   # denials + appeals only
.venv/bin/python download_pda_reports.py --retry-errors   # re-download failed rows
```

- **Output layout:** `data/pdfs/<ReportType>/<Year>/<filename>.pdf`, where type
  is `MajorDisaster` / `Emergency` / `Expedited` / `Denials` / `AppealDenials` /
  `Other`, and year comes from the report's posted date on the index.
- **Manifest:** `data/manifest.csv` records every report (state, type, year,
  URL, local path, download status).
- **Resumable:** completed index pages are checkpointed in `data/.progress` and
  existing PDFs are skipped, so re-running continues where it left off.
- **Errors:** failures are written to the manifest with an `error:` status and
  logged to `data/download.log`; `--retry-errors` re-attempts only those rows
  without re-crawling the slow index pages.

## 2. Parse — `parse_pda_reports.py`

For each report, the **native PDF** is sent to **Claude Opus 4.8**
(`claude-opus-4-8`) — with adaptive thinking and `effort: "high"` for accuracy —
as a `document` content block, so the model reads both the text layer and the
visual layout (the two-column IA fields and the inline county lists extract more
reliably than from flattened text). Output is constrained by **structured
outputs** (`output_config.format`) to a JSON Schema generated from the Pydantic
models in `pda/schema.py`, so the response is schema-validated. No local text
extraction (OCR/`pdfplumber`) is used — a full-corpus check confirmed every PDF
has a text layer, and the model reads the PDF directly.

```bash
.venv/bin/python parse_pda_reports.py                       # all PDFs, resumable
.venv/bin/python parse_pda_reports.py --limit 5             # first 5 not-yet-done
.venv/bin/python parse_pda_reports.py --glob 'data/pdfs/Denials/**/*.pdf'
```

The run is **resumable and idempotent**: PDFs already present in
`data/reports.csv` are skipped, so a stopped run continues without re-billing
finished reports. A failure on any one report is logged to stderr and the run
continues.

### Code layout (`pda/` package)

| Module | Responsibility |
| --- | --- |
| `pda/schema.py` | Pydantic `PdaReport` + `County` — the extraction contract and the JSON Schema. Single source of truth for fields. |
| `pda/extract.py` | Builds and sends the Claude request for one PDF; returns a validated `PdaReport`. The only module that touches the network. |
| `pda/provenance.py` | Derives `report_type` (folder), `url`, `posted_date` (from `data/manifest.csv`). |
| `pda/flatten.py` | Splits a `PdaReport` into one `reports` row + N `report_counties` rows. |
| `pda/columns.py` | The exact ordered column lists for the two CSVs. |
| `pda/io.py` | Append-only CSV writers + the resume set. |
| `parse_pda_reports.py` | CLI: glob → skip-done → extract → flatten → append. |

### Extraction rules (enforced by the system prompt)

- Numbers: strip `$`, `%`, thousands separators. A dash `-` or `N/A` → **null**,
  never `0`. A block headed `… – (Not Requested)` sets that program's
  `*_requested` flag false and its numbers null (preserving "not requested" vs
  "genuinely zero").
- `disaster_number` is the numeric part of e.g. `FEMA-4807-DR` → `4807`, and
  `declaration_type` keeps `DR`/`EM` (null for denials/appeals with no number).
- **Requested vs granted per county.** `requested_ia`/`requested_pa` record what
  the Governor requested; `granted_ia`/`granted_pa` record what was actually made
  available (approvals only). Denials and appeal denials have every
  `granted_*` = false (enforced again in `flatten.py`).
- **Legacy vs modern IA fields.** Older reports print only
  `ia_pct_low_income` / `ia_pct_elderly`; modern reports print
  poverty/SSI/SNAP/ownership/unemployment/age/disability/ICC. Whichever the
  report prints is populated; the rest are null. Legacy and modern fields are
  **never** conflated.
- `geo_type` captures non-county jurisdictions (parish, borough, tribe,
  reservation, city-county, municipality) so they can be filtered from the
  county-level political analysis.
- `needs_review` (+ `review_note`) flags any report that departs from the
  expected structure, for auditing rather than blind trust.
- No derived fields — only raw facts. Ratios, FIPS, and joins are computed
  downstream. **FIPS resolution is intentionally not the model's job** (Phase 2),
  to avoid silent hallucinated joins.

### Output tables

**`data/reports.csv`** — one row per report. Provenance columns the model does
not see (`source_pdf`, `report_type`, `url`, `posted_date`, `parser_model`,
`extracted_at`) are added by the pipeline. Columns: `source_pdf`, `report_type`,
`url`, `posted_date`, `report_outcome`, `decision_date`, `jurisdiction_name`,
`state_abbr`, `requestor_type`, `requestor_name`, `incident_name`,
`incident_begin`, `incident_end`, `request_date`, `disaster_number`,
`declaration_type`, `denial_reason`, `original_denial_date`, `appeal_date`,
`ia_requested`, `pa_requested`, `hm_requested`, `pa_categories_requested`,
`ia_residences_total`, `ia_destroyed`, `ia_major`, `ia_minor`, `ia_affected`,
`ia_pct_insured`, `ia_pct_flood_insured`, `ia_pct_poverty`, `ia_pct_ssi`,
`ia_pct_snap`, `ia_pct_ownership`, `ia_unemployment`, `ia_pct_age_65_plus`,
`ia_pct_age_18_under`, `ia_pct_disability`, `ia_icc_ratio`, `ia_pct_low_income`,
`ia_pct_elderly`, `ia_cost_estimate`, `pa_primary_impact`, `pa_cost_estimate`,
`pa_statewide_per_capita`, `pa_statewide_per_capita_indicator`,
`pa_countywide_per_capita_indicator`, `needs_review`, `review_note`,
`parser_model`, `extracted_at`.

**`data/report_counties.csv`** — one row per (report × county), foreign-keyed to
`reports` on `source_pdf`. Columns: `source_pdf`, `county_name`, `geo_type`,
`per_capita_impact`, `requested_ia`, `requested_pa`, `granted_ia`, `granted_pa`,
`source`. (`fips` is added in Phase 2, not by the model.)

## 3. Resolve & join (Phase 2)

- **County → FIPS.** Resolve each `report_counties.county_name` to a FIPS code
  via a Census state+county crosswalk with fuzzy matching, flagging unmatched
  names for review. Tribal/non-county units (`geo_type`) are excluded from the
  county-level political join.
- **Denials / appeal denials** → [`DeclarationDenials`](https://www.fema.gov/api/open/v1/DeclarationDenials)
  on `state_abbr` + `decision_date` (matches `stateAbbreviation` +
  `requestStatusDate`).
- **Approved declarations** → [`DisasterDeclarationsSummaries`](https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries)
  on `disaster_number`.
- **County political lean** → `dataverse_files/countypres_2000-2024.csv`
  (FIPS-keyed), joining each report's `decision_date` year to the nearest
  preceding presidential election.

## Data sources

- FEMA PDA reports index: <https://www.fema.gov/disaster/how-declared/preliminary-damage-assessments/reports>
- OpenFEMA API (no key required): <https://www.fema.gov/about/openfema/api>
