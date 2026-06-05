# Disaster_Denials

Research on FEMA disaster declaration outcomes — **major disaster declarations,
emergency declarations, denials, and appeal denials** — built from FEMA's
[Preliminary Damage Assessment (PDA) reports](https://www.fema.gov/disaster/how-declared/preliminary-damage-assessments/reports).

The goal is a **county-level, machine-readable dataset** of the Individual
Assistance (IA) and Public Assistance (PA) information in each report, suitable
for ML and data visualization, and joinable to FEMA's OpenFEMA datasets.

## Pipeline

```
FEMA PDA reports (PDF)  ──►  1. download  ──►  data/pdfs/<Type>/<Year>/*.pdf
                                                      │
                                                      ▼
                            2. parse (LLM)  ──►  data/denial_counties.csv
                                                      │
                                                      ▼
                            3. join (optional) ─►  OpenFEMA DeclarationDenials /
                                                   DisasterDeclarationsSummaries
```

1. **Download** every PDA report PDF, organized by type and year.
2. **Parse** each PDF into structured, county-level rows using Claude with
   structured outputs (one row per county per report).
3. **Join** (optional) to OpenFEMA on state + decision date (denials) or
   disaster number (approved declarations).

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

For each report, the **native PDF** is sent to **Claude Opus 4.7**
(`claude-opus-4-7`) — with adaptive thinking and `effort: "high"` for accuracy —
as a `document` content block, so the model reads both the text layer and the
visual layout (the two-column IA fields and the county lists extract more
reliably than from flattened text). It returns the data via a strict JSON tool
schema, which the script explodes
into **one CSV row per county per report**. Counties come primarily from the PA
"Countywide per capita impact" list (which carries the per-county dollar
amount) and are unioned with counties named in the narrative; reports naming no
counties yield a single row with a null county.

Null semantics: a `-` or `N/A` becomes an empty/null value (not zero), and the
`(Not Requested)` section label sets `ia_requested` / `pa_requested` to false —
preserving the difference between "not requested" and "genuinely zero".

Quality levers: Opus 4.7 with adaptive thinking and `effort: "high"`; 2–3
few-shot examples covering the sparse-denial and rich-approved extremes; and a
`needs_review` flag so low-confidence extractions can be audited rather than
trusted blindly. Prompt caching (stable system + examples + tool schema prefix)
is applied too, but as a side benefit — accuracy is the priority.

### Precise parsing prompt

The parser sends the following to the Anthropic Messages API. The **system
prompt** defines the extraction rules; the report PDF is sent as a `document`
block in the user message; and the model is required to return its answer by
calling the `record_pda_report` tool, whose `input_schema` is the exact output
contract. The system prompt, few-shot examples, and tool schema form the cached
prefix; the PDF is the per-request tail.

**System prompt:**

```text
You are a meticulous data-extraction assistant. You are given one FEMA
Preliminary Damage Assessment (PDA) report as a PDF. Extract the fields defined
by the `record_pda_report` tool and return them by calling that tool. Never
infer, estimate, or fabricate a value that is not present in the document.

Document structure:
- The report opens with a title, a "<State> – <incident name>" line, and an
  outcome line: "Declared <date>", "Denied on <date>", or a "Denial of Appeal"
  followed by "Denied on <date>".
- One or more narrative paragraphs describe who requested what (Individual
  Assistance, Public Assistance, Hazard Mitigation) and for which counties.
- Two data blocks follow — Individual Assistance and Public Assistance — each a
  bulleted list of fields. A block headed "... – (Not Requested)" means that
  program was not requested; treat all its values as not-requested.
- Numbered footnotes (1, 2, 3, ...) at the bottom are legal boilerplate. Ignore
  them entirely.

Extraction rules:
- Numbers: strip "$", "%", and thousands separators and return a number. A dash
  "-" or "N/A" means there is no value: return null. Never coerce "-" to 0.
- report_outcome: "Declared", "Denied", or "Denial of Appeal", from the outcome
  line. decision_date: the date on that line, as YYYY-MM-DD.
- disaster_number: the FEMA disaster number if present (e.g. "FEMA-4807-DR" ->
  4807); null for denials/appeals that have no number.
- ia_requested / pa_requested: false if that block is marked "(Not Requested)",
  otherwise true. When a block is not requested, set its numeric fields to null.
- Percentages with two parts (e.g. "73.2% Flood 10.9%") map to the two distinct
  fields (ia_pct_insured and ia_pct_flood_insured). "8.1% SSI / 16.3% SNAP" map
  to ia_pct_ssi and ia_pct_snap.
- Counties: return one entry per county. Set county_ia true if the county is
  designated/requested for Individual Assistance, county_pa true if for Public
  Assistance (a county can be both). The PA "Countywide per capita impact" list
  implies county_pa true.
  * Primary source is the PA "Countywide per capita impact" list: each
    "County Name ($amount)" becomes {county_name, per_capita_impact,
    source: "per_capita", county_pa: true}.
  * Also include counties named in the narrative request sentence(s) with
    source "narrative" and county_ia / county_pa set from how they're listed.
    If the narrative gives only a count ("18 counties") with no names, do not
    invent names.
  * If a county appears in both the per-capita list and the narrative, set
    source to "both" and keep the per-capita amount.
  * If no county is named anywhere, return exactly one entry with
    county_name null and source "none".
- Set needs_review to true if the document departs from this structure or any
  required field is ambiguous, and put a short note in review_note.
```

**User message** (a `document` block carrying the PDF, then the instruction):

```python
{
  "role": "user",
  "content": [
    {
      "type": "document",
      "source": {"type": "base64", "media_type": "application/pdf", "data": "<base64 PDF>"},
    },
    {"type": "text", "text": "Extract this FEMA PDA report by calling record_pda_report."},
  ],
}
```

**Tool `input_schema` (the output contract):**

```json
{
  "name": "record_pda_report",
  "description": "Record the structured data extracted from one FEMA PDA report.",
  "input_schema": {
    "type": "object",
    "properties": {
      "report_outcome": {"type": ["string", "null"], "enum": ["Declared", "Denied", "Denial of Appeal", null]},
      "decision_date": {"type": ["string", "null"], "description": "YYYY-MM-DD of the Declared/Denied line"},
      "state": {"type": ["string", "null"]},
      "incident_name": {"type": ["string", "null"]},
      "disaster_number": {"type": ["integer", "null"]},
      "request_date": {"type": ["string", "null"], "description": "YYYY-MM-DD the governor requested"},
      "incident_begin": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
      "incident_end": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
      "denial_reason": {"type": ["string", "null"], "description": "the stated basis for denial, if any"},

      "ia_requested": {"type": "boolean"},
      "ia_residences_total": {"type": ["number", "null"]},
      "ia_destroyed": {"type": ["number", "null"]},
      "ia_major": {"type": ["number", "null"]},
      "ia_minor": {"type": ["number", "null"]},
      "ia_affected": {"type": ["number", "null"]},
      "ia_pct_insured": {"type": ["number", "null"]},
      "ia_pct_flood_insured": {"type": ["number", "null"]},
      "ia_pct_poverty": {"type": ["number", "null"]},
      "ia_pct_ssi": {"type": ["number", "null"]},
      "ia_pct_snap": {"type": ["number", "null"]},
      "ia_pct_ownership": {"type": ["number", "null"]},
      "ia_unemployment": {"type": ["number", "null"]},
      "ia_pct_age_65_plus": {"type": ["number", "null"]},
      "ia_pct_age_18_under": {"type": ["number", "null"]},
      "ia_pct_disability": {"type": ["number", "null"]},
      "ia_icc_ratio": {"type": ["number", "null"]},
      "ia_cost_estimate": {"type": ["number", "null"]},

      "pa_requested": {"type": "boolean"},
      "pa_primary_impact": {"type": ["string", "null"]},
      "pa_cost_estimate": {"type": ["number", "null"]},
      "pa_statewide_per_capita": {"type": ["number", "null"]},
      "pa_statewide_per_capita_indicator": {"type": ["number", "null"]},
      "pa_countywide_per_capita_indicator": {"type": ["number", "null"]},
      "hm_requested": {"type": "boolean"},

      "counties": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "county_name": {"type": ["string", "null"]},
            "source": {"type": "string", "enum": ["per_capita", "narrative", "both", "none"]},
            "per_capita_impact": {"type": ["number", "null"]},
            "county_ia": {"type": "boolean", "description": "designated/requested for Individual Assistance"},
            "county_pa": {"type": "boolean", "description": "designated/requested for Public Assistance"}
          },
          "required": ["county_name", "source", "per_capita_impact", "county_ia", "county_pa"]
        }
      },

      "needs_review": {"type": "boolean"},
      "review_note": {"type": ["string", "null"]}
    },
    "required": ["report_outcome", "decision_date", "state", "ia_requested", "pa_requested", "counties", "needs_review"]
  }
}
```

### Few-shot examples (in the cached system prompt)

Two contrasting real reports are appended to the system prompt as worked
`document → record_pda_report(...)` examples, so the model sees both extremes.
The boilerplate footnotes are omitted here for brevity.

**Example 1 — a sparse denial** (`Idaho – Gwen Fire`, IA populated, PA *not
requested*; note the per-capita *indicators* are still present, and the only
county comes from the narrative):

```json
{
  "report_outcome": "Denied", "decision_date": "2024-11-20",
  "state": "Idaho", "incident_name": "Gwen Fire", "disaster_number": null,
  "request_date": "2024-10-07", "incident_begin": "2024-07-24", "incident_end": "2024-08-09",
  "denial_reason": "The event was not of such severity and magnitude as to be beyond the capabilities of the state, affected local governments, and voluntary agencies.",
  "ia_requested": true, "ia_residences_total": 70, "ia_destroyed": 42, "ia_major": 0,
  "ia_minor": 1, "ia_affected": 27, "ia_pct_insured": 66.5, "ia_pct_flood_insured": null,
  "ia_pct_poverty": 14.1, "ia_pct_ssi": 7.8, "ia_pct_snap": 9.2, "ia_pct_ownership": 99.0,
  "ia_unemployment": 3.2, "ia_pct_age_65_plus": 20.3, "ia_pct_age_18_under": 21.4,
  "ia_pct_disability": 17.7, "ia_icc_ratio": 8.57, "ia_cost_estimate": 1066144,
  "pa_requested": false, "pa_primary_impact": null, "pa_cost_estimate": null,
  "pa_statewide_per_capita": null, "pa_statewide_per_capita_indicator": 1.84,
  "pa_countywide_per_capita_indicator": 4.60, "hm_requested": true,
  "counties": [
    {"county_name": "Nez Perce", "source": "narrative", "per_capita_impact": null, "county_ia": true, "county_pa": false}
  ],
  "needs_review": false, "review_note": null
}
```

**Example 2 — a rich approved declaration** (`South Dakota; FEMA-4807-DR`,
`Declared`, the two-part `73.2% Flood 10.9%` insured split, and the full
"Countywide per capita impact" list → one county row each; the four IA counties
also appear in the PA list, so they are `source: "both"` with both flags true).
The county array enumerates **all** listed counties; a representative slice:

```json
{
  "report_outcome": "Declared", "decision_date": "2024-08-15",
  "state": "South Dakota", "incident_name": "Severe Storms, Straight-line Winds, and Flooding",
  "disaster_number": 4807, "ia_requested": true, "ia_residences_total": 185,
  "ia_destroyed": 45, "ia_major": 71, "ia_minor": 49, "ia_affected": 20,
  "ia_pct_insured": 73.2, "ia_pct_flood_insured": 10.9, "ia_pct_poverty": 10.6,
  "ia_pct_ssi": 8.1, "ia_pct_snap": 16.3, "ia_pct_ownership": 65.1, "ia_unemployment": 3.9,
  "ia_pct_age_65_plus": 19.3, "ia_pct_age_18_under": 22.8, "ia_pct_disability": 21.9,
  "ia_icc_ratio": 32.56, "ia_cost_estimate": 2419564,
  "pa_requested": true, "pa_primary_impact": "Damage to roads and bridges",
  "pa_cost_estimate": 19122256, "pa_statewide_per_capita": 21.57,
  "pa_statewide_per_capita_indicator": 1.84, "pa_countywide_per_capita_indicator": 4.60,
  "hm_requested": true,
  "counties": [
    {"county_name": "Aurora", "source": "per_capita", "per_capita_impact": 701.83, "county_ia": false, "county_pa": true},
    {"county_name": "Davison", "source": "both", "per_capita_impact": 106.90, "county_ia": true, "county_pa": true},
    {"county_name": "Lincoln", "source": "both", "per_capita_impact": 6.74, "county_ia": true, "county_pa": true}
  ],
  "needs_review": false, "review_note": null
}
```

### Output: `data/denial_counties.csv`

One row per county per report. The script adds provenance columns the model
does not see (`source_pdf`, `report_type`, `state_abbr` from the filename), then
flattens each county entry alongside the report-level IA/PA fields (which the
PDFs report at the state level and which repeat across a report's county rows).

Columns: `source_pdf`, `report_type`, `report_outcome`, `disaster_number`,
`state`, `state_abbr`, `incident_name`, `decision_date`, `request_date`,
`incident_begin`, `incident_end`, `denial_reason`, `county_name`,
`county_source`, `county_per_capita_impact`, `county_ia`, `county_pa`, `ia_requested`,
`ia_residences_total`, `ia_destroyed`, `ia_major`, `ia_minor`, `ia_affected`,
`ia_pct_insured`, `ia_pct_flood_insured`, `ia_pct_poverty`, `ia_pct_ssi`,
`ia_pct_snap`, `ia_pct_ownership`, `ia_unemployment`, `ia_pct_age_65_plus`,
`ia_pct_age_18_under`, `ia_pct_disability`, `ia_icc_ratio`, `ia_cost_estimate`,
`pa_requested`, `pa_primary_impact`, `pa_cost_estimate`,
`pa_statewide_per_capita`, `pa_statewide_per_capita_indicator`,
`pa_countywide_per_capita_indicator`, `hm_requested`, `needs_review`,
`review_note`.

## 3. Join to OpenFEMA (optional)

- **Denials / appeal denials** → [`DeclarationDenials`](https://www.fema.gov/api/open/v1/DeclarationDenials)
  on `state_abbr` + `decision_date` (matches `stateAbbreviation` +
  `requestStatusDate`).
- **Approved declarations** → [`DisasterDeclarationsSummaries`](https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries)
  on `disaster_number`.

## Data sources

- FEMA PDA reports index: <https://www.fema.gov/disaster/how-declared/preliminary-damage-assessments/reports>
- OpenFEMA API (no key required): <https://www.fema.gov/about/openfema/api>
