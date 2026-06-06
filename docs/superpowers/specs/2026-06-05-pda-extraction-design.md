# PDA Report Extraction — Design Spec

**Date:** 2026-06-05
**Status:** Approved (brainstorming) — ready for implementation planning
**Phase:** 1 of 3 (Extraction). Phases 2 (resolution & joins) and 3 (analysis) are
noted as follow-ons but out of scope for this spec.

## 1. Goal

Turn FEMA Preliminary Damage Assessment (PDA) report PDFs into a clean,
machine-readable, **normalized** dataset using an LLM (Claude via the Anthropic
API). This is the extraction layer that downstream phases build on.

The ultimate research question driving the project is: **do FEMA disaster-denial
outcomes correlate with a county's political lean?** That goal shapes the
*scope* and the *join plan* below, but extraction itself captures only raw facts
from each report — no derived or joined values.

## 2. Scope

- **Input:** all 1,378 downloaded PDA report PDFs under
  `data/pdfs/<ReportType>/<Year>/*.pdf`, across all five report types
  (`MajorDisaster`, `Denials`, `AppealDenials`, `Expedited`, `Other`).
- **Why all types, not just denials:** the political-lean analysis needs
  approvals as a **control group** ("denied vs granted, given similar damage and
  lean"). Denials are the treatment; approvals are the comparison.
- **Output:** two normalized CSV tables — `reports` and `report_counties`
  (defined in §4).
- **Out of scope (later phases):** county→FIPS resolution, joins to the
  presidential-returns and OpenFEMA datasets, and the correlation analysis.

## 3. Source-document facts (verified against the corpus)

- **Length:** nearly all reports are 2 pages; a small number are 3. Single
  incident per report.
- **Text layer:** every report has an extractable text layer — a full-corpus
  scan of all 1,378 PDFs found **0 image-only / low-text files** (every PDF
  yielded > 200 characters of extractable text). OCR is therefore not part of
  this pipeline. (If any image-only reports surface later, they are handled by
  the `needs_review` path, not by adding an OCR stage.)
- **Consistent skeleton across all types and years:**
  1. Header — `Jurisdiction – Incident name`, an outcome line
     (`Declared <date>` / `Denied on <date>` / `Denial of Appeal` then
     `Denied on <date>`), and a disaster number for approvals (`FEMA-4909-DR`).
  2. Narrative paragraph(s) — who requested what (IA / PA categories / Hazard
     Mitigation), for which counties, request date, incident dates; and for
     approvals, what the President made available and where.
  3. Individual Assistance block — residence damage counts + a set of
     percentages + a cost estimate.
  4. Public Assistance block — primary impact, cost estimate, statewide &
     countywide per-capita impacts and indicators, and an inline county
     per-capita list.
  5. Footnotes — identical legal boilerplate everywhere. **Ignored.**

### 3.1 Three realities the schema must absorb

1. **IA fields drifted over ~20 years.** Older reports (e.g. 2007, 2014) list
   only *insured / low-income / elderly* percentages. Modern reports (~2016+)
   replace those with *poverty, SSI/SNAP, ownership, unemployment, age 65+,
   age 18−, disability, IHP Cost-to-Capacity (ICC) ratio*. Insured is sometimes
   split with a second "Flood X%" number. The schema is a **superset** that keeps
   legacy and modern fields as **distinct columns** — we do **not** conflate
   "low income"→"poverty" or "elderly"→"age 65+", because the definitions and
   methodology changed.
2. **The geographic unit isn't always a county.** Tribal nations (requested by a
   Tribal Chairman, not a Governor), reservations, Louisiana parishes, Alaska
   boroughs, and "City and County of …" all appear. A `geo_type` column captures
   this; tribal/non-county units are cleanly excludable from the county-level
   political analysis later.
3. **Expedited reports are mostly empty of damage data** — declared before PDAs
   complete, so IA/PA numbers are largely `N/A`/`-`. Their value is the
   narrative. This is expected, not an extraction error.

## 4. Data model — two normalized tables

Rationale: almost every *number* in a report is **state/report-level** (all IA
damage counts, all IA percentages, both cost estimates, the statewide
per-capita). The **only genuinely per-county datum** is the PA countywide
per-capita dollar figure, plus which counties were requested/granted for IA vs
PA. Storing one flat row per county would duplicate ~40 state-level columns
across every county row. We normalize and can materialize a flat ML view via a
join on demand.

### 4.1 `reports` — one row per PDF

Keyed by `source_pdf`.

**Identity & outcome**

| Column | Type | Notes |
|---|---|---|
| `source_pdf` | string | Relative path / filename. Primary key. |
| `report_type` | enum | MajorDisaster / Denials / AppealDenials / Expedited / Other (from folder). |
| `url` | string | From `manifest.csv`. |
| `posted_date` | date | From `manifest.csv`. |
| `report_outcome` | enum\|null | Declared / Denied / Denial of Appeal (from outcome line). |
| `decision_date` | date\|null | The date on the outcome line (YYYY-MM-DD). |
| `jurisdiction_name` | string\|null | State or tribe name. |
| `state_abbr` | string\|null | Two-letter (from filename/manifest). |
| `requestor_type` | enum\|null | Governor / Tribal Chairman / Acting Chairman / … |
| `requestor_name` | string\|null | Named requestor. |
| `incident_name` | string\|null | e.g. "Gwen Fire". |
| `incident_begin` | date\|null | YYYY-MM-DD. |
| `incident_end` | date\|null | YYYY-MM-DD. |
| `request_date` | date\|null | When the request was made. |
| `disaster_number` | integer\|null | Numeric part only (e.g. 4909); null for denials. |
| `declaration_type` | enum\|null | DR / EM — kept, not discarded. |
| `denial_reason` | string\|null | Stated basis for denial (free text). |
| `original_denial_date` | date\|null | Appeal reports: date of the original denial. |
| `appeal_date` | date\|null | Appeal reports: date the appeal was filed. |

**Requested programs** (from narrative)

| Column | Type | Notes |
|---|---|---|
| `ia_requested` | bool | false if IA block marked "(Not Requested)". |
| `pa_requested` | bool | false if PA block marked "(Not Requested)". |
| `hm_requested` | bool | Hazard Mitigation requested. |
| `pa_categories_requested` | string\|null | e.g. "A,B" or "A-F". |

**Individual Assistance** (state-level)

| Column | Type | Notes |
|---|---|---|
| `ia_residences_total` | number\|null | Total residences impacted. |
| `ia_destroyed` | number\|null | |
| `ia_major` | number\|null | |
| `ia_minor` | number\|null | |
| `ia_affected` | number\|null | |
| `ia_pct_insured` | number\|null | |
| `ia_pct_flood_insured` | number\|null | The second "Flood X%" value when present. |
| `ia_pct_poverty` | number\|null | Modern. |
| `ia_pct_ssi` | number\|null | Modern. |
| `ia_pct_snap` | number\|null | Modern. |
| `ia_pct_ownership` | number\|null | Modern. |
| `ia_unemployment` | number\|null | Modern (pre-disaster). |
| `ia_pct_age_65_plus` | number\|null | Modern. |
| `ia_pct_age_18_under` | number\|null | Modern. |
| `ia_pct_disability` | number\|null | Modern. |
| `ia_icc_ratio` | number\|null | Modern (IHP Cost-to-Capacity). |
| `ia_pct_low_income` | number\|null | **Legacy** (older reports only). |
| `ia_pct_elderly` | number\|null | **Legacy** (older reports only). |
| `ia_cost_estimate` | number\|null | Total IA cost estimate ($). |

**Public Assistance** (state-level)

| Column | Type | Notes |
|---|---|---|
| `pa_primary_impact` | string\|null | Free text (e.g. "Damage to roads and bridges"). |
| `pa_cost_estimate` | number\|null | $ |
| `pa_statewide_per_capita` | number\|null | $ |
| `pa_statewide_per_capita_indicator` | number\|null | $ (FY threshold). |
| `pa_countywide_per_capita_indicator` | number\|null | $ (FY threshold). |

**Provenance**

| Column | Type | Notes |
|---|---|---|
| `needs_review` | bool | Set when the document departs from structure or a field is ambiguous. |
| `review_note` | string\|null | Short reason. |
| `parser_model` | string | Model id used. |
| `extracted_at` | datetime | Extraction timestamp. |

### 4.2 `report_counties` — one row per (report × county)

| Column | Type | Notes |
|---|---|---|
| `source_pdf` | string | Foreign key → `reports`. |
| `county_name` | string\|null | Raw name as printed; null only when no county is named anywhere. |
| `geo_type` | enum | county / parish / borough / tribe / reservation / city-county / municipality. |
| `per_capita_impact` | number\|null | $ from the PA countywide per-capita list. |
| `requested_ia` | bool | County requested for Individual Assistance. |
| `requested_pa` | bool | County requested for Public Assistance. |
| `granted_ia` | bool | IA actually made available here (approvals); false for denials. |
| `granted_pa` | bool | PA actually made available here (approvals); false for denials. |
| `source` | enum | per_capita / narrative / both / none — where this county was found. |

`fips` is intentionally **absent** — it is added by the Phase 2 resolution step,
not guessed by the LLM (see §6).

## 5. Extraction rules (LLM contract)

- **Numbers:** strip `$`, `%`, and thousands separators; return a number. A dash
  `-` or `N/A` means **no value → null**. Never coerce `-` to 0.
- **Null semantics:** a block headed "… – (Not Requested)" sets that program's
  `*_requested` flag false and all its numeric fields null — preserving the
  difference between "not requested" and "genuinely zero".
- **Dates** → `YYYY-MM-DD`.
- **`disaster_number`**: numeric part of `FEMA-4909-DR` → 4909; `declaration_type`
  → DR/EM; null for denials/appeals with no number.
- **Insured split:** "62.4% Flood 6.0%" → `ia_pct_insured` = 62.4,
  `ia_pct_flood_insured` = 6.0. "8.1% SSI / 16.3% SNAP" → `ia_pct_ssi`,
  `ia_pct_snap`.
- **Legacy vs modern:** populate whichever percentage fields the report actually
  prints; leave the others null. Do not map legacy→modern.
- **Counties:**
  - Primary source is the PA "Countywide per capita impact" list: each
    `County Name ($amount)` → a row with `per_capita_impact`, `source` =
    per_capita, `requested_pa`/`granted_pa` set from context.
  - Also include counties named in the narrative (request and grant sentences),
    `source` = narrative, with `requested_*`/`granted_*` set from how they are
    listed. **Requested ≠ granted:** an approval can request more counties than
    it grants; capture both. Denials have all `granted_*` = false.
  - A county in both the per-capita list and narrative → `source` = both, keep
    the per-capita amount.
  - If the narrative gives only a count ("18 counties") with no names, do not
    invent names.
  - If no county is named anywhere, emit exactly one row with `county_name` null
    and `source` = none.
  - `geo_type` from how the unit is described (County / Parish / Borough / Tribe
    / Reservation / City and County / Municipality).
- **`needs_review`:** true when the document departs from this structure or any
  required field is ambiguous; put a short reason in `review_note`.
- **No derived fields:** the model returns only raw facts. Ratios, "cleared the
  bar" flags, election-cycle joins, etc. are computed downstream.

## 6. LLM approach

- **Model:** Claude (Opus tier) with adaptive thinking and high effort, prioritizing
  accuracy over token cost (consistent with the project's stated parsing
  preference). Exact model id pinned during implementation planning.
- **Input:** the **native PDF** sent as a `document` content block (base64), so
  the model reads both the text layer and the two-column visual layout (the IA
  fields and county lists extract more reliably from layout than from flattened
  text).
- **Output contract:** a strict tool schema (`record_pda_report`) the model must
  call — report-level fields plus a `counties` array, validated at the tool-call
  layer so the model retries on mismatch.
- **Few-shot:** 2–3 worked examples spanning the extremes — a sparse denial, a
  rich approved declaration, and a legacy-format report — appended to a cached
  system prefix (system prompt + examples + tool schema). PDF is the per-request
  tail. Prompt caching is a cost side-benefit; accuracy is the priority.
- **FIPS resolution is NOT the LLM's job.** The model emits raw `county_name` +
  `geo_type`; a deterministic Phase 2 step resolves FIPS via a Census
  state+county crosswalk with fuzzy matching, flagging unmatched names for
  review. Letting the model guess FIPS would create silent, wrong joins.

## 7. Output & assembly

- The parser writes `reports.csv` and `report_counties.csv` under `data/`.
- Provenance columns the model does not see (`source_pdf`, `report_type`,
  `state_abbr`, `url`, `posted_date`) are added from the folder path and
  `manifest.csv`.
- A run is **resumable** and **idempotent**: already-extracted reports are
  skipped; `--retry-review` re-runs only `needs_review` rows.

## 8. Follow-on phases (out of scope here)

2. **Resolution & joins** — county→FIPS crosswalk; join `report_counties` to the
   FIPS-keyed presidential returns (`countypres_2000-2024.csv`) keyed by the
   report's `decision_date` year → nearest preceding presidential election; join
   to OpenFEMA (`DeclarationDenials` on state+decision date for denials;
   `DisasterDeclarationsSummaries` on `disaster_number` for approvals).
3. **Analysis** — denial-vs-political-lean correlation, with approvals as control
   and tribal/non-county units excluded via `geo_type`.

## 9. Deliverable note

The current `README.md` documents an earlier **single flat table** design. It
must be updated to this normalized two-table model once the parser is built.
