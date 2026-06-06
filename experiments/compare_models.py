"""
Title: Opus-vs-Haiku Extraction Comparison Harness
Description: Runs a fixed sample of FEMA PDA report PDFs through two Claude
    models — Opus 4.8 (the production config: adaptive thinking + effort:high)
    and Haiku 4.5 (a cheaper config: no thinking, no effort, which Haiku does
    not accept) — using the IDENTICAL system prompt, tool, and schema from
    `pda/extract.py`. For each PDF it validates both responses into PdaReport
    objects, diffs every report-level and county-level field, and records token
    usage and dollar cost per model. It then writes a findings report to
    `docs/model_comparison_opus_vs_haiku.md`.

    Opus is used as the comparison *baseline* (it is the trusted production
    config), not as verified ground truth: a field where the two models differ
    is a candidate error for whichever model is wrong, to be adjudicated against
    the source PDF. The sample is deliberately chosen to exercise the subtle
    fields most likely to trip a smaller model (legacy low-income/elderly
    percentages, multi-county per-capita lists, the appeal date chain, tribal
    geo_type, expedited nulls, and the denial granted-false rule).

    Usage:
        .venv/bin/python experiments/compare_models.py

    Cost: ~20 API calls (10 per model); roughly $1 total.

Changelog:
    2026-06-06  Initial version.
"""

import datetime
import os
import sys

# Make the repo root importable so `pda` resolves no matter the working dir
# (running a script puts the script's own dir on sys.path, not the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from dotenv import load_dotenv

from pda.extract import MODEL as OPUS_MODEL
from pda.extract import TOOL_NAME, build_request
from pda.schema import PdaReport

HAIKU_MODEL = "claude-haiku-4-5"

# Per-million-token pricing ($). Cache write = 1.25x input, cache read = 0.10x.
PRICING = {
    OPUS_MODEL: {"in": 5.0, "out": 25.0},
    HAIKU_MODEL: {"in": 1.0, "out": 5.0},
}

# Sample chosen to stress the subtle fields (see module docstring).
SAMPLE_PDFS = [
    "data/pdfs/MajorDisaster/2026/PDAReport_FEMA4909DR-HI.pdf",       # modern, multi-county, IA+PA
    "data/pdfs/MajorDisaster/2017/FEMA4304DRKS.pdf",                  # 23-county per-capita list (stress)
    "data/pdfs/MajorDisaster/2013/PDA_Report_FEMA-4127-DR-MT.pdf",    # 15-county per-capita list
    "data/pdfs/MajorDisaster/2010/pda_report_FEMA-1926-DR-OK.pdf",    # older approval
    "data/pdfs/MajorDisaster/2007/1730_0.pdf",                        # legacy IA fields (low-income/elderly)
    "data/pdfs/Denials/2024/PDAReport_Denial-ID.pdf",                 # county + tribal reservation, PA not requested
    "data/pdfs/Denials/2007/101608_tx_denial.pdf",                    # legacy denial, 4-county per-capita list
    "data/pdfs/AppealDenials/2026/FY26PDAReport_AppealDenial-CO.pdf", # appeal chain (original_denial_date/appeal_date)
    "data/pdfs/Expedited/2014/PDA_Report_FEMA-4174-DR-AR_Expedited.pdf",  # expedited: mostly N/A -> null
    "data/pdfs/Other/2024/PDAReport_FEMA4773DR-HoopaVAlleyTribe.pdf", # tribal requestor + geo_type
]

# Report-level fields singled out in the report because they are the ones a
# weaker model is most likely to get wrong (subtle reading / mapping rules).
HARD_REPORT_FIELDS = {
    "disaster_number", "declaration_type", "original_denial_date", "appeal_date",
    "ia_pct_low_income", "ia_pct_elderly", "ia_pct_flood_insured",
    "pa_categories_requested", "ia_requested", "pa_requested", "hm_requested",
}

REPORT_OUTPUT = "docs/model_comparison_opus_vs_haiku.md"


def haiku_request(pdf_bytes: bytes) -> dict:
    """Build a Haiku request from the Opus request, dropping Opus-only params.

    Haiku 4.5 rejects `output_config.effort` and does not take adaptive
    thinking, so we reuse the identical system prompt / tool / document block
    from `build_request` and strip `thinking` and `output_config`.

    Args:
        pdf_bytes: Raw PDF bytes.
    Returns:
        Messages API kwargs targeting Haiku.
    """
    request = dict(build_request(pdf_bytes))
    request["model"] = HAIKU_MODEL
    request.pop("thinking", None)
    request.pop("output_config", None)
    return request


def run_one(client: anthropic.Anthropic, request: dict) -> tuple:
    """Send one request and return (PdaReport or None, usage dict, error str).

    Args:
        client: Anthropic client.
        request: Messages API kwargs.
    Returns:
        (report, usage, error) — report is None if the tool was not called or
        validation failed, in which case error explains why.
    """
    response = client.messages.create(**request)
    usage = response.usage
    usage_dict = {
        "in": getattr(usage, "input_tokens", 0) or 0,
        "cache_w": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_r": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "out": getattr(usage, "output_tokens", 0) or 0,
    }
    tool_use = next(
        (b for b in response.content
         if b.type == "tool_use" and b.name == TOOL_NAME), None)
    if tool_use is None:
        return None, usage_dict, f"no tool_use (stop={response.stop_reason})"
    try:
        return PdaReport.model_validate(tool_use.input), usage_dict, ""
    except Exception as error:  # noqa: BLE001 — record validation failure
        return None, usage_dict, f"validation error: {error}"


def request_cost(model: str, usage: dict) -> float:
    """Dollar cost of one request from its usage and the model's pricing."""
    price = PRICING[model]
    return (
        usage["in"] * price["in"]
        + usage["cache_w"] * price["in"] * 1.25
        + usage["cache_r"] * price["in"] * 0.10
        + usage["out"] * price["out"]
    ) / 1e6


def diff_reports(opus: PdaReport, haiku: PdaReport) -> tuple:
    """Diff two reports field by field.

    Args:
        opus: The baseline report.
        haiku: The candidate report.
    Returns:
        (report_field_diffs, county_diffs) where:
        - report_field_diffs: list of (field, opus_value, haiku_value)
        - county_diffs: dict with keys 'only_opus', 'only_haiku' (county-name
          lists) and 'field' (list of (county, field, opus_val, haiku_val))
    """
    opus_data = opus.model_dump()
    haiku_data = haiku.model_dump()

    report_field_diffs = []
    for field in opus_data:
        if field == "counties":
            continue
        if opus_data[field] != haiku_data.get(field):
            report_field_diffs.append((field, opus_data[field], haiku_data.get(field)))

    def by_name(report):
        return {(c.county_name or "<none>"): c for c in report.counties}

    opus_counties = by_name(opus)
    haiku_counties = by_name(haiku)
    only_opus = sorted(set(opus_counties) - set(haiku_counties))
    only_haiku = sorted(set(haiku_counties) - set(opus_counties))

    county_field_diffs = []
    for name in sorted(set(opus_counties) & set(haiku_counties)):
        oc = opus_counties[name].model_dump()
        hc = haiku_counties[name].model_dump()
        for field in oc:
            if oc[field] != hc.get(field):
                county_field_diffs.append((name, field, oc[field], hc.get(field)))

    return report_field_diffs, {
        "only_opus": only_opus,
        "only_haiku": only_haiku,
        "field": county_field_diffs,
    }


def main() -> int:
    """Run the comparison and write the findings report."""
    load_dotenv()
    client = anthropic.Anthropic()

    rows = []                       # per-PDF result records
    per_report_field_disagree = {}  # report field -> count of PDFs where it differs
    total_county_field_diffs = 0
    total_missing_counties = 0
    total_extra_counties = 0
    opus_cost = haiku_cost = 0.0
    opus_out = haiku_out = 0

    for path in SAMPLE_PDFS:
        with open(path, "rb") as handle:
            pdf_bytes = handle.read()

        opus_report, opus_usage, opus_err = run_one(client, build_request(pdf_bytes))
        haiku_report, haiku_usage, haiku_err = run_one(client, haiku_request(pdf_bytes))

        opus_c = request_cost(OPUS_MODEL, opus_usage)
        haiku_c = request_cost(HAIKU_MODEL, haiku_usage)
        opus_cost += opus_c
        haiku_cost += haiku_c
        opus_out += opus_usage["out"]
        haiku_out += haiku_usage["out"]

        record = {
            "path": path, "opus_err": opus_err, "haiku_err": haiku_err,
            "opus_cost": opus_c, "haiku_cost": haiku_c,
            "opus_counties": len(opus_report.counties) if opus_report else None,
            "haiku_counties": len(haiku_report.counties) if haiku_report else None,
            "report_diffs": [], "county_diffs": None,
        }
        if opus_report and haiku_report:
            report_diffs, county_diffs = diff_reports(opus_report, haiku_report)
            record["report_diffs"] = report_diffs
            record["county_diffs"] = county_diffs
            for field, _, _ in report_diffs:
                per_report_field_disagree[field] = per_report_field_disagree.get(field, 0) + 1
            total_county_field_diffs += len(county_diffs["field"])
            total_missing_counties += len(county_diffs["only_opus"])
            total_extra_counties += len(county_diffs["only_haiku"])
        rows.append(record)
        print(f"done: {path}  opus_diffs="
              f"{len(record['report_diffs'])}  "
              f"county_field_diffs="
              f"{len(record['county_diffs']['field']) if record['county_diffs'] else 'n/a'}")

    _write_report(rows, per_report_field_disagree, total_county_field_diffs,
                  total_missing_counties, total_extra_counties,
                  opus_cost, haiku_cost, opus_out, haiku_out)
    print(f"\nReport written to {REPORT_OUTPUT}")
    return 0


def _write_report(rows, per_report_field_disagree, total_county_field_diffs,
                  total_missing, total_extra, opus_cost, haiku_cost,
                  opus_out, haiku_out) -> None:
    """Render the findings markdown report."""
    n = len(rows)
    both_ok = [r for r in rows if not r["opus_err"] and not r["haiku_err"]]
    total_report_diffs = sum(len(r["report_diffs"]) for r in rows)
    perfect = sum(1 for r in both_ok
                  if not r["report_diffs"] and r["county_diffs"]
                  and not r["county_diffs"]["field"]
                  and not r["county_diffs"]["only_opus"]
                  and not r["county_diffs"]["only_haiku"])

    lines = []
    add = lines.append
    add("# Model Comparison — Opus 4.8 vs Haiku 4.5 for PDA Extraction\n")
    add(f"_Generated by `experiments/compare_models.py` on "
        f"{datetime.date.today().isoformat()}. Sample of {n} PDFs._\n")

    add("## Methodology\n")
    add("Each PDF was extracted twice with the **identical** system prompt, "
        "`record_pda_report` tool, and JSON schema from `pda/extract.py`. The "
        "only differences between the two runs are the model and the params "
        "Haiku does not accept:\n")
    add(f"- **Opus** (`{OPUS_MODEL}`): adaptive thinking + `effort: high` "
        "(the production config).")
    add(f"- **Haiku** (`{HAIKU_MODEL}`): no thinking, no `effort` "
        "(Haiku 4.5 rejects both).\n")
    add("Opus is the **baseline** for the diff — not verified ground truth. A "
        "disagreement flags a field for human adjudication against the source "
        "PDF; it counts against whichever model is actually wrong.\n")
    add("The sample deliberately exercises the subtle fields most likely to "
        "trip a smaller model: legacy low-income/elderly percentages "
        "(pre-~2016 reports), multi-county per-capita dollar lists, the appeal "
        "date chain, tribal `geo_type`, expedited `N/A`→null, and the "
        "denial `granted_*`=false rule.\n")

    add("## Cost\n")
    add("| Model | Total (10 PDFs) | Avg / report | Avg output tok | Full corpus (1,378) |")
    add("|---|---|---|---|---|")
    add(f"| Opus 4.8 | ${opus_cost:.2f} | ${opus_cost/n:.4f} | "
        f"{opus_out/n:.0f} | ${opus_cost/n*1378:.0f} |")
    add(f"| Haiku 4.5 | ${haiku_cost:.2f} | ${haiku_cost/n:.4f} | "
        f"{haiku_out/n:.0f} | ${haiku_cost/n*1378:.0f} |")
    savings = (opus_cost - haiku_cost) / n * 1378
    add(f"\nHaiku would save **~${savings:.0f}** across the full corpus "
        f"(~{(1 - haiku_cost/opus_cost)*100:.0f}% cheaper).\n")

    add("## Agreement summary\n")
    add(f"- PDFs where both models returned a valid extraction: "
        f"**{len(both_ok)}/{n}**")
    add(f"- PDFs with **zero** disagreements (report fields AND counties): "
        f"**{perfect}/{len(both_ok)}**")
    add(f"- Total report-level field disagreements: **{total_report_diffs}**")
    add(f"- Total county-level field disagreements: **{total_county_field_diffs}**")
    add(f"- Counties Opus found but Haiku missed: **{total_missing}**")
    add(f"- Counties Haiku found but Opus did not: **{total_extra}**\n")

    if per_report_field_disagree:
        add("### Report fields by disagreement frequency\n")
        add("| Field | # PDFs differing | Hard field? |")
        add("|---|---|---|")
        for field, count in sorted(per_report_field_disagree.items(),
                                   key=lambda kv: -kv[1]):
            hard = "yes" if field in HARD_REPORT_FIELDS else ""
            add(f"| `{field}` | {count}/{n} | {hard} |")
        add("")

    add("## Per-PDF detail\n")
    for r in rows:
        short = r["path"].split("/", 2)[2]
        add(f"### `{short}`")
        add(f"- Cost: Opus ${r['opus_cost']:.4f} / Haiku ${r['haiku_cost']:.4f}; "
            f"counties Opus={r['opus_counties']} Haiku={r['haiku_counties']}")
        if r["opus_err"]:
            add(f"- ⚠️ Opus failed: {r['opus_err']}")
        if r["haiku_err"]:
            add(f"- ⚠️ Haiku failed: {r['haiku_err']}")
        if not r["report_diffs"] and r["county_diffs"] \
                and not r["county_diffs"]["field"] \
                and not r["county_diffs"]["only_opus"] \
                and not r["county_diffs"]["only_haiku"]:
            add("- ✅ Identical extraction.")
        if r["report_diffs"]:
            add("- Report-field differences (field — Opus | Haiku):")
            for field, ov, hv in r["report_diffs"]:
                mark = " **(hard)**" if field in HARD_REPORT_FIELDS else ""
                add(f"  - `{field}`{mark}: `{ov!r}` | `{hv!r}`")
        if r["county_diffs"]:
            cd = r["county_diffs"]
            if cd["only_opus"]:
                add(f"  - Counties only Opus found: {cd['only_opus']}")
            if cd["only_haiku"]:
                add(f"  - Counties only Haiku found: {cd['only_haiku']}")
            for name, field, ov, hv in cd["field"]:
                add(f"  - county `{name}` `{field}`: `{ov!r}` | `{hv!r}`")
        add("")

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
