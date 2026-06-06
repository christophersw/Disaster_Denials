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

# The model returns its answer by calling this tool. We use non-strict tool use
# (not structured outputs / output_config.format) because the structured-output
# schema compiler caps union-typed parameters at 16, and our schema has ~41
# nullable fields. Tool use does not compile the schema, so the cap doesn't
# apply; we re-validate the tool input with Pydantic instead.
TOOL_NAME = "record_pda_report"

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
    "Extract this FEMA PDA report by calling the record_pda_report tool with "
    "every field filled in (use null where a value is absent)."
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
        "tools": [{
            "name": TOOL_NAME,
            "description": (
                "Record the structured data extracted from one FEMA PDA report."
            ),
            "input_schema": json_schema(),
        }],
        "tool_choice": {"type": "auto"},
        "output_config": {"effort": "high"},
    }


def report_from_message(message) -> PdaReport:
    """Validate the record_pda_report tool call in a Messages response.

    Works on any Messages API response object — a live messages.create result
    or a batch result's `.message` — so the parsing lives in one place.

    Args:
        message: A response with `.content` (content blocks) and `.stop_reason`.
    Returns:
        The validated PdaReport.
    Raises:
        ValueError: if the model did not call the record_pda_report tool.
        pydantic.ValidationError: if the tool input does not match the schema.
    """
    tool_use = next(
        (block for block in message.content
         if block.type == "tool_use" and block.name == TOOL_NAME), None)
    if tool_use is None:
        raise ValueError(
            f"Model did not call {TOOL_NAME} (stop_reason={message.stop_reason})")
    return PdaReport.model_validate(tool_use.input)


def extract_report(client: anthropic.Anthropic, pdf_bytes: bytes) -> PdaReport:
    """Extract one PDF into a validated PdaReport via a live API call.

    Args:
        client: An Anthropic client.
        pdf_bytes: Raw PDF bytes.
    Returns:
        The validated PdaReport.
    Raises:
        pydantic.ValidationError: if the response does not match the schema.
    """
    response = client.messages.create(**build_request(pdf_bytes))
    return report_from_message(response)
