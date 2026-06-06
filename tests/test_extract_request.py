"""Tests that the request is shaped correctly, without calling the API."""

import base64

import pytest

from pda.extract import (
    MODEL,
    TOOL_NAME,
    build_request,
    extract_report,
    report_from_message,
)


class _Block:
    """Minimal stand-in for a response content block."""

    def __init__(self, block_type: str, name: str | None = None,
                 input: dict | None = None):
        self.type = block_type
        self.name = name
        self.input = input


class _Response:
    """Minimal stand-in for a Messages API response."""

    def __init__(self, content: list, stop_reason: str = "end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class _FakeClient:
    """A client whose messages.create returns a preset response (no network)."""

    def __init__(self, response: _Response):
        self._response = response

        class _Messages:
            def create(_self, **_kwargs):
                return response

        self.messages = _Messages()


def test_build_request_has_pdf_document_and_tool_schema():
    req = build_request(b"%PDF-1.4 fake bytes")
    assert req["model"] == MODEL
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"]["effort"] == "high"
    # Non-strict tool use (not structured outputs) — avoids the 16-union cap.
    assert "format" not in req["output_config"]
    assert req["tools"][0]["name"] == TOOL_NAME
    assert req["tools"][0]["input_schema"]["type"] == "object"

    content = req["messages"][0]["content"]
    doc = next(b for b in content if b["type"] == "document")
    assert doc["source"]["media_type"] == "application/pdf"
    # data is base64 of the input bytes
    assert base64.b64decode(doc["source"]["data"]) == b"%PDF-1.4 fake bytes"


def test_system_prompt_is_cacheable():
    req = build_request(b"x")
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_extract_report_parses_tool_use_input():
    """extract_report should validate the tool_use block's input into a PdaReport."""
    payload = {
        "report_outcome": "Denied", "decision_date": "2024-11-20",
        "jurisdiction_name": "Idaho", "state_abbr": "ID",
        "requestor_type": "Governor", "requestor_name": "Brad Little",
        "incident_name": "Gwen Fire", "incident_begin": "2024-07-24",
        "incident_end": "2024-08-09", "request_date": "2024-10-07",
        "disaster_number": None, "declaration_type": None,
        "denial_reason": "x", "original_denial_date": None, "appeal_date": None,
        "ia_requested": True, "pa_requested": False, "hm_requested": True,
        "pa_categories_requested": None,
        "ia_residences_total": 70, "ia_destroyed": 42, "ia_major": 0,
        "ia_minor": 1, "ia_affected": 27, "ia_pct_insured": 66.5,
        "ia_pct_flood_insured": None, "ia_pct_poverty": 14.1, "ia_pct_ssi": 7.8,
        "ia_pct_snap": 9.2, "ia_pct_ownership": 99.0, "ia_unemployment": 3.2,
        "ia_pct_age_65_plus": 20.3, "ia_pct_age_18_under": 21.4,
        "ia_pct_disability": 17.7, "ia_icc_ratio": 8.57,
        "ia_pct_low_income": None, "ia_pct_elderly": None,
        "ia_cost_estimate": 1066144, "pa_primary_impact": None,
        "pa_cost_estimate": None, "pa_statewide_per_capita": None,
        "pa_statewide_per_capita_indicator": 1.84,
        "pa_countywide_per_capita_indicator": 4.60,
        "counties": [], "needs_review": False, "review_note": None,
    }
    response = _Response(
        content=[_Block("tool_use", name=TOOL_NAME, input=payload)])
    report = extract_report(_FakeClient(response), b"%PDF fake")
    assert report.report_outcome == "Denied"
    assert report.state_abbr == "ID"


def test_extract_report_raises_clear_error_when_tool_not_called():
    """A response with no record_pda_report tool_use should raise a clear ValueError."""
    response = _Response(content=[_Block("text")], stop_reason="end_turn")
    with pytest.raises(ValueError, match="did not call"):
        extract_report(_FakeClient(response), b"%PDF fake")


def test_report_from_message_parses_any_message():
    """report_from_message validates a tool_use block from any Messages response
    (used for both live calls and batch results)."""
    payload = {c: None for c in (
        "report_outcome", "decision_date", "jurisdiction_name", "state_abbr",
        "requestor_type", "requestor_name", "incident_name", "incident_begin",
        "incident_end", "request_date", "disaster_number", "declaration_type",
        "denial_reason", "original_denial_date", "appeal_date",
        "pa_categories_requested", "ia_residences_total", "ia_destroyed",
        "ia_major", "ia_minor", "ia_affected", "ia_pct_insured",
        "ia_pct_flood_insured", "ia_pct_poverty", "ia_pct_ssi", "ia_pct_snap",
        "ia_pct_ownership", "ia_unemployment", "ia_pct_age_65_plus",
        "ia_pct_age_18_under", "ia_pct_disability", "ia_icc_ratio",
        "ia_pct_low_income", "ia_pct_elderly", "ia_cost_estimate",
        "pa_primary_impact", "pa_cost_estimate", "pa_statewide_per_capita",
        "pa_statewide_per_capita_indicator", "pa_countywide_per_capita_indicator",
        "review_note")}
    payload.update({
        "report_outcome": "Declared", "ia_requested": True, "pa_requested": True,
        "hm_requested": False, "counties": [], "needs_review": False,
    })
    message = _Response(content=[_Block("tool_use", name=TOOL_NAME, input=payload)])
    report = report_from_message(message)
    assert report.report_outcome == "Declared"


def test_report_from_message_raises_when_tool_not_called():
    message = _Response(content=[_Block("text")], stop_reason="max_tokens")
    with pytest.raises(ValueError, match="did not call"):
        report_from_message(message)
