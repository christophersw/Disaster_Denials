# tests/test_llm_match.py
"""Tests for LLM-assisted county→FIPS matching of the residual unmatched rows.

The live Anthropic call is exercised only through a stub client; everything else
(candidate construction, prompt assembly, hallucination-guarding validation, and
the database write-back) is plain, deterministic logic tested directly.
"""

import sqlite3

from pda.llm_match import (
    MODEL, TOOL_NAME,
    build_candidates, build_user_prompt, build_request,
    validate_matches, matches_from_message, match_state,
    gather_unmatched, apply_matches,
)


def _mit_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE county_presidential_returns "
        "(state_po TEXT, county_name TEXT, county_fips TEXT)")
    conn.executemany(
        "INSERT INTO county_presidential_returns VALUES (?, ?, ?)",
        [
            ("VA", "FAIRFAX", "51059"),
            ("VA", "FAIRFAX CITY", "51600"),
            ("VA", "FAIRFAX", "51059"),    # duplicate row → one candidate
            ("VA", "FEDERAL UNIT", "NA"),  # non-county FIPS → excluded
        ])
    conn.commit()
    return conn


def test_build_candidates_dedupes_pads_and_excludes_non_county():
    candidates = build_candidates(_mit_conn(), "VA")
    assert ("FAIRFAX", "51059") in candidates
    assert ("FAIRFAX CITY", "51600") in candidates
    assert len(candidates) == 2                       # dup collapsed, 'NA' dropped
    assert all(f.isdigit() and len(f) == 5 for _, f in candidates)


def test_build_user_prompt_lists_names_and_candidates():
    prompt = build_user_prompt(
        "VA", ["Fairfax County", "Independent City of Roanoke"],
        [("FAIRFAX", "51059"), ("ROANOKE CITY", "51770")])
    assert "VA" in prompt
    assert "Fairfax County" in prompt and "Independent City of Roanoke" in prompt
    assert "51059" in prompt and "ROANOKE CITY" in prompt


def test_build_request_uses_opus_with_thinking_and_offers_the_tool():
    req = build_request("VA", ["Fairfax County"], [("FAIRFAX", "51059")])
    assert req["model"] == MODEL == "claude-opus-4-8"
    # auto tool_choice keeps adaptive thinking valid (forced tool use + thinking 400s)
    assert req["tool_choice"] == {"type": "auto"}
    assert req["thinking"] == {"type": "adaptive"}
    assert req["tools"][0]["name"] == TOOL_NAME


def test_validate_matches_filters_hallucinations_and_none():
    tool_input = {"matches": [
        {"pda_name": "Fairfax County", "fips": "51059", "confidence": 0.95,
         "reasoning": "Fairfax County, VA"},
        {"pda_name": "Made Up", "fips": "99999", "confidence": 0.9,
         "reasoning": "not a real candidate"},          # fips not offered → drop
        {"pda_name": "Adair County", "fips": "NONE", "confidence": 0.1,
         "reasoning": "no Adair county in VA"},          # explicit no-match → drop
    ]}
    out = validate_matches(
        tool_input,
        requested_names=["Fairfax County", "Made Up", "Adair County"],
        candidate_fips={"51059", "51770"})
    assert out == [{"name": "Fairfax County", "fips": "51059",
                    "confidence": 0.95, "reasoning": "Fairfax County, VA"}]


def test_validate_matches_ignores_unrequested_names():
    tool_input = {"matches": [
        {"pda_name": "Surprise County", "fips": "51059", "confidence": 0.9,
         "reasoning": "not asked about"},
    ]}
    out = validate_matches(tool_input, requested_names=["Fairfax County"],
                           candidate_fips={"51059"})
    assert out == []


class _Block:
    type = "tool_use"
    name = TOOL_NAME

    def __init__(self, payload):
        self.input = payload


class _Message:
    stop_reason = "tool_use"

    def __init__(self, payload):
        self.content = [_Block(payload)]


def test_matches_from_message_returns_empty_when_tool_not_called():
    class _NoTool:
        stop_reason = "end_turn"
        content = []
    assert matches_from_message(_NoTool(), ["Fairfax County"], {"51059"}) == []


def test_match_state_calls_opus_and_returns_validated_matches():
    captured = {}
    payload = {"matches": [
        {"pda_name": "Fairfax County", "fips": "51059", "confidence": 0.92,
         "reasoning": "Fairfax County, VA"}]}

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Message(payload)

    client = type("StubClient", (), {"messages": _Messages()})()
    results = match_state(client, "VA", ["Fairfax County"], [("FAIRFAX", "51059")])
    assert captured["model"] == "claude-opus-4-8"
    assert results == [{"name": "Fairfax County", "fips": "51059",
                        "confidence": 0.92, "reasoning": "Fairfax County, VA"}]


def _report_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE reports (source_pdf TEXT PRIMARY KEY, state_abbr TEXT)")
    conn.execute(
        "CREATE TABLE report_counties (county_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "source_pdf TEXT, county_name TEXT, geo_type TEXT, county_fips TEXT, "
        "fips_match_method TEXT, fuzzy_score REAL, llm_confidence REAL, "
        "llm_reasoning TEXT)")
    conn.execute("INSERT INTO reports VALUES ('a.pdf', 'VA')")
    conn.execute("INSERT INTO reports VALUES ('b.pdf', 'VA')")
    rows = [
        ("a.pdf", "Fairfax County", "county", "unmatched"),  # id 1
        ("b.pdf", "Fairfax County", "county", "unmatched"),  # id 2 (same name)
        ("a.pdf", "Already Matched", "county", "exact_normalized"),  # id 3
        ("a.pdf", "", "county", "unmatched"),                # id 4 (blank name)
    ]
    conn.executemany(
        "INSERT INTO report_counties "
        "(source_pdf, county_name, geo_type, fips_match_method) VALUES (?, ?, ?, ?)",
        rows)
    conn.commit()
    return conn


def test_gather_unmatched_groups_named_rows_by_state():
    grouped = gather_unmatched(_report_conn())
    assert set(grouped) == {"VA"}
    items = grouped["VA"]
    assert len(items) == 1                       # both "Fairfax County" rows, one group
    assert items[0]["name"] == "Fairfax County"
    assert sorted(items[0]["county_ids"]) == [1, 2]


def test_apply_matches_writes_fips_method_and_audit_fields():
    conn = _report_conn()
    n = apply_matches(conn, [
        {"county_ids": [1, 2], "fips": "51059", "confidence": 0.92,
         "reasoning": "Fairfax County, VA"}])
    assert n == 2
    rows = conn.execute(
        "SELECT county_fips, fips_match_method, llm_confidence, llm_reasoning "
        "FROM report_counties WHERE county_id IN (1, 2) ORDER BY county_id").fetchall()
    assert rows[0] == ("51059", "llm_match", 0.92, "Fairfax County, VA")
    assert rows[1] == ("51059", "llm_match", 0.92, "Fairfax County, VA")
    # untouched rows keep their prior state
    assert conn.execute(
        "SELECT fips_match_method FROM report_counties WHERE county_id = 3"
    ).fetchone()[0] == "exact_normalized"
