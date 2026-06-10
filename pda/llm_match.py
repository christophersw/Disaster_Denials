# pda/llm_match.py
"""
Title: LLM-Assisted County → FIPS Matching
Description: Resolves the residual unmatched PDA counties (after exact + fuzzy
    matching) by asking Claude Opus 4.8 to pick the correct county/independent
    city from a state-scoped candidate list drawn from the MIT data. The model
    is constrained to choose a FIPS *from the offered candidates* or return
    "NONE"; every returned code is validated against the candidate set before it
    is written, so a hallucinated FIPS can never be applied. Matches are tagged
    'llm_match' with the model's confidence and rationale for review.
Changelog:
    2026-06-09  Initial version.
"""

from pda.fips import is_county_fips, pad_fips

# Match the extraction pipeline's conventions (pda/extract.py): Opus 4.8, tool
# use, adaptive thinking, high effort.
MODEL = "claude-opus-4-8"
TOOL_NAME = "record_county_matches"

SYSTEM_PROMPT = (
    "You are a careful U.S. geographic record-linkage assistant. You match a "
    "county or place name extracted from a FEMA disaster report to the correct "
    "county-equivalent in one specific state, choosing only from a provided "
    "candidate list.\n"
    "Rules:\n"
    "- Choose the FIPS code of the single best-matching candidate, or \"NONE\" "
    "if none of the candidates is the same place.\n"
    "- NEVER invent a FIPS code. Only return a code that appears in the "
    "candidate list, exactly as written.\n"
    "- Independent cities are distinct from like-named counties (e.g. Baltimore "
    "city vs Baltimore County); match the one the name indicates, preferring the "
    "county when the name says \"County\".\n"
    "- Account for renames (e.g. Oglala Lakota was formerly Shannon) and obvious "
    "misspellings.\n"
    "- If the name cannot belong to this state (wrong-state data error) or has no "
    "real counterpart, return \"NONE\".\n"
    "- Give a confidence from 0 to 1 and a one-line reason for every name."
)

# Strict tool schema — the model must return one decision per requested name.
MATCH_TOOL = {
    "name": TOOL_NAME,
    "description": (
        "Record the FIPS match decision for each PDA county name provided."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pda_name": {"type": "string"},
                        "fips": {
                            "type": "string",
                            "description": "A 5-digit FIPS from the candidates, or 'NONE'.",
                        },
                        "confidence": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["pda_name", "fips", "confidence", "reasoning"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["matches"],
        "additionalProperties": False,
    },
}


def build_candidates(conn, state_abbr: str) -> list:
    """List the real county-equivalents for one state as (name, FIPS) pairs.

    Args:
        conn: open sqlite3 connection with county_presidential_returns.
        state_abbr: two-letter state (matches MIT state_po).
    Returns:
        Sorted, de-duplicated list of (county_name, padded_fips) for candidates
        whose FIPS is a real 5-digit county code.
    """
    rows = conn.execute(
        "SELECT DISTINCT county_name, county_fips FROM county_presidential_returns "
        "WHERE state_po = ?", (state_abbr,)).fetchall()
    candidates = {
        (name, pad_fips(fips)) for name, fips in rows if is_county_fips(fips)
    }
    return sorted(candidates)


def build_user_prompt(state_abbr: str, names: list, candidates: list) -> str:
    """Assemble the user message: the names to resolve and the candidate list.

    Args:
        state_abbr: two-letter state.
        names: PDA county names to resolve.
        candidates: (name, FIPS) pairs offered as the only valid answers.
    Returns:
        The prompt string.
    """
    candidate_lines = "\n".join(f"  {fips}  {name}" for name, fips in candidates)
    name_lines = "\n".join(f"  - {name}" for name in names)
    return (
        f"State: {state_abbr}\n\n"
        f"Resolve each of these extracted place names to one candidate FIPS "
        f"below (or \"NONE\"):\n{name_lines}\n\n"
        f"Candidates (FIPS  NAME) — choose only from these:\n{candidate_lines}\n\n"
        f"Call {TOOL_NAME} with one entry per name, in the same order."
    )


def build_request(state_abbr: str, names: list, candidates: list,
                  model: str = MODEL) -> dict:
    """Build the kwargs for one Messages API call resolving one state's names.

    Args:
        state_abbr: two-letter state.
        names: PDA county names to resolve.
        candidates: (name, FIPS) candidate pairs.
        model: model id (defaults to Opus 4.8).
    Returns:
        A dict of keyword arguments for client.messages.create.
    """
    return {
        "model": model,
        "max_tokens": 16000,
        "thinking": {"type": "adaptive"},
        "system": [{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": [{
            "role": "user",
            "content": build_user_prompt(state_abbr, names, candidates),
        }],
        "tools": [MATCH_TOOL],
        # 'auto' (not a forced tool_choice) so adaptive thinking stays enabled —
        # the API rejects thinking + forced tool use. The prompt instructs the
        # model to call the tool, and matches_from_message tolerates a miss.
        "tool_choice": {"type": "auto"},
        "output_config": {"effort": "high"},
    }


def validate_matches(tool_input: dict, requested_names: list,
                     candidate_fips: set) -> list:
    """Keep only confident, in-candidate decisions for names we actually asked.

    A decision is accepted only when its name was requested and its FIPS is one
    of the offered candidates — this rejects both hallucinated codes and "NONE".

    Args:
        tool_input: the model's tool-call input ({"matches": [...]}).
        requested_names: the names sent for this state.
        candidate_fips: the set of valid padded FIPS for this state.
    Returns:
        list of {name, fips, confidence, reasoning} for accepted matches.
    """
    requested = set(requested_names)
    accepted = []
    for match in tool_input.get("matches", []):
        name = match.get("pda_name")
        fips = (match.get("fips") or "").strip()
        if name in requested and fips in candidate_fips:
            accepted.append({
                "name": name,
                "fips": fips,
                "confidence": match.get("confidence"),
                "reasoning": match.get("reasoning"),
            })
    return accepted


def matches_from_message(message, requested_names: list,
                         candidate_fips: set) -> list:
    """Extract and validate the tool call from a Messages API response.

    Args:
        message: a response object with `.content` content blocks.
        requested_names: the names sent for this state.
        candidate_fips: valid padded FIPS for this state.
    Returns:
        Validated matches (empty list if the model did not call the tool).
    """
    tool_use = next(
        (block for block in message.content
         if block.type == "tool_use" and block.name == TOOL_NAME), None)
    if tool_use is None:
        return []
    return validate_matches(tool_use.input, requested_names, candidate_fips)


def match_state(client, state_abbr: str, names: list, candidates: list,
                model: str = MODEL) -> list:
    """Resolve one state's unmatched names via a single Opus 4.8 tool call.

    Args:
        client: an Anthropic client.
        state_abbr: two-letter state.
        names: PDA county names to resolve.
        candidates: (name, FIPS) candidate pairs.
        model: model id (defaults to Opus 4.8).
    Returns:
        Validated matches (see validate_matches).
    """
    candidate_fips = {fips for _, fips in candidates}
    response = client.messages.create(
        **build_request(state_abbr, names, candidates, model))
    return matches_from_message(response, names, candidate_fips)


def gather_unmatched(conn) -> dict:
    """Collect still-unmatched, named report_counties rows, grouped by state.

    Args:
        conn: open sqlite3 connection.
    Returns:
        dict state_abbr → list of {name, county_ids}. Blank names and rows
        already resolved (any method other than 'unmatched') are excluded.
    """
    rows = conn.execute(
        "SELECT rc.county_id, rc.county_name, r.state_abbr "
        "FROM report_counties rc LEFT JOIN reports r "
        "ON rc.source_pdf = r.source_pdf "
        "WHERE rc.fips_match_method = 'unmatched' "
        "AND rc.county_name IS NOT NULL AND TRIM(rc.county_name) <> ''").fetchall()
    grouped: dict = {}
    for county_id, county_name, state_abbr in rows:
        by_name = grouped.setdefault(state_abbr, {})
        by_name.setdefault(county_name, []).append(county_id)
    return {
        state: [{"name": name, "county_ids": ids} for name, ids in by_name.items()]
        for state, by_name in grouped.items()
    }


def apply_matches(conn, records: list) -> int:
    """Write accepted LLM matches into report_counties.

    Each record applies one FIPS to a set of county_ids, tagging them
    'llm_match' and recording the model's confidence and reasoning.

    Args:
        conn: open sqlite3 connection (audit columns already added).
        records: list of {county_ids, fips, confidence, reasoning}.
    Returns:
        Number of report_counties rows updated.
    """
    updates = []
    for record in records:
        for county_id in record["county_ids"]:
            updates.append((
                record["fips"], record.get("confidence"),
                record.get("reasoning"), county_id))
    with conn:
        conn.executemany(
            "UPDATE report_counties SET county_fips = ?, "
            "fips_match_method = 'llm_match', llm_confidence = ?, "
            "llm_reasoning = ? WHERE county_id = ?", updates)
    return len(updates)


def run(conn, client, model: str = MODEL) -> dict:
    """Resolve all residual unmatched named counties, state by state.

    Args:
        conn: open sqlite3 connection.
        client: an Anthropic client.
        model: model id (defaults to Opus 4.8).
    Returns:
        dict with 'applied' (rows updated) and 'details' (list of applied
        records enriched with state, for the review report).
    """
    grouped = gather_unmatched(conn)
    applied_rows = 0
    details = []
    for state_abbr in sorted(grouped):
        items = grouped[state_abbr]
        names = [item["name"] for item in items]
        candidates = build_candidates(conn, state_abbr)
        if not candidates:
            continue  # state absent from MIT data — nothing to match against
        ids_by_name = {item["name"]: item["county_ids"] for item in items}
        results = match_state(client, state_abbr, names, candidates, model)
        records = [
            {**result, "state": state_abbr, "county_ids": ids_by_name[result["name"]]}
            for result in results if result["name"] in ids_by_name
        ]
        applied_rows += apply_matches(conn, records)
        details.extend(records)
    return {"applied": applied_rows, "details": details}
