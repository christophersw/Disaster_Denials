# pda/modeling/jurisdiction.py
"""
Title: pda/modeling/jurisdiction.py — Jurisdiction class and alignment flags.
Description:
    Classifies each request as state / territory / federal_district / tribal and
    marks where presidential and gubernatorial partisan alignment are
    structurally defined (spec §5.7). Tribes are sovereign (no state alignment,
    no governor); territories have a governor but cast no electoral votes; DC
    votes for president but has a mayor, not a governor. These flags let the
    models treat 'not applicable' distinctly from 'missing'.
Changelog:
    2026-06-28  Initial version.
"""

TERRITORY_ABBRS = {"PR", "GU", "VI", "AS", "MP", "FM", "MH", "PW"}
DISTRICT_ABBRS = {"DC"}

# Requester titles that indicate a sovereign tribal nation rather than a state.
_TRIBAL_MARKERS = (
    "tribal", "chief", "chairman", "chairwoman", "chairperson", "chair",
    "council", "village", "principal", "ira",
)


def _classify(state_abbr, requestor_type):
    """Return the jurisdiction_type for one row.

    Args:
        state_abbr: two-letter postal/territory code (may be None/blank).
        requestor_type: parsed requester title (may be None/blank).
    Returns:
        One of 'tribal', 'territory', 'federal_district', 'state'.
    """
    # Guard against pandas NaN (float) — coerce non-string values to "".
    title = (requestor_type if isinstance(requestor_type, str) else "").lower()
    if any(marker in title for marker in _TRIBAL_MARKERS):
        return "tribal"
    code = (state_abbr if isinstance(state_abbr, str) else "").upper()
    if code in DISTRICT_ABBRS:
        return "federal_district"
    if code in TERRITORY_ABBRS:
        return "territory"
    return "state"


_PRESIDENTIAL_APPLICABLE = {"state", "federal_district"}
_GUBERNATORIAL_APPLICABLE = {"state", "territory"}


def add_jurisdiction_features(df):
    """Add jurisdiction_type and the two alignment-applicability flags.

    Args:
        df: frame with 'state_abbr' and 'requestor_type' columns.
    Returns:
        A copy of df with 'jurisdiction_type',
        'presidential_alignment_applicable', and
        'gubernatorial_alignment_applicable' added.
    """
    out = df.copy()
    out["jurisdiction_type"] = [
        _classify(s, r)
        for s, r in zip(out["state_abbr"], out["requestor_type"])
    ]
    out["presidential_alignment_applicable"] = (
        out["jurisdiction_type"].isin(_PRESIDENTIAL_APPLICABLE).astype(int)
    )
    out["gubernatorial_alignment_applicable"] = (
        out["jurisdiction_type"].isin(_GUBERNATORIAL_APPLICABLE).astype(int)
    )
    return out
