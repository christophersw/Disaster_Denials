# pda/fips.py
"""
Title: PDA County → FIPS Mapping
Description: Normalizes county names, builds a (state, normalized-name) → FIPS
    crosswalk from the imported MIT presidential-returns table, and tags
    report_counties rows with a canonical 5-digit county FIPS code. Tribe and
    reservation rows are skipped by design (no county FIPS exists); names that
    do not match exactly after normalization are left NULL and reported.
Changelog:
    2026-06-07  Initial version.
"""

import re

# Trailing geographic suffix words removed before matching. Order matters:
# multi-word suffixes are tried before single words.
_SUFFIX_PATTERNS = [
    r"CITY AND BOROUGH",
    r"CENSUS AREA",
    r"COUNTY",
    r"PARISH",
    r"BOROUGH",
    r"MUNICIPALITY",
    r"MUNICIPIO",
]
_SUFFIX_RE = re.compile(r"\s+(?:%s)$" % "|".join(_SUFFIX_PATTERNS))
_SAINT_RE = re.compile(r"\bST[E]?\b")  # ST or STE as a whole word (periods already removed)


def normalize_county_name(name: str) -> str:
    """Normalize a county name for cross-source matching.

    Uppercases, removes periods and apostrophes, collapses whitespace,
    standardizes Saint abbreviations (ST./STE. → SAINT), and strips a trailing
    geographic suffix word (County, Parish, Borough, etc.).

    Args:
        name: A county name from either PDA or MIT data.
    Returns:
        The normalized, suffix-free uppercase name. Empty string for falsy input.
    """
    if not name:
        return ""
    text = (name.upper().replace(".", "")
            .replace("'", "").replace("’", "").replace("‘", ""))
    text = re.sub(r"\s+", " ", text).strip()
    text = _SAINT_RE.sub("SAINT", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _SUFFIX_RE.sub("", text).strip()
    return text


def pad_fips(raw_fips) -> str:
    """Zero-pad a county FIPS code to the canonical 5 characters.

    Args:
        raw_fips: FIPS value as published (str or int), e.g. "1001" or 1001.
    Returns:
        The 5-character zero-padded string, e.g. "01001".
    """
    return str(raw_fips).strip().zfill(5)


def is_county_fips(raw) -> bool:
    """Return True if a raw FIPS value denotes a real 5-digit county.

    MIT ships non-county reporting units (CT/ME/RI overseas & write-in precincts
    share the placeholder 'NA'; MO Kansas City uses a 7-digit '2938000'). Those
    are not counties and must be excluded from a county-keyed rollup. A genuine
    county code is all digits and exactly 5 characters once zero-padded.

    Args:
        raw: FIPS value as published (str, int, or None).
    Returns:
        True for a real county FIPS, False for placeholders / non-county units.
    """
    text = str(raw).strip()
    return text.isdigit() and len(pad_fips(text)) == 5


def build_state_index(mit_rows) -> dict:
    """Build a per-state index of normalized county name → set of FIPS codes.

    The index is the search space for matching: scoped by state first, then by
    normalized name. A normalized name that maps to more than one FIPS (e.g. VA
    'RICHMOND' is labelled bare for both the county and the independent city in
    different MIT years) keeps all of them, so the matcher can recognize the
    name as ambiguous and decline to guess.

    Args:
        mit_rows: iterable of (state_po, county_name, county_fips) tuples.
    Returns:
        dict mapping state_po → {normalized_name: {padded_fips, ...}}.
    """
    index: dict = {}
    for state_po, county_name, county_fips in mit_rows:
        names = index.setdefault(state_po, {})
        names.setdefault(normalize_county_name(county_name), set()).add(
            pad_fips(county_fips))
    return index


# States whose MIT presidential data is not county-based and so can never yield
# a real county FIPS: Alaska reports by State House District, not borough.
# (US territories are simply absent from the MIT data and handled the same way.)
_NON_COUNTY_SOURCE_STATES = {"AK"}

_NON_COUNTY_GEO_TYPES = {"tribe", "reservation"}

# Fuzzy-match guardrails: a candidate must score at least this well and beat the
# runner-up by at least this margin, or the row is left unmatched rather than
# guessed.
_FUZZY_MIN_SCORE = 0.90
_FUZZY_MIN_MARGIN = 0.05


def _fuzzy_match(normalized_name: str, names: dict):
    """Find the single best fuzzy candidate for a name within one state.

    Args:
        normalized_name: the normalized PDA county name to match.
        names: {normalized_name: {fips, ...}} for one state.
    Returns:
        (padded_fips, score) if a confident, unambiguous winner clears the score
        and margin thresholds; otherwise (None, None).
    """
    import difflib
    scored = sorted(
        ((difflib.SequenceMatcher(None, normalized_name, candidate).ratio(), candidate)
         for candidate in names),
        reverse=True)
    if not scored:
        return None, None
    best_score, best_name = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    candidate_fips = names[best_name]
    if (best_score >= _FUZZY_MIN_SCORE
            and best_score - runner_up >= _FUZZY_MIN_MARGIN
            and len(candidate_fips) == 1):
        return next(iter(candidate_fips)), best_score
    return None, None


def match_county_name(state_abbr: str, county_name: str, geo_type, state_index: dict):
    """Resolve one PDA county to a FIPS code, state first then name.

    Order of resolution:
      1. Tribe/reservation rows are 'na_non_county' (no county FIPS exists).
      2. A state absent from the MIT data, or one whose data is not county-based
         (Alaska), is 'no_source_county' — unresolvable from this source.
      3. An exact normalized-name hit that maps to a single FIPS is
         'exact_normalized'. A name that maps to several FIPS is ambiguous and
         left 'unmatched'.
      4. Otherwise a confident, unambiguous fuzzy candidate (see _fuzzy_match) is
         'fuzzy_match'; anything else is 'unmatched'.

    Args:
        state_abbr: the report's two-letter state (from reports.state_abbr).
        county_name: the PDA county name to resolve.
        geo_type: the PDA geo_type ('county', 'tribe', 'reservation', ...).
        state_index: output of build_state_index.
    Returns:
        (fips, method, score): fips is None unless matched; score is a float only
        for a fuzzy_match, else None.
    """
    if (geo_type or "").lower() in _NON_COUNTY_GEO_TYPES:
        return None, "na_non_county", None
    if state_abbr in _NON_COUNTY_SOURCE_STATES or state_abbr not in state_index:
        return None, "no_source_county", None

    names = state_index[state_abbr]
    normalized = normalize_county_name(county_name)
    candidate_fips = names.get(normalized)
    if candidate_fips and len(candidate_fips) == 1:
        return next(iter(candidate_fips)), "exact_normalized", None
    if candidate_fips:  # name present but maps to several FIPS → ambiguous
        return None, "unmatched", None

    fips, score = _fuzzy_match(normalized, names)
    if fips:
        return fips, "fuzzy_match", score
    return None, "unmatched", None


def add_fips_columns(conn) -> None:
    """Add the FIPS-mapping columns to report_counties if not already present.

    Adds county_fips, fips_match_method, and fuzzy_score (the difflib ratio,
    populated only for fuzzy_match rows so they can be reviewed by confidence).

    Args:
        conn: open sqlite3 connection.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(report_counties)")}
    if "county_fips" not in existing:
        conn.execute("ALTER TABLE report_counties ADD COLUMN county_fips TEXT")
    if "fips_match_method" not in existing:
        conn.execute("ALTER TABLE report_counties ADD COLUMN fips_match_method TEXT")
    if "fuzzy_score" not in existing:
        conn.execute("ALTER TABLE report_counties ADD COLUMN fuzzy_score REAL")
    conn.commit()


def map_report_counties(conn, state_index: dict) -> dict:
    """Tag every report_counties row with a FIPS code, match method, and score.

    Each row is resolved by match_county_name (state-scoped, exact then fuzzy).
    Confident fuzzy matches are applied directly with method 'fuzzy_match' and
    their difflib score, so they can be reviewed and reversed if wrong.

    Args:
        conn: open sqlite3 connection (columns already added).
        state_index: output of build_state_index.
    Returns:
        dict of counts keyed by method (exact_normalized, fuzzy_match,
        na_non_county, no_source_county, unmatched).
    """
    rows = conn.execute(
        "SELECT rc.county_id, rc.county_name, rc.geo_type, r.state_abbr "
        "FROM report_counties rc LEFT JOIN reports r "
        "ON rc.source_pdf = r.source_pdf").fetchall()
    counts = {"exact_normalized": 0, "fuzzy_match": 0, "na_non_county": 0,
              "no_source_county": 0, "unmatched": 0}
    updates = []
    for county_id, county_name, geo_type, state_abbr in rows:
        fips, method, score = match_county_name(
            state_abbr, county_name, geo_type, state_index)
        counts[method] += 1
        updates.append((fips, method, score, county_id))
    with conn:
        conn.executemany(
            "UPDATE report_counties SET county_fips = ?, fips_match_method = ?, "
            "fuzzy_score = ? WHERE county_id = ?", updates)
    return counts
