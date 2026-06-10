"""Tests for FIPS crosswalk construction and PDA county mapping."""

from pda.fips import normalize_county_name


def test_normalize_strips_suffix_and_uppercases():
    assert normalize_county_name("Barton County") == "BARTON"
    assert normalize_county_name("Jefferson Parish") == "JEFFERSON"
    assert normalize_county_name("Kenai Peninsula Borough") == "KENAI PENINSULA"


def test_normalize_standardizes_saint():
    assert normalize_county_name("St. Louis County") == "SAINT LOUIS"
    assert normalize_county_name("Ste. Genevieve County") == "SAINT GENEVIEVE"
    assert normalize_county_name("ST. LOUIS") == "SAINT LOUIS"


def test_normalize_collapses_whitespace_and_punctuation():
    assert normalize_county_name("  De  Soto   County ") == "DE SOTO"
    assert normalize_county_name("Prince George's County") == "PRINCE GEORGES"


def test_normalize_handles_census_area_and_city_and_borough():
    assert normalize_county_name("Valdez-Cordova Census Area") == "VALDEZ-CORDOVA"
    assert normalize_county_name("Sitka City and Borough") == "SITKA"


from pda.fips import pad_fips, is_county_fips, build_state_index, match_county_name


def test_normalize_strips_curly_apostrophe():
    # PDF extraction yields a curly apostrophe (U+2019); it must normalize the
    # same as a straight one so "O'Brien" matches MIT's "OBRIEN".
    assert normalize_county_name("O’Brien") == "OBRIEN"
    assert normalize_county_name("O'Brien") == normalize_county_name("O’Brien")


def test_build_state_index_groups_by_state_and_keeps_ambiguous_as_set():
    mit_rows = [
        ("AL", "AUTAUGA", "1001"),
        ("VA", "RICHMOND", "51159"),       # Richmond County
        ("VA", "RICHMOND CITY", "51760"),  # distinct normalized name
        ("VA", "RICHMOND", "51760"),       # MIT also labels the city bare → ambiguous
    ]
    index = build_state_index(mit_rows)
    assert index["AL"]["AUTAUGA"] == {"01001"}
    assert index["VA"]["RICHMOND"] == {"51159", "51760"}   # ambiguous, kept as a set
    assert index["VA"]["RICHMOND CITY"] == {"51760"}


# A small index reused by the matcher tests.
_INDEX = {
    "AL": {"AUTAUGA": {"01001"}, "BALDWIN": {"01003"}},
    "KY": {"BRECKINRIDGE": {"21027"}, "BRACKEN": {"21023"}},
    "VA": {"RICHMOND": {"51159", "51760"}},  # ambiguous
}


def test_match_exact_returns_exact_normalized():
    fips, method, score = match_county_name("AL", "Autauga County", "county", _INDEX)
    assert (fips, method) == ("01001", "exact_normalized")
    assert score is None


def test_match_tribe_is_na_non_county():
    fips, method, score = match_county_name("AL", "Some Reservation", "reservation", _INDEX)
    assert (fips, method, score) == (None, "na_non_county", None)


def test_match_state_absent_from_data_is_no_source_county():
    fips, method, score = match_county_name("PR", "Bayamon", "county", _INDEX)
    assert (fips, method) == (None, "no_source_county")


def test_match_alaska_is_no_source_county_even_though_present():
    # AK appears in MIT only as State House Districts, not boroughs, so it can
    # never resolve to a real county FIPS.
    ak_index = dict(_INDEX, AK={"DISTRICT 01": {"02001"}})
    fips, method, score = match_county_name("AK", "Kenai Peninsula Borough", "county", ak_index)
    assert (fips, method) == (None, "no_source_county")


def test_match_fuzzy_recovers_misspelling_with_score():
    # "Breckenridge" (PDA) vs the real "Breckinridge" (KY 21027).
    fips, method, score = match_county_name("KY", "Breckenridge", "county", _INDEX)
    assert (fips, method) == ("21027", "fuzzy_match")
    assert score >= 0.90


def test_match_ambiguous_name_stays_unmatched():
    fips, method, score = match_county_name("VA", "Richmond", "county", _INDEX)
    assert (fips, method) == (None, "unmatched")


def test_match_unknown_name_is_unmatched():
    fips, method, score = match_county_name("AL", "Nowhere", "county", _INDEX)
    assert (fips, method) == (None, "unmatched")


def test_is_county_fips_accepts_real_codes_rejects_placeholders():
    # A real county FIPS is all digits and exactly 5 chars once zero-padded.
    assert is_county_fips("01001") is True
    assert is_county_fips("1001") is True        # unpadded real county (AL)
    assert is_county_fips(1001) is True
    assert is_county_fips("48201") is True
    # Non-county MIT reporting units that must be dropped from a county rollup:
    assert is_county_fips("NA") is False         # CT/ME/RI overseas/write-in units
    assert is_county_fips("000NA") is False      # padded form of the above
    assert is_county_fips("2938000") is False    # MO Kansas City (7 digits)
    assert is_county_fips("") is False
    assert is_county_fips(None) is False


def test_pad_fips_zero_pads_to_five():
    assert pad_fips("1001") == "01001"
    assert pad_fips("48201") == "48201"
    assert pad_fips(1001) == "01001"


# tests/test_fips.py  (append)
import sqlite3

from pda.fips import add_fips_columns, map_report_counties


def _setup_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE reports (source_pdf TEXT PRIMARY KEY, state_abbr TEXT)")
    conn.execute(
        "CREATE TABLE report_counties (county_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "source_pdf TEXT, county_name TEXT, geo_type TEXT)")
    conn.execute("INSERT INTO reports VALUES ('a.pdf', 'MO')")
    conn.execute("INSERT INTO reports VALUES ('b.pdf', 'KY')")
    conn.execute("INSERT INTO reports VALUES ('c.pdf', 'PR')")
    counties = [
        ("a.pdf", "St. Louis County", "county"),      # exact → MO/SAINT LOUIS
        ("a.pdf", "Nowhere County", "county"),         # unmatched
        ("b.pdf", "Breckenridge", "county"),           # fuzzy → KY/BRECKINRIDGE
        ("b.pdf", "Some Reservation", "reservation"),  # na_non_county
        ("c.pdf", "Bayamon", "county"),                # no_source_county (territory)
    ]
    conn.executemany(
        "INSERT INTO report_counties (source_pdf, county_name, geo_type) "
        "VALUES (?, ?, ?)", counties)
    conn.commit()
    return conn


_DB_INDEX = {
    "MO": {"SAINT LOUIS": {"29189"}},
    "KY": {"BRECKINRIDGE": {"21027"}, "BRACKEN": {"21023"}},
}


def test_add_fips_columns_is_idempotent():
    conn = _setup_db()
    add_fips_columns(conn)
    add_fips_columns(conn)  # second call must not raise
    cols = {r[1] for r in conn.execute("PRAGMA table_info(report_counties)")}
    assert {"county_fips", "fips_match_method", "fuzzy_score"} <= cols


def test_map_report_counties_classifies_and_applies_fuzzy():
    conn = _setup_db()
    add_fips_columns(conn)
    counts = map_report_counties(conn, _DB_INDEX)
    assert counts == {"exact_normalized": 1, "fuzzy_match": 1,
                      "na_non_county": 1, "no_source_county": 1, "unmatched": 1}
    result = dict(conn.execute(
        "SELECT county_name, county_fips FROM report_counties").fetchall())
    assert result["St. Louis County"] == "29189"   # exact
    assert result["Breckenridge"] == "21027"        # fuzzy applied
    assert result["Nowhere County"] is None
    assert result["Some Reservation"] is None
    assert result["Bayamon"] is None
    methods = dict(conn.execute(
        "SELECT county_name, fips_match_method FROM report_counties").fetchall())
    assert methods["Breckenridge"] == "fuzzy_match"
    assert methods["Bayamon"] == "no_source_county"
    assert methods["Some Reservation"] == "na_non_county"
    # fuzzy_score recorded only for the fuzzy row
    score = conn.execute(
        "SELECT fuzzy_score FROM report_counties WHERE county_name='Breckenridge'").fetchone()[0]
    assert score >= 0.90
