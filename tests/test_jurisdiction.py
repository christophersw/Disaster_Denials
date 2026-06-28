# tests/test_jurisdiction.py
"""
Title: tests/test_jurisdiction.py — Tests for jurisdiction classification.
Description: Verifies jurisdiction_type and the alignment-applicability flags
    for states, territories, DC, and tribal requesters (spec §5.7).
Changelog:
    2026-06-28  Initial version.
"""

import pandas as pd

from pda.modeling.jurisdiction import _classify, add_jurisdiction_features


def _frame():
    return pd.DataFrame(
        {
            "state_abbr": ["TX", "PR", "DC", "AZ", "GU"],
            "requestor_type": [
                "Governor", "Governor", "Mayor", "Tribal Chairman", "Governor",
            ],
        }
    )


def test_jurisdiction_type_assignment():
    out = add_jurisdiction_features(_frame())
    assert out["jurisdiction_type"].tolist() == [
        "state", "territory", "federal_district", "tribal", "territory",
    ]


def test_presidential_applicability():
    out = add_jurisdiction_features(_frame())
    # State + DC vote for president; territories and tribes do not.
    assert out["presidential_alignment_applicable"].tolist() == [1, 0, 1, 0, 0]


def test_gubernatorial_applicability():
    out = add_jurisdiction_features(_frame())
    # State + territory have governors; DC (mayor) and tribal do not.
    assert out["gubernatorial_alignment_applicable"].tolist() == [1, 1, 0, 0, 1]


# --- NaN / None / blank guard tests for _classify ---


def test_classify_handles_nan_state_abbr():
    """Float NaN for state_abbr must not raise and must fall through to 'state'."""
    assert _classify(float("nan"), "Governor") == "state"


def test_classify_handles_none_and_blank_state_abbr():
    """None and empty string for state_abbr must fall through to 'state'."""
    assert _classify(None, "Governor") == "state"
    assert _classify("", "Governor") == "state"


def test_classify_handles_nan_requestor_type():
    """Float NaN for requestor_type must not raise; no tribal markers → geography decides."""
    assert _classify("TX", float("nan")) == "state"


def test_classify_tribal_still_detected_with_valid_inputs():
    """Tribal markers in requestor_type must still be detected even when state_abbr is NaN."""
    assert _classify(float("nan"), "Tribal Chairman") == "tribal"
