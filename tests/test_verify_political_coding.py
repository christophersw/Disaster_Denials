# tests/test_verify_political_coding.py
"""
Title: tests/test_verify_political_coding.py — Tests for the coding-check helper.
Description: Verifies the denial-rate crosstab used by the EDA gate (spec §11).
Changelog:
    2026-06-28  Initial version.
"""

import pandas as pd

from scripts.verify_political_coding import crosstab_denial_rate


def test_crosstab_denial_rate():
    df = pd.DataFrame(
        {
            "denied": [1, 0, 0, 1, 0],
            "flag": [1, 1, 1, 0, 0],
        }
    )
    out = crosstab_denial_rate(df, "flag")
    # flag=1: 1 denial / 3 = 0.333 ; flag=0: 1 denial / 2 = 0.5
    assert abs(out.loc[1, "denial_rate"] - 1 / 3) < 1e-9
    assert abs(out.loc[0, "denial_rate"] - 0.5) < 1e-9
    assert out.loc[1, "n"] == 3
