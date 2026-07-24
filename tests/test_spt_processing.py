"""Test SPT: N200 rule."""
import pandas as pd
import pytest

from core.spt_processing import extract_spt, _clean_n


def test_n200_penetration_refusal():
    # penetration < 450 mm = refusal per VBA
    n, flag = _clean_n(50, 400, "")
    assert n == 200
    assert flag == "N200"


def test_n200_100_blows_no_penetration():
    n, flag = _clean_n("100", 300, "100 blows no penetration")
    assert n == 200
    assert flag == "N200"


def test_n200_total_blows():
    n, flag = _clean_n(250, 300, "")
    assert n == 200
    assert flag == "N200"


def test_normal_n():
    n, flag = _clean_n(25, 450, "")
    assert n == 25
    assert flag == ""


def test_missing_penetration_does_not_force_n200():
    groups = {
        "ISPT": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH1",
                    "ISPT_TOP": "3.10",
                    "ISPT_NVAL": "16",
                    "ISPT_TYPE": "N",
                }
            ]
        )
    }

    out = extract_spt(groups)

    assert out.loc[0, "N_raw"] == "16"
    assert out.loc[0, "N_effective"] == 16
    assert out.loc[0, "Flag"] == ""
