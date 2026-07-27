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


def _ispt_groups(ispt_row, samples=None):
    groups = {"ISPT": pd.DataFrame([ispt_row])}
    if samples is not None:
        groups["SAMP"] = pd.DataFrame(samples)
    return groups


def _ispt_row(**overrides):
    row = {
        "LOCA_ID": "BH1",
        "ISPT_TOP": "3.10",
        "ISPT_NVAL": "16",
        "ISPT_TYPE": "N",
    }
    row.update(overrides)
    return row


def test_samp_spt_fallback_derives_penetration_from_matching_geometry():
    out = extract_spt(_ispt_groups(
        _ispt_row(),
        [{"LOCA_ID": "BH1", "SAMP_ID": "S1", "SAMP_TYPE": " sPt ",
          "SAMP_TOP": "3.10", "SAMP_BASE": "3.55"}],
    ))

    assert out.loc[0, "Penetration_mm"] == 450


def test_ispt_npen_takes_precedence_over_matching_samp_geometry():
    out = extract_spt(_ispt_groups(
        _ispt_row(ISPT_NPEN="300", ISPT_SEAT="325"),
        [{"LOCA_ID": "BH1", "SAMP_TYPE": "SPT", "SAMP_TOP": 3.10, "SAMP_BASE": 3.55}],
    ))

    assert out.loc[0, "Penetration_mm"] == 300


def test_legacy_ispt_penetration_alias_takes_precedence_over_seat_and_samp():
    out = extract_spt(_ispt_groups(
        _ispt_row(PENETRATION="300", ISPT_SEAT="325"),
        [{"LOCA_ID": "BH1", "SAMP_TYPE": "SPT", "SAMP_TOP": 3.10, "SAMP_BASE": 3.55}],
    ))

    assert out.loc[0, "Penetration_mm"] == 300


def test_ispt_seat_takes_precedence_over_matching_samp_geometry_when_npen_absent():
    out = extract_spt(_ispt_groups(
        _ispt_row(ISPT_SEAT="325"),
        [{"LOCA_ID": "BH1", "SAMP_TYPE": "SPT", "SAMP_TOP": 3.10, "SAMP_BASE": 3.55}],
    ))

    assert out.loc[0, "Penetration_mm"] == 325


@pytest.mark.parametrize("sample", [
    {"LOCA_ID": "BH1", "SAMP_TYPE": "U100", "SAMP_TOP": 3.10, "SAMP_BASE": 3.55},
    {"LOCA_ID": "OTHER", "SAMP_TYPE": "SPT", "SAMP_TOP": 3.10, "SAMP_BASE": 3.55},
    {"LOCA_ID": "BH1", "SAMP_TYPE": "SPT", "SAMP_TOP": 3.11, "SAMP_BASE": 3.56},
])
def test_samp_fallback_requires_matching_spt_borehole_and_top_depth(sample):
    out = extract_spt(_ispt_groups(_ispt_row(), [sample]))

    assert pd.isna(out.loc[0, "Penetration_mm"])
