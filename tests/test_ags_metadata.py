import pandas as pd

from core.ags_metadata import extract_ags_metadata


def test_extract_ags_version_from_tran_group():
    groups = {
        "TRAN": pd.DataFrame(
            [
                {
                    "TRAN_AGS": "4.0.4",
                    "TRAN_STAT": "FINAL",
                    "TRAN_DATE": "2026-07-13",
                    "TRAN_PROD": "Advance JV Construction Limited",
                    "TRAN_RECV": "GEO HK submission",
                }
            ]
        )
    }

    meta = extract_ags_metadata(groups)

    assert meta["AGS Version"] == "4.0.4"
    assert meta["Status"] == "FINAL"
    assert meta["Date"] == "2026-07-13"
    assert meta["Producer"] == "Advance JV Construction Limited"
    assert meta["Receiver"] == "GEO HK submission"
