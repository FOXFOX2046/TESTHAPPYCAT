from __future__ import annotations

import pandas as pd


def extract_ags_metadata(groups: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Extract display metadata from the AGS TRAN group."""
    tran = groups.get("TRAN")
    if tran is None or tran.empty:
        return {"AGS Version": "Unknown"}

    row = tran.iloc[0]
    fields = {
        "AGS Version": "TRAN_AGS",
        "Status": "TRAN_STAT",
        "Date": "TRAN_DATE",
        "Producer": "TRAN_PROD",
        "Receiver": "TRAN_RECV",
    }
    meta: dict[str, str] = {}
    for label, col in fields.items():
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            meta[label] = str(value).strip()
    if "AGS Version" not in meta:
        meta["AGS Version"] = "Unknown"
    return meta
