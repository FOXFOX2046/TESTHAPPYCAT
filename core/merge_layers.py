"""
Adjacent layer merge: same Borehole_ID, same Normalized_Code, Depth_To == next Depth_From.
"""
from __future__ import annotations

import pandas as pd


def merge_adjacent_layers(
    df: pd.DataFrame,
    tol: float = 1e-6,
    bh_col: str = "Borehole_ID",
    code_col: str = "Normalized_Code",
    from_col: str = "Depth_From",
    to_col: str = "Depth_To",
    desc_col: str | None = "Description",
) -> pd.DataFrame:
    """
    Merge adjacent layers with same Borehole_ID and Normalized_Code
    where abs(prev.Depth_To - next.Depth_From) <= tol.
    """
    # Resolve column names (AGS may use GEOL_TOP/GEOL_BASE etc.)
    bh = bh_col if bh_col in df.columns else _find_col(df, ["borehole", "loca", "hole"])
    code = code_col if code_col in df.columns else _find_col(df, ["normalized", "code", "abbr"])
    from_ = from_col if from_col in df.columns else _find_col(df, ["top", "from", "geol_top"])
    to_ = to_col if to_col in df.columns else _find_col(df, ["base", "to", "geol_base", "btm"])
    desc = desc_col if desc_col and desc_col in df.columns else _find_col(df, ["desc", "description"], optional=True)

    if bh is None or code is None or from_ is None or to_ is None:
        return df.copy()

    # Coerce depth columns to float (may come as string from AGS)
    out = df.copy()
    for c in [from_, to_]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out = out.sort_values([bh, from_]).reset_index(drop=True)
    merged_rows = []
    i = 0

    while i < len(out):
        row = out.iloc[i].to_dict()
        start_from = row[from_]
        start_to = row[to_]
        count = 1
        descs = [str(row.get(desc, ""))] if desc else []

        j = i + 1
        while j < len(out):
            next_row = out.iloc[j]
            if next_row[bh] != row[bh] or next_row[code] != row[code]:
                break
            if abs(next_row[from_] - start_to) <= tol:
                start_to = next_row[to_]
                count += 1
                if desc:
                    descs.append(str(next_row.get(desc, "")))
                j += 1
            else:
                break

        merged = {
            bh: row[bh],
            from_: start_from,
            to_: start_to,
            "Thickness": start_to - start_from,
            code: row[code],
            "Merged_Count": count,
        }
        if desc:
            merged["Description_Merged"] = " | ".join(d for d in descs if d)
        merged_rows.append(merged)
        i = j

    result = pd.DataFrame(merged_rows)
    rename = {from_: "Depth_From", to_: "Depth_To"}
    if bh != "Borehole_ID":
        rename[bh] = "Borehole_ID"
    if code != "Normalized_Code":
        rename[code] = "Normalized_Code"
    result = result.rename(columns=rename)
    return result


def _find_col(df: pd.DataFrame, keywords: list[str], optional: bool = False) -> str | None:
    for c in df.columns:
        cu = str(c).upper()
        for kw in keywords:
            if kw.upper() in cu:
                return c
    return None if not optional else (df.columns[-1] if len(df.columns) else None)
