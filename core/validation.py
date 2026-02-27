"""
Validation: errors (blocking) and warnings (non-blocking).
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def validate_layers(
    df: pd.DataFrame,
    bh_col: str = "Borehole_ID",
    from_col: str = "Depth_From",
    to_col: str = "Depth_To",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (errors_df, warnings_df).
    Errors: Depth_From >= Depth_To, overlaps within same borehole.
    Warnings: gaps, missing groups.
    """
    bh = _col(df, bh_col, ["borehole", "hole", "loca"])
    from_ = _col(df, from_col, ["top", "from", "geol_top"])
    to_ = _col(df, to_col, ["base", "to", "geol_base", "btm"])

    errors = []
    warnings = []

    for idx, row in df.iterrows():
        f, t = row[from_], row[to_]
        if pd.isna(f) or pd.isna(t):
            continue
        if f >= t:
            errors.append({"severity": "error", "message": "Depth_From >= Depth_To", "borehole": row[bh], "row_index": idx})

    # Overlaps: within same borehole, layers overlap
    for borehole, grp in df.groupby(bh):
        grp = grp.sort_values(from_)
        prev_to = None
        for idx, row in grp.iterrows():
            f, t = row[from_], row[to_]
            if prev_to is not None and f < prev_to - 1e-6:
                errors.append({"severity": "error", "message": "Overlap within same borehole", "borehole": borehole, "row_index": idx})
            prev_to = t

    # Gaps
    for borehole, grp in df.groupby(bh):
        grp = grp.sort_values(from_)
        prev_to = None
        for idx, row in grp.iterrows():
            f, t = row[from_], row[to_]
            if prev_to is not None and f > prev_to + 1e-6:
                warnings.append({"severity": "warning", "message": f"Gap between {prev_to} and {f}", "borehole": borehole, "row_index": idx})
            prev_to = t

    errors_df = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["severity", "message", "borehole", "row_index"])
    warnings_df = pd.DataFrame(warnings) if warnings else pd.DataFrame(columns=["severity", "message", "borehole", "row_index"])
    return errors_df, warnings_df


def _col(df: pd.DataFrame, name: str, keywords: list[str]) -> str:
    if name in df.columns:
        return name
    for c in df.columns:
        for kw in keywords:
            if kw.upper() in str(c).upper():
                return c
    return df.columns[0]
