"""
SPT processing: ISPT/SPT groups, N200/refusal rule.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def extract_spt(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract SPT data: prefer ISPT, fallback to SPT."""
    if "ISPT" in groups:
        df = groups["ISPT"].copy()
    elif "SPT" in groups:
        df = groups["SPT"].copy()
    else:
        return pd.DataFrame(columns=["Borehole_ID", "Depth", "N_raw", "N_effective", "Flag"])

    # Resolve column names (AGS: HOLE_ID, ISPT_TOP, ISPT_NVAL, ISPT_NPEN)
    bh_col = _find_col(df, ["hole_id", "loca_id", "borehole", "hole", "loca"])
    depth_col = _find_col(df, ["ispt_top", "ispt_dpth", "depth", "dep"])
    pen_col = _find_col(df, ["ispt_npen", "pen", "penetration", "thk"])
    # Prefer exact ISPT_NVAL (N value); avoid matching ISPT_NPEN/ISPT_TOP
    n_col = None
    for c in df.columns:
        cu = str(c).upper()
        if cu == "ISPT_NVAL" or (cu.endswith("_NVAL") and "ISPT" in cu):
            n_col = c
            break
    n_col = n_col or _find_col(df, ["ispt_nval", "n_value", "spt_n", "blow"])
    if n_col and (n_col == depth_col or n_col == pen_col):
        n_col = None
    # Coerce numeric columns (AGS may have header-continuation rows as data)
    df = df.copy()
    df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
    if pen_col:
        df[pen_col] = pd.to_numeric(df[pen_col], errors="coerce")
    df = df.dropna(subset=[depth_col])
    rep_col = _find_col(df, ["ispt_rep", "rep", "remark"])
    rows = []
    for _, r in df.iterrows():
        bh = r[bh_col]
        depth = float(r[depth_col])
        n_raw = r.get(n_col, None) if n_col else None
        if pd.isna(n_raw) or str(n_raw).strip() in ("", "nan"):
            # Fallback: parse N from ISPT_REP e.g. "(1,1,1,1,2,3) N=7"
            rep_text = str(r.get(rep_col, "")) if rep_col else str(r.to_dict())
            n_raw = _parse_n_from_text(rep_text)
        penetration = r.get(pen_col, 300) if pen_col else 300

        n_eff, flag = _clean_n(n_raw, penetration, str(r.to_dict()))
        # When UNPARSED, try n_raw as int if it looks numeric
        if pd.isna(n_eff) and n_raw is not None and str(n_raw).strip():
            try:
                v = int(float(str(n_raw).strip()))
                if 0 < v <= 200:
                    n_eff, flag = v, ""
            except (ValueError, TypeError):
                pass

        # Seat depth: ISPT_SEAT if available, else ISPT_NPEN
        seat_col = _find_col(df, ["ispt_seat", "seat"])
        seat_val = float(r.get(seat_col, 0) or 0) if seat_col else 0
        try:
            pen_mm = float(penetration)
        except (ValueError, TypeError):
            pen_mm = 0
        rows.append({
            "Borehole_ID": bh,
            "Depth": depth,
            "Penetration_mm": pen_mm,
            "N_raw": n_raw,
            "N_effective": n_eff,
            "Flag": flag,
        })

    return pd.DataFrame(rows)


def _clean_n(n_raw: Any, penetration: Any, full_text: str) -> tuple[int | float, str]:
    """
    Apply N200 rule:
    - penetration < 450 mm (refusal) => N_effective=200, Flag="N200"
    - total blows >= 200 OR ("100 blows" and "no penetration") => N_effective=200, Flag="N200"
    - Unparsable => Flag="UNPARSED"
    """
    full_text = str(n_raw) + " " + str(full_text)
    full_upper = full_text.upper()

    # N200: penetration < 450 mm (refusal per VBA)
    try:
        pen = float(penetration)
        if pen < 450:
            return 200, "N200"
    except (ValueError, TypeError):
        pass

    # N200: 100 blows + no penetration
    if "100" in full_text and "NO PENETRATION" in full_upper:
        return 200, "N200"

    # N200: total blows >= 200
    try:
        val = int(float(n_raw))
        if val >= 200:
            return 200, "N200"
        return val, ""
    except (ValueError, TypeError):
        pass

    # Try to parse from text
    m = re.search(r"\b(\d{1,3})\b", str(n_raw))
    if m:
        val = int(m.group(1))
        if val >= 200:
            return 200, "N200"
        return val, ""

    return float("nan"), "UNPARSED"


def _parse_n_from_text(text: str) -> str | None:
    """Extract N value from text like '(1,1,1,1,2,3) N=7' or 'N=200'."""
    if not text:
        return None
    m = re.search(r"\bN\s*=\s*(\d{1,3})\b", str(text), re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{1,3})\s+blows?\b", str(text), re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def clean_spt(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for extract_spt output format; ensures columns exist."""
    required = ["Borehole_ID", "Depth", "Penetration_mm", "N_raw", "N_effective", "Flag"]
    for c in required:
        if c not in df.columns:
            df[c] = None
    return df[required]


def _find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for c in df.columns:
        cu = str(c).upper()
        for kw in keywords:
            if kw.upper() in cu:
                return c
    return None
