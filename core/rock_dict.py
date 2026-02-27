"""
Rock dictionary: load Igneous/Sedimentary/Metamorphic sets from AGS-completed.csv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

# Repo-relative path for rock dictionary
CORE_DIR = Path(__file__).resolve().parent
DATA_DIR = CORE_DIR.parent / "data"
ROCK_DICT_CSV = DATA_DIR / "AGS-completed.csv"


def load_rock_sets(csv_path: Path | None = None) -> Tuple[set[str], set[str], set[str]]:
    """
    Load rock type sets from AGS-completed.csv.
    Returns (IGNEOUS_SET, SEDIMENTARY_SET, METAMORPHIC_SET) - sets of uppercase rock names.
    """
    path = csv_path or ROCK_DICT_CSV
    df = pd.read_csv(path, header=None)

    # Find column indices: row 0 has "Igneous Rocks", "Sedimentary Rocks", "Metamorphic Rocks"
    ign_col = sed_col = meta_col = None
    for c in range(df.shape[1]):
        val = str(df.iloc[0, c]).strip()
        if "igneous" in val.lower() and "rock" in val.lower():
            ign_col = c
        if "sedimentary" in val.lower() and "rock" in val.lower():
            sed_col = c
        if "metamorphic" in val.lower() and "rock" in val.lower():
            meta_col = c

    def _collect(col: int | None) -> set[str]:
        if col is None:
            return set()
        vals = df.iloc[1:, col].dropna().astype(str).str.strip()
        return {v.upper() for v in vals if v and not v.startswith("#") and v != "nan"}

    igneous = _collect(ign_col)
    sedimentary = _collect(sed_col)
    metamorphic = _collect(meta_col)

    return igneous, sedimentary, metamorphic


def match_rock_type(
    text: str,
    label2: str | None,
    igneous: set[str],
    sedimentary: set[str],
    metamorphic: set[str],
) -> tuple[str | None, str | None]:
    """
    Match text to rock type. Prefer label2 (bracket) first, then full description.
    Case-insensitive, longest match wins.
    Returns (Rock_Type, Rock_Category) or (None, None).
    """
    if not text and not label2:
        return None, None

    def _find_match(s: str, sets_dict: dict) -> tuple[str | None, str | None]:
        if not s or not s.strip():
            return None, None
        sup = s.upper()
        best_match = None
        best_cat = None
        for cat, st in sets_dict.items():
            for rock in st:
                if rock in sup and (best_match is None or len(rock) > len(best_match)):
                    best_match = rock
                    best_cat = cat
        return best_match, best_cat

    sets_dict = {
        "IGNEOUS": igneous,
        "SEDIMENTARY": sedimentary,
        "METAMORPHIC": metamorphic,
    }

    # Try label2 first (bracket content)
    if label2:
        m, c = _find_match(label2, sets_dict)
        if m:
            return m, c

    # Then full text
    combined = f"{label2 or ''} {text or ''}".strip()
    return _find_match(combined, sets_dict)
