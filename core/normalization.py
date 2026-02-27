"""
Normalized_Code with strict precedence: CORESTONE > decomp > shortcuts > rock family > fallback.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .rock_dict import load_rock_sets

# Decomp keywords (order matters for precedence)
DECOMP_MAP = {
    "COMPLETELY DECOMPOSED": "CD",
    "HIGHLY DECOMPOSED": "HD",
    "MODERATELY DECOMPOSED": "MD",
    "SLIGHTLY DECOMPOSED": "SD",
}

# Lithology shortcuts (only when decomp exists)
LITH_SHORTCUTS = [
    ("TUFF", "T"),
    ("GRANITE", "G"),
    ("VOLCANIC", "V"),
]

# Non-decomp soil types
SOIL_TYPES = ["FILL", "ALLUVIUM", "MARINE DEPOSIT", "MARINE DEPOSITS", "COLLUVIUM", "ROCK FILL"]

# Primary soil keywords for subtype extraction (order = priority, last uppercase match wins)
_PRIMARY_SOILS = ["CLAY", "SILT", "SAND", "GRAVEL", "COBBLES", "BOULDERS"]

# Deposit types that should carry a subtype suffix like ALLUVIUM (CLAY)
_DEPOSIT_BASES = {"ALLUVIUM", "MARINE_DEPOSIT", "FILL", "CL"}


def _get_decomp_class(desc: str) -> str | None:
    """Extract decomposition class from description."""
    ud = desc.upper()
    for kw, code in DECOMP_MAP.items():
        if kw in ud:
            return code
    return None


def _get_lithology_shortcut(desc: str) -> str | None:
    """Check lithology shortcuts (TUFF/GRANITE/VOLCANIC)."""
    ud = desc.upper()
    for kw, suffix in LITH_SHORTCUTS:
        if kw in ud:
            return suffix
    return None


def _match_bracket_soil(label2: str) -> str | None:
    """Match FILL/ALLUVIUM/MARINE DEPOSIT etc from label2."""
    if not label2:
        return None
    u = label2.upper()
    for st in SOIL_TYPES:
        if st in u:
            if "ALLUVIUM" in st or st == "ALLUVIUM":
                return "ALLUVIUM"
            if "MARINE" in st:
                return "MARINE_DEPOSIT"
            if st == "FILL":
                return "FILL"
            if st == "ROCK FILL":
                return "RFILL"
            if st == "COLLUVIUM":
                return "CL"
    return None


def _extract_primary_soil(desc: str) -> str | None:
    """Find the primary soil type from a geotechnical description.

    In standard descriptions the main material is written in ALL CAPS,
    e.g. "slightly sandy, silty CLAY" → CLAY.
    We search for standalone uppercase soil keywords and return the last
    match (which is the primary noun in English geotechnical convention).
    """
    if not desc:
        return None
    for kw in _PRIMARY_SOILS:
        if re.search(rf"\b{kw}\b", desc):
            return kw
    return None


def _clean_label(raw: str) -> str:
    """Uppercase, strip symbols (keep letters, digits, spaces, underscores)."""
    s = re.sub(r"[^A-Za-z0-9\s_]", " ", str(raw))
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s.replace(" ", "_") if s else ""


def _normalize_single(
    desc: str,
    label1: str,
    label2: str,
    final_label: str,
    rock_sets: tuple,
    fill_subtype: bool = False,
) -> dict[str, Any]:
    """Compute Normalized_Code and related fields for one row."""
    from .rock_dict import match_rock_type

    ud = desc.upper() if desc else ""

    # (0) CORESTONE dominance
    if "CORESTONE" in ud:
        return {
            "Normalized_Code": "CORESTONE",
            "Decomp_Class": None,
            "Rock_Type": None,
            "Rock_Category": None,
            "Lithology_Unknown": False,
        }

    decomp = _get_decomp_class(desc)
    ign, sed, meta = rock_sets
    rock_type, rock_cat = match_rock_type(label1, label2, ign, sed, meta)
    lith_shortcut = _get_lithology_shortcut(desc)

    lith_unknown = False

    if decomp:
        if lith_shortcut:
            code = decomp + lith_shortcut
        elif rock_cat:
            suffix = {"IGNEOUS": "I", "SEDIMENTARY": "S", "METAMORPHIC": "M"}.get(rock_cat, "S")
            code = decomp + suffix
        else:
            lith_unknown = True
            code = decomp + "S"
    else:
        bracket = _match_bracket_soil(label2)
        if bracket:
            code = bracket
        elif "FILL" in ud:
            code = "FILL"
        elif "ALLUVIUM" in ud or "AL-" in ud:
            code = "ALLUVIUM"
        elif "MARINE" in ud:
            code = "MARINE_DEPOSIT"
        else:
            code = "UNKNOWN"

    # Fallback chain: never leave UNKNOWN — use best available label
    if code == "UNKNOWN":
        lbl2 = _clean_label(label2)
        lbl1 = _clean_label(label1)
        fl = _clean_label(final_label)
        if lbl2:
            code = lbl2
        elif lbl1:
            code = lbl1
        elif fl:
            code = fl

    # For deposit types, append primary soil subtype: ALLUVIUM → ALLUVIUM (CLAY)
    # FILL only gets subtype when fill_subtype flag is True
    if code in _DEPOSIT_BASES:
        if code != "FILL" or fill_subtype:
            primary = _extract_primary_soil(ud)
            if primary:
                code = f"{code} ({primary})"

    return {
        "Normalized_Code": code,
        "Decomp_Class": decomp,
        "Rock_Type": rock_type,
        "Rock_Category": rock_cat,
        "Lithology_Unknown": lith_unknown,
    }


def normalize_layers(
    layers_df: pd.DataFrame,
    rock_sets: tuple | None = None,
    desc_col: str = "Description",
    label1_col: str = "Label1",
    label2_col: str = "Label2",
    final_label_col: str = "FinalLabel",
    fill_subtype: bool = False,
) -> pd.DataFrame:
    """
    Add Normalized_Code, Decomp_Class, Rock_Type, Rock_Category, Lithology_Unknown.
    """
    from .rock_dict import load_rock_sets as _load

    rock_sets = rock_sets or _load()
    out = layers_df.copy()

    if desc_col not in out.columns:
        cand = [c for c in out.columns if "desc" in str(c).lower() or "geol" in str(c).lower()]
        desc_col = cand[0] if cand else out.columns[-1]
    if label1_col not in out.columns:
        out[label1_col] = ""
    if label2_col not in out.columns:
        out[label2_col] = ""
    if final_label_col not in out.columns:
        out[final_label_col] = ""

    results = []
    for _, row in out.iterrows():
        r = _normalize_single(
            str(row.get(desc_col, "")),
            str(row.get(label1_col, "")),
            str(row.get(label2_col, "")),
            str(row.get(final_label_col, "")),
            rock_sets,
            fill_subtype=fill_subtype,
        )
        results.append(r)

    for k in ["Normalized_Code", "Decomp_Class", "Rock_Type", "Rock_Category", "Lithology_Unknown"]:
        out[k] = [r[k] for r in results]

    return out
