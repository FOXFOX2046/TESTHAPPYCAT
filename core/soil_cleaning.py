"""
Soil cleaning and label extraction (VBA CreateSoilLabels logic).
"""
from __future__ import annotations

import re
from typing import Any


def clean_description(text: str) -> str:
    """Normalize whitespace, remove duplicate commas, preserve meaning."""
    if not text or not isinstance(text, str):
        return ""
    s = text.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r",\s*,", ",", s)
    return s.strip()


def extract_labels(text: str) -> dict[str, Any]:
    """
    Extract Label1, Label2, FinalLabel from description.
    - Label1: bracket-removed, ALL CAPS phrases joined by /
    - Label2: ALL CAPS phrase from last bracket group at end
    - FinalLabel: Label2 + " (" + Label1 + ")" if Label2 else Label1
    """
    desc_orig = str(text).strip() if text else ""
    desc_clean = clean_description(desc_orig)

    # Label1: remove brackets, extract ALL CAPS phrases
    desc_no_brackets = re.sub(r"\(.*?\)", "", desc_clean)
    caps_pattern = r"\b(?:[A-Z]+(?:[-\s][A-Z]+)*)\b"
    matches1 = re.findall(caps_pattern, desc_no_brackets)
    label1 = "/".join(matches1) if matches1 else ""

    # Label2: last bracket group at end, ALL CAPS inside
    bracket_match = re.findall(r"\(([^)]*)\)\s*$", desc_clean)
    label2 = ""
    if bracket_match:
        inner = bracket_match[-1]
        matches2 = re.findall(caps_pattern, inner)
        label2 = matches2[-1] if matches2 else ""

    # FinalLabel: Label2优先 (VBA: label2 & " (" & label1 & ")" when no rock intersection)
    if label2:
        final = f"{label2} ({label1})" if label1 else label2
    else:
        final = label1 or desc_clean

    return {
        "Description_Original": desc_orig,
        "Description_Clean": desc_clean,
        "Label1": label1,
        "Label2": label2,
        "FinalLabel": final,
    }
