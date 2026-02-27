"""
Summary report: "due to the lower level of <LAYER>".
Layer selection: deepest non-UNKNOWN major layer (deterministic).
"""
from __future__ import annotations

import pandas as pd

# Major non-UNKNOWN codes (excluding fill/alluvium trivial)
MAJOR_CODES = {"CDG", "HDG", "MDG", "SDG", "CDI", "HDI", "MDI", "SDI", "CDS", "HDS", "MDS", "SDS", "CDM", "HDM", "MDM", "SDM", "CDT", "HDT", "MDT", "SDT", "CDV", "HDV", "MDV", "SDV", "CORESTONE"}


def build_summary_report(
    layers_merged_df: pd.DataFrame,
    gl_df: pd.DataFrame | None = None,
    bh_col: str = "Borehole_ID",
    code_col: str = "Normalized_Code",
    from_col: str = "Depth_From",
    to_col: str = "Depth_To",
) -> str:
    """Generate summary text with exact phrase 'due to the lower level of <LAYER>'."""
    lines = []
    bh = bh_col if bh_col in layers_merged_df.columns else _find(layers_merged_df.columns, ["borehole", "loca", "hole"])
    code = code_col if code_col in layers_merged_df.columns else _find(layers_merged_df.columns, ["normalized", "code"])
    from_ = from_col if from_col in layers_merged_df.columns else _find(layers_merged_df.columns, ["from", "top"])
    to_ = to_col if to_col in layers_merged_df.columns else _find(layers_merged_df.columns, ["to", "base", "btm"])

    if not bh or not code:
        return "No layers to summarize."

    for borehole, grp in layers_merged_df.groupby(bh):
        grp = grp.sort_values(from_)
        seq = grp[code].tolist()
        lines.append(f"Borehole: {borehole}")
        lines.append(f"  Encountered sequence (depth order): {', '.join(str(c) for c in seq)}")

        # Select deepest major non-UNKNOWN layer
        selected = None
        for _, r in grp.iterrows():
            c = str(r[code]).strip()
            if c in MAJOR_CODES:
                selected = c

        if selected:
            lines.append(f"  Adopted design layer: due to the lower level of {selected}.")
        else:
            lines.append("  Adopted design layer: (no major layer identified)")
        lines.append("")

    return "\n".join(lines)


def _find(cols, keywords):
    for c in cols:
        cu = str(c).upper()
        for kw in keywords:
            if kw.upper() in cu:
                return c
    return cols[0] if len(cols) else None
