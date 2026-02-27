"""
Static ON-SCALE Strip Log PNG using matplotlib.
Layout: left legend panel + boreholes with strata, boundary markers, SPT labels.
Level = GL - Depth (Depth in meters below ground).

Adaptive layout tiers ensure readability at any borehole count:
  Tier 1 (<=8 BHs):  generous spacing, 4 SPT lanes, large font
  Tier 2 (9-15 BHs): medium spacing, 2 SPT lanes, medium font
  Tier 3 (>15 BHs):  compact spacing, 2 SPT lanes, small font, 1-decimal levels
"""
from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle, Polygon

from core.plots_striplog import build_color_map


def _col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for c in df.columns:
        for kw in keywords:
            if kw.upper() in str(c).upper():
                return c
    return df.columns[0] if len(df.columns) > 0 else None


def _resolve_adaptive_params(n_bh: int) -> dict:
    """Return layout parameters tuned to the number of boreholes."""
    if n_bh <= 8:
        return dict(
            col_w=80, gap=200, spt_pad=210, right_pad=120,
            lanes=4, lane_offset=90, spt_font=8, bnd_font=8,
            header_font=10, gl_font=9, min_gap_m=1.2,
            bnd_decimals=2, bnd_min_gap_m=1.5, bnd_max=12,
            bnd_right=10, spt_format="N={n} @ {lev}",
            header_max_chars=99, tri_size=3,
            left_panel=250, legend_font=8,
        )
    elif n_bh <= 15:
        return dict(
            col_w=70, gap=260, spt_pad=180, right_pad=100,
            lanes=2, lane_offset=70, spt_font=7, bnd_font=7,
            header_font=8, gl_font=7, min_gap_m=1.8,
            bnd_decimals=2, bnd_min_gap_m=2.5, bnd_max=8,
            bnd_right=12, spt_format="N={n}@{lev}",
            header_max_chars=16, tri_size=3,
            left_panel=220, legend_font=7,
        )
    else:
        return dict(
            col_w=55, gap=260, spt_pad=160, right_pad=80,
            lanes=2, lane_offset=55, spt_font=6, bnd_font=6,
            header_font=7, gl_font=6, min_gap_m=2.0,
            bnd_decimals=1, bnd_min_gap_m=4.0, bnd_max=6,
            bnd_right=8, spt_format="N={n}@{lev}",
            header_max_chars=12, tri_size=2,
            left_panel=180, legend_font=6,
        )


def render_striplog_png(
    layers_df: pd.DataFrame,
    spt_df: pd.DataFrame | None,
    gl_df: pd.DataFrame | None,
    borehole_ids: list[str],
    legend_codes: list[str] | None = None,
    color_map: dict[str, str] | None = None,
    dpi: int = 200,
    px_per_m: float = 12.0,
    spt_level_mode: str = "TOP",
) -> bytes:
    """Render static ON-SCALE strip log PNG.  Returns PNG bytes."""
    bh_col = _col(layers_df, ["borehole", "loca", "hole"]) or layers_df.columns[0]
    from_col = _col(layers_df, ["from", "top"]) or "Depth_From"
    to_col = _col(layers_df, ["to", "base", "btm"]) or "Depth_To"
    code_col = _col(layers_df, ["normalized", "code"]) or "Normalized_Code"

    gl_map: dict[str, float] = {}
    if gl_df is not None:
        gh = _col(gl_df, ["borehole", "loca", "hole"]) or gl_df.columns[0]
        gg = _col(gl_df, ["gl", "ground", "level", "elev"]) or (
            gl_df.columns[1] if len(gl_df.columns) > 1 else gl_df.columns[0]
        )
        for _, r in gl_df.iterrows():
            key = str(r[gh]).strip()
            if key and not key.startswith("*"):
                try:
                    gl_map[key] = float(r[gg]) if pd.notna(r.get(gg)) else 0.0
                except (ValueError, TypeError):
                    gl_map[key] = 0.0

    codes = (
        legend_codes
        if legend_codes is not None
        else sorted(
            layers_df[layers_df[bh_col].isin(borehole_ids)][code_col]
            .dropna().unique().astype(str).tolist(),
            key=str.upper,
        )
    )
    colors = color_map if color_map else build_color_map(codes)

    # --- Vertical datum ---
    n_bh = len(borehole_ids)
    P = _resolve_adaptive_params(n_bh)

    top_candidates = [gl_map.get(bh, 0.0) for bh in borehole_ids]
    bottom_candidates: list[float] = []
    for bh in borehole_ids:
        gl = gl_map.get(bh, 0.0)
        grp = layers_df[layers_df[bh_col] == bh]
        for _, row in grp.iterrows():
            try:
                bottom_candidates.append(gl - float(row[to_col]))
            except (ValueError, TypeError):
                pass
        if spt_df is not None and len(spt_df) > 0:
            sd_col = _col(spt_df, ["borehole", "loca", "hole"])
            sd_dep = _col(spt_df, ["depth"])
            if sd_col and sd_dep:
                for _, sr in spt_df[spt_df[sd_col] == bh].iterrows():
                    try:
                        bottom_candidates.append(gl - float(sr[sd_dep]))
                    except (ValueError, TypeError):
                        pass

    top_level = (max(top_candidates) if top_candidates else 5.0) + 1.0
    bottom_level = (min(bottom_candidates) - 1.0) if bottom_candidates else top_level - 21.0

    H_m = top_level - bottom_level
    top_margin = 110
    bot_margin = 70
    H_px = int(H_m * px_per_m) + top_margin + bot_margin

    def y_px(level: float) -> float:
        return top_margin + (top_level - level) * px_per_m

    # --- Horizontal layout ---
    col_w = P["col_w"]
    gap = P["gap"]
    spt_pad = P["spt_pad"]
    left_panel = P["left_panel"]
    W_px = left_panel + spt_pad + n_bh * (col_w + gap) + P["right_pad"]

    fig = plt.figure(figsize=(W_px / dpi, H_px / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_px)
    ax.set_ylim(H_px, 0)
    ax.axis("off")

    # --- Legend panel ---
    leg_x = left_panel // 2
    lfs = P["legend_font"]
    ax.text(leg_x, 30, "TopBound Level (mPD)", fontsize=lfs - 1, ha="center", va="top")
    ax.text(leg_x, 48, f"{top_level - 1.0:.2f}", fontsize=lfs, ha="center", va="top", fontweight="bold")
    ax.text(leg_x, 68, "LowBound Level (mPD)", fontsize=lfs - 1, ha="center", va="top")
    ax.text(leg_x, 86, f"{bottom_level + 1.0:.2f}", fontsize=lfs, ha="center", va="top", fontweight="bold")
    ax.text(leg_x, 110, "Soil Legend", fontsize=lfs, ha="center", va="top", fontweight="bold")

    swatch = 10
    row_h = 16
    for i, code in enumerate(codes):
        c = str(code).strip()
        y_row = 140 + i * row_h
        rect = Rectangle((leg_x - 60, y_row - swatch // 2), swatch, swatch,
                          facecolor=colors.get(c, "#888"), edgecolor="k", linewidth=0.4)
        ax.add_patch(rect)
        ax.text(leg_x - 45, y_row, c, fontsize=lfs - 1, ha="left", va="center")

    # --- Boreholes ---
    def _fmt_level(lev: float) -> str:
        return f"{lev:+.2f}" if lev >= 0 else f"{lev:.2f}"

    for idx, bh in enumerate(borehole_ids):
        gl = gl_map.get(bh, 0.0)
        x0 = left_panel + spt_pad + idx * (col_w + gap)
        x1 = x0 + col_w
        x_spt_base = x0 - 30
        x_bnd_text = x1 + P["bnd_right"]
        x_mid = (x0 + x1) / 2

        # Header
        y_head = top_margin - 60
        bh_str = str(bh)
        if len(bh_str) > P["header_max_chars"]:
            bh_str = bh_str[:P["header_max_chars"]] + ".."
        ax.text(x_mid, y_head, bh_str, fontsize=P["header_font"],
                ha="center", va="bottom", fontweight="bold")
        gl_str = f"+{gl:.2f}" if gl >= 0 else f"{gl:.2f}"
        ax.text(x_mid, y_head + 16, f"G.L{gl_str}", fontsize=P["gl_font"],
                ha="center", va="bottom")

        # Strata rectangles — collect ALL layer boundaries
        grp = layers_df[layers_df[bh_col] == bh].sort_values(from_col)
        boundaries: list[float] = [gl]
        for _, row in grp.iterrows():
            try:
                f, t = float(row[from_col]), float(row[to_col])
            except (ValueError, TypeError):
                continue
            if f >= t:
                continue
            code = str(row.get(code_col, "")).strip()
            y_top = y_px(gl - f)
            y_bot = y_px(gl - t)
            rect = Rectangle((x0, y_top), col_w, y_bot - y_top,
                              facecolor=colors.get(code, "#888"), edgecolor="k", linewidth=0.3)
            ax.add_patch(rect)
            boundaries.append(gl - f)
            boundaries.append(gl - t)

        # De-duplicate within 0.005m tolerance, keep all
        boundaries = sorted(set(round(b, 2) for b in boundaries), reverse=True)

        # Boundary triangles + level labels for EVERY boundary (fixed x alignment)
        ts = P["tri_size"]
        for level in boundaries:
            y = y_px(level)
            tri = Polygon([(x_mid, y), (x_mid - ts, y - ts * 1.6), (x_mid + ts, y - ts * 1.6)],
                          closed=True, facecolor="black", edgecolor="black")
            ax.add_patch(tri)
            ax.text(x_bnd_text, y + 2, _fmt_level(level),
                    fontsize=P["bnd_font"], ha="left", va="center")

        # SPT labels with lane staggering (capped to gap width)
        if spt_df is not None and len(spt_df) > 0:
            sd_col = _col(spt_df, ["borehole", "loca", "hole"])
            sd_dep = _col(spt_df, ["depth"])
            n_eff_col = _col(spt_df, ["n_effective"])
            n_raw_col = "N_raw" if "N_raw" in spt_df.columns else None
            pen_col = "Penetration_mm" if "Penetration_mm" in spt_df.columns else None
            bh_spt = spt_df[spt_df[sd_col] == bh].copy()
            bh_spt["_level"] = bh_spt[sd_dep].apply(
                lambda d: gl - float(d) if pd.notna(d) else None)
            if spt_level_mode == "SEAT" and pen_col:
                bh_spt["_level"] = bh_spt["_level"] - bh_spt[pen_col].fillna(0) / 1000.0
            bh_spt = bh_spt.dropna(subset=["_level"]).sort_values("_level", ascending=False)

            for _, sr in bh_spt.iterrows():
                level = sr["_level"]
                n_val = sr.get(n_eff_col, None) if n_eff_col else None
                if (pd.isna(n_val) or str(n_val).strip() in ("", "nan")) and n_raw_col:
                    n_val = sr.get(n_raw_col, None)
                try:
                    n_display = int(float(n_val)) if pd.notna(n_val) and str(n_val).strip() not in ("", "nan") else "?"
                except (ValueError, TypeError):
                    n_display = "?"

                lev_str = _fmt_level(level)
                text = P["spt_format"].format(n=n_display, lev=lev_str)
                y_label = y_px(level)

                ax.text(x_spt_base, y_label, text, fontsize=P["spt_font"],
                        ha="right", va="center", color="#333")
                ax_x = x_spt_base + 5
                ax.annotate("", xy=(ax_x, y_label + 28),
                            xytext=(ax_x, y_label + 2),
                            arrowprops=dict(arrowstyle="->, head_length=0.6, head_width=0.4",
                                            color="#888", lw=0.5))

    # Horizontal guide lines every 10m
    for lv in range(int(bottom_level), int(top_level) + 1, 10):
        y = y_px(float(lv))
        if top_margin < y < H_px - bot_margin:
            ax.axhline(y=y, xmin=(left_panel + spt_pad) / W_px, xmax=1.0,
                        color="lightgray", linewidth=0.4, zorder=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# A3 Landscape print-ready strip log
# ---------------------------------------------------------------------------
MM_PER_INCH = 25.4
A3_WIDTH_MM = 420
A3_HEIGHT_MM = 297
A3_DPI = 300
LEFT_MARGIN_MM = 40
RIGHT_MARGIN_MM = 20
TOP_MARGIN_MM = 38
BOTTOM_MARGIN_MM = 10
MM_PER_M_1_200 = 1000 / 200
MIN_COL_WIDTH_MM = 5
PRINTABLE_HEIGHT_MM = A3_HEIGHT_MM - TOP_MARGIN_MM - BOTTOM_MARGIN_MM
PRINTABLE_WIDTH_MM = A3_WIDTH_MM - LEFT_MARGIN_MM - RIGHT_MARGIN_MM


DEFAULT_BH_PER_PAGE = 8


def _a3_params(n_bh: int, gap_mm: float) -> dict:
    """Adaptive parameters for A3 layout, computed from actual gap between columns."""
    spt_budget = gap_mm * 0.55
    if n_bh <= 8:
        lanes = min(3, max(1, int(spt_budget / 4)))
        return dict(
            spt_font=7, bnd_font=6, header_font=9, gl_font=7,
            lanes=lanes, lane_step=max(1, spt_budget / max(1, lanes)),
            min_spt_gap=1.0,
            bnd_dec=2, bnd_gap=2.0, bnd_max=10,
            bnd_right=min(gap_mm * 0.3, 3),
            header_max=99,
        )
    elif n_bh <= 12:
        lanes = min(2, max(1, int(spt_budget / 3)))
        return dict(
            spt_font=6, bnd_font=5.5, header_font=7, gl_font=6,
            lanes=lanes, lane_step=max(1, spt_budget / max(1, lanes)),
            min_spt_gap=1.2,
            bnd_dec=2, bnd_gap=2.5, bnd_max=8,
            bnd_right=min(gap_mm * 0.3, 2.5),
            header_max=18,
        )
    else:
        lanes = min(2, max(1, int(spt_budget / 3)))
        return dict(
            spt_font=5.5, bnd_font=5, header_font=6, gl_font=5,
            lanes=lanes, lane_step=max(1, spt_budget / max(1, lanes)),
            min_spt_gap=1.5,
            bnd_dec=1, bnd_gap=3.0, bnd_max=6,
            bnd_right=min(gap_mm * 0.3, 2),
            header_max=14,
        )


def render_striplog_a3_png(
    layers_df: pd.DataFrame,
    spt_df: pd.DataFrame | None,
    gl_df: pd.DataFrame | None,
    borehole_ids: list[str],
    legend_codes: list[str] | None = None,
    color_map: dict[str, str] | None = None,
    spt_level_mode: str = "TOP",
) -> tuple[bytes, str]:
    """Render print-ready A3 landscape strip log.  Returns (png_bytes, filename)."""
    bh_col = _col(layers_df, ["borehole", "loca", "hole"]) or layers_df.columns[0]
    from_col = _col(layers_df, ["from", "top"]) or "Depth_From"
    to_col = _col(layers_df, ["to", "base", "btm"]) or "Depth_To"
    code_col = _col(layers_df, ["normalized", "code"]) or "Normalized_Code"

    gl_map: dict[str, float] = {}
    if gl_df is not None:
        gh = _col(gl_df, ["borehole", "loca", "hole"]) or gl_df.columns[0]
        gg = _col(gl_df, ["gl", "ground", "level", "elev"]) or (
            gl_df.columns[1] if len(gl_df.columns) > 1 else gl_df.columns[0]
        )
        for _, r in gl_df.iterrows():
            key = str(r[gh]).strip()
            if key and not key.startswith("*"):
                try:
                    gl_map[key] = float(r[gg]) if pd.notna(r.get(gg)) else 0.0
                except (ValueError, TypeError):
                    gl_map[key] = 0.0

    codes = (
        legend_codes
        if legend_codes is not None
        else sorted(
            layers_df[layers_df[bh_col].isin(borehole_ids)][code_col]
            .dropna().unique().astype(str).tolist(), key=str.upper)
    )
    colors = color_map if color_map else build_color_map(codes)

    n_bh = max(1, len(borehole_ids))

    # Vertical bounds: computed from ALL boreholes in layers_df (global range)
    all_bhs = layers_df[bh_col].dropna().unique().tolist()
    top_candidates = [gl_map.get(str(b).strip(), 0.0) for b in all_bhs if gl_map.get(str(b).strip())]
    bottom_candidates: list[float] = []
    for bh in all_bhs:
        gl = gl_map.get(str(bh).strip(), 0.0)
        grp = layers_df[layers_df[bh_col] == bh]
        for _, row in grp.iterrows():
            try:
                bottom_candidates.append(gl - float(row[to_col]))
            except (ValueError, TypeError):
                pass
        if spt_df is not None and len(spt_df) > 0:
            sd_col = _col(spt_df, ["borehole", "loca", "hole"])
            sd_dep = _col(spt_df, ["depth"])
            if sd_col and sd_dep:
                for _, sr in spt_df[spt_df[sd_col] == bh].iterrows():
                    try:
                        bottom_candidates.append(gl - float(sr[sd_dep]))
                    except (ValueError, TypeError):
                        pass

    top_level = max(top_candidates) if top_candidates else 5.0
    bottom_level = min(bottom_candidates) if bottom_candidates else top_level - 20.0
    depth_range_m = top_level - bottom_level

    required_mm = depth_range_m * MM_PER_M_1_200
    if required_mm > PRINTABLE_HEIGHT_MM:
        mm_per_m = PRINTABLE_HEIGHT_MM / depth_range_m
        scale_label = f"1:{int(1000 / mm_per_m)}"
    else:
        mm_per_m = MM_PER_M_1_200
        scale_label = "1:200"

    def y_mm(level: float) -> float:
        return TOP_MARGIN_MM + (top_level - level) * mm_per_m

    # Horizontal: distribute columns evenly, then derive all spacing from gap
    if n_bh == 1:
        col_w = min(9, PRINTABLE_WIDTH_MM)
        gap_mm = 20.0
    else:
        col_w = max(MIN_COL_WIDTH_MM, min(9, (PRINTABLE_WIDTH_MM * 0.35) / n_bh))
        gap_mm = (PRINTABLE_WIDTH_MM - n_bh * col_w) / max(1, n_bh - 1)

    P = _a3_params(n_bh, gap_mm)

    fig = plt.figure(figsize=(A3_WIDTH_MM / MM_PER_INCH, A3_HEIGHT_MM / MM_PER_INCH),
                     dpi=A3_DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A3_WIDTH_MM)
    ax.set_ylim(A3_HEIGHT_MM, 0)
    ax.axis("off")

    # Grid lines
    px_min = LEFT_MARGIN_MM
    px_max = A3_WIDTH_MM - RIGHT_MARGIN_MM
    for lv in range(int(bottom_level) - 5, int(top_level) + 10, 5):
        y = y_mm(float(lv))
        if TOP_MARGIN_MM < y < A3_HEIGHT_MM - BOTTOM_MARGIN_MM:
            ax.axhline(y=y, xmin=px_min / A3_WIDTH_MM, xmax=px_max / A3_WIDTH_MM,
                       color="lightgray", linewidth=0.3, zorder=0)
        if lv % 10 == 0 and int(bottom_level) <= lv <= int(top_level) + 5:
            if TOP_MARGIN_MM < y < A3_HEIGHT_MM - BOTTOM_MARGIN_MM:
                ax.text(5, y, f"{lv:+d}m", fontsize=6, ha="left", va="center", fontfamily="monospace")

    # Title: top-left corner (avoids overlap with borehole headers in the center)
    ax.text(5, 5, "Borehole Log",
            fontsize=9, ha="left", va="top", fontweight="bold")
    ax.text(5, 12, f"Top Level: {top_level:.2f} mPD    Bottom Level: {bottom_level:.2f} mPD",
            fontsize=6, ha="left", va="top", color="#333")

    # Legend: above soil stick (horizontal layout), with gap below Top Level
    leg_y = 22
    ax.text(5, leg_y - 6, "Soil Legend:", fontsize=6, ha="left", va="top", fontweight="bold")
    max_legend = min(16, len(codes))
    for i, code in enumerate(codes[:max_legend]):
        x_leg = 5 + i * 22
        rect = Rectangle((x_leg, leg_y - 3), 3, 3,
                          facecolor=colors.get(str(code).strip(), "#888"),
                          edgecolor="k", linewidth=0.2)
        ax.add_patch(rect)
        ax.text(x_leg + 4, leg_y - 1.5, str(code).strip(),
                fontsize=5, ha="left", va="center")

    def _fmt(lev: float) -> str:
        return f"{lev:+.2f}" if lev >= 0 else f"{lev:.2f}"

    # Boreholes
    for i, bh in enumerate(borehole_ids):
        gl = gl_map.get(bh, 0.0)
        x0 = LEFT_MARGIN_MM + i * (col_w + gap_mm)
        x1 = x0 + col_w
        xm = (x0 + x1) / 2
        x_bnd = x1 + P["bnd_right"]

        # Header
        y_h = TOP_MARGIN_MM - 7
        bh_str = str(bh)
        if len(bh_str) > P["header_max"]:
            bh_str = bh_str[:P["header_max"]] + ".."
        ax.text(xm, y_h, bh_str, fontsize=P["header_font"],
                ha="center", va="bottom", fontweight="bold")
        gl_s = f"+{gl:.2f}" if gl >= 0 else f"{gl:.2f}"
        ax.text(xm, y_h + 3.5, f"G.L{gl_s}", fontsize=P["gl_font"], ha="center", va="bottom")

        # Strata — collect ALL boundaries (from and to)
        grp = layers_df[layers_df[bh_col] == bh].sort_values(from_col)
        boundaries: list[float] = [gl]
        for _, row in grp.iterrows():
            try:
                f, t = float(row[from_col]), float(row[to_col])
            except (ValueError, TypeError):
                continue
            if f >= t:
                continue
            code = str(row.get(code_col, "")).strip()
            yt = y_mm(gl - f)
            yb = y_mm(gl - t)
            rect = Rectangle((x0, yt), col_w, yb - yt,
                              facecolor=colors.get(code, "#888"), edgecolor="k", linewidth=0.25)
            ax.add_patch(rect)
            boundaries.append(gl - f)
            boundaries.append(gl - t)

        # De-duplicate within rounding tolerance, keep ALL
        boundaries = sorted(set(round(b, 2) for b in boundaries), reverse=True)

        # Boundary markers + level labels for EVERY boundary (fixed x alignment)
        for level in boundaries:
            y = y_mm(level)
            tri = Polygon([(xm, y), (xm - 0.8, y - 1.3), (xm + 0.8, y - 1.3)],
                          closed=True, facecolor="black", edgecolor="black")
            ax.add_patch(tri)
            ax.text(x_bnd, y, _fmt(level), fontsize=P["bnd_font"], ha="left", va="center")

        # SPT labels with lane staggering and arrows
        if spt_df is not None and len(spt_df) > 0:
            sd_col = _col(spt_df, ["borehole", "loca", "hole"])
            sd_dep = _col(spt_df, ["depth"])
            n_eff_col = _col(spt_df, ["n_effective"])
            n_raw_col = "N_raw" if "N_raw" in spt_df.columns else None
            pen_col = "Penetration_mm" if "Penetration_mm" in spt_df.columns else None
            bh_spt = spt_df[spt_df[sd_col] == bh].copy()
            bh_spt["_level"] = bh_spt[sd_dep].apply(lambda d: gl - float(d) if pd.notna(d) else None)
            if spt_level_mode == "SEAT" and pen_col:
                bh_spt["_level"] = bh_spt["_level"] - bh_spt[pen_col].fillna(0) / 1000.0
            bh_spt = bh_spt.dropna(subset=["_level"]).sort_values("_level", ascending=False)

            x_spt = x0 - 4.0

            for _, sr in bh_spt.iterrows():
                level = sr["_level"]
                n_val = sr.get(n_eff_col) or (sr.get(n_raw_col) if n_raw_col else None)
                try:
                    nd = int(float(n_val)) if pd.notna(n_val) and str(n_val).strip() else "?"
                except (ValueError, TypeError):
                    nd = "?"
                yl = y_mm(level)
                lev_s = _fmt(level)
                ax.text(x_spt, yl, f"N={nd}@{lev_s}", fontsize=P["spt_font"],
                        ha="right", va="center", fontfamily="monospace", color="#333")
                ax_x = x_spt + 0.8
                ax.annotate("", xy=(ax_x, yl + 4.0),
                            xytext=(ax_x, yl + 0.3),
                            arrowprops=dict(arrowstyle="->, head_length=0.5, head_width=0.35",
                                            color="#888", lw=0.3))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=A3_DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read(), "striplog_A3.png"


def render_striplog_a3_pages(
    layers_df: pd.DataFrame,
    spt_df: pd.DataFrame | None,
    gl_df: pd.DataFrame | None,
    borehole_ids: list[str],
    bh_per_page: int = DEFAULT_BH_PER_PAGE,
    legend_codes: list[str] | None = None,
    color_map: dict[str, str] | None = None,
    spt_level_mode: str = "TOP",
) -> list[tuple[bytes, str]]:
    """Split boreholes into groups of bh_per_page and render each as a separate A3 PNG.

    Preserves the order of borehole_ids as passed by the caller
    (caller is responsible for sorting, e.g. natural numeric order).

    Returns list of (png_bytes, filename) tuples.
    """
    bh_per_page = max(1, bh_per_page)

    chunks = [
        borehole_ids[i : i + bh_per_page]
        for i in range(0, len(borehole_ids), bh_per_page)
    ]
    results: list[tuple[bytes, str]] = []
    for page_idx, chunk in enumerate(chunks, start=1):
        png_bytes, base_name = render_striplog_a3_png(
            layers_df, spt_df, gl_df, chunk,
            legend_codes=legend_codes, color_map=color_map,
            spt_level_mode=spt_level_mode,
        )
        stem = base_name.rsplit(".", 1)[0]
        page_name = f"{stem}_page{page_idx}.png"
        results.append((png_bytes, page_name))
    return results
