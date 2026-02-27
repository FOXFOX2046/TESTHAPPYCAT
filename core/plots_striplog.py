"""
Strip log plot: multi-borehole, colored strata, Level = GL - Depth.
Dynamic Plotly version with explicit, adaptive scaling.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

# Layout constants for dynamic Plotly striplog.
SPACING_BASE = 2.2
COL_W = 1.0
LEFT_LABEL_PAD = 2.2
RIGHT_LABEL_PAD = 0.7
TOP_BUFFER = 0.3
BOTTOM_BUFFER = 0.3

# Color map (from Definition.bas + PDF Soil Legend: AGS6 Auto)
CODE_COLORS = {
    "WB": "#FFFFFF",
    "RFILL": "#FFE699",
    "FILL": "#FFFF00",
    "AL-C": "#C6E0B4",
    "AL-G": "#A9D08E",
    "MD-C": "#D9D9D9",
    "MD-G": "#BFBFBF",
    "CL": "#339733",
    "CDG": "#B57EDC",
    "HDG": "#9B59B6",
    "MDG": "#7D3C98",
    "SDG": "#5B2C6F",
    "CDT": "#D35400",
    "HDT": "#E67E22",
    "MDT": "#EB984E",
    "SDT": "#F5B041",
    "CDTUFF": "#D35400",
    "HDTUFF": "#E67E22",
    "MDTUFF": "#EB984E",
    "SDTUFF": "#F5B041",
    "HDMETATUFF": "#8E44AD",
    "MDMETATUFF": "#9B59B6",
    "CDV": "#1ABC9C",
    "HDV": "#16A085",
    "MDV": "#138D75",
    "SDV": "#117A65",
    "CDS": "#CC9900",
    "HDS": "#996600",
    "MDS": "#990000",
    "SDS": "#800000",
    "CDI": "#D297FF",
    "HDI": "#C06DFF",
    "MDI": "#A225FF",
    "SDI": "#7200C8",
    "CDM": "#66CCFF",
    "HDM": "#3399FF",
    "MDM": "#0066FF",
    "SDM": "#0000FF",
    "ALLUVIUM": "#27AE60",
    "ALLUVIUM SAND": "#2ECC71",
    "ALLUVIUM SILT": "#58D68D",
    "ALLUVIUM GRAVEL": "#1E8449",
    "MARINE_DEPOSIT": "#3498DB",
    "VEIN": "#FF0000",
    "QUARTZ VEIN": "#E74C3C",
    "OTHER": "#7B7B7B",
    "CORESTONE": "#808080",
    "UNKNOWN": "#95A5A6",
}


def build_color_map(codes: list[str]) -> dict[str, str]:
    """Build code -> hex color map, assign defaults for unknown."""
    out = {}
    for c in codes:
        c = str(c).strip()
        cu = c.upper()
        # User preferred defaults (still overrideable from Streamlit UI):
        # - FILL XX        -> #da79da
        # - ALLUVIUM XX    -> #E67E22
        # - MARINE DEPOSIT -> #63BEFF
        # - POND DEPOSIT   -> #1871B3
        # - CDXX / MDXX / SDXX families
        if "FILL" in cu:
            out[c] = "#da79da"
        elif cu.startswith("ALLUVIUM"):
            out[c] = "#E67E22"
        elif cu.startswith("MARINE DEPOSIT") or cu.startswith("MARINE_DEPOSIT"):
            out[c] = "#63BEFF"
        elif cu.startswith("POND DEPOSIT") or cu.startswith("POND_DEPOSIT"):
            out[c] = "#1871B3"
        elif cu.startswith("HD"):
            out[c] = "#00FF05"
        elif cu.startswith("CD"):
            out[c] = "#FFFF00"
        elif cu.startswith("MD"):
            out[c] = "#3ba176"
        elif cu.startswith("SD"):
            out[c] = "#4FCE99"
        elif c in CODE_COLORS:
            out[c] = CODE_COLORS[c]
        elif cu in CODE_COLORS:
            out[c] = CODE_COLORS[cu]
        else:
            h = abs(hash(c)) % 0xFFFFFF
            out[c] = f"#{h:06x}"
    return out


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _resolve_dynamic_scale(depth_range: float, n_bh: int) -> tuple[int, int, float]:
    """Return (height_px, width_px, y_dtick_m) for dynamic plot."""
    if depth_range <= 12:
        px_per_m, dtick = 34, 0.5
    elif depth_range <= 25:
        px_per_m, dtick = 28, 1.0
    elif depth_range <= 50:
        px_per_m, dtick = 22, 2.0
    elif depth_range <= 100:
        px_per_m, dtick = 16, 5.0
    elif depth_range <= 180:
        px_per_m, dtick = 12, 10.0
    else:
        px_per_m, dtick = 10, 20.0

    if n_bh >= 12:
        px_per_m = max(8, px_per_m - 2)

    height = int(_clamp(depth_range * px_per_m, 760, 4600))
    width = int(_clamp(360 + n_bh * (170 if n_bh <= 10 else 150), 1100, 6200))
    return height, width, dtick


def _resolve_dynamic_params(n_bh: int) -> dict[str, Any]:
    """
    Mirror static tier logic, mapped to Plotly axis units.
    """
    if n_bh <= 8:
        return dict(
            col_w=1.10, spacing=3.25, spt_pad=0.55, right_pad=1.30,
            lanes=4, lane_offset=0.18, spt_font=9, bnd_font=9,
            header_font=11, gl_font=10, min_gap_m=1.2,
            bnd_decimals=2, bnd_right=0.22, spt_format="N={n} @ {lev}",
            header_max_chars=99, tri_size=9, left_panel=1.8, legend_font=9,
        )
    if n_bh <= 15:
        return dict(
            col_w=1.00, spacing=2.85, spt_pad=0.50, right_pad=1.20,
            lanes=2, lane_offset=0.14, spt_font=8, bnd_font=8,
            header_font=9, gl_font=8, min_gap_m=1.8,
            bnd_decimals=2, bnd_right=0.20, spt_format="N={n}@{lev}",
            header_max_chars=16, tri_size=8, left_panel=1.6, legend_font=8,
        )
    return dict(
        col_w=0.90, spacing=2.45, spt_pad=0.45, right_pad=1.00,
        lanes=2, lane_offset=0.12, spt_font=7, bnd_font=7,
        header_font=8, gl_font=7, min_gap_m=2.0,
        bnd_decimals=1, bnd_right=0.16, spt_format="N={n}@{lev}",
        header_max_chars=12, tri_size=7, left_panel=1.3, legend_font=7,
    )


def debug_scale_summary(
    top: float, bottom: float, depth_range: float, height: int,
    spacing: float, col_w: float, n_bh: int
) -> dict[str, Any]:
    """Return dict of scale params for quick self-check."""
    return {
        "top": top,
        "bottom": bottom,
        "depth_range": depth_range,
        "height": height,
        "spacing": spacing,
        "col_w": col_w,
        "n_bh": n_bh,
    }


def plot_striplog(
    layers_df: pd.DataFrame,
    spt_df: pd.DataFrame | None,
    gl_df: pd.DataFrame | None,
    borehole_ids: list[str] | None = None,
    color_override: dict[str, str] | None = None,
    return_debug: bool = False,
    spt_level_mode: str = "TOP",
) -> go.Figure | tuple[go.Figure, dict[str, Any]]:
    """
    Rebuilt dynamic Plotly strip log.
    Level = GL - Depth, explicit axis range, adaptive annotation density.
    """
    bh_col = _col(layers_df, ["borehole", "loca", "hole"])
    from_col = _col(layers_df, ["from", "top"])
    to_col = _col(layers_df, ["to", "base", "btm"])
    code_col = _col(layers_df, ["normalized", "code"])

    if borehole_ids is None:
        borehole_ids = layers_df[bh_col].dropna().astype(str).str.strip().unique().tolist()
    borehole_ids = [str(bh).strip() for bh in borehole_ids if str(bh).strip()]

    if not borehole_ids or len(layers_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No borehole data to display",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16),
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=60, r=60), height=300)
        if return_debug:
            return fig, {"top": 0, "bottom": 0, "depth_range": 0, "height": 300, "spacing": 0, "col_w": 0, "n_bh": 0}
        return fig

    layers_sel = layers_df[layers_df[bh_col].astype(str).str.strip().isin(borehole_ids)].copy()
    if layers_sel.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No matching borehole rows",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16),
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=60, r=60), height=300)
        if return_debug:
            return fig, {"top": 0, "bottom": 0, "depth_range": 0, "height": 300, "spacing": 0, "col_w": 0, "n_bh": 0}
        return fig

    gl_map: dict[str, float] = {}
    if gl_df is not None and len(gl_df) > 0:
        gh = _col(gl_df, ["borehole", "loca", "hole"])
        gg = _col(gl_df, ["gl", "ground", "level", "elev"])
        for _, r in gl_df.iterrows():
            key = str(r[gh]).strip()
            if not key:
                continue
            try:
                gl_map[key] = float(r[gg]) if pd.notna(r.get(gg)) else 0.0
            except (ValueError, TypeError):
                gl_map[key] = 0.0

    codes = layers_sel[code_col].dropna().astype(str).str.strip().unique().tolist()
    colors = build_color_map(codes)
    if color_override:
        for k, v in color_override.items():
            if v and (k in colors or k in [str(c).strip() for c in codes]):
                colors[k] = str(v).strip()

    # Vertical extent from static-style logic:
    # top from GL only; bottom from layer bottoms and SPT levels.
    top_candidates: list[float] = [gl_map.get(bh, 0.0) for bh in borehole_ids]
    bottom_candidates: list[float] = []
    spt_bh_col = _col(spt_df, ["borehole", "loca", "hole"]) if spt_df is not None and len(spt_df) > 0 else None
    spt_dep_col = _col(spt_df, ["depth"]) if spt_df is not None and len(spt_df) > 0 else None
    spt_pen_col = "Penetration_mm" if spt_df is not None and "Penetration_mm" in spt_df.columns else None

    for bh in borehole_ids:
        gl = gl_map.get(bh, 0.0)
        grp = layers_sel[layers_sel[bh_col].astype(str).str.strip() == bh]
        for _, row in grp.iterrows():
            try:
                f = float(row[from_col])
                t = float(row[to_col])
                if f >= t:
                    continue
                bottom_candidates.append(gl - t)
            except (ValueError, TypeError):
                pass
        if spt_df is not None and spt_bh_col and spt_dep_col:
            bh_spt = spt_df[spt_df[spt_bh_col].astype(str).str.strip() == bh]
            for _, sr in bh_spt.iterrows():
                try:
                    lev = gl - float(sr[spt_dep_col])
                    if spt_level_mode == "SEAT" and spt_pen_col:
                        lev -= float(sr.get(spt_pen_col, 0) or 0) / 1000.0
                    bottom_candidates.append(lev)
                except (ValueError, TypeError):
                    pass

    global_top = max(top_candidates) if top_candidates else 5.0
    global_bottom = min(bottom_candidates) if bottom_candidates else global_top - 21.0
    top = global_top + 1.0
    bottom = global_bottom - 1.0
    depth_range = max(0.5, top - bottom)

    n_bh = len(borehole_ids)
    P = _resolve_dynamic_params(n_bh)
    height, width, y_dtick = _resolve_dynamic_scale(depth_range, n_bh)
    spacing = float(P["spacing"])
    col_w = float(P["col_w"])
    x_min = -float(P["left_panel"]) - float(P["spt_pad"]) - 0.6
    x_max = (n_bh - 1) * spacing + col_w + float(P["right_pad"])
    min_spt_gap = float(P["min_gap_m"])
    fs_base = int(P["header_font"])
    fs_anno = int(P["bnd_font"])
    bnd_decimals = int(P["bnd_decimals"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[bottom, top],
        mode="markers",
        marker=dict(size=0, opacity=0, color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=1.12,
        text=f"TopBound Level (mPD) {top - 1.0:.2f}<br>LowBound Level (mPD) {bottom + 1.0:.2f}",
        showarrow=False,
        font=dict(size=8, color="#333"),
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        borderpad=4,
        align="left",
    )

    for bh_idx, bh in enumerate(borehole_ids):
        gl = gl_map.get(bh, 0.0)
        grp = layers_sel[layers_sel[bh_col].astype(str).str.strip() == bh].sort_values(from_col)

        x0 = bh_idx * spacing
        x1 = x0 + col_w
        x_center = (x0 + x1) / 2.0
        x_spt_base = x0 - max(0.35, min(0.75, float(P["spt_pad"])))
        x_bnd = x1 + float(P["bnd_right"])

        header_y = min(global_top + 0.5, top - 0.5)
        gl_str = f"+{gl:.2f}" if gl >= 0 else f"{gl:.2f}"
        bh_str = str(bh)
        if len(bh_str) > int(P["header_max_chars"]):
            bh_str = bh_str[: int(P["header_max_chars"])] + ".."
        fig.add_annotation(
            x=x_center, y=header_y, text=bh_str,
            xref="x", yref="y", showarrow=False,
            font=dict(size=int(P["header_font"]), color="#111"),
            bgcolor="rgba(0,0,0,0)",
            borderpad=2,
        )
        fig.add_annotation(
            x=x_center, y=header_y - 0.35, text=f"G.L{gl_str}",
            xref="x", yref="y", showarrow=False,
            font=dict(size=int(P["gl_font"]), color="#111"),
            bgcolor="rgba(0,0,0,0)",
            borderpad=2,
        )

        fig.add_shape(
            type="line",
            x0=x0 - 0.1, x1=x1 + 0.1, y0=gl, y1=gl,
            line=dict(color="#222", width=1.0),
        )

        boundaries: set[float] = {round(gl, 2)}
        for _, row in grp.iterrows():
            try:
                f = float(row[from_col])
                t = float(row[to_col])
            except (ValueError, TypeError):
                continue
            if f >= t:
                continue

            code = str(row[code_col]).strip()
            level_top = gl - f
            level_btm = gl - t
            fig.add_shape(
                type="rect",
                x0=x0, y0=level_btm, x1=x1, y1=level_top,
                fillcolor=colors.get(code, "#888888"),
                line=dict(color="#444", width=0.8),
                layer="below",
            )
            boundaries.add(round(level_top, 2))
            boundaries.add(round(level_btm, 2))

        # Static logic: show every boundary after de-dup to configured decimal.
        boundaries_all = sorted(set(round(b, bnd_decimals) for b in boundaries), reverse=True)
        for lev in boundaries_all:
            if bnd_decimals == 1:
                level_str = f"{lev:+.1f}" if lev >= 0 else f"{lev:.1f}"
            else:
                level_str = f"{lev:+.2f}" if lev >= 0 else f"{lev:.2f}"
            fig.add_annotation(
                x=x_bnd, y=lev, text=level_str,
                xref="x", yref="y", showarrow=False,
                font=dict(size=int(P["bnd_font"]), color="#111"),
                bgcolor="rgba(0,0,0,0)",
                borderpad=2,
                xanchor="left",
            )
            fig.add_annotation(
                x=x_center, y=lev, text="v",
                xref="x", yref="y", showarrow=False,
                font=dict(size=int(P["tri_size"]), color="black"),
            )

        if spt_df is not None and len(spt_df) > 0:
            sd_col = _col(spt_df, ["borehole", "loca", "hole"])
            sd_dep = _col(spt_df, ["depth"])
            n_eff_col = _col(spt_df, ["n_effective"])
            n_raw_col = "N_raw" if "N_raw" in spt_df.columns else None
            pen_col = "Penetration_mm" if "Penetration_mm" in spt_df.columns else None
            bh_spt = spt_df[spt_df[sd_col].astype(str).str.strip() == bh].copy()
            bh_spt["_level"] = bh_spt[sd_dep].apply(lambda d: gl - float(d) if pd.notna(d) else None)
            if spt_level_mode == "SEAT" and pen_col:
                bh_spt["_level"] = bh_spt["_level"] - bh_spt[pen_col].fillna(0) / 1000.0
            bh_spt = bh_spt.dropna(subset=["_level"]).sort_values("_level", ascending=False)

            last_y = None
            for _, sr in bh_spt.iterrows():
                level = sr["_level"]
                if last_y is not None and abs(level - last_y) < min_spt_gap:
                    continue
                n_val = sr.get(n_eff_col, None) if n_eff_col else None
                if (pd.isna(n_val) or str(n_val).strip() in ("", "nan")) and n_raw_col:
                    n_val = sr.get(n_raw_col, None)
                try:
                    n_display = int(float(n_val)) if pd.notna(n_val) and str(n_val).strip() not in ("", "nan") else "?"
                except (ValueError, TypeError):
                    n_display = "?"

                last_y = level
                if bnd_decimals == 1:
                    level_str = f"{level:+.1f}" if level >= 0 else f"{level:.1f}"
                else:
                    level_str = f"{level:+.2f}" if level >= 0 else f"{level:.2f}"
                # Keep all SPT labels on one vertical line (no lane staggering).
                x_spt = x_spt_base
                spt_txt = str(P["spt_format"]).format(n=n_display, lev=level_str)
                fig.add_annotation(
                    x=x_spt, y=level,
                    text=spt_txt,
                    xref="x", yref="y",
                    showarrow=False,
                    font=dict(size=int(P["spt_font"]), color="#222"),
                    bgcolor="rgba(0,0,0,0)",
                    borderpad=2,
                    xanchor="right",
                )
                # Draw SPT arrow with line segments so it aligns cleanly with text.
                x_arrow = x_spt + 0.10
                y_top = level - 0.02
                y_tip = level - 0.30
                y_head = level - 0.22
                dx_head = 0.03
                fig.add_trace(go.Scatter(
                    x=[x_arrow, x_arrow, None, x_arrow, x_arrow - dx_head, None, x_arrow, x_arrow + dx_head],
                    y=[y_top, y_tip, None, y_tip, y_head, None, y_tip, y_head],
                    mode="lines",
                    line=dict(color="#666", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    show_legend = True
    for code in sorted(codes, key=lambda c: str(c).upper()):
        c = str(code).strip()
        fig.add_trace(go.Scatter(
            x=[x_min - 0.35], y=[bottom],
            mode="markers",
            marker=dict(size=8, color=colors.get(c, "#888888"), symbol="square", line=dict(width=1, color="#333")),
            name=c,
            showlegend=show_legend,
            legendgroup="soil",
        ))


    fig.update_layout(
        template="plotly_white",
        colorway=list(dict.fromkeys(CODE_COLORS.values())),
        title=dict(text="Borehole Log", font=dict(size=fs_base + 2)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        autosize=False,
        width=width,
        xaxis=dict(
            showticklabels=False,
            zeroline=False,
            range=[x_min, x_max],
            showgrid=False,
            autorange=False,
        ),
        yaxis=dict(
            title="Level (mPD)",
            range=[bottom, top],
            autorange=False,
            dtick=y_dtick,
            gridcolor="rgba(0,0,0,0.1)",
            zerolinecolor="#ccc",
            showline=True,
            linecolor="#666",
            tickfont=dict(size=fs_base),
        ),
        legend=dict(
            title=dict(text="Soil Legend"),
            orientation="h",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=-0.02,
            entrywidth=80,
            entrywidthmode="pixels",
            itemsizing="constant",
            font=dict(size=8, color="#222"),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
        margin=dict(l=100, r=40, t=60, b=60),
        height=height,
        hovermode="closest",
    )

    # Static-like horizontal guides every 10m.
    for lv in range(int(bottom), int(top) + 1, 10):
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            xref="paper",
            y0=float(lv),
            y1=float(lv),
            yref="y",
            line=dict(color="lightgray", width=0.5),
            layer="below",
        )

    debug = debug_scale_summary(top, bottom, depth_range, height, spacing, col_w, n_bh)
    if return_debug:
        return fig, debug
    return fig


def _col(df: pd.DataFrame, keywords: list[str]) -> str:
    for c in df.columns:
        for kw in keywords:
            if kw.upper() in str(c).upper():
                return c
    return df.columns[0]
