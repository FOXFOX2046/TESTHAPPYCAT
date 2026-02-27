"""
SPT N vs Depth plot (optional).
"""
from __future__ import annotations

import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_VIVID_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf",
]


def _pick_col(df: pd.DataFrame, candidates: list[str], fallback: str | None = None) -> str | None:
    """Pick first existing column from candidates (case-insensitive)."""
    name_map = {str(c).upper(): c for c in df.columns}
    for c in candidates:
        cu = c.upper()
        if cu in name_map:
            return name_map[cu]
    if fallback and fallback in df.columns:
        return fallback
    return None


def _attach_soil_type_from_layers(
    spt_df: pd.DataFrame,
    layers_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach Soil_Type to each SPT row by interval matching in merged layers."""
    out = spt_df.copy()
    out["Soil_Type"] = "UNCLASSIFIED"
    if layers_df is None or layers_df.empty:
        return out

    bh_col = _pick_col(out, ["Borehole_ID", "HOLE_ID", "LOCA_ID"])
    dep_col = _pick_col(out, ["Depth", "ISPT_TOP", "ISPT_DPTH"])
    if not bh_col or not dep_col:
        return out

    layer_bh = _pick_col(layers_df, ["Borehole_ID", "HOLE_ID", "LOCA_ID"])
    layer_from = _pick_col(layers_df, ["Depth_From", "GEOL_TOP", "FROM"])
    layer_to = _pick_col(layers_df, ["Depth_To", "GEOL_BASE", "TO"])
    layer_code = _pick_col(layers_df, ["Normalized_Code", "Code"], fallback="Normalized_Code")
    if not layer_bh or not layer_from or not layer_to or not layer_code:
        return out

    for bh, spt_idx in out.groupby(bh_col).groups.items():
        lyr = layers_df[layers_df[layer_bh].astype(str) == str(bh)]
        if lyr.empty:
            continue
        lyr = lyr.sort_values(layer_from).reset_index(drop=True)
        for idx in spt_idx:
            d = out.at[idx, dep_col]
            if pd.isna(d):
                continue
            hit = lyr[(lyr[layer_from] <= d) & (d < lyr[layer_to])]
            if hit.empty:
                hit = lyr[(lyr[layer_from] <= d) & (d <= lyr[layer_to])]
            if not hit.empty:
                code = str(hit.iloc[0][layer_code]).strip()
                out.at[idx, "Soil_Type"] = code if code else "UNCLASSIFIED"

    return out


def prepare_spt_with_soil_type(
    spt_df: pd.DataFrame,
    merged_layers_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return SPT dataframe with Soil_Type attached from merged layers."""
    return _attach_soil_type_from_layers(spt_df, merged_layers_df)


def _linear_regression(x_vals: list[float], y_vals: list[float]) -> tuple[float, float]:
    """Least-squares fit: y = slope*x + intercept."""
    n = len(x_vals)
    if n == 0:
        return 0.0, 0.0
    sx = sum(x_vals)
    sy = sum(y_vals)
    sxy = sum(x * y for x, y in zip(x_vals, y_vals))
    sxx = sum(x * x for x in x_vals)
    den = n * sxx - sx * sx
    if den == 0:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _color_map_for(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i, v in enumerate(values):
        out[str(v)] = _VIVID_COLORS[i % len(_VIVID_COLORS)]
    return out


def _round_half_away_from_zero(x: float) -> int:
    """Stable rounding: .5 always goes away from zero (not banker's rounding)."""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def _enforce_bounds_order(
    x_low: list[float],
    x_high: list[float],
) -> tuple[list[float], list[float]]:
    """Ensure lower bound is always <= upper bound at each level point."""
    low_out: list[float] = []
    high_out: list[float] = []
    for lo, hi in zip(x_low, x_high):
        a = float(lo)
        b = float(hi)
        if a <= b:
            low_out.append(a)
            high_out.append(b)
        else:
            low_out.append(b)
            high_out.append(a)
    return low_out, high_out


def _build_spt_bounds(
    df: pd.DataFrame,
    level_col: str,
    n_col: str,
    bin_size: float = 1.0,
    p_low: float = 0.05,
    p_high: float = 0.95,
    min_level: float | None = None,
    max_level: float | None = None,
    min_n: float | None = None,
    max_n: float | None = None,
    exclude_n200: bool = True,
) -> tuple[list[float], list[float], list[float], list[float], float, float] | None:
    """
    VBA AddSPTNBounds equivalent:
    - Bin by level
    - Compute low/high percentile per bin (exclude N>=200)
    - Fit regression lines x(level) for low/high and design midpoint.
    Returns (levels, x_low_fit, x_high_fit, x_design, m_design, b_design).
    """
    if df.empty:
        return None
    bins: dict[float, list[float]] = {}
    for _, r in df.iterrows():
        lev = r[level_col]
        spt = r[n_col]
        if pd.isna(lev) or pd.isna(spt):
            continue
        # Zone limits are inclusive.
        if min_level is not None and lev < min_level:
            continue
        if max_level is not None and lev > max_level:
            continue
        if min_n is not None and float(spt) < float(min_n):
            continue
        if max_n is not None and float(spt) > float(max_n):
            continue
        if exclude_n200 and float(spt) >= 200:
            continue
        key = _round_half_away_from_zero(float(lev) / float(bin_size)) * float(bin_size)
        bins.setdefault(key, []).append(float(spt))

    if len(bins) < 2:
        return None

    levels = sorted(bins.keys())
    x_low = [pd.Series(bins[k]).quantile(p_low, interpolation="linear") for k in levels]
    x_high = [pd.Series(bins[k]).quantile(p_high, interpolation="linear") for k in levels]
    x_low, x_high = _enforce_bounds_order(x_low, x_high)

    # Fit x = m*level + b (same orientation as VBA logic).
    m_low, b_low = _linear_regression(levels, x_low)
    m_high, b_high = _linear_regression(levels, x_high)
    x_low_fit = [m_low * lv + b_low for lv in levels]
    x_high_fit = [m_high * lv + b_high for lv in levels]
    x_low_fit, x_high_fit = _enforce_bounds_order(x_low_fit, x_high_fit)
    x_design = [(a + b) / 2.0 for a, b in zip(x_low_fit, x_high_fit)]
    m_design, b_design = _linear_regression(levels, x_design)
    return levels, x_low_fit, x_high_fit, x_design, m_design, b_design


def compute_spt_design_equation(
    spt_df: pd.DataFrame,
    merged_layers_df: pd.DataFrame | None = None,
    selected_boreholes: list[str] | None = None,
    selected_soil_types: list[str] | None = None,
    bounds_bin_size: float = 1.0,
    bounds_p_low: float = 0.05,
    bounds_p_high: float = 0.95,
    bounds_y_max: float | None = None,
    bounds_y_min: float | None = None,
    bounds_x_min: float | None = None,
    bounds_x_max: float | None = None,
) -> tuple[float, float] | None:
    """Return auto-calculated (m, c) for design line N = c + mZ."""
    bounds_eq = compute_spt_bounds_equations(
        spt_df=spt_df,
        merged_layers_df=merged_layers_df,
        selected_boreholes=selected_boreholes,
        selected_soil_types=selected_soil_types,
        bounds_bin_size=bounds_bin_size,
        bounds_p_low=bounds_p_low,
        bounds_p_high=bounds_p_high,
        bounds_y_max=bounds_y_max,
        bounds_y_min=bounds_y_min,
        bounds_x_min=bounds_x_min,
        bounds_x_max=bounds_x_max,
    )
    if bounds_eq is None:
        return None
    _, _, m_design, c_design, _, _ = bounds_eq
    return float(m_design), float(c_design)


def compute_spt_bounds_equations(
    spt_df: pd.DataFrame,
    merged_layers_df: pd.DataFrame | None = None,
    selected_boreholes: list[str] | None = None,
    selected_soil_types: list[str] | None = None,
    bounds_bin_size: float = 1.0,
    bounds_p_low: float = 0.05,
    bounds_p_high: float = 0.95,
    bounds_y_max: float | None = None,
    bounds_y_min: float | None = None,
    bounds_x_min: float | None = None,
    bounds_x_max: float | None = None,
) -> tuple[float, float, float, float, float, float] | None:
    """Return auto (m,c) for lower/design/upper lines."""
    if spt_df is None or spt_df.empty:
        return None

    df = prepare_spt_with_soil_type(spt_df, merged_layers_df)
    n_col = _pick_col(df, ["N_effective", "N", "ISPT_NVAL"], fallback="N_effective")
    y_col = _pick_col(df, ["Level", "Level_Top", "Depth"], fallback="Depth")
    bh_col = _pick_col(df, ["Borehole_ID", "HOLE_ID", "LOCA_ID"], fallback="Borehole_ID")
    if not n_col or not y_col or not bh_col:
        return None

    if selected_boreholes is not None:
        wanted_bh = {str(v) for v in selected_boreholes}
        df = df[df[bh_col].astype(str).isin(wanted_bh)]
    if selected_soil_types is not None:
        wanted_soil = {str(v) for v in selected_soil_types}
        df = df[df["Soil_Type"].astype(str).isin(wanted_soil)]

    df = df.dropna(subset=[n_col, y_col])
    if bounds_x_min is not None:
        df = df[pd.to_numeric(df[n_col], errors="coerce") >= float(bounds_x_min)]
    if bounds_x_max is not None:
        df = df[pd.to_numeric(df[n_col], errors="coerce") <= float(bounds_x_max)]
    if df.empty:
        return None

    min_level = None
    max_level = None
    if bounds_y_max is not None and bounds_y_min is not None:
        min_level = min(bounds_y_min, bounds_y_max)
        max_level = max(bounds_y_min, bounds_y_max)

    bounds = _build_spt_bounds(
        df=df,
        level_col=y_col,
        n_col=n_col,
        bin_size=bounds_bin_size,
        p_low=bounds_p_low,
        p_high=bounds_p_high,
        min_level=min_level,
        max_level=max_level,
        min_n=bounds_x_min,
        max_n=bounds_x_max,
        exclude_n200=True,
    )
    if bounds is None:
        bounds = _build_spt_bounds(
            df=df,
            level_col=y_col,
            n_col=n_col,
            bin_size=max(bounds_bin_size * 2.0, bounds_bin_size + 0.5),
            p_low=bounds_p_low,
            p_high=bounds_p_high,
            min_level=min_level,
            max_level=max_level,
            min_n=bounds_x_min,
            max_n=bounds_x_max,
            exclude_n200=True,
        )
    if bounds is None:
        return None

    levels, x_low_fit, x_high_fit, x_design, m_design, b_design = bounds
    m_low, c_low = _linear_regression(levels, x_low_fit)
    m_high, c_high = _linear_regression(levels, x_high_fit)
    return (
        float(m_low),
        float(c_low),
        float(m_design),
        float(b_design),
        float(m_high),
        float(c_high),
    )


def plot_spt_vs_level_panels(
    spt_df: pd.DataFrame,
    merged_layers_df: pd.DataFrame | None = None,
    selected_boreholes: list[str] | None = None,
    selected_soil_types: list[str] | None = None,
    x_min: float = 0.0,
    x_max: float = 200.0,
    x_dtick: float | None = None,
    marker_symbol: str = "circle",
    marker_size: float = 7.0,
    soil_color_map: dict[str, str] | None = None,
    y_top: float | None = None,
    y_bottom: float | None = None,
    y_dtick: float | None = None,
    show_bounds: bool = False,
    bounds_bin_size: float = 1.0,
    bounds_p_low: float = 0.05,
    bounds_p_high: float = 0.95,
    bounds_y_max: float | None = None,
    bounds_y_min: float | None = None,
    bounds_zones: list[tuple[float, float]] | None = None,
    bounds_zone_bin_sizes: list[float] | None = None,
    bounds_x_min: float | None = None,
    bounds_x_max: float | None = None,
    zone_manual_params: list[dict[str, float]] | None = None,
    design_equation_m: float | None = None,
    design_equation_c: float | None = None,
    lower_bound_m: float | None = None,
    lower_bound_c: float | None = None,
    upper_bound_m: float | None = None,
    upper_bound_c: float | None = None,
    bounds_line_color: str = "#808080",
    design_line_color: str = "#000080",
) -> go.Figure:
    """
    Dual-panel SPT scatter:
    - Left: SPTN vs Level grouped by Borehole_ID
    - Right: SPTN vs Level grouped by Soil_Type
    """
    if spt_df is None or spt_df.empty:
        return go.Figure()

    df = prepare_spt_with_soil_type(spt_df, merged_layers_df)

    n_col = _pick_col(df, ["N_effective", "N", "ISPT_NVAL"], fallback="N_effective")
    y_col = _pick_col(df, ["Level", "Level_Top", "Depth"], fallback="Depth")
    bh_col = _pick_col(df, ["Borehole_ID", "HOLE_ID", "LOCA_ID"], fallback="Borehole_ID")
    if not n_col or not y_col or not bh_col:
        return go.Figure()

    if selected_boreholes is not None:
        wanted_bh = {str(v) for v in selected_boreholes}
        df = df[df[bh_col].astype(str).isin(wanted_bh)]
    if selected_soil_types is not None:
        wanted_soil = {str(v) for v in selected_soil_types}
        df = df[df["Soil_Type"].astype(str).isin(wanted_soil)]

    df = df.dropna(subset=[n_col, y_col])
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.08,
        subplot_titles=("SPTN vs Level (by Borehole)", "SPTN vs Level (by Soil_Type)"),
    )

    bh_keys = sorted(df[bh_col].astype(str).unique(), key=str.upper)
    soil_keys = sorted(df["Soil_Type"].astype(str).unique(), key=str.upper)
    bh_colors = _color_map_for(bh_keys)
    soil_colors = _color_map_for(soil_keys)
    if soil_color_map:
        for s in soil_keys:
            ss = str(s)
            if ss in soil_color_map and str(soil_color_map[ss]).strip():
                soil_colors[ss] = str(soil_color_map[ss]).strip()

    for bh in bh_keys:
        part = df[df[bh_col].astype(str) == bh]
        fig.add_trace(
            go.Scatter(
                x=part[n_col],
                y=part[y_col],
                mode="markers",
                name=bh,
                legendgroup=f"BH::{bh}",
                marker=dict(
                    size=marker_size,
                    symbol=marker_symbol,
                    color=bh_colors.get(str(bh)),
                    line=dict(width=0.5, color=bh_colors.get(str(bh))),
                ),
                hovertemplate=f"Borehole: {bh}<br>SPTN: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    for soil in soil_keys:
        part = df[df["Soil_Type"].astype(str) == soil]
        fig.add_trace(
            go.Scatter(
                x=part[n_col],
                y=part[y_col],
                mode="markers",
                name=soil,
                legendgroup=f"SOIL::{soil}",
                marker=dict(
                    size=marker_size,
                    symbol=marker_symbol,
                    color=soil_colors.get(str(soil)),
                    line=dict(width=0.5, color=soil_colors.get(str(soil))),
                ),
                hovertemplate=f"Soil_Type: {soil}<br>SPTN: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    eq_zone_tags: list[tuple[str, list[tuple[float, float]]]] = []

    zone_pairs: list[tuple[float, float]] = []
    if bounds_zones:
        for z in bounds_zones:
            try:
                zone_pairs.append((float(z[0]), float(z[1])))
            except Exception:
                continue
    elif bounds_y_max is not None and bounds_y_min is not None:
        zone_pairs.append((float(bounds_y_max), float(bounds_y_min)))

    if show_bounds:
        bounds_df = df
        if bounds_x_min is not None:
            bounds_df = bounds_df[pd.to_numeric(bounds_df[n_col], errors="coerce") >= float(bounds_x_min)]
        if bounds_x_max is not None:
            bounds_df = bounds_df[pd.to_numeric(bounds_df[n_col], errors="coerce") <= float(bounds_x_max)]
        if not zone_pairs:
            if y_top is not None and y_bottom is not None:
                zone_pairs = [(float(y_top), float(y_bottom))]
            else:
                zone_pairs = []

        any_zone_drawn = False
        for zone_idx, (z_top, z_bottom) in enumerate(zone_pairs):
            y_max_val = float(max(z_top, z_bottom))
            y_min_val = float(min(z_top, z_bottom))
            zone_bin = float(bounds_bin_size)
            if bounds_zone_bin_sizes and zone_idx < len(bounds_zone_bin_sizes):
                try:
                    zb = float(bounds_zone_bin_sizes[zone_idx])
                    if zb > 0:
                        zone_bin = zb
                except Exception:
                    pass
            for row_no in (1, 2):
                fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[y_max_val, y_max_val],
                        mode="lines",
                        name=f"YMAX zone Z{zone_idx + 1}",
                        line=dict(color="#1f77b4", width=1.5, dash="dash"),
                        showlegend=(row_no == 1 and zone_idx == 0),
                        legendgroup="SPT_ZONE",
                        hovertemplate=f"YMAX zone: {y_max_val:.2f} mPD<extra></extra>",
                    ),
                    row=row_no,
                    col=1,
                )
                fig.add_annotation(
                    xref=f"x{row_no}",
                    yref=f"y{row_no}",
                    x=float(x_max),
                    y=y_max_val,
                    xanchor="right",
                    yanchor="middle",
                    xshift=-6,
                    showarrow=False,
                    text=f"Y MAX ZONE LINE LEVEL @ {y_max_val:.2f}",
                    font=dict(size=10, color="#1f77b4"),
                    bgcolor="rgba(255,255,255,0.75)",
                    bordercolor="#cfcfcf",
                    borderwidth=1,
                )
                fig.add_annotation(
                    xref=f"x{row_no}",
                    yref=f"y{row_no}",
                    x=float(x_max),
                    y=y_min_val,
                    xanchor="right",
                    yanchor="middle",
                    xshift=-6,
                    showarrow=False,
                    text=f"Y MIN ZONE LINE LEVEL @ {y_min_val:.2f}",
                    font=dict(size=10, color="#1f77b4"),
                    bgcolor="rgba(255,255,255,0.75)",
                    bordercolor="#cfcfcf",
                    borderwidth=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[y_min_val, y_min_val],
                        mode="lines",
                        name=f"YMIN zone Z{zone_idx + 1}",
                        line=dict(color="#1f77b4", width=1.5, dash="dash"),
                        showlegend=(row_no == 1 and zone_idx == 0),
                        legendgroup="SPT_ZONE",
                        hovertemplate=f"YMIN zone: {y_min_val:.2f} mPD<extra></extra>",
                    ),
                    row=row_no,
                    col=1,
                )

            bounds = _build_spt_bounds(
                df=bounds_df,
                level_col=y_col,
                n_col=n_col,
                bin_size=zone_bin,
                p_low=bounds_p_low,
                p_high=bounds_p_high,
                min_level=y_min_val,
                max_level=y_max_val,
                min_n=bounds_x_min,
                max_n=bounds_x_max,
                exclude_n200=True,
            )
            if bounds is None:
                bounds = _build_spt_bounds(
                    df=bounds_df,
                    level_col=y_col,
                    n_col=n_col,
                    bin_size=max(zone_bin * 2.0, zone_bin + 0.5),
                    p_low=bounds_p_low,
                    p_high=bounds_p_high,
                    min_level=y_min_val,
                    max_level=y_max_val,
                    min_n=bounds_x_min,
                    max_n=bounds_x_max,
                    exclude_n200=True,
                )
            if bounds is None:
                continue

            any_zone_drawn = True
            levels, x_low_fit, x_high_fit, x_design, m_design, b_design = bounds
            zone_manual = None
            if zone_manual_params and zone_idx < len(zone_manual_params):
                zone_manual = zone_manual_params[zone_idx]

            use_zone_manual_design = (
                zone_manual is not None
                and zone_manual.get("design_m") is not None
                and zone_manual.get("design_c") is not None
            )
            use_global_manual_design = (
                design_equation_m is not None
                and design_equation_c is not None
                and len(zone_pairs) <= 1
            )

            if use_zone_manual_design:
                m_design_adj = float(zone_manual["design_m"])
                b_design_adj = float(zone_manual["design_c"])
            elif use_global_manual_design:
                m_design_adj = float(design_equation_m)
                b_design_adj = float(design_equation_c)
            else:
                m_design_adj = float(m_design)
                b_design_adj = float(b_design)
            eq_text = f"N = {b_design_adj:.3f} + {abs(m_design_adj):.3f} Z"

            line_levels = [y_min_val, y_max_val]
            m_low, b_low = _linear_regression(levels, x_low_fit)
            m_high, b_high = _linear_regression(levels, x_high_fit)
            x_low_fit = [m_low * lv + b_low for lv in line_levels]
            x_high_fit = [m_high * lv + b_high for lv in line_levels]
            x_design = [m_design_adj * lv + b_design_adj for lv in line_levels]

            if use_zone_manual_design or use_global_manual_design:
                z_low_m = zone_manual.get("lower_m") if zone_manual else None
                z_low_c = zone_manual.get("lower_c") if zone_manual else None
                z_up_m = zone_manual.get("upper_m") if zone_manual else None
                z_up_c = zone_manual.get("upper_c") if zone_manual else None
                if z_low_m is not None and z_low_c is not None:
                    x_low_fit = [float(z_low_m) * lv + float(z_low_c) for lv in line_levels]
                elif lower_bound_m is not None and lower_bound_c is not None:
                    x_low_fit = [float(lower_bound_m) * lv + float(lower_bound_c) for lv in line_levels]
                else:
                    x_design_auto = [float(m_design) * lv + float(b_design) for lv in line_levels]
                    low_offsets = [xl - xd for xl, xd in zip(x_low_fit, x_design_auto)]
                    low_off = float(pd.Series(low_offsets).median()) if low_offsets else 0.0
                    x_low_fit = [xd + low_off for xd in x_design]
                if z_up_m is not None and z_up_c is not None:
                    x_high_fit = [float(z_up_m) * lv + float(z_up_c) for lv in line_levels]
                elif upper_bound_m is not None and upper_bound_c is not None:
                    x_high_fit = [float(upper_bound_m) * lv + float(upper_bound_c) for lv in line_levels]
                else:
                    x_design_auto = [float(m_design) * lv + float(b_design) for lv in line_levels]
                    high_offsets = [xh - xd for xh, xd in zip(x_high_fit, x_design_auto)]
                    high_off = float(pd.Series(high_offsets).median()) if high_offsets else 0.0
                    x_high_fit = [xd + high_off for xd in x_design]

            x_low_fit, x_high_fit = _enforce_bounds_order(x_low_fit, x_high_fit)
            eq_zone_tags.append((eq_text, [(float(x), float(y)) for x, y in zip(x_design, line_levels)]))
            for row_no in (1, 2):
                fig.add_trace(
                    go.Scatter(
                        x=x_low_fit, y=line_levels, mode="lines",
                        name="Lower bound",
                        line=dict(color=bounds_line_color, width=2),
                        showlegend=(row_no == 1 and zone_idx == 0),
                        legendgroup="SPT_BOUNDS",
                    ),
                    row=row_no, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_high_fit, y=line_levels, mode="lines",
                        name="Upper bound",
                        line=dict(color=bounds_line_color, width=2),
                        showlegend=(row_no == 1 and zone_idx == 0),
                        legendgroup="SPT_BOUNDS",
                    ),
                    row=row_no, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_design, y=line_levels, mode="lines",
                        name="Design profile",
                        line=dict(color=design_line_color, width=3),
                        hovertemplate=f"Design profile<br>{eq_text}<br>N: %{{x:.3f}}<br>Z: %{{y:.3f}}<extra></extra>",
                        showlegend=(row_no == 1 and zone_idx == 0),
                        legendgroup="SPT_BOUNDS",
                    ),
                    row=row_no, col=1,
                )
        if not any_zone_drawn:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.01,
                y=1.02,
                showarrow=False,
                text="Bounds not drawn: insufficient valid SPT bins after filters.",
                font=dict(size=11, color="#b00020"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#d0d0d0",
                borderwidth=1,
            )

    y_title = "Level (mPD)" if y_col.lower().startswith("level") else "Depth (m)"

    # Keep user-entered Y top/bottom semantics:
    # "top" stays at top of the plot, "bottom" stays at bottom.
    axis_y_top = float(y_top) if y_top is not None else None
    axis_y_bottom = float(y_bottom) if y_bottom is not None else None
    # Only expand axis with Function 5 zone limits when bounds are enabled.
    if show_bounds and zone_pairs and axis_y_top is not None and axis_y_bottom is not None:
        vals: list[float] = [axis_y_top, axis_y_bottom]
        for z_top, z_bottom in zone_pairs:
            vals.extend([float(z_top), float(z_bottom)])
        if axis_y_top >= axis_y_bottom:
            axis_y_top = max(vals)
            axis_y_bottom = min(vals)
        else:
            axis_y_top = min(vals)
            axis_y_bottom = max(vals)

    x_axis_kwargs = {"title_text": "SPTN", "range": [x_min, x_max]}
    if x_dtick is not None and x_dtick > 0:
        x_axis_kwargs["dtick"] = x_dtick
    fig.update_xaxes(row=1, col=1, **x_axis_kwargs)
    fig.update_xaxes(row=2, col=1, **x_axis_kwargs)
    if axis_y_top is not None and axis_y_bottom is not None:
        # Plotly y-range is [bottom, top] when we want "top" value shown at top.
        y_axis_kwargs = {
            "title_text": y_title,
            "range": [axis_y_bottom, axis_y_top],
            "autorange": False,
        }
        if y_dtick is not None and y_dtick > 0:
            y_axis_kwargs["dtick"] = y_dtick
        fig.update_yaxes(row=1, col=1, **y_axis_kwargs)
        fig.update_yaxes(row=2, col=1, **y_axis_kwargs)
    else:
        y_axis_kwargs = {"title_text": y_title, "autorange": "reversed"}
        if y_dtick is not None and y_dtick > 0:
            y_axis_kwargs["dtick"] = y_dtick
        fig.update_yaxes(row=1, col=1, **y_axis_kwargs)
        fig.update_yaxes(row=2, col=1, **y_axis_kwargs)

    # Equation tags: one tag per zone.
    if eq_zone_tags:
        x_lo = float(x_min)
        x_hi = float(x_max)
        y_lo = float(min(axis_y_top, axis_y_bottom)) if (axis_y_top is not None and axis_y_bottom is not None) else float(df[y_col].min())
        y_hi = float(max(axis_y_top, axis_y_bottom)) if (axis_y_top is not None and axis_y_bottom is not None) else float(df[y_col].max())
        xs = max(1e-9, x_hi - x_lo)
        ys = max(1e-9, y_hi - y_lo)
        for idx, (eq_text, eq_line_points) in enumerate(eq_zone_tags):
            p0 = eq_line_points[0]
            p1 = eq_line_points[-1]
            cx = p0[0] + (p1[0] - p0[0]) * 0.55
            cy = p0[1] + (p1[1] - p0[1]) * 0.55
            # Small stagger to avoid overlap when multiple zones are close.
            cy = cy + (idx % 2) * 0.02 * ys - 0.01 * ys
            cx = max(x_lo + 0.02 * xs, min(x_hi - 0.02 * xs, cx))
            cy = max(y_lo + 0.02 * ys, min(y_hi - 0.02 * ys, cy))
            x_anchor = "left" if cx <= (x_lo + x_hi) / 2 else "right"
            for xref, yref in (("x1", "y1"), ("x2", "y2")):
                fig.add_annotation(
                    xref=xref,
                    yref=yref,
                    x=cx,
                    y=cy,
                    xanchor=x_anchor,
                    yanchor="middle",
                    showarrow=False,
                    text=eq_text,
                    font=dict(size=12, color=design_line_color),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#d0d0d0",
                    borderwidth=1,
                )

    fig.update_layout(
        template="plotly_white",
        colorway=_VIVID_COLORS,
        height=1180,
        margin=dict(l=20, r=20, t=70, b=70),
    )
    return fig


def plot_spt(
    spt_df: pd.DataFrame,
    borehole_id: str | None = None,
) -> go.Figure:
    """SPT N vs Depth scatter, N200 highlighted."""
    if "Borehole_ID" not in spt_df.columns:
        bh_col = [c for c in spt_df.columns if "borehole" in str(c).lower() or "hole" in str(c).lower()]
        bh_col = bh_col[0] if bh_col else spt_df.columns[0]
    else:
        bh_col = "Borehole_ID"

    depth_col = [c for c in spt_df.columns if "depth" in str(c).lower()][0] if any("depth" in str(c).lower() for c in spt_df.columns) else "Depth"
    n_col = [c for c in spt_df.columns if "n_" in str(c).lower() or c == "N"][0] if any("n" in str(c).lower() for c in spt_df.columns) else "N_effective"
    flag_col = "Flag" if "Flag" in spt_df.columns else None

    plot_df = spt_df.copy()
    if borehole_id:
        plot_df = plot_df[plot_df[bh_col] == borehole_id]

    plot_df = plot_df.dropna(subset=[depth_col, n_col])
    if len(plot_df) == 0:
        return go.Figure()

    plot_df["_is_n200"] = plot_df[flag_col] == "N200" if flag_col else (plot_df[n_col] >= 200)

    fig = go.Figure()
    normal = plot_df[~plot_df["_is_n200"]]
    n200 = plot_df[plot_df["_is_n200"]]
    if len(normal) > 0:
        fig.add_trace(go.Scatter(x=normal[n_col], y=normal[depth_col], mode="markers", name="SPT N", marker=dict(color="blue")))
    if len(n200) > 0:
        fig.add_trace(go.Scatter(x=n200[n_col], y=n200[depth_col], mode="markers", name="N200", marker=dict(color="red", symbol="diamond")))

    fig.update_layout(
        title=f"SPT N vs Depth" + (f" - {borehole_id}" if borehole_id else ""),
        xaxis_title="N (blows)",
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),
    )
    return fig
