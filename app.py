"""
FoxGI WebApp v1 - Streamlit UI for AGS/CSV processing.
"""
from __future__ import annotations

import io
import inspect
import json
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st


def _natural_sort_key(s: str):
    """Sort key that orders numeric parts numerically: BH1, BH3, BH9, BH11..."""
    return [int(tok) if tok.isdigit() else tok.upper() for tok in re.split(r"(\d+)", str(s))]


def _safe_key(s: str) -> str:
    """Convert free text to a stable Streamlit key suffix."""
    k = re.sub(r"[^0-9A-Za-z_]+", "_", str(s)).strip("_").lower()
    return k or "item"


def _parse_bounds_zones(text: str, default_bin_size: float) -> tuple[list[tuple[float, float, float]], list[str]]:
    """
    Parse multi-zone text into [(ymax, ymin, bin_size), ...].
    Accepts lines like:
    +5,-5,1
    -5,-10,1
    -10,-20
    """
    zones: list[tuple[float, float, float]] = []
    bad_lines: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+", line)
        if len(nums) < 2:
            bad_lines.append(line)
            continue
        try:
            a = float(nums[0])
            b = float(nums[1])
        except ValueError:
            bad_lines.append(line)
            continue
        if a == b:
            bad_lines.append(line)
            continue
        try:
            bin_size = float(nums[2]) if len(nums) >= 3 else float(default_bin_size)
        except ValueError:
            bad_lines.append(line)
            continue
        if bin_size <= 0:
            bad_lines.append(line)
            continue
        zones.append((a, b, bin_size))
    return zones, bad_lines


def _pages_png_to_pdf_bytes(page_pngs: list[tuple[bytes, str]]) -> bytes:
    """Merge PNG page bytes into a single multi-page PDF."""
    from PIL import Image

    images: list[Image.Image] = []
    try:
        for png_bytes, _name in page_pngs:
            with Image.open(io.BytesIO(png_bytes)) as im:
                if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                    # Flatten transparency to white for reliable PDF output.
                    alpha = im.split()[-1] if im.mode in ("RGBA", "LA") else None
                    rgb = Image.new("RGB", im.size, "white")
                    rgb.paste(im, mask=alpha)
                    images.append(rgb)
                else:
                    images.append(im.convert("RGB"))

        if not images:
            return b""

        out = io.BytesIO()
        images[0].save(
            out,
            format="PDF",
            save_all=True,
            append_images=images[1:],
            resolution=300.0,
        )
        return out.getvalue()
    finally:
        for im in images:
            try:
                im.close()
            except Exception:
                pass

# Core imports
from core.ags_parser import parse_ags_text
from core.merge_layers import merge_adjacent_layers
from core.normalization import normalize_layers
from core.reporting import build_summary_report
from core.rock_dict import load_rock_sets, ROCK_DICT_CSV
from core.soil_cleaning import extract_labels
from core.spt_processing import extract_spt
from core.validation import validate_layers
from core.plots_striplog import plot_striplog, build_color_map
from core.plots_spt import plot_spt_vs_level_panels, prepare_spt_with_soil_type, compute_spt_bounds_equations
from core.striplog_static import render_striplog_png, render_striplog_a3_png, render_striplog_a3_pages, DEFAULT_BH_PER_PAGE

st.set_page_config(page_title="FoxGI WebApp", layout="wide")


# Cache rock dict
@st.cache_data
def get_rock_sets():
    return load_rock_sets(ROCK_DICT_CSV)


def _def_col(df, *keys):
    for k in keys:
        for c in df.columns:
            if k.upper() in str(c).upper():
                return c
    return None


def _pick_spt_n_col(df: pd.DataFrame) -> str | None:
    """Pick SPT N-value column with exact-name priority to avoid false matches."""
    if df is None or df.empty:
        return None
    exact_priority = ["N_effective", "N", "ISPT_NVAL"]
    name_map = {str(c).strip().upper(): c for c in df.columns}
    for n in exact_priority:
        if n.upper() in name_map:
            return name_map[n.upper()]
    # fallback: broader match, but avoid plain single-letter "N" contains issues
    for c in df.columns:
        cu = str(c).upper()
        if "N_EFFECTIVE" in cu or "ISPT_NVAL" in cu:
            return c
    return None


def _run_pipeline_impl(
    files_data: tuple[tuple[str, str], ...],
    mode: str,
    progress_cb: object = None,
    fill_subtype: bool = False,
) -> tuple:
    """Core pipeline logic. progress_cb(percent, message) called at each step."""
    def _report(pct: float, msg: str):
        if progress_cb:
            progress_cb(pct, msg)

    _report(0, "Loading rock dictionary...")
    rock_sets = get_rock_sets()

    _report(5, "Parsing AGS/CSV files...")
    all_groups = {}
    n_files = len(files_data)
    for i, (name, text) in enumerate(files_data):
        if progress_cb:
            progress_cb(5 + 20 * (i + 1) / n_files, f"Parsing {name}...")
        if mode == "AGS":
            groups = parse_ags_text(text)
            for k, v in groups.items():
                if k not in all_groups:
                    all_groups[k] = []
                v = v.copy()
                v["Source_File"] = name
                all_groups[k].append(v)
        else:
            df = pd.read_csv(io.StringIO(text))
            df["Source_File"] = name
            if "GEOL" not in all_groups:
                all_groups["GEOL"] = []
            all_groups["GEOL"].append(df)

    _report(28, "Combining groups...")
    groups = {k: pd.concat(v, ignore_index=True) for k, v in all_groups.items()}

    if "GEOL" not in groups or groups["GEOL"].empty:
        return None, None, None, None, None, "No GEOL data found."

    geol = groups["GEOL"]
    col = lambda df, *keys: _def_col(df, *keys)

    bh_c = col(geol, "HOLE_ID", "LOCA_ID", "HOLE", "LOCA", "BOREHOLE") or geol.columns[0]
    top_c = col(geol, "GEOL_TOP", "TOP", "FROM")
    base_c = col(geol, "GEOL_BASE", "BASE", "BTM", "TO")
    desc_c = col(geol, "GEOL_DESC", "DESC", "DESCRIPTION") or geol.columns[-1]

    layers = geol.copy()
    rename = {bh_c: "Borehole_ID"}
    if top_c:
        rename[top_c] = "Depth_From"
    if base_c:
        rename[base_c] = "Depth_To"
    if desc_c:
        rename[desc_c] = "Description"
    layers = layers.rename(columns=rename)
    for dc in ["Depth_From", "Depth_To"]:
        if dc in layers.columns:
            layers[dc] = pd.to_numeric(layers[dc], errors="coerce").fillna(0)
    if "Depth_From" not in layers.columns:
        layers["Depth_From"] = layers.iloc[:, 1] if layers.shape[1] > 1 else 0
    if "Depth_To" not in layers.columns:
        layers["Depth_To"] = layers.iloc[:, 2] if layers.shape[1] > 2 else 0

    _report(35, "Soil description cleaning...")
    labels = [extract_labels(str(d)) for d in layers["Description"]]
    for k in ["Description_Original", "Description_Clean", "Label1", "Label2", "FinalLabel"]:
        layers[k] = [lb[k] for lb in labels]

    _report(55, "Normalizing soil codes...")
    layers = normalize_layers(layers, rock_sets=rock_sets, desc_col="Description", label1_col="Label1", label2_col="Label2", fill_subtype=fill_subtype)

    _report(70, "Merging adjacent layers...")
    merged = merge_adjacent_layers(layers, bh_col="Borehole_ID", code_col="Normalized_Code", from_col="Depth_From", to_col="Depth_To", desc_col="Description")

    _report(80, "Extracting SPT data...")
    spt_df = extract_spt(groups)

    gl_data = []
    _report(85, "Loading ground levels...")
    for grp_name in ["HOLE", "LOCA"]:
        if grp_name in groups and not gl_data:
            tbl = groups[grp_name].copy()
            lh = col(tbl, "HOLE_ID", "LOCA_ID", "HOLE", "LOCA", "ID")
            lg = col(tbl, "HOLE_GL", "LOCA_GL", "GL", "GROUND", "ELEV")
            if lh and lg:
                tbl[lg] = pd.to_numeric(tbl[lg], errors="coerce")
                tbl = tbl.dropna(subset=[lg])
                seen = set()
                for _, r in tbl.iterrows():
                    bh_id = str(r[lh]).strip()
                    if bh_id and bh_id not in seen and not bh_id.startswith("*"):
                        seen.add(bh_id)
                        gl_val = r[lg]
                        gl_data.append({"Borehole_ID": bh_id, "GL": float(gl_val)})
            break

    gl_df = pd.DataFrame(gl_data) if gl_data else None

    # Compute GL + Level columns (Level = GL - Depth) for layers, merged, spt
    if gl_df is not None and not gl_df.empty:
        gl_map = dict(zip(gl_df["Borehole_ID"], gl_df["GL"]))
        for df in (layers, merged):
            df["GL"] = df["Borehole_ID"].map(gl_map).round(2)
            df["Level_From"] = (df["GL"] - df["Depth_From"]).round(2)
            df["Level_To"] = (df["GL"] - df["Depth_To"]).round(2)
        if spt_df is not None and not spt_df.empty:
            spt_df["GL"] = spt_df["Borehole_ID"].map(gl_map).round(2)
            spt_df["Level_Top"] = (spt_df["GL"] - spt_df["Depth"]).round(2)
            pen_m = spt_df["Penetration_mm"].fillna(0) / 1000.0
            spt_df["Level_Seat"] = (spt_df["Level_Top"] - pen_m).round(2)

    _report(92, "Validating layers...")
    errors_df, warnings_df = validate_layers(merged, bh_col="Borehole_ID", from_col="Depth_From", to_col="Depth_To")
    _report(96, "Building summary report...")
    report = build_summary_report(merged, gl_df)

    _report(100, "Done")
    return layers, merged, spt_df, gl_df, (errors_df, warnings_df), report, groups


@st.cache_data(ttl=3600)
def run_pipeline_cached(files_data: tuple[tuple[str, str], ...], mode: str, fill_subtype: bool = False):
    """Cached pipeline (no progress). Used for quick re-run on cache hit."""
    return _run_pipeline_impl(files_data, mode, progress_cb=None, fill_subtype=fill_subtype)


# Sidebar
_logo_path = Path(__file__).parent / "MadFoxLogo.png"
if _logo_path.exists():
    try:
        from PIL import Image
        img = Image.open(_logo_path).convert("RGBA")
        data = img.load()
        w, h = img.size
        for y in range(h):
            for x in range(w):
                r, g, b, a = data[x, y]
                if r > 250 and g > 250 and b > 250:
                    data[x, y] = (r, g, b, 0)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        st.sidebar.image(buf, use_container_width=True)
    except Exception:
        st.sidebar.image(str(_logo_path), use_container_width=True)
st.sidebar.title("FoxGI WebApp")
uploaded = st.sidebar.file_uploader("Upload AGS/CSV", type=["ags", "csv"], accept_multiple_files=True)

# Sample: Sample/11 0210 03 R002.ags (Yuen Long AGS)
_sample_path = Path(__file__).parent / "Sample" / "11 0210 03 R002.ags"
if _sample_path.exists():
    st.sidebar.caption("Sample: Sample/11 0210 03 R002.ags")

mode = st.sidebar.radio("Mode", ["AGS", "CSV"])
fill_subtype = st.sidebar.checkbox("FILL with subtype (e.g. FILL (SAND))", value=False)
spt_level_mode = st.sidebar.radio(
    "SPT N-value Level",
    ["TOP", "SEAT"],
    help="TOP = GL − Depth_Top;  SEAT = TOP Level − Penetration/1000",
)

# AGS input tab (sidebar) dark background.
if mode == "AGS":
    st.markdown(
        """
        <style>
        :root {
            --neon-green: #39FF14;
        }
        section[data-testid="stSidebar"] {
            background: #000000 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #f2f2f2 !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="input"] input,
        section[data-testid="stSidebar"] textarea {
            background: #121212 !important;
            color: #f2f2f2 !important;
            border: 1px solid #ffffff !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="input"] {
            border: 1px solid #ffffff !important;
            border-radius: 6px !important;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stCaption {
            color: var(--neon-green) !important;
        }
        section[data-testid="stSidebar"] button {
            background: var(--neon-green) !important;
            color: #000000 !important;
            border: 1px solid var(--neon-green) !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
            background: var(--neon-green) !important;
            color: #000000 !important;
            border: 1px solid var(--neon-green) !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] [data-testid^="stBaseButton-"] {
            background: var(--neon-green) !important;
            color: #000000 !important;
            border: 1px solid var(--neon-green) !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] [data-testid^="stBaseButton-"] *,
        section[data-testid="stSidebar"] button[kind],
        section[data-testid="stSidebar"] button[kind] * {
            color: #000000 !important;
        }
        section[data-testid="stSidebar"] [data-testid^="stBaseButton-"]:hover,
        section[data-testid="stSidebar"] button[kind]:hover {
            background: var(--neon-green) !important;
            border-color: var(--neon-green) !important;
        }
        section[data-testid="stSidebar"] button[kind="primary"],
        section[data-testid="stSidebar"] button[kind="primary"]:hover,
        section[data-testid="stSidebar"] button[kind="primary"]:focus,
        section[data-testid="stSidebar"] button[kind="primary"]:active {
            background: #ff2b2b !important;
            border-color: #ff2b2b !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] button[kind="primary"] * {
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] button[kind="secondary"],
        section[data-testid="stSidebar"] button[kind="secondary"]:hover,
        section[data-testid="stSidebar"] button[kind="secondary"]:focus,
        section[data-testid="stSidebar"] button[kind="secondary"]:active {
            background: var(--neon-green) !important;
            border-color: var(--neon-green) !important;
            color: #000000 !important;
        }
        section[data-testid="stSidebar"] button[kind="secondary"] * {
            color: #000000 !important;
        }
        section[data-testid="stSidebar"] button * {
            color: #000000 !important;
        }
        section[data-testid="stSidebar"] button:hover {
            filter: brightness(1.12);
        }
        section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] button,
        section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] button:hover,
        section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] button:focus,
        section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] button:active {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] button *,
        section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        section[data-testid="stSidebar"] input[type="checkbox"],
        section[data-testid="stSidebar"] input[type="radio"] {
            accent-color: var(--neon-green) !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div {
            background: transparent !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
            color: #f2f2f2 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *,
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] [role="button"],
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] [role="button"] * {
            color: #000000 !important;
        }
        section[data-testid="stSidebar"] input::placeholder,
        section[data-testid="stSidebar"] textarea::placeholder {
            color: #f2f2f2 !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Filters (applied after run)
st.sidebar.subheader("Filters")
with st.sidebar.expander("Filter help", expanded=False):
    st.caption("Filters apply to results after Run. Leave blank = show all.")
    st.caption("Borehole_ID: e.g. BH1, BH11 (partial match)")
    st.caption("Normalized_Code: e.g. CDG, FILL, ALLUVIUM")
    st.caption("Depth: layer depth range (m below ground)")
    st.caption("Level: level range (mPD)")
filter_bh = st.sidebar.text_input("Borehole_ID filter", "", placeholder="e.g. BH1 or BH11")
filter_code = st.sidebar.text_input("Normalized_Code filter", "", placeholder="e.g. CDG, FILL")
filter_depth_min = st.sidebar.number_input("Depth min (m)", value=None, format="%.2f", placeholder="0", help="Min depth to include (m below GL)")
filter_depth_max = st.sidebar.number_input("Depth max (m)", value=None, format="%.2f", placeholder="150", help="Max depth to include (m below GL)")
filter_level_min = st.sidebar.number_input("Level min (mPD)", value=None, format="%.2f", placeholder="-100", help="Min level to include (mPD)")
filter_level_max = st.sidebar.number_input("Level max (mPD)", value=None, format="%.2f", placeholder="+100", help="Max level to include (mPD)")

run = st.sidebar.button("Run", type="secondary")
restart = st.sidebar.button(
    "Restart",
    type="primary",
    help="Clear cache and reset. Upload files and Run again.",
)

# Persist pipeline results so color picker / filter changes don't require re-run
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = None
if "pipeline_files" not in st.session_state:
    st.session_state.pipeline_files = None

# Run pipeline when Run clicked (with % progress and ETA)
if run and uploaded:
    files_data = tuple((up.name, up.read().decode("utf-8", errors="replace")) for up in uploaded)
    cache_key = (tuple((n, t) for n, t in files_data), mode)
    if "_pipe_cache" not in st.session_state:
        st.session_state._pipe_cache = {}

    if cache_key in st.session_state._pipe_cache:
        layers, merged, spt_df, gl_df, val, report, groups = st.session_state._pipe_cache[cache_key]
        st.toast("Using cached result (same files).")
    else:
        progress_bar = st.progress(0, text="Starting...")
        status_placeholder = st.empty()
        start_t = time.perf_counter()

        def _progress(pct: float, msg: str):
            pct_clamped = min(100, max(0, pct))
            progress_bar.progress(pct_clamped / 100, text=f"{int(pct_clamped)}% — {msg}")
            elapsed = time.perf_counter() - start_t
            eta_str = ""
            if pct_clamped > 5 and pct_clamped < 99:
                eta_sec = (elapsed / (pct_clamped / 100)) - elapsed if pct_clamped else 0
                eta_str = f"  ·  ~{max(0, int(eta_sec))}s left"
            status_placeholder.caption(f"{int(pct_clamped)}% — {msg}{eta_str}")

        layers, merged, spt_df, gl_df, val, report, groups = _run_pipeline_impl(files_data, mode, progress_cb=_progress, fill_subtype=fill_subtype)
        progress_bar.progress(1.0, text="Done")
        status_placeholder.caption("")
        progress_bar.empty()
        status_placeholder.empty()
        st.session_state._pipe_cache[cache_key] = (layers, merged, spt_df, gl_df, val, report, groups)

    if layers is not None:
        st.session_state.pipeline_data = (layers, merged, spt_df, gl_df, val, report, groups)
        st.session_state.pipeline_files = tuple(f.name for f in uploaded)
    else:
        st.session_state.pipeline_data = None
        st.error(report or "Processing failed.")

# Restart: clear cached data and color picks
if restart:
    st.session_state.pipeline_data = None
    st.session_state.pipeline_files = None
    if "_pipe_cache" in st.session_state:
        del st.session_state["_pipe_cache"]
    if "striplog_colors" in st.session_state:
        del st.session_state["striplog_colors"]

# Clear cache if files changed or removed
if uploaded:
    current_files = tuple(f.name for f in uploaded)
    if st.session_state.pipeline_files != current_files:
        st.session_state.pipeline_data = None
else:
    st.session_state.pipeline_data = None

# Main: show results if we have cached data (from Run or previous interaction)
if st.session_state.pipeline_data is not None:
    layers, merged, spt_df, gl_df, val, report, groups = st.session_state.pipeline_data

    # File prefix for download names (from first uploaded AGS/CSV file)
    file_prefix = "output"
    if st.session_state.pipeline_files:
        file_prefix = Path(st.session_state.pipeline_files[0]).stem or "output"

    # Apply filters
    def apply_filters(df, is_layer=True):
        if df is None or df.empty:
            return df
        d = df.copy()
        if filter_bh and "Borehole_ID" in d.columns:
            d = d[d["Borehole_ID"].astype(str).str.contains(filter_bh, case=False, na=False)]
        if filter_code and "Normalized_Code" in d.columns:
            d = d[d["Normalized_Code"].astype(str).str.contains(filter_code, case=False, na=False)]
        if is_layer and "Depth_From" in d.columns and "Depth_To" in d.columns:
            if filter_depth_min is not None:
                d = d[d["Depth_To"] > filter_depth_min]
            if filter_depth_max is not None:
                d = d[d["Depth_From"] < filter_depth_max]
            if filter_depth_min is not None or filter_depth_max is not None:
                if filter_depth_min is not None:
                    d["Depth_From"] = d["Depth_From"].clip(lower=filter_depth_min)
                if filter_depth_max is not None:
                    d["Depth_To"] = d["Depth_To"].clip(upper=filter_depth_max)
                d = d[d["Depth_From"] < d["Depth_To"]]
                if "GL" in d.columns:
                    d["Level_From"] = (d["GL"] - d["Depth_From"]).round(2)
                    d["Level_To"] = (d["GL"] - d["Depth_To"]).round(2)
            if "Level_From" in d.columns and "Level_To" in d.columns:
                level_low = d[["Level_From", "Level_To"]].min(axis=1)
                level_high = d[["Level_From", "Level_To"]].max(axis=1)
                if filter_level_min is not None:
                    d = d[level_high >= filter_level_min]
                if filter_level_max is not None:
                    d = d[level_low <= filter_level_max]
                if (filter_level_min is not None or filter_level_max is not None) and ("GL" in d.columns):
                    if filter_level_min is not None:
                        d["Level_To"] = d["Level_To"].clip(lower=filter_level_min)
                        d["Level_From"] = d["Level_From"].clip(lower=filter_level_min)
                    if filter_level_max is not None:
                        d["Level_To"] = d["Level_To"].clip(upper=filter_level_max)
                        d["Level_From"] = d["Level_From"].clip(upper=filter_level_max)
                    d = d[d["Level_From"] != d["Level_To"]]
                    d["Depth_From"] = (d["GL"] - d["Level_From"]).round(2)
                    d["Depth_To"] = (d["GL"] - d["Level_To"]).round(2)
                    d["Depth_From"], d["Depth_To"] = (
                        d[["Depth_From", "Depth_To"]].min(axis=1),
                        d[["Depth_From", "Depth_To"]].max(axis=1),
                    )
                    d["Level_From"] = (d["GL"] - d["Depth_From"]).round(2)
                    d["Level_To"] = (d["GL"] - d["Depth_To"]).round(2)
        elif not is_layer and "Depth" in d.columns:
            if filter_depth_min is not None:
                d = d[d["Depth"] >= filter_depth_min]
            if filter_depth_max is not None:
                d = d[d["Depth"] <= filter_depth_max]
            level_col = None
            if spt_level_mode == "SEAT" and "Level_Seat" in d.columns:
                level_col = "Level_Seat"
            elif "Level_Top" in d.columns:
                level_col = "Level_Top"
            elif "Level" in d.columns:
                level_col = "Level"
            if level_col:
                if filter_level_min is not None:
                    d = d[d[level_col] >= filter_level_min]
                if filter_level_max is not None:
                    d = d[d[level_col] <= filter_level_max]
        return d

    layers_f = apply_filters(layers)
    merged_f = apply_filters(merged)
    spt_f = apply_filters(spt_df, False) if spt_df is not None else None

    # Apply SPT level mode (TOP or SEAT) — set the active "Level" column
    if spt_f is not None and not spt_f.empty and "Level_Top" in spt_f.columns:
        if spt_level_mode == "SEAT":
            spt_f = spt_f.copy()
            spt_f["Level"] = spt_f["Level_Seat"]
        else:
            spt_f = spt_f.copy()
            spt_f["Level"] = spt_f["Level_Top"]

    # Tab selector with session_state so GEN (form submit) keeps us on Strip Log Plot
    tab_labels = ["Project Info", "Layers (Clean)", "Layers (Merged)", "SPT", "SPT Plot", "SPT Plot (Soil)", "Validation", "Summary", "Strip Log Plot"]
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Strip Log Plot"
    tab_choice = st.radio(
        "Tab",
        tab_labels,
        key="active_tab",
        horizontal=True,
        label_visibility="collapsed",
    )

    if tab_choice == "Project Info":
        st.subheader("Summary of GI Results")
        proj = groups.get("PROJ")
        hole = groups.get("HOLE")
        _field_map = {
            "Project": ["PROJ_NAME"],
            "Location": ["PROJ_LOC"],
            "Client": ["PROJ_CLNT"],
            "Engineer": ["PROJ_ENG"],
            "Contractor": ["PROJ_CONT"],
        }
        info_rows = []
        if proj is not None and not proj.empty:
            row0 = proj.iloc[0]
            for label, cols in _field_map.items():
                val_str = ""
                for c in cols:
                    if c in row0.index and pd.notna(row0[c]) and str(row0[c]).strip():
                        val_str = str(row0[c]).strip()
                        break
                info_rows.append({"Field": label, "Value": val_str})

        if hole is not None and not hole.empty:
            star_col = _def_col(hole, "HOLE_STAR", "STAR")
            end_col = _def_col(hole, "HOLE_ENDD", "ENDD")
            dates = []
            if star_col:
                dates.extend(hole[star_col].dropna().astype(str).str.strip().tolist())
            if end_col:
                dates.extend(hole[end_col].dropna().astype(str).str.strip().tolist())
            dates = [d for d in dates if d and d != "nan"]
            if dates:
                info_rows.append({"Field": "Period", "Value": f"{min(dates)} to {max(dates)}"})
            info_rows.append({"Field": "No. of Boreholes", "Value": str(hole["HOLE_ID"].nunique() if "HOLE_ID" in hole.columns else len(hole))})

        if info_rows:
            info_df = pd.DataFrame(info_rows)
            st.dataframe(info_df, use_container_width=True, hide_index=True)
            st.download_button("Download Summary of GI Results", info_df.to_csv(index=False).encode(), f"{file_prefix}_Summary_of_GI_Results.csv", "text/csv", key="dl_gi_results")
        else:
            st.info("No PROJ group found in AGS data.")

        if hole is not None and not hole.empty:
            st.subheader("Borehole Summary")
            bh_display_cols = [c for c in hole.columns if any(k in c.upper() for k in ["HOLE_ID", "TYPE", "NATE", "NATN", "GL", "FDEP", "STAR", "ENDD"])]
            if bh_display_cols:
                bh_df = hole[bh_display_cols]
                st.dataframe(bh_df, use_container_width=True, hide_index=True)
                st.download_button("Download Borehole Summary", bh_df.to_csv(index=False).encode(), f"{file_prefix}_Borehole_Summary.csv", "text/csv", key="dl_borehole_summary")

    elif tab_choice == "Layers (Clean)":
        st.dataframe(layers_f, use_container_width=True, height=700)
        st.download_button("Download soil_layers_clean.csv", layers_f.to_csv(index=False).encode(), f"{file_prefix}_soil_layers_clean.csv", "text/csv")

    elif tab_choice == "Layers (Merged)":
        st.dataframe(merged_f, use_container_width=True, height=700)
        st.download_button("Download soil_layers_merged.csv", merged_f.to_csv(index=False).encode(), f"{file_prefix}_soil_layers_merged.csv", "text/csv")

    elif tab_choice == "SPT":
        if spt_f is not None and not spt_f.empty:
            spt_tab_df = prepare_spt_with_soil_type(spt_f, merged_f)
            st.dataframe(spt_tab_df, use_container_width=True, height=700)
            st.download_button("Download spt_clean.csv", spt_tab_df.to_csv(index=False).encode(), f"{file_prefix}_spt_clean.csv", "text/csv")
        else:
            st.info("No SPT data.")

    elif tab_choice == "SPT Plot":
        if spt_f is not None and not spt_f.empty:
            spt_for_filter = prepare_spt_with_soil_type(spt_f, merged_f)
            bh_col = _def_col(spt_for_filter, "Borehole_ID", "HOLE_ID", "LOCA_ID")
            n_col_spt_plot = _pick_spt_n_col(spt_for_filter)
            bh_options = (
                sorted(spt_for_filter[bh_col].dropna().astype(str).unique().tolist(), key=_natural_sort_key)
                if bh_col else []
            )
            if "Soil_Type" in spt_for_filter.columns and n_col_spt_plot and n_col_spt_plot in spt_for_filter.columns:
                soil_n = pd.to_numeric(spt_for_filter[n_col_spt_plot], errors="coerce")
                soil_counts = (
                    pd.DataFrame({"Soil_Type": spt_for_filter["Soil_Type"].astype(str), "_n": soil_n})
                    .dropna(subset=["_n"])
                    .groupby("Soil_Type", as_index=False)
                    .size()
                )
                soil_options = sorted(
                    soil_counts.loc[soil_counts["size"] > 0, "Soil_Type"].astype(str).tolist(),
                    key=str.upper,
                )
            else:
                soil_options = []

            c1, c2 = st.columns(2)
            with c1:
                sel_bh = st.multiselect(
                    "Select borehole(s)",
                    options=bh_options,
                    default=bh_options,
                    key="spt_plot_boreholes",
                )
            with c2:
                sel_soil = st.multiselect(
                    "Select soil type(s)",
                    options=soil_options,
                    default=soil_options,
                    key="spt_plot_soils",
                )

            c3, c4 = st.columns(2)
            n_col_for_axis = _def_col(spt_for_filter, "N_effective", "N", "ISPT_NVAL")
            if n_col_for_axis and n_col_for_axis in spt_for_filter.columns:
                n_vals = pd.to_numeric(spt_for_filter[n_col_for_axis], errors="coerce").dropna()
                spt_x_max_default = float(n_vals.max()) if not n_vals.empty else 200.0
            else:
                spt_x_max_default = 200.0
            with c3:
                spt_x_min = st.number_input(
                    "X-axis min",
                    value=0.0,
                    step=1.0,
                    key="spt_plot_x_min",
                )
            with c4:
                spt_x_max = st.number_input(
                    "X-axis max",
                    value=spt_x_max_default,
                    step=1.0,
                    key="spt_plot_x_max",
                )
            c3i, c4i = st.columns(2)
            with c3i:
                spt_x_interval = st.number_input(
                    "X-axis interval",
                    min_value=0.1,
                    value=10.0,
                    step=0.5,
                    key="spt_plot_x_interval",
                )
            with c4i:
                spt_y_interval = st.number_input(
                    "Y-axis interval",
                    min_value=0.1,
                    value=1.0,
                    step=0.5,
                    key="spt_plot_y_interval",
                )
            symbol_map = {
                "Dot": "circle",
                "X": "x",
                "Cross": "cross",
                "Diamond": "diamond",
                "Square": "square",
                "Triangle Up": "triangle-up",
            }
            c_mark1, c_mark2 = st.columns(2)
            with c_mark1:
                spt_marker_label = st.selectbox(
                    "Marker symbol",
                    options=list(symbol_map.keys()),
                    index=0,
                    key="spt_plot_marker_symbol",
                )
            with c_mark2:
                spt_marker_size = st.number_input(
                    "Marker size",
                    min_value=1.0,
                    max_value=20.0,
                    value=7.0,
                    step=1.0,
                    key="spt_plot_marker_size",
                )

            y_col = "Level" if "Level" in spt_for_filter.columns else (
                "Level_Top" if "Level_Top" in spt_for_filter.columns else (
                    "Depth" if "Depth" in spt_for_filter.columns else None
                )
            )
            y_vals = pd.to_numeric(spt_for_filter[y_col], errors="coerce").dropna() if y_col else pd.Series(dtype=float)
            if not y_vals.empty:
                if str(y_col).lower().startswith("level"):
                    y_top_default = float(y_vals.max())
                    y_bottom_default = float(y_vals.min())
                else:
                    y_top_default = float(y_vals.min())
                    y_bottom_default = float(y_vals.max())
            else:
                y_top_default, y_bottom_default = 0.0, 50.0

            c5, c6 = st.columns(2)
            with c5:
                spt_y_top = st.number_input(
                    "Y-axis top",
                    value=y_top_default,
                    step=1.0,
                    key="spt_plot_y_top",
                )
            with c6:
                spt_y_bottom = st.number_input(
                    "Y-axis bottom",
                    value=y_bottom_default,
                    step=1.0,
                    key="spt_plot_y_bottom",
                )

            side_col, main_col = st.columns([1, 4], vertical_alignment="top")
            with side_col:
                st.markdown("**Percentile Bounds / Design Profile**")
                c_show, c_reset = st.columns([3, 1])
                with c_reset:
                    if st.button("Reset", key="spt_plot_bounds_reset_btn", use_container_width=True):
                        st.session_state["spt_plot_show_bounds"] = False
                        st.session_state["spt_plot_bounds_line_color"] = "#808080"
                        st.session_state["spt_plot_design_line_color"] = "#000080"
                        st.session_state["spt_plot_bounds_bin_size"] = 1.0
                        st.session_state["spt_plot_bounds_p_low"] = 0.05
                        st.session_state["spt_plot_bounds_p_high"] = 0.95
                        st.session_state["spt_plot_bounds_y_max"] = float(spt_y_top)
                        st.session_state["spt_plot_bounds_y_min"] = float(spt_y_bottom)
                        st.session_state["spt_plot_bounds_x_min"] = float(spt_x_min)
                        st.session_state["spt_plot_bounds_x_max"] = float(spt_x_max)
                        st.session_state["spt_plot_bounds_zones_text"] = f"{float(spt_y_top):g},{float(spt_y_bottom):g},1"
                        st.session_state["spt_plot_use_manual_design_eq"] = False
                        st.session_state["spt_plot_design_c"] = 19.752
                        st.session_state["spt_plot_design_m"] = -2.390
                        st.session_state["spt_plot_lower_c"] = 15.000
                        st.session_state["spt_plot_lower_m"] = -2.390
                        st.session_state["spt_plot_upper_c"] = 25.000
                        st.session_state["spt_plot_upper_m"] = -2.390
                        st.rerun()
                with c_show:
                    spt_show_bounds = st.checkbox(
                        "Show bounds lines",
                        value=False,
                        key="spt_plot_show_bounds",
                    )
                c_bound_color, c_design_color = st.columns(2)
                with c_bound_color:
                    st.caption("Bound")
                    spt_bounds_line_color = st.color_picker(
                        "Bound line color",
                        value="#808080",
                        key="spt_plot_bounds_line_color",
                        label_visibility="collapsed",
                    )
                with c_design_color:
                    st.caption("Design")
                    spt_design_line_color = st.color_picker(
                        "Design line color",
                        value="#000080",
                        key="spt_plot_design_line_color",
                        label_visibility="collapsed",
                    )
                spt_bin_size = st.number_input(
                    "Bin size (m)",
                    min_value=0.1,
                    value=1.0,
                    step=0.1,
                    key="spt_plot_bounds_bin_size",
                )
                c_plow, c_phigh = st.columns(2)
                with c_plow:
                    spt_p_low = st.number_input(
                        "Low percentile",
                        min_value=0.0,
                        max_value=0.49,
                        value=0.05,
                        step=0.01,
                        key="spt_plot_bounds_p_low",
                    )
                with c_phigh:
                    spt_p_high = st.number_input(
                        "High percentile",
                        min_value=0.51,
                        max_value=1.0,
                        value=0.95,
                        step=0.01,
                        key="spt_plot_bounds_p_high",
                    )
                c_by1, c_by2 = st.columns(2)
                with c_by1:
                    spt_bounds_y_max = st.number_input(
                        "YMAX",
                        value=float(spt_y_top),
                        step=1.0,
                        key="spt_plot_bounds_y_max",
                    )
                with c_by2:
                    spt_bounds_y_min = st.number_input(
                        "YMIN",
                        value=float(spt_y_bottom),
                        step=1.0,
                        key="spt_plot_bounds_y_min",
                    )
                spt_zone_text = st.text_area(
                    "Zones (YMAX,YMIN,BIN; one per line)",
                    value=f"{float(spt_bounds_y_max):g},{float(spt_bounds_y_min):g},{float(spt_bin_size):g}",
                    key="spt_plot_bounds_zones_text",
                    height=96,
                )
                spt_zone_specs, spt_bad_zone_lines = _parse_bounds_zones(spt_zone_text, float(spt_bin_size))
                spt_bounds_zones = [(a, b) for a, b, _ in spt_zone_specs]
                spt_zone_bin_sizes = [zbin for _, _, zbin in spt_zone_specs]
                if spt_zone_specs:
                    zone_preview = " | ".join([f"{a:g},{b:g},{zbin:g}" for a, b, zbin in spt_zone_specs[:5]])
                    st.caption(f"Parsed zones: {len(spt_bounds_zones)} ({zone_preview})")
                c_bx1, c_bx2 = st.columns(2)
                with c_bx1:
                    spt_bounds_x_min = st.number_input(
                        "XMIN limit",
                        value=float(spt_x_min),
                        step=1.0,
                        key="spt_plot_bounds_x_min",
                    )
                with c_bx2:
                    spt_bounds_x_max = st.number_input(
                        "XMAX limit",
                        value=float(spt_x_max),
                        step=1.0,
                        key="spt_plot_bounds_x_max",
                    )
                use_manual_design_eq = st.checkbox(
                    "Use manual design equation (N = C + mZ)",
                    value=False,
                    key="spt_plot_use_manual_design_eq",
                )
                auto_eq = compute_spt_bounds_equations(
                    spt_f,
                    merged_f,
                    selected_boreholes=sel_bh,
                    selected_soil_types=sel_soil,
                    bounds_bin_size=float(spt_bin_size),
                    bounds_p_low=float(spt_p_low),
                    bounds_p_high=float(spt_p_high),
                    bounds_y_max=float(spt_bounds_y_max),
                    bounds_y_min=float(spt_bounds_y_min),
                    bounds_x_min=float(spt_bounds_x_min),
                    bounds_x_max=float(spt_bounds_x_max),
                )
                if auto_eq is not None:
                    low_m, low_c, auto_m, auto_c, up_m, up_c = auto_eq
                    st.caption(f"Auto: N = {auto_c:.3f} + {abs(auto_m):.3f} Z")
                    if not use_manual_design_eq:
                        st.session_state["spt_plot_design_c"] = float(auto_c)
                        st.session_state["spt_plot_design_m"] = float(-abs(auto_m))
                        st.session_state["spt_plot_lower_c"] = float(low_c)
                        st.session_state["spt_plot_lower_m"] = float(-abs(low_m))
                        st.session_state["spt_plot_upper_c"] = float(up_c)
                        st.session_state["spt_plot_upper_m"] = float(-abs(up_m))
                    elif "spt_plot_design_c" not in st.session_state:
                        st.session_state["spt_plot_design_c"] = float(auto_c)
                        st.session_state["spt_plot_design_m"] = float(-abs(auto_m))
                        st.session_state["spt_plot_lower_c"] = float(low_c)
                        st.session_state["spt_plot_lower_m"] = float(-abs(low_m))
                        st.session_state["spt_plot_upper_c"] = float(up_c)
                        st.session_state["spt_plot_upper_m"] = float(-abs(up_m))
                else:
                    st.caption("Auto: insufficient data for design equation")
                spt_design_c = float(st.session_state.get("spt_plot_design_c", 19.752))
                spt_design_m = float(st.session_state.get("spt_plot_design_m", -2.390))
                spt_lower_c = float(st.session_state.get("spt_plot_lower_c", 15.000))
                spt_lower_m = float(st.session_state.get("spt_plot_lower_m", -2.390))
                spt_upper_c = float(st.session_state.get("spt_plot_upper_c", 25.000))
                spt_upper_m = float(st.session_state.get("spt_plot_upper_m", -2.390))
                spt_zone_manual_params = None
                if use_manual_design_eq:
                    if len(spt_bounds_zones) > 1:
                        rows = []
                        for (z_top, z_bottom), z_bin in zip(spt_bounds_zones, spt_zone_bin_sizes):
                            z_auto = compute_spt_bounds_equations(
                                spt_f,
                                merged_f,
                                selected_boreholes=sel_bh,
                                selected_soil_types=sel_soil,
                                bounds_bin_size=float(z_bin),
                                bounds_p_low=float(spt_p_low),
                                bounds_p_high=float(spt_p_high),
                                bounds_y_max=float(z_top),
                                bounds_y_min=float(z_bottom),
                                bounds_x_min=float(spt_bounds_x_min),
                                bounds_x_max=float(spt_bounds_x_max),
                            )
                            if z_auto is not None:
                                low_m, low_c, d_m, d_c, up_m, up_c = z_auto
                                rows.append(
                                    {
                                        "zone": f"{z_top:g},{z_bottom:g},{z_bin:g}",
                                        "design_c": float(d_c),
                                        "design_m": float(-abs(d_m)),
                                        "lower_c": float(low_c),
                                        "lower_m": float(-abs(low_m)),
                                        "upper_c": float(up_c),
                                        "upper_m": float(-abs(up_m)),
                                    }
                                )
                            else:
                                rows.append(
                                    {
                                        "zone": f"{z_top:g},{z_bottom:g},{z_bin:g}",
                                        "design_c": float(spt_design_c),
                                        "design_m": float(spt_design_m),
                                        "lower_c": float(spt_lower_c),
                                        "lower_m": float(spt_lower_m),
                                        "upper_c": float(spt_upper_c),
                                        "upper_m": float(spt_upper_m),
                                    }
                                )
                        zdf = pd.DataFrame(rows)
                        zdf_edit = st.data_editor(
                            zdf,
                            hide_index=True,
                            use_container_width=True,
                            num_rows="fixed",
                            column_config={"zone": st.column_config.TextColumn(disabled=True)},
                            key="spt_plot_zone_manual_editor",
                        )
                        spt_zone_manual_params = []
                        for _, zr in zdf_edit.iterrows():
                            spt_zone_manual_params.append(
                                {
                                    "design_c": float(zr["design_c"]),
                                    "design_m": float(zr["design_m"]),
                                    "lower_c": float(zr["lower_c"]),
                                    "lower_m": float(zr["lower_m"]),
                                    "upper_c": float(zr["upper_c"]),
                                    "upper_m": float(zr["upper_m"]),
                                }
                            )
                    else:
                        c_dc, c_dm = st.columns(2)
                        with c_dc:
                            spt_design_c = st.number_input(
                                "C (intercept)",
                                value=float(st.session_state.get("spt_plot_design_c", 19.752)),
                                step=0.001,
                                format="%.3f",
                                key="spt_plot_design_c",
                            )
                        with c_dm:
                            spt_design_m = st.number_input(
                                "m (slope) (Negetive)",
                                value=float(st.session_state.get("spt_plot_design_m", -2.390)),
                                step=0.001,
                                format="%.3f",
                                key="spt_plot_design_m",
                            )
                        c_lc, c_lm = st.columns(2)
                        with c_lc:
                            spt_lower_c = st.number_input(
                                "Lower C",
                                value=float(st.session_state.get("spt_plot_lower_c", 15.000)),
                                step=0.001,
                                format="%.3f",
                                key="spt_plot_lower_c",
                            )
                        with c_lm:
                            spt_lower_m = st.number_input(
                                "Lower m",
                                value=float(st.session_state.get("spt_plot_lower_m", -2.390)),
                                step=0.001,
                                format="%.3f",
                                key="spt_plot_lower_m",
                            )
                        c_uc, c_um = st.columns(2)
                        with c_uc:
                            spt_upper_c = st.number_input(
                                "Upper C",
                                value=float(st.session_state.get("spt_plot_upper_c", 25.000)),
                                step=0.001,
                                format="%.3f",
                                key="spt_plot_upper_c",
                            )
                        with c_um:
                            spt_upper_m = st.number_input(
                                "Upper m",
                                value=float(st.session_state.get("spt_plot_upper_m", -2.390)),
                                step=0.001,
                                format="%.3f",
                                key="spt_plot_upper_m",
                            )

            if spt_x_max <= spt_x_min:
                st.warning("X-axis max must be greater than X-axis min.")
            elif spt_x_interval <= 0 or spt_y_interval <= 0:
                st.warning("Axis interval must be greater than 0.")
            elif spt_y_top == spt_y_bottom:
                st.warning("Y-axis top and bottom cannot be the same.")
            elif spt_p_high <= spt_p_low:
                st.warning("High percentile must be greater than low percentile.")
            elif spt_show_bounds and (not spt_bounds_zones):
                st.warning("Please input at least one valid zone (YMAX to YMIN).")
            elif spt_show_bounds and spt_bad_zone_lines:
                st.warning(f"Invalid zone line(s): {', '.join(spt_bad_zone_lines[:3])}")
            elif spt_bounds_x_max <= spt_bounds_x_min:
                st.warning("Function 5 XMAX limit must be greater than XMIN limit.")
            else:
                eq_m = None
                eq_c = None
                if spt_show_bounds and use_manual_design_eq:
                    eq_c = float(spt_design_c)
                    eq_m = float(spt_design_m)
                striplog_custom = st.session_state.get("striplog_colors", {})
                striplog_defaults = build_color_map(soil_options) if soil_options else {}
                soil_colors_for_spt = {
                    str(s): str(striplog_custom.get(str(s), striplog_defaults.get(str(s), "#888888")))
                    for s in soil_options
                }

                plot_sig = inspect.signature(plot_spt_vs_level_panels)
                plot_kwargs = {
                    "selected_boreholes": sel_bh,
                    "selected_soil_types": sel_soil,
                }
                if "soil_color_map" in plot_sig.parameters:
                    plot_kwargs["soil_color_map"] = dict(soil_colors_for_spt)
                if "x_min" in plot_sig.parameters:
                    plot_kwargs["x_min"] = float(spt_x_min)
                if "x_max" in plot_sig.parameters:
                    plot_kwargs["x_max"] = float(spt_x_max)
                if "x_dtick" in plot_sig.parameters:
                    plot_kwargs["x_dtick"] = float(spt_x_interval)
                if "marker_symbol" in plot_sig.parameters:
                    plot_kwargs["marker_symbol"] = symbol_map.get(spt_marker_label, "circle")
                if "marker_size" in plot_sig.parameters:
                    plot_kwargs["marker_size"] = float(spt_marker_size)
                if "y_top" in plot_sig.parameters:
                    plot_kwargs["y_top"] = float(spt_y_top)
                if "y_bottom" in plot_sig.parameters:
                    plot_kwargs["y_bottom"] = float(spt_y_bottom)
                if "y_dtick" in plot_sig.parameters:
                    plot_kwargs["y_dtick"] = float(spt_y_interval)
                if "show_bounds" in plot_sig.parameters:
                    plot_kwargs["show_bounds"] = bool(spt_show_bounds)
                if "bounds_bin_size" in plot_sig.parameters:
                    plot_kwargs["bounds_bin_size"] = float(spt_bin_size)
                if "bounds_p_low" in plot_sig.parameters:
                    plot_kwargs["bounds_p_low"] = float(spt_p_low)
                if "bounds_p_high" in plot_sig.parameters:
                    plot_kwargs["bounds_p_high"] = float(spt_p_high)
                if "bounds_y_max" in plot_sig.parameters:
                    plot_kwargs["bounds_y_max"] = float(spt_bounds_y_max)
                if "bounds_y_min" in plot_sig.parameters:
                    plot_kwargs["bounds_y_min"] = float(spt_bounds_y_min)
                if "bounds_zones" in plot_sig.parameters:
                    plot_kwargs["bounds_zones"] = list(spt_bounds_zones)
                if "bounds_zone_bin_sizes" in plot_sig.parameters:
                    plot_kwargs["bounds_zone_bin_sizes"] = list(spt_zone_bin_sizes)
                if "bounds_x_min" in plot_sig.parameters:
                    plot_kwargs["bounds_x_min"] = float(spt_bounds_x_min)
                if "bounds_x_max" in plot_sig.parameters:
                    plot_kwargs["bounds_x_max"] = float(spt_bounds_x_max)
                if "bounds_line_color" in plot_sig.parameters:
                    plot_kwargs["bounds_line_color"] = str(spt_bounds_line_color)
                if "design_line_color" in plot_sig.parameters:
                    plot_kwargs["design_line_color"] = str(spt_design_line_color)
                if "design_equation_m" in plot_sig.parameters and eq_m is not None:
                    plot_kwargs["design_equation_m"] = float(eq_m)
                if "design_equation_c" in plot_sig.parameters and eq_c is not None:
                    plot_kwargs["design_equation_c"] = float(eq_c)
                if spt_show_bounds and use_manual_design_eq and spt_zone_manual_params is not None:
                    if "zone_manual_params" in plot_sig.parameters:
                        plot_kwargs["zone_manual_params"] = list(spt_zone_manual_params)
                elif spt_show_bounds and use_manual_design_eq:
                    if "lower_bound_m" in plot_sig.parameters:
                        plot_kwargs["lower_bound_m"] = float(spt_lower_m)
                    if "lower_bound_c" in plot_sig.parameters:
                        plot_kwargs["lower_bound_c"] = float(spt_lower_c)
                    if "upper_bound_m" in plot_sig.parameters:
                        plot_kwargs["upper_bound_m"] = float(spt_upper_m)
                    if "upper_bound_c" in plot_sig.parameters:
                        plot_kwargs["upper_bound_c"] = float(spt_upper_c)

                fig_spt = plot_spt_vs_level_panels(
                    spt_f,
                    merged_f,
                    **plot_kwargs,
                )
                with main_col:
                    if len(fig_spt.data) == 0:
                        st.info("No SPT points match the selected borehole/soil filters.")
                    else:
                        st.plotly_chart(fig_spt, use_container_width=True, config={"scrollZoom": False})
                        try:
                            img = fig_spt.to_image(format="png", engine="kaleido")
                            with side_col:
                                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                                st.download_button(
                                    "Download spt_plot.png",
                                    img,
                                    f"{file_prefix}_spt_plot.png",
                                    "image/png",
                                    key="dl_spt_plot_png",
                                )
                        except Exception:
                            with side_col:
                                st.caption("Install kaleido for PNG export: pip install kaleido")
        else:
            st.info("No SPT data.")

    elif tab_choice == "SPT Plot (Soil)":
        if spt_f is not None and not spt_f.empty:
            spt_for_filter = prepare_spt_with_soil_type(spt_f, merged_f)
            bh_col = _def_col(spt_for_filter, "Borehole_ID", "HOLE_ID", "LOCA_ID")
            n_col_soil_pick = _pick_spt_n_col(spt_for_filter)
            bh_options = (
                sorted(spt_for_filter[bh_col].dropna().astype(str).unique().tolist(), key=_natural_sort_key)
                if bh_col else []
            )

            c_top1, c_top2 = st.columns(2)
            with c_top1:
                sel_bh_soil = st.multiselect(
                    "Select borehole(s)",
                    options=bh_options,
                    default=bh_options,
                    key="spt_soil_plot_boreholes",
                )
            soil_pick_df = spt_for_filter.copy()
            if bh_col and sel_bh_soil:
                soil_pick_df = soil_pick_df[soil_pick_df[bh_col].astype(str).isin({str(v) for v in sel_bh_soil})]
            if "Soil_Type" in soil_pick_df.columns and n_col_soil_pick and n_col_soil_pick in soil_pick_df.columns:
                soil_n = pd.to_numeric(soil_pick_df[n_col_soil_pick], errors="coerce")
                soil_counts = (
                    pd.DataFrame({"Soil_Type": soil_pick_df["Soil_Type"].astype(str), "_n": soil_n})
                    .dropna(subset=["_n"])
                    .groupby("Soil_Type", as_index=False)
                    .size()
                )
                soil_options = sorted(
                    soil_counts.loc[soil_counts["size"] > 0, "Soil_Type"].astype(str).tolist(),
                    key=str.upper,
                )
            else:
                soil_options = []
            with c_top2:
                sel_soils_plot = st.multiselect(
                    "Select soil(s) to plot",
                    options=soil_options,
                    default=soil_options,
                    key="spt_soil_plot_soils",
                )

            symbol_map = {
                "Dot": "circle",
                "X": "x",
                "Cross": "cross",
                "Diamond": "diamond",
                "Square": "square",
                "Triangle Up": "triangle-up",
            }
            c_mark1, c_mark2 = st.columns(2)
            with c_mark1:
                spt_soil_marker_label = st.selectbox(
                    "Marker symbol",
                    options=list(symbol_map.keys()),
                    index=0,
                    key="spt_soil_plot_marker_symbol",
                )
            with c_mark2:
                spt_soil_marker_size = st.number_input(
                    "Marker size",
                    min_value=1.0,
                    max_value=20.0,
                    value=7.0,
                    step=1.0,
                    key="spt_soil_plot_marker_size",
                )

            if not sel_soils_plot:
                st.info("Select at least one soil type.")
            else:
                plot_sig = inspect.signature(plot_spt_vs_level_panels)
                skipped_soils: list[str] = []
                for soil_name in sel_soils_plot:
                    soil_key = _safe_key(soil_name)
                    soil_df = spt_for_filter[spt_for_filter["Soil_Type"].astype(str) == str(soil_name)].copy()
                    if bh_col and sel_bh_soil:
                        soil_df = soil_df[soil_df[bh_col].astype(str).isin({str(v) for v in sel_bh_soil})]
                    n_col_soil_check = _def_col(soil_df, "N_effective", "N", "ISPT_NVAL")
                    y_col = "Level" if "Level" in soil_df.columns else (
                        "Level_Top" if "Level_Top" in soil_df.columns else (
                            "Depth" if "Depth" in soil_df.columns else None
                        )
                    )
                    if (
                        soil_df.empty
                        or not n_col_soil_check
                        or y_col is None
                        or pd.to_numeric(soil_df[n_col_soil_check], errors="coerce").dropna().empty
                        or pd.to_numeric(soil_df[y_col], errors="coerce").dropna().empty
                    ):
                        skipped_soils.append(str(soil_name))
                        continue

                    st.markdown(f"### Soil: {soil_name}")
                    side_col, main_col = st.columns([1, 4], vertical_alignment="top")
                    y_vals = pd.to_numeric(soil_df[y_col], errors="coerce").dropna() if y_col else pd.Series(dtype=float)
                    if not y_vals.empty:
                        if str(y_col).lower().startswith("level"):
                            y_top_default = float(y_vals.max())
                            y_bottom_default = float(y_vals.min())
                        else:
                            y_top_default = float(y_vals.min())
                            y_bottom_default = float(y_vals.max())
                    else:
                        y_top_default, y_bottom_default = 0.0, 50.0

                    with side_col:
                        n_col_soil_axis = _def_col(soil_df, "N_effective", "N", "ISPT_NVAL")
                        if n_col_soil_axis and n_col_soil_axis in soil_df.columns:
                            n_vals_soil = pd.to_numeric(soil_df[n_col_soil_axis], errors="coerce").dropna()
                            soil_x_max_default = float(n_vals_soil.max()) if not n_vals_soil.empty else 200.0
                        else:
                            soil_x_max_default = 200.0
                        c_sx1, c_sx2 = st.columns(2)
                        with c_sx1:
                            sx_min = st.number_input("X-axis min", value=0.0, step=1.0, key=f"spt_soil_{soil_key}_x_min")
                        with c_sx2:
                            sx_max = st.number_input("X-axis max", value=soil_x_max_default, step=1.0, key=f"spt_soil_{soil_key}_x_max")

                        c_si1, c_si2 = st.columns(2)
                        with c_si1:
                            sx_int = st.number_input("X-axis interval", min_value=0.1, value=10.0, step=0.5, key=f"spt_soil_{soil_key}_x_int")
                        with c_si2:
                            sy_int = st.number_input("Y-axis interval", min_value=0.1, value=1.0, step=0.5, key=f"spt_soil_{soil_key}_y_int")

                        c_sy1, c_sy2 = st.columns(2)
                        with c_sy1:
                            sy_top = st.number_input("Y-axis top", value=y_top_default, step=1.0, key=f"spt_soil_{soil_key}_y_top")
                        with c_sy2:
                            sy_bottom = st.number_input("Y-axis bottom", value=y_bottom_default, step=1.0, key=f"spt_soil_{soil_key}_y_bottom")

                        st.markdown("**Percentile Bounds / Design Profile**")
                        c_show_s, c_reset_s = st.columns([3, 1])
                        with c_reset_s:
                            if st.button("Reset", key=f"spt_soil_{soil_key}_bounds_reset_btn", use_container_width=True):
                                st.session_state[f"spt_soil_{soil_key}_show_bounds"] = False
                                st.session_state[f"spt_soil_{soil_key}_bound_color"] = "#808080"
                                st.session_state[f"spt_soil_{soil_key}_design_color"] = "#000080"
                                st.session_state[f"spt_soil_{soil_key}_bin"] = 1.0
                                st.session_state[f"spt_soil_{soil_key}_plow"] = 0.05
                                st.session_state[f"spt_soil_{soil_key}_phigh"] = 0.95
                                st.session_state[f"spt_soil_{soil_key}_ymax"] = float(sy_top)
                                st.session_state[f"spt_soil_{soil_key}_ymin"] = float(sy_bottom)
                                st.session_state[f"spt_soil_{soil_key}_xmin_lim"] = float(sx_min)
                                st.session_state[f"spt_soil_{soil_key}_xmax_lim"] = float(sx_max)
                                st.session_state[f"spt_soil_{soil_key}_zones_text"] = f"{float(sy_top):g},{float(sy_bottom):g},1"
                                st.session_state[f"spt_soil_{soil_key}_use_manual"] = False
                                st.session_state[f"spt_soil_{soil_key}_design_c"] = 19.752
                                st.session_state[f"spt_soil_{soil_key}_design_m"] = -2.390
                                st.session_state[f"spt_soil_{soil_key}_lower_c"] = 15.000
                                st.session_state[f"spt_soil_{soil_key}_lower_m"] = -2.390
                                st.session_state[f"spt_soil_{soil_key}_upper_c"] = 25.000
                                st.session_state[f"spt_soil_{soil_key}_upper_m"] = -2.390
                                st.rerun()
                        with c_show_s:
                            s_show_bounds = st.checkbox("Show bounds lines", value=False, key=f"spt_soil_{soil_key}_show_bounds")
                        c_bc, c_dc, c_pt = st.columns(3)
                        with c_bc:
                            st.caption("Bound")
                            s_bound_color = st.color_picker(
                                "Bound line color",
                                value="#808080",
                                key=f"spt_soil_{soil_key}_bound_color",
                                label_visibility="collapsed",
                            )
                        with c_dc:
                            st.caption("Design")
                            s_design_color = st.color_picker(
                                "Design line color",
                                value="#000080",
                                key=f"spt_soil_{soil_key}_design_color",
                                label_visibility="collapsed",
                            )
                        with c_pt:
                            st.caption("Dot")
                            striplog_custom = st.session_state.get("striplog_colors", {})
                            striplog_default = build_color_map([str(soil_name)]).get(str(soil_name), "#888888")
                            soil_dot_default = str(striplog_custom.get(str(soil_name), striplog_default))
                            s_dot_color = st.color_picker(
                                "Dot color",
                                value=soil_dot_default,
                                key=f"spt_soil_{soil_key}_dot_color",
                                label_visibility="collapsed",
                            )
                        s_bin = st.number_input("Bin size (m)", min_value=0.1, value=1.0, step=0.1, key=f"spt_soil_{soil_key}_bin")
                        c_pl, c_ph = st.columns(2)
                        with c_pl:
                            s_plow = st.number_input("Low percentile", min_value=0.0, max_value=0.49, value=0.05, step=0.01, key=f"spt_soil_{soil_key}_plow")
                        with c_ph:
                            s_phigh = st.number_input("High percentile", min_value=0.51, max_value=1.0, value=0.95, step=0.01, key=f"spt_soil_{soil_key}_phigh")
                        c_y1, c_y2 = st.columns(2)
                        with c_y1:
                            s_ymax = st.number_input("YMAX", value=float(sy_top), step=1.0, key=f"spt_soil_{soil_key}_ymax")
                        with c_y2:
                            s_ymin = st.number_input("YMIN", value=float(sy_bottom), step=1.0, key=f"spt_soil_{soil_key}_ymin")
                        s_zone_text = st.text_area(
                            "Zones (YMAX,YMIN,BIN; one per line)",
                            value=f"{float(s_ymax):g},{float(s_ymin):g},{float(s_bin):g}",
                            key=f"spt_soil_{soil_key}_zones_text",
                            height=96,
                        )
                        s_zone_specs, s_bad_zone_lines = _parse_bounds_zones(s_zone_text, float(s_bin))
                        s_bounds_zones = [(a, b) for a, b, _ in s_zone_specs]
                        s_zone_bin_sizes = [zbin for _, _, zbin in s_zone_specs]
                        if s_zone_specs:
                            zone_preview = " | ".join([f"{a:g},{b:g},{zbin:g}" for a, b, zbin in s_zone_specs[:5]])
                            st.caption(f"Parsed zones: {len(s_bounds_zones)} ({zone_preview})")
                        c_x1, c_x2 = st.columns(2)
                        with c_x1:
                            s_xmin_lim = st.number_input("XMIN limit", value=float(sx_min), step=1.0, key=f"spt_soil_{soil_key}_xmin_lim")
                        with c_x2:
                            s_xmax_lim = st.number_input("XMAX limit", value=float(sx_max), step=1.0, key=f"spt_soil_{soil_key}_xmax_lim")

                        s_use_manual = st.checkbox(
                            "Use manual design equation (N = C + mZ)",
                            value=False,
                            key=f"spt_soil_{soil_key}_use_manual",
                        )
                        auto_eq = compute_spt_bounds_equations(
                            soil_df,
                            merged_f,
                            selected_boreholes=sel_bh_soil,
                            selected_soil_types=None,
                            bounds_bin_size=float(s_bin),
                            bounds_p_low=float(s_plow),
                            bounds_p_high=float(s_phigh),
                            bounds_y_max=float(s_ymax),
                            bounds_y_min=float(s_ymin),
                            bounds_x_min=float(s_xmin_lim),
                            bounds_x_max=float(s_xmax_lim),
                        )
                        if auto_eq is not None:
                            low_m, low_c, auto_m, auto_c, up_m, up_c = auto_eq
                            st.caption(f"Auto: N = {auto_c:.3f} + {abs(auto_m):.3f} Z")
                            if not s_use_manual:
                                st.session_state[f"spt_soil_{soil_key}_design_c"] = float(auto_c)
                                st.session_state[f"spt_soil_{soil_key}_design_m"] = float(-abs(auto_m))
                                st.session_state[f"spt_soil_{soil_key}_lower_c"] = float(low_c)
                                st.session_state[f"spt_soil_{soil_key}_lower_m"] = float(-abs(low_m))
                                st.session_state[f"spt_soil_{soil_key}_upper_c"] = float(up_c)
                                st.session_state[f"spt_soil_{soil_key}_upper_m"] = float(-abs(up_m))
                        else:
                            st.caption("Auto: insufficient data for design equation")
                        s_dc = float(st.session_state.get(f"spt_soil_{soil_key}_design_c", 19.752))
                        s_dm = float(st.session_state.get(f"spt_soil_{soil_key}_design_m", -2.390))
                        s_lc = float(st.session_state.get(f"spt_soil_{soil_key}_lower_c", 15.000))
                        s_lm = float(st.session_state.get(f"spt_soil_{soil_key}_lower_m", -2.390))
                        s_uc = float(st.session_state.get(f"spt_soil_{soil_key}_upper_c", 25.000))
                        s_um = float(st.session_state.get(f"spt_soil_{soil_key}_upper_m", -2.390))
                        s_zone_manual_params = None
                        if s_use_manual:
                            if len(s_bounds_zones) > 1:
                                rows = []
                                for (z_top, z_bottom), z_bin in zip(s_bounds_zones, s_zone_bin_sizes):
                                    z_auto = compute_spt_bounds_equations(
                                        soil_df,
                                        merged_f,
                                        selected_boreholes=sel_bh_soil,
                                        selected_soil_types=None,
                                        bounds_bin_size=float(z_bin),
                                        bounds_p_low=float(s_plow),
                                        bounds_p_high=float(s_phigh),
                                        bounds_y_max=float(z_top),
                                        bounds_y_min=float(z_bottom),
                                        bounds_x_min=float(s_xmin_lim),
                                        bounds_x_max=float(s_xmax_lim),
                                    )
                                    if z_auto is not None:
                                        low_m, low_c, d_m, d_c, up_m, up_c = z_auto
                                        rows.append(
                                            {
                                                "zone": f"{z_top:g},{z_bottom:g},{z_bin:g}",
                                                "design_c": float(d_c),
                                                "design_m": float(-abs(d_m)),
                                                "lower_c": float(low_c),
                                                "lower_m": float(-abs(low_m)),
                                                "upper_c": float(up_c),
                                                "upper_m": float(-abs(up_m)),
                                            }
                                        )
                                    else:
                                        rows.append(
                                            {
                                                "zone": f"{z_top:g},{z_bottom:g},{z_bin:g}",
                                                "design_c": float(s_dc),
                                                "design_m": float(s_dm),
                                                "lower_c": float(s_lc),
                                                "lower_m": float(s_lm),
                                                "upper_c": float(s_uc),
                                                "upper_m": float(s_um),
                                            }
                                        )
                                zdf = pd.DataFrame(rows)
                                zdf_edit = st.data_editor(
                                    zdf,
                                    hide_index=True,
                                    use_container_width=True,
                                    num_rows="fixed",
                                    column_config={"zone": st.column_config.TextColumn(disabled=True)},
                                    key=f"spt_soil_{soil_key}_zone_manual_editor",
                                )
                                s_zone_manual_params = []
                                for _, zr in zdf_edit.iterrows():
                                    s_zone_manual_params.append(
                                        {
                                            "design_c": float(zr["design_c"]),
                                            "design_m": float(zr["design_m"]),
                                            "lower_c": float(zr["lower_c"]),
                                            "lower_m": float(zr["lower_m"]),
                                            "upper_c": float(zr["upper_c"]),
                                            "upper_m": float(zr["upper_m"]),
                                        }
                                    )
                            else:
                                c1m, c2m = st.columns(2)
                                with c1m:
                                    s_dc = st.number_input("C (intercept)", value=float(st.session_state.get(f"spt_soil_{soil_key}_design_c", 19.752)), step=0.001, format="%.3f", key=f"spt_soil_{soil_key}_design_c")
                                with c2m:
                                    s_dm = st.number_input("m (slope) (Negetive)", value=float(st.session_state.get(f"spt_soil_{soil_key}_design_m", -2.390)), step=0.001, format="%.3f", key=f"spt_soil_{soil_key}_design_m")
                                c3m, c4m = st.columns(2)
                                with c3m:
                                    s_lc = st.number_input("Lower C", value=float(st.session_state.get(f"spt_soil_{soil_key}_lower_c", 15.000)), step=0.001, format="%.3f", key=f"spt_soil_{soil_key}_lower_c")
                                with c4m:
                                    s_lm = st.number_input("Lower m", value=float(st.session_state.get(f"spt_soil_{soil_key}_lower_m", -2.390)), step=0.001, format="%.3f", key=f"spt_soil_{soil_key}_lower_m")
                                c5m, c6m = st.columns(2)
                                with c5m:
                                    s_uc = st.number_input("Upper C", value=float(st.session_state.get(f"spt_soil_{soil_key}_upper_c", 25.000)), step=0.001, format="%.3f", key=f"spt_soil_{soil_key}_upper_c")
                                with c6m:
                                    s_um = st.number_input("Upper m", value=float(st.session_state.get(f"spt_soil_{soil_key}_upper_m", -2.390)), step=0.001, format="%.3f", key=f"spt_soil_{soil_key}_upper_m")
                        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                    invalid_msg = None
                    if sx_max <= sx_min:
                        invalid_msg = "X-axis max must be greater than X-axis min."
                    elif sx_int <= 0 or sy_int <= 0:
                        invalid_msg = "Axis interval must be greater than 0."
                    elif sy_top == sy_bottom:
                        invalid_msg = "Y-axis top and bottom cannot be the same."
                    elif s_phigh <= s_plow:
                        invalid_msg = "High percentile must be greater than low percentile."
                    elif s_show_bounds and (not s_bounds_zones):
                        invalid_msg = "Please input at least one valid zone (YMAX to YMIN)."
                    elif s_show_bounds and s_bad_zone_lines:
                        invalid_msg = f"Invalid zone line(s): {', '.join(s_bad_zone_lines[:3])}"
                    elif s_xmax_lim <= s_xmin_lim:
                        invalid_msg = "Function 5 XMAX limit must be greater than XMIN limit."

                    if invalid_msg:
                        with side_col:
                            st.warning(invalid_msg)
                        continue

                    cached_fig_key = f"spt_soil_cached_fig_{soil_key}"
                    has_cached_fig = cached_fig_key in st.session_state
                    should_recompute = True
                    if not should_recompute and has_cached_fig:
                        fig_soil = st.session_state[cached_fig_key]
                    else:
                        eq_m = float(s_dm) if (s_show_bounds and s_use_manual) else None
                        eq_c = float(s_dc) if (s_show_bounds and s_use_manual) else None
                        striplog_custom = st.session_state.get("striplog_colors", {})
                        striplog_defaults = build_color_map([soil_name])
                        soil_colors_for_spt = {
                            str(soil_name): str(striplog_custom.get(str(soil_name), striplog_defaults.get(str(soil_name), "#888888")))
                        }
                        plot_kwargs = {
                            "selected_boreholes": sel_bh_soil,
                        }
                        if "soil_color_map" in plot_sig.parameters:
                            plot_kwargs["soil_color_map"] = dict(soil_colors_for_spt)
                        if "x_min" in plot_sig.parameters:
                            plot_kwargs["x_min"] = float(sx_min)
                        if "x_max" in plot_sig.parameters:
                            plot_kwargs["x_max"] = float(sx_max)
                        if "x_dtick" in plot_sig.parameters:
                            plot_kwargs["x_dtick"] = float(sx_int)
                        if "marker_symbol" in plot_sig.parameters:
                            plot_kwargs["marker_symbol"] = symbol_map.get(spt_soil_marker_label, "circle")
                        if "marker_size" in plot_sig.parameters:
                            plot_kwargs["marker_size"] = float(spt_soil_marker_size)
                        if "y_top" in plot_sig.parameters:
                            plot_kwargs["y_top"] = float(sy_top)
                        if "y_bottom" in plot_sig.parameters:
                            plot_kwargs["y_bottom"] = float(sy_bottom)
                        if "y_dtick" in plot_sig.parameters:
                            plot_kwargs["y_dtick"] = float(sy_int)
                        if "show_bounds" in plot_sig.parameters:
                            plot_kwargs["show_bounds"] = bool(s_show_bounds)
                        if "bounds_bin_size" in plot_sig.parameters:
                            plot_kwargs["bounds_bin_size"] = float(s_bin)
                        if "bounds_p_low" in plot_sig.parameters:
                            plot_kwargs["bounds_p_low"] = float(s_plow)
                        if "bounds_p_high" in plot_sig.parameters:
                            plot_kwargs["bounds_p_high"] = float(s_phigh)
                        if "bounds_y_max" in plot_sig.parameters:
                            plot_kwargs["bounds_y_max"] = float(s_ymax)
                        if "bounds_y_min" in plot_sig.parameters:
                            plot_kwargs["bounds_y_min"] = float(s_ymin)
                        if "bounds_zones" in plot_sig.parameters:
                            plot_kwargs["bounds_zones"] = list(s_bounds_zones)
                        if "bounds_zone_bin_sizes" in plot_sig.parameters:
                            plot_kwargs["bounds_zone_bin_sizes"] = list(s_zone_bin_sizes)
                        if "bounds_x_min" in plot_sig.parameters:
                            plot_kwargs["bounds_x_min"] = float(s_xmin_lim)
                        if "bounds_x_max" in plot_sig.parameters:
                            plot_kwargs["bounds_x_max"] = float(s_xmax_lim)
                        if "bounds_line_color" in plot_sig.parameters:
                            plot_kwargs["bounds_line_color"] = str(s_bound_color)
                        if "design_line_color" in plot_sig.parameters:
                            plot_kwargs["design_line_color"] = str(s_design_color)
                        if "design_equation_m" in plot_sig.parameters and eq_m is not None:
                            plot_kwargs["design_equation_m"] = float(eq_m)
                        if "design_equation_c" in plot_sig.parameters and eq_c is not None:
                            plot_kwargs["design_equation_c"] = float(eq_c)
                        if s_show_bounds and s_use_manual and s_zone_manual_params is not None:
                            if "zone_manual_params" in plot_sig.parameters:
                                plot_kwargs["zone_manual_params"] = list(s_zone_manual_params)
                        elif s_show_bounds and s_use_manual:
                            if "lower_bound_m" in plot_sig.parameters:
                                plot_kwargs["lower_bound_m"] = float(s_lm)
                            if "lower_bound_c" in plot_sig.parameters:
                                plot_kwargs["lower_bound_c"] = float(s_lc)
                            if "upper_bound_m" in plot_sig.parameters:
                                plot_kwargs["upper_bound_m"] = float(s_um)
                            if "upper_bound_c" in plot_sig.parameters:
                                plot_kwargs["upper_bound_c"] = float(s_uc)

                        fig_soil = plot_spt_vs_level_panels(
                            soil_df,
                            merged_f,
                            **plot_kwargs,
                        )
                        # Soil tab requirement:
                        # - one soil => one graph only
                        # - all points use same color
                        uniform_point_color = str(s_dot_color)
                        row1_traces = []
                        for tr in fig_soil.data:
                            tr_yaxis = getattr(tr, "yaxis", None)
                            if tr_yaxis in (None, "y"):
                                if getattr(tr, "mode", "") and "markers" in str(tr.mode):
                                    tr.marker.color = uniform_point_color
                                    if getattr(tr.marker, "line", None) is not None:
                                        tr.marker.line.color = uniform_point_color
                                    tr.showlegend = False
                                row1_traces.append(tr)
                        fig_soil.data = tuple(row1_traces)

                        anns = []
                        for ann in list(fig_soil.layout.annotations or []):
                            xref = getattr(ann, "xref", None)
                            text = str(getattr(ann, "text", "") or "")
                            if xref == "x2":
                                continue
                            if "by Soil_Type" in text:
                                continue
                            anns.append(ann)
                        fig_soil.update_layout(annotations=anns)
                        fig_soil.update_xaxes(domain=[0.0, 1.0], row=1, col=1)
                        fig_soil.update_yaxes(domain=[0.0, 1.0], row=1, col=1)
                        fig_soil.update_layout(
                            xaxis2=dict(visible=False),
                            yaxis2=dict(visible=False),
                            height=760,
                            margin=dict(l=20, r=20, t=70, b=70),
                        )
                        st.session_state[cached_fig_key] = fig_soil

                    with main_col:
                        st.plotly_chart(fig_soil, use_container_width=True, config={"scrollZoom": False})
                    with side_col:
                        try:
                            img = fig_soil.to_image(format="png", engine="kaleido")
                            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                            st.download_button(
                                f"Download {soil_name}.png",
                                img,
                                f"{file_prefix}_spt_plot_{_safe_key(soil_name)}.png",
                                "image/png",
                                key=f"dl_spt_soil_{soil_key}",
                            )
                        except Exception:
                            st.caption("Install kaleido for PNG export: pip install kaleido")
                    st.markdown("---")
                if skipped_soils:
                    st.caption(f"Skipped soils with no data: {', '.join(skipped_soils)}")
        else:
            st.info("No SPT data.")

    elif tab_choice == "Validation":
        err_df, warn_df = val
        st.subheader("Errors")
        st.dataframe(err_df, use_container_width=True)
        st.subheader("Warnings")
        st.dataframe(warn_df, use_container_width=True)
        val_json = json.dumps({"errors": err_df.to_dict("records"), "warnings": warn_df.to_dict("records")})
        st.download_button("Download validation.json", val_json.encode(), f"{file_prefix}_validation.json", "application/json")

    elif tab_choice == "Summary":
        st.text(report)
        st.download_button("Download summary_report.txt", report.encode(), f"{file_prefix}_summary_report.txt", "text/plain")

    else:  # Strip Log Plot
        # Color customization: select colors then click GEN to apply
        code_col = "Normalized_Code" if "Normalized_Code" in merged_f.columns else (
            "Code" if "Code" in merged_f.columns else merged_f.columns[0]
        )
        codes_in_data = sorted(merged_f[code_col].dropna().unique().astype(str).tolist(), key=str.upper)
        default_colors = build_color_map(codes_in_data)
        if "striplog_colors" not in st.session_state:
            st.session_state.striplog_colors = {}
        color_overrides = {}
        with st.expander("Customize colors", expanded=False):
            with st.form("color_form"):
                n_cols = min(4, max(1, len(codes_in_data)))
                cols = st.columns(n_cols)
                for i, code in enumerate(codes_in_data):
                    with cols[i % n_cols]:
                        current = st.session_state.striplog_colors.get(code, default_colors.get(code, "#888888"))
                        color_overrides[code] = st.color_picker(code, value=current, key=f"striplog_cp_{code}")
                gen_clicked = st.form_submit_button("GEN")
            if gen_clicked:
                st.session_state.striplog_colors = dict(color_overrides)
        # Use committed colors for chart (from GEN) or defaults
        colors_for_chart = {c: st.session_state.striplog_colors.get(c, default_colors.get(c, "#888888")) for c in codes_in_data}
        st.markdown("**Current color preferences**")
        if colors_for_chart:
            pref_cols = st.columns(min(4, max(1, len(colors_for_chart))))
            for i, code in enumerate(sorted(colors_for_chart.keys(), key=str.upper)):
                color_val = str(colors_for_chart.get(code, "#888888"))
                with pref_cols[i % len(pref_cols)]:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0;'>"
                        f"<span style='display:inline-block;width:14px;height:14px;border:1px solid #999;background:{color_val};'></span>"
                        f"<span><b>{code}</b> {color_val}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No soil codes found for color preferences.")

        use_static_png = st.checkbox("Static ON-SCALE PNG (recommended)", value=True, key="striplog_static_png")
        show_nvalue = st.checkbox("Show N-value on plot", value=True, key="striplog_show_nvalue")
        spt_for_plot = spt_f if show_nvalue else None

        if use_static_png:
            bh_ids = sorted(merged_f["Borehole_ID"].unique().tolist(), key=_natural_sort_key) if "Borehole_ID" in merged_f.columns else []
            try:
                png_bytes = render_striplog_png(
                    merged_f,
                    spt_for_plot,
                    gl_df,
                    bh_ids,
                    legend_codes=codes_in_data,
                    color_map=colors_for_chart,
                    spt_level_mode=spt_level_mode,
                )
                st.image(png_bytes, use_container_width=True)
                st.download_button("Download striplog.png", png_bytes, f"{file_prefix}_striplog.png", "image/png", key="dl_static_png")

                st.markdown("---")
                st.subheader("A3 Print-ready PNG")
                bh_per_page = st.slider(
                    "Boreholes per A3 page",
                    min_value=4, max_value=15, value=DEFAULT_BH_PER_PAGE,
                    help="Split boreholes into multiple A3 pages for readable output",
                    key="a3_bh_per_page",
                )
                try:
                    pages = render_striplog_a3_pages(
                        merged_f, spt_for_plot, gl_df, bh_ids,
                        bh_per_page=bh_per_page,
                        legend_codes=codes_in_data, color_map=colors_for_chart,
                        spt_level_mode=spt_level_mode,
                    )
                    n_pages = len(pages)
                    st.caption(f"{len(bh_ids)} boreholes → {n_pages} page(s), {bh_per_page} BH/page")
                    for pg_idx, (pg_bytes, pg_name) in enumerate(pages):
                        with st.expander(f"Page {pg_idx + 1} / {n_pages}  ({pg_name})", expanded=(pg_idx == 0)):
                            st.image(pg_bytes, use_container_width=True)
                            st.download_button(
                                f"Download {pg_name}", pg_bytes, f"{file_prefix}_{pg_name}", "image/png",
                                key=f"dl_a3_pg_{pg_idx}",
                            )
                    try:
                        pdf_bytes = _pages_png_to_pdf_bytes(pages)
                        if pdf_bytes:
                            st.download_button(
                                "Export all A3 pages (.pdf)",
                                pdf_bytes,
                                f"{file_prefix}_striplog_a3_all.pdf",
                                "application/pdf",
                                key="dl_a3_all_pdf",
                            )
                    except Exception as e_pdf:
                        st.caption(f"A3 PDF export unavailable: {e_pdf}")
                except Exception as e2:
                    st.error(f"A3 generation failed: {e2}")
                    import traceback
                    st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"Static PNG failed: {e}")
                import traceback
                st.code(traceback.format_exc())

        else:
            # Plotly interactive version
            st.markdown(
                """
                <style>
                div[data-testid="stVerticalBlock"]:has(div.js-plotly-plot) {
                    max-height: 88vh;
                    overflow-y: auto;
                    overflow-x: auto;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 8px;
                }
                div[data-testid="stVerticalBlock"]:has(div.js-plotly-plot)::-webkit-scrollbar {
                    width: 10px;
                    height: 10px;
                }
                div[data-testid="stVerticalBlock"]:has(div.js-plotly-plot)::-webkit-scrollbar-track {
                    background: #f1f1f1;
                    border-radius: 4px;
                }
                div[data-testid="stVerticalBlock"]:has(div.js-plotly-plot)::-webkit-scrollbar-thumb {
                    background: #888;
                    border-radius: 4px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            fig = plot_striplog(
                merged_f, spt_for_plot, gl_df,
                sorted(merged_f["Borehole_ID"].unique().tolist(), key=_natural_sort_key) if "Borehole_ID" in merged_f.columns else None,
                color_override=colors_for_chart,
                spt_level_mode=spt_level_mode,
            )
            st.plotly_chart(fig, use_container_width=False, config={"scrollZoom": False})
            try:
                img = fig.to_image(format="png", engine="kaleido")
                st.download_button("Download striplog.png", img, f"{file_prefix}_striplog.png", "image/png", key="dl_plotly_png")
            except Exception:
                st.caption("Install kaleido for PNG export: pip install kaleido")

else:
    st.info("Upload AGS or CSV files and click Run.")
