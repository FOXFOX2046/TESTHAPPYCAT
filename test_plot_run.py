"""Quick test: run full pipeline and generate striplog to verify fix."""
import os

import pandas as pd

from core.ags_parser import parse_ags_text
from core.merge_layers import merge_adjacent_layers
from core.normalization import normalize_layers
from core.plots_striplog import plot_striplog
from core.rock_dict import load_rock_sets, ROCK_DICT_CSV
from core.soil_cleaning import extract_labels
from core.spt_processing import extract_spt

sample = "Sample/11 0210 03 R002.ags"
if not os.path.exists(sample):
    sample = "Sample/NTM.ags"
print(f"Using sample: {sample}")

with open(sample, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

groups = parse_ags_text(text)
geol = groups.get("GEOL")
if geol is None or geol.empty:
    print("No GEOL data"); exit(1)

def col(df, *keys):
    for k in keys:
        for c in df.columns:
            if k.upper() in str(c).upper():
                return c
    return None

bh_c = col(geol, "HOLE_ID", "LOCA_ID") or geol.columns[0]
top_c = col(geol, "GEOL_TOP", "TOP")
base_c = col(geol, "GEOL_BASE", "BASE")
desc_c = col(geol, "GEOL_DESC", "DESC") or geol.columns[-1]
rename = {bh_c: "Borehole_ID"}
if top_c: rename[top_c] = "Depth_From"
if base_c: rename[base_c] = "Depth_To"
if desc_c: rename[desc_c] = "Description"

layers = geol.rename(columns=rename)
for dc in ["Depth_From", "Depth_To"]:
    if dc in layers.columns:
        layers[dc] = pd.to_numeric(layers[dc], errors="coerce").fillna(0)
labels = [extract_labels(str(d)) for d in layers["Description"]]
for k in ["Description_Original", "Description_Clean", "Label1", "Label2", "FinalLabel"]:
    layers[k] = [lb[k] for lb in labels]
rock_sets = load_rock_sets(ROCK_DICT_CSV)
layers = normalize_layers(layers, rock_sets=rock_sets, desc_col="Description", label1_col="Label1", label2_col="Label2")
merged = merge_adjacent_layers(layers, bh_col="Borehole_ID", code_col="Normalized_Code", from_col="Depth_From", to_col="Depth_To", desc_col="Description")
spt_df = extract_spt(groups)

gl_data = []
for grp_name in ["HOLE", "LOCA"]:
    if grp_name in groups and not gl_data:
        tbl = groups[grp_name].copy()
        lh = col(tbl, "HOLE_ID", "LOCA_ID")
        lg = col(tbl, "HOLE_GL", "LOCA_GL", "GL")
        if lh and lg:
            tbl[lg] = pd.to_numeric(tbl[lg], errors="coerce")
            tbl = tbl.dropna(subset=[lg])
            for _, r in tbl.iterrows():
                bh_id = str(r[lh]).strip()
                if bh_id and not bh_id.startswith("*"):
                    gl_data.append({"Borehole_ID": bh_id, "GL": float(r[lg])})
        break
gl_df = pd.DataFrame(gl_data) if gl_data else None

bh_ids = merged["Borehole_ID"].unique().tolist()
print(f" merged rows: {len(merged)}, boreholes: {bh_ids}")

fig = plot_striplog(merged, spt_df, gl_df, bh_ids)
fig.write_html("test_striplog_out.html")
print("Wrote test_striplog_out.html")
