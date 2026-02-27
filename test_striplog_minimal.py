"""Minimal striplog test - no AGS parsing."""
import pandas as pd
from core.plots_striplog import plot_striplog

# Minimal test data
merged = pd.DataFrame({
    "Borehole_ID": ["BH1", "BH1", "BH1"],
    "Depth_From": [0, 2, 5],
    "Depth_To": [2, 5, 10],
    "Normalized_Code": ["FILL", "CDG", "CDG"],
})
gl_df = pd.DataFrame({"Borehole_ID": ["BH1"], "GL": [10.0]})
spt_df = pd.DataFrame({"Borehole_ID": ["BH1"], "Depth": [3.0], "N_effective": [25], "Flag": [""]})

fig = plot_striplog(merged, spt_df, gl_df, ["BH1"])
fig.write_html("test_striplog_minimal.html")
print("Wrote test_striplog_minimal.html")
