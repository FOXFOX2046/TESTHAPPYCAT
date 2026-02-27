"""Test normalization: CORESTONE, CDI/CDS/CDM families."""
import pandas as pd
import pytest

from core.normalization import normalize_layers
from core.rock_dict import load_rock_sets
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ROCK_CSV = DATA_DIR / "AGS-completed.csv"


def test_corestone_dominance():
    rock_sets = load_rock_sets(ROCK_CSV)
    df = pd.DataFrame({
        "Borehole_ID": ["BH1"],
        "Depth_From": [0],
        "Depth_To": [5],
        "Description": ["Completely decomposed granite with CORESTONE core"],
        "Label1": ["GRANITE"],
        "Label2": ["CDG"],
    })
    out = normalize_layers(df, rock_sets=rock_sets)
    assert out.iloc[0]["Normalized_Code"] == "CORESTONE"


def test_moderately_decomposed_arkose_mds():
    rock_sets = load_rock_sets(ROCK_CSV)
    df = pd.DataFrame({
        "Borehole_ID": ["BH1"],
        "Depth_From": [0],
        "Depth_To": [5],
        "Description": ["Moderately decomposed ARKOSE"],
        "Label1": ["ARKOSE"],
        "Label2": ["MDS"],
    })
    out = normalize_layers(df, rock_sets=rock_sets)
    assert out.iloc[0]["Normalized_Code"] == "MDS"
    assert out.iloc[0]["Rock_Category"] == "SEDIMENTARY"


def test_completely_decomposed_andesite_cdi():
    rock_sets = load_rock_sets(ROCK_CSV)
    df = pd.DataFrame({
        "Borehole_ID": ["BH1"],
        "Depth_From": [0],
        "Depth_To": [5],
        "Description": ["Completely decomposed ANDESITE"],
        "Label1": ["ANDESITE"],
        "Label2": ["CDI"],
    })
    out = normalize_layers(df, rock_sets=rock_sets)
    assert out.iloc[0]["Normalized_Code"] == "CDI"
    assert out.iloc[0]["Rock_Category"] == "IGNEOUS"


def test_highly_decomposed_blueschist_hdm():
    rock_sets = load_rock_sets(ROCK_CSV)
    df = pd.DataFrame({
        "Borehole_ID": ["BH1"],
        "Depth_From": [0],
        "Depth_To": [5],
        "Description": ["Highly decomposed BLUESCHIST"],
        "Label1": ["BLUESCHIST"],
        "Label2": ["HDM"],
    })
    out = normalize_layers(df, rock_sets=rock_sets)
    assert out.iloc[0]["Normalized_Code"] == "HDM"
    assert out.iloc[0]["Rock_Category"] == "METAMORPHIC"
