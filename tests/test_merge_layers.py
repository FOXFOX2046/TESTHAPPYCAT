"""Test merge: adjacency rule."""
import pandas as pd
import pytest

from core.merge_layers import merge_adjacent_layers


def test_merge_adjacent_same_code():
    df = pd.DataFrame({
        "Borehole_ID": ["BH1", "BH1"],
        "Depth_From": [0, 3],
        "Depth_To": [3, 6],
        "Normalized_Code": ["CDG", "CDG"],
    })
    out = merge_adjacent_layers(df)
    assert len(out) == 1
    assert out.iloc[0]["Depth_From"] == 0
    assert out.iloc[0]["Depth_To"] == 6
    assert out.iloc[0]["Merged_Count"] == 2


def test_no_merge_different_code():
    df = pd.DataFrame({
        "Borehole_ID": ["BH1", "BH1"],
        "Depth_From": [0, 3],
        "Depth_To": [3, 6],
        "Normalized_Code": ["CDG", "HDG"],
    })
    out = merge_adjacent_layers(df)
    assert len(out) == 2


def test_no_merge_gap():
    df = pd.DataFrame({
        "Borehole_ID": ["BH1", "BH1"],
        "Depth_From": [0, 4],
        "Depth_To": [3, 6],
        "Normalized_Code": ["CDG", "CDG"],
    })
    out = merge_adjacent_layers(df)
    assert len(out) == 2
