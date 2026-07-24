import pandas as pd

from core.filters import apply_text_filter


def test_comma_separated_borehole_filter_matches_each_id_exactly():
    df = pd.DataFrame({"Borehole_ID": ["BH1", "TP1", "TP2", "TP3", "TP4", "TP10"]})

    out = apply_text_filter(df, "Borehole_ID", "TP1,TP2,TP3,TP4")

    assert out["Borehole_ID"].tolist() == ["TP1", "TP2", "TP3", "TP4"]


def test_single_text_filter_keeps_partial_match():
    df = pd.DataFrame({"Borehole_ID": ["BH1", "TP1", "TP2"]})

    out = apply_text_filter(df, "Borehole_ID", "TP")

    assert out["Borehole_ID"].tolist() == ["TP1", "TP2"]
