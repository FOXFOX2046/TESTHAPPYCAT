"""Test AGS parser: <CONT> concatenation, quote-safe CSV, and AGS 4 records."""
from pathlib import Path

import pandas as pd
import pytest

from core.ags_parser import parse_ags_text, safe_csv_split


def test_safe_csv_split_quotes():
    line = 'a,"b,c",d'
    assert safe_csv_split(line) == ["a", "b,c", "d"]


def test_cont_concatenation():
    text = '''"**GEOL"
"*","*"
"A","B","C"
"x","1","2"
"<CONT>"," more"," text",""
'''
    groups = parse_ags_text(text)
    assert "GEOL" in groups
    df = groups["GEOL"]
    assert len(df) == 1
    assert df.iloc[0]["A"] == "x"
    assert df.iloc[0]["B"] == "1 more"
    assert df.iloc[0]["C"] == "2 text"


def test_cont_append_correct_columns():
    text = '''"**GEOL"
"A","B","C"
"v1","v2","v3"
"<CONT>","a","b","c"
'''
    groups = parse_ags_text(text)
    df = groups["GEOL"]
    assert len(df) == 1
    # CONT field 0 is a marker; fields 1..N append at their original indexes.
    assert df.iloc[0]["A"] == "v1"
    assert df.iloc[0]["B"] == "v2a"
    assert df.iloc[0]["C"] == "v3b"


def test_cont_preserves_geol_depth_fields_and_appends_description_and_legend():
    text = '''"**GEOL"
"HOLE_ID","GEOL_TOP","GEOL_BASE","Description","GEOL_LEG"
"BH10","17.55","36.65","Sand","S"
"<CONT>","",""," with gravel"," and cobbles"
'''

    df = parse_ags_text(text)["GEOL"]

    assert df.iloc[0]["HOLE_ID"] == "BH10"
    assert df.iloc[0]["GEOL_TOP"] == "17.55"
    assert df.iloc[0]["GEOL_BASE"] == "36.65"
    assert df.iloc[0]["Description"] == "Sand with gravel"
    assert df.iloc[0]["GEOL_LEG"] == "S and cobbles"


def test_parse_ags4_group_heading_and_data_records():
    text = '''"GROUP","GEOL"
"HEADING","LOCA_ID","GEOL_TOP"
"DATA","BH1","1.20"
'''

    groups = parse_ags_text(text)

    assert groups["GEOL"].to_dict("records") == [
        {"LOCA_ID": "BH1", "GEOL_TOP": "1.20"}
    ]


def test_parse_uploaded_ags4_file_finds_required_groups():
    path = Path(__file__).parents[2] / "No. 1 Selkirk Road.ags"

    groups = parse_ags_text(path.read_text(encoding="utf-8"))

    assert {"GEOL", "LOCA", "ISPT"}.issubset(groups)
