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
    assert df.iloc[0]["A"] == "x more"
    assert df.iloc[0]["B"] == "1 text"


def test_cont_append_correct_columns():
    text = '''"**GEOL"
"A","B","C"
"v1","v2","v3"
"<CONT>","a","b","c"
'''
    groups = parse_ags_text(text)
    df = groups["GEOL"]
    assert len(df) == 1
    # CONT appends: col0+=parts[1]=a, col1+=parts[2]=b, col2+=parts[3]=c
    assert df.iloc[0]["A"] == "v1a"
    assert df.iloc[0]["B"] == "v2b"
    assert df.iloc[0]["C"] == "v3c"


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
