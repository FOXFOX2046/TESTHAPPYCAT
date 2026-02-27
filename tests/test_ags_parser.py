"""Test AGS parser: <CONT> concatenation and quote-safe CSV."""
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
