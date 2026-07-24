from __future__ import annotations

import pandas as pd


def apply_text_filter(df: pd.DataFrame, column: str, raw_filter: str) -> pd.DataFrame:
    """Filter a text column by single partial text or comma-separated exact tokens."""
    if not raw_filter or column not in df.columns:
        return df

    value = str(raw_filter).strip()
    tokens = [tok.strip() for tok in value.replace("\n", ",").split(",") if tok.strip()]
    series = df[column].astype(str)

    if len(tokens) > 1:
        wanted = {tok.upper() for tok in tokens}
        return df[series.str.upper().isin(wanted)]

    return df[series.str.contains(value, case=False, na=False, regex=False)]
