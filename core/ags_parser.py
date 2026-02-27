"""
AGS Parser: parse .ags files with quote-safe CSV and <CONT> continuation.
Port of VBA BHInfo.bas getData + ParseCSVLine logic.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

import pandas as pd


def safe_csv_split(line: str) -> list[str]:
    """
    Quote-safe CSV line split (equivalent to VBA ParseCSVLine).
    Respects quoted fields; commas inside quotes are not delimiters.
    """
    reader = csv.reader(io.StringIO(line))
    row = next(reader)
    return list(row)


def parse_ags_text(text: str) -> dict[str, pd.DataFrame]:
    """
    Parse AGS text content into group DataFrames.
    - Group start: lines beginning with "**" (e.g. "**GEOL", "**LOCA", "**ISPT", "**SPT")
    - Ignore lines beginning with "*" (single) and <UNITS>
    - <CONT>: first field "<CONT>" appends to previous row
    """
    groups: dict[str, pd.DataFrame] = {}
    active_group: str | None = None
    headers: list[str] = []
    rows: list[list[str]] = []
    prev_row: list[str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            # Save current group before clearing (empty line ends group in AGS)
            if active_group and headers and rows:
                df = pd.DataFrame(rows, columns=headers)
                groups[active_group] = df
            active_group = None
            headers = []
            rows = []
            prev_row = None
            continue

        # Detect group start: "**GROUPNAME"
        m = re.match(r'^"\*\*(.+?)"', line)
        if m:
            # Save previous group if any
            if active_group and headers and rows:
                df = pd.DataFrame(rows, columns=headers)
                groups[active_group] = df

            active_group = m.group(1).strip()
            headers = []
            rows = []
            prev_row = None
            continue

        # Skip only <UNITS> rows (AGS 4.0 uses *FIELD_NAME as headers - don't skip those)
        parts = safe_csv_split(line)
        if parts and str(parts[0]).strip().upper() == "<UNITS>":
            continue

        if active_group is None:
            continue

        # <CONT> continuation: append to previous row.
        # AGS CONT lines have the same field layout as data lines;
        # parts[0] = "<CONT>" sits in the HOLE_ID position, and
        # parts[1..N] correspond to the remaining columns 1..N.
        if parts and str(parts[0]).strip().upper() == "<CONT>":
            if prev_row is not None:
                for i in range(len(prev_row)):
                    if i >= len(parts):
                        break
                    cont_val = str(parts[i]) if parts[i] else ""
                    if i == 0 or not cont_val:
                        continue
                    prev_row[i] = (str(prev_row[i]) if prev_row[i] else "") + cont_val
            continue

        # Normal data row - use as header if no headers yet, else data
        if not headers:
            # Strip * prefix from AGS field names (e.g. *HOLE_ID -> HOLE_ID)
            headers = [str(p).strip().strip('"').lstrip("*") for p in parts if str(p).strip()]
            if not headers:
                continue
        else:
            first_val = str(parts[0]).strip() if parts else ""
            if first_val.startswith("*"):
                # Header continuation line (multi-line headers) — extend
                extra = [str(p).strip().strip('"').lstrip("*") for p in parts if str(p).strip()]
                headers.extend(extra)
                continue
            if len(parts) >= len(headers):
                row = [str(parts[i]).replace('"', "").strip() if i < len(parts) else "" for i in range(len(headers))]
                prev_row = row
                rows.append(row)
            elif len(parts) > 0:
                row = [str(parts[i]).replace('"', "").strip() if i < len(parts) else "" for i in range(len(headers))]
                prev_row = row
                rows.append(row)

    # Save last group
    if active_group and headers and rows:
        df = pd.DataFrame(rows, columns=headers)
        groups[active_group] = df

    return groups


def parse_ags_file(path: str | Path) -> dict[str, pd.DataFrame]:
    """Parse AGS file from path."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_ags_text(text)
