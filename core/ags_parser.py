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


def parse_ags4_text(text: str) -> dict[str, pd.DataFrame]:
    """Parse AGS 4 ``GROUP``/``HEADING``/``DATA`` records into DataFrames."""
    groups: dict[str, pd.DataFrame] = {}
    active_group: str | None = None
    headers: list[str] = []
    rows: list[list[str]] = []

    def save_group() -> None:
        if active_group and headers:
            groups[active_group] = pd.DataFrame(rows, columns=headers)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = safe_csv_split(line)
        if not parts:
            continue

        record = str(parts[0]).strip().upper()
        if record == "GROUP":
            save_group()
            active_group = str(parts[1]).strip() if len(parts) > 1 else ""
            headers = []
            rows = []
        elif record == "HEADING" and active_group:
            headers = [str(value).strip().lstrip("*") for value in parts[1:]]
        elif record == "DATA" and active_group and headers:
            rows.append([
                str(parts[index]).replace('"', "").strip() if index < len(parts) else ""
                for index in range(1, len(headers) + 1)
            ])

    save_group()
    return groups


def parse_ags_text(text: str) -> dict[str, pd.DataFrame]:
    """
    Parse AGS text content into group DataFrames.
    - Group start: lines beginning with "**" (e.g. "**GEOL", "**LOCA", "**ISPT", "**SPT")
    - Ignore lines beginning with "*" (single) and <UNITS>
    - <CONT>: first field "<CONT>" appends to previous row
    """
    if any(
        safe_csv_split(line.strip())[:1] == ["GROUP"]
        for line in text.splitlines()
        if line.strip()
    ):
        return parse_ags4_text(text)

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
        if parts and (
            str(parts[0]).strip().upper() == "<UNITS>"
            or all(str(part).strip() == "*" for part in parts if str(part).strip())
        ):
            continue

        if active_group is None:
            continue

        # <CONT> continuation: append to previous row.
        # Field 0 is the <CONT> marker.  Continuation values retain their
        # original field indexes, so field 1 appends to column 1, etc.
        if parts and str(parts[0]).strip().upper() == "<CONT>":
            if prev_row is not None:
                for i in range(1, len(prev_row)):
                    if i >= len(parts):
                        break
                    cont_val = str(parts[i]) if parts[i] else ""
                    if not cont_val:
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
