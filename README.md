# FoxGI WebApp

Streamlit app for AGS/CSV geotechnical processing, SPT plotting, and strip log generation.

## Live app

- https://testhappycat-2046.streamlit.app/

## What it does

- Parse AGS groups and CSV inputs
- Clean and normalize geology layers
- Merge adjacent layers
- Process SPT and compute `N_effective` / `N200`
- Validate depth continuity (errors + warnings)
- Plot:
  - `SPT Plot`
  - `SPT Plot (Soil)`
  - `Strip Log Plot` (interactive + static on-scale)
- Export images and A3 multi-page PDF (strip log)

## Repository layout

- `app.py`: Streamlit entrypoint
- `core/`: processing and plotting modules
- `tests/`: pytest test suite
- `scripts/`: Windows helper scripts
- `Sample/`: sample AGS files
- `.streamlit/config.toml`: Streamlit settings

## Requirements

- Python 3.11+ recommended
- Dependencies from `requirements.txt`

Install:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Windows shortcut:

```bat
scripts\run_app.bat
```

Alternate port:

```bat
scripts\run_app_port.bat 8502
```

## Input notes

- AGS mode: upload one or more `.ags` files
- CSV mode: expects borehole/depth/description style columns
- Sample file: `Sample/11 0210 03 R002.ags`

## Filters

Sidebar filters apply after `Run`:

- Borehole ID text filter
- Normalized code text filter
- Depth range (`Depth min/max`)
- Level range (`Level min/max`, mPD)

## Testing

Run all tests:

```bash
python -m pytest tests -q
```

Windows shortcut:

```bat
scripts\smoke_test.bat
```

## Packaging

Build zip:

```bat
scripts\build_zip.bat
```

Output: `foxgi_web_app.zip`

## GitHub deploy checklist

1. Ensure local run works: `streamlit run app.py`
2. Run tests: `python -m pytest tests -q`
3. Do not commit local artifacts: `.venv/`
4. Do not commit local artifacts: `.pytest_cache/`
5. Do not commit local artifacts: `__pycache__/`
6. Do not commit generated HTML/PNG test outputs
7. Commit source files (`app.py`, `core/`, `tests/`, `scripts/`, `requirements.txt`, `README.md`, `.streamlit/`)
8. Push to GitHub

## Known issue to fix before strict CI gate

- `tests/test_ags_parser.py` currently has failures around `<CONT>` concatenation behavior.
