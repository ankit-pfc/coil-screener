# Agent Notes

## Project

This repository is a research-to-product buildout for long-duration "coil effect" stock screening.

The current implementation has two layers:

- `screen_monthly.py`: quantitative monthly OHLCV screener and CLI.
- `app.py` plus `static/`: FastAPI backend and plain HTML/CSS/JS review UI.

## Current State

- The pipeline is a working prototype, not a production app.
- Market data currently comes from `yfinance`.
- Scoring criteria now flow through `ScreenConfig` in `screen_monthly.py`.
- The UI exposes ticker/universe inputs, saved CSV runs, ranked results, chart inspection, feature readout, and a first set of scoring controls.
- Live API/UI screens persist local run artifacts under ignored `storage/runs/<run_id>/` folders by default.
- Each persisted run stores `metadata.json` and `results.csv`.
- Runtime storage is platform-neutral and configured with `COIL_STORAGE_DIR`, `COIL_RUNS_DIR`, and `COIL_CACHE_DIR`.

## Product Direction

Build toward a reliable app where a pro user can:

- run the monthly screening pipeline repeatedly
- inspect per-ticker chart geometry
- see why a ticker scored well or poorly
- adjust screening criteria with visible knobs and switches
- save runs with their exact parameter set
- compare outputs across configurations

## Engineering Preferences

- Avoid Streamlit for this internal tool.
- Prefer a simple HTML/CSS/JS frontend.
- Prefer FastAPI APIs; split into routers when the backend grows.
- Keep changes scoped and preserve the research notes.
- Use explicit config objects for scoring parameters before adding more heuristics.

## Near-Term Backlog

1. Add a cached market-data layer under `COIL_CACHE_DIR` so repeated runs are less dependent on live downloads.
2. Add stronger per-ticker error details from the market-data fetch layer.
3. Add smoke tests for CLI, feature computation, API serialization, and run persistence.
4. Add saved config presets and compare-runs behavior.
5. Add deployment docs for Railway/Docker/VPS using only env-configured storage paths.
6. Decide whether benchmark CSVs should remain committed or move to ignored/generated data.

## Run Commands

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the CLI:

```bash
python screen_monthly.py AER AVT CAT
```

Run the app:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8010
```
