from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from history_cache import get_history_payload
from screen_monthly import (
    DEFAULT_TICKERS,
    build_ticker_list,
    compute_features,
    fetch_monthly_history,
    run_screen,
)

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"

# The frontend auto-loads runs[0] from /api/saved-runs as the default view. On a
# fresh deploy (e.g. Railway) every file gets the same checkout mtime, so an
# mtime sort is non-deterministic and may surface a large non-curated run whose
# tickers aren't in seed_cache. Pin the curated demo run first so the demo always
# boots into the seeded coil universe; everything else stays mtime-sorted below.
DEMO_DEFAULT_RUN = "demo_curated_coils_results.csv"

app = FastAPI(title="Coil Screening")


class ScreenRequest(BaseModel):
    tickers: list[str] = []
    universe: Literal["sp500"] | None = None
    limit: int | None = None


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def serialize_frame(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({key: clean_value(value) for key, value in row.items()})
    return records


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/default-tickers")
def default_tickers() -> dict[str, list[str]]:
    return {"tickers": DEFAULT_TICKERS}


@app.get("/api/saved-runs")
def saved_runs() -> dict[str, list[dict[str, Any]]]:
    # Curated demo run first (deterministic default), then the rest newest-first.
    files = sorted(
        PROJECT_ROOT.glob("*.csv"),
        key=lambda path: (path.name == DEMO_DEFAULT_RUN, path.stat().st_mtime),
        reverse=True,
    )
    return {
        "runs": [
            {
                "name": file.name,
                "size_bytes": file.stat().st_size,
            }
            for file in files
        ]
    }


@app.get("/api/saved-runs/{filename}")
def saved_run(filename: str) -> dict[str, Any]:
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = PROJECT_ROOT / filename
    if not file_path.exists() or file_path.suffix.lower() != ".csv":
        raise HTTPException(status_code=404, detail="Saved run not found.")

    df = pd.read_csv(file_path)
    return {
        "name": filename,
        "count": len(df),
        "results": serialize_frame(df),
    }


@app.post("/api/screen")
def screen(request: ScreenRequest) -> dict[str, Any]:
    tickers = build_ticker_list(
        explicit_tickers=request.tickers,
        universe=request.universe,
        limit=request.limit,
    )
    df = run_screen(tickers)
    return {
        "count": len(df),
        "tickers": tickers,
        "results": serialize_frame(df),
    }


@app.get("/api/history/{ticker}")
def history(ticker: str, max_bars: int = 180) -> dict[str, Any]:
    symbol = ticker.strip().upper()
    payload = get_history_payload(symbol, fetch_monthly_history, compute_features)
    if payload is None:
        raise HTTPException(status_code=404, detail="No monthly history found.")

    bars = payload["bars"]
    trimmed = bars[-max_bars:] if max_bars and max_bars < len(bars) else bars

    return {
        "ticker": symbol,
        "bars": trimmed,
        "features": payload["features"],
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
