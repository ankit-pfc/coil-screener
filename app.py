from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from coil_analysis import analyze_coil
from history_cache import get_history_payload
from screen_monthly import (
    DEFAULT_TICKERS,
    build_ticker_list,
    compute_features,
    fetch_monthly_history,
    run_screen,
)
from vision.run import VisionRunConfig, run_vision_pipeline
from vision.storage import VisionRunStore

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
VISION_RUNS_DIR = PROJECT_ROOT / "vision_runs"

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


class VisionRunRequest(BaseModel):
    tickers: list[str] = []
    saved_run: str | None = None
    interval: Literal["1M", "3M", "6M", "1Y"] = "3M"
    timeframe: Literal["1Y", "2Y", "5Y", "10Y", "All"] = "10Y"
    chart_type: Literal["candles", "bars", "line", "area"] = "candles"
    base_url: str = "http://127.0.0.1:5173"
    limit: int | None = None
    run_id: str | None = None
    headless: bool = True
    max_highs: int = 3
    confidence: float = 0.35
    max_trendlines: int = 5
    touch_tolerance_pct: float = 1.5


class VisionReviewRequest(BaseModel):
    run_id: str
    ticker: str
    interval: Literal["1M", "3M", "6M", "1Y"] = "3M"
    decision: Literal["accepted", "rejected", "edited"]
    accepted_highs: list[dict[str, Any]] = []
    notes: str | None = None


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


@app.get("/api/coil/{ticker}")
def coil(ticker: str, as_of: str | None = None) -> dict[str, Any]:
    """Deterministic major-top / resistance-slope / coiling analysis.

    ``as_of`` truncates the history first (YYYY-MM-DD, inclusive), which lets
    the UI and validation scripts replay the pre-breakout state of a chart.
    """
    if as_of is not None and (len(as_of) != 10 or as_of[4] != "-" or as_of[7] != "-"):
        raise HTTPException(status_code=400, detail="as_of must be YYYY-MM-DD.")

    symbol = ticker.strip().upper()
    payload = get_history_payload(symbol, fetch_monthly_history, compute_features)
    if payload is None:
        raise HTTPException(status_code=404, detail="No monthly history found.")

    return {"ticker": symbol, **analyze_coil(payload["bars"], as_of=as_of)}


def vision_store() -> VisionRunStore:
    return VisionRunStore(VISION_RUNS_DIR)


@app.post("/api/vision/run")
def vision_run(request: VisionRunRequest | None = None) -> dict[str, Any]:
    req = request or VisionRunRequest()
    try:
        return run_vision_pipeline(
            VisionRunConfig(
                project_root=PROJECT_ROOT,
                tickers=req.tickers,
                saved_run=req.saved_run,
                interval=req.interval,
                timeframe=req.timeframe,
                chart_type=req.chart_type,
                base_url=req.base_url,
                limit=req.limit,
                run_id=req.run_id,
                headless=req.headless,
                max_highs=req.max_highs,
                confidence=req.confidence,
                max_trendlines=req.max_trendlines,
                touch_tolerance_pct=req.touch_tolerance_pct,
            ),
            store=vision_store(),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/vision/runs")
def vision_runs() -> dict[str, Any]:
    return {"runs": vision_store().list_runs()}


@app.get("/api/vision/runs/{run_id}")
def vision_run_detail(run_id: str) -> dict[str, Any]:
    try:
        return vision_store().read_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Vision run not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/vision/predictions/{ticker}")
def vision_prediction(
    ticker: str,
    interval: Literal["1M", "3M", "6M", "1Y"] = "3M",
    run_id: str = "latest",
) -> dict[str, Any]:
    try:
        return vision_store().read_prediction(ticker, interval, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Vision prediction not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/vision/reviews")
def vision_review(request: VisionReviewRequest) -> dict[str, Any]:
    try:
        review = vision_store().append_review(request.model_dump())
        return {"review": review}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
