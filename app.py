from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from screen_monthly import (
    DEFAULT_TICKERS,
    DEFAULT_CACHE_MAX_AGE_HOURS,
    build_ticker_list,
    build_config,
    compute_features,
    default_config_dict,
    fetch_monthly_history,
    run_screen_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
STORAGE_DIR = Path(os.getenv("COIL_STORAGE_DIR", PROJECT_ROOT / "storage"))
RUNS_DIR = Path(os.getenv("COIL_RUNS_DIR", STORAGE_DIR / "runs"))
CACHE_DIR = Path(os.getenv("COIL_CACHE_DIR", STORAGE_DIR / "market_cache"))
LEGACY_RUNS_DIR = PROJECT_ROOT / "runs"

app = FastAPI(title="Coil Screening")


class ScreenRequest(BaseModel):
    tickers: list[str] = []
    universe: Literal["sp500"] | None = None
    limit: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    use_cache: bool = True
    cache_max_age_hours: float = DEFAULT_CACHE_MAX_AGE_HOURS
    force_refresh: bool = False


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def create_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def run_dir(run_id: str, base_dir: Path = RUNS_DIR) -> Path:
    return base_dir / run_id


def metadata_path(run_id: str, base_dir: Path = RUNS_DIR) -> Path:
    return run_dir(run_id, base_dir=base_dir) / "metadata.json"


def results_path(run_id: str, base_dir: Path = RUNS_DIR) -> Path:
    return run_dir(run_id, base_dir=base_dir) / "results.csv"


def validate_run_id(run_id: str) -> str:
    if "/" in run_id or "\\" in run_id or run_id.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid run id.")
    return run_id


def write_run_artifacts(run_id: str, metadata: dict[str, Any], df: pd.DataFrame) -> None:
    path = run_dir(run_id)
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path(run_id), index=False)
    metadata_path(run_id).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def read_run_metadata(run_id: str) -> dict[str, Any]:
    validate_run_id(run_id)
    path = find_metadata_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def run_lookup_dirs() -> list[Path]:
    dirs = [RUNS_DIR]
    if LEGACY_RUNS_DIR != RUNS_DIR:
        dirs.append(LEGACY_RUNS_DIR)
    return dirs


def find_metadata_path(run_id: str) -> Path:
    for base_dir in run_lookup_dirs():
        path = metadata_path(run_id, base_dir=base_dir)
        if path.exists():
            return path
    return metadata_path(run_id)


def find_results_path(run_id: str) -> Path:
    for base_dir in run_lookup_dirs():
        path = results_path(run_id, base_dir=base_dir)
        if path.exists():
            return path
    return results_path(run_id)


def list_persisted_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    run_dirs: list[Path] = []
    for base_dir in run_lookup_dirs():
        if base_dir.exists():
            run_dirs.extend(path for path in base_dir.iterdir() if path.is_dir())

    for path in sorted(run_dirs, key=lambda item: item.name, reverse=True):
        if path.name in seen:
            continue
        meta = path / "metadata.json"
        if not meta.exists():
            continue
        seen.add(path.name)
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        runs.append(
            {
                "id": data.get("run_id", path.name),
                "created_at": data.get("created_at"),
                "result_count": data.get("result_count", 0),
                "failure_count": data.get("failure_count", 0),
                "status": data.get("status", "unknown"),
                "source": "persisted_run",
                "name": f"{data.get('created_at', path.name)} · {data.get('result_count', 0)} rows",
            }
        )
    return runs


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/default-tickers")
def default_tickers() -> dict[str, list[str]]:
    return {"tickers": DEFAULT_TICKERS}


@app.get("/api/default-config")
def default_config() -> dict[str, Any]:
    return {"config": default_config_dict()}


@app.get("/api/storage")
def storage() -> dict[str, str]:
    return {
        "storage_dir": str(STORAGE_DIR),
        "runs_dir": str(RUNS_DIR),
        "cache_dir": str(CACHE_DIR),
        "cache_max_age_hours_default": str(DEFAULT_CACHE_MAX_AGE_HOURS),
    }


@app.get("/api/saved-runs")
def saved_runs() -> dict[str, list[dict[str, Any]]]:
    files = sorted(PROJECT_ROOT.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return {
        "runs": [
            {
                "name": file.name,
                "size_bytes": file.stat().st_size,
            }
            for file in files
        ]
    }


@app.get("/api/runs")
def runs() -> dict[str, list[dict[str, Any]]]:
    return {"runs": list_persisted_runs()}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    metadata = read_run_metadata(run_id)
    result_file = find_results_path(run_id)
    try:
        results = serialize_frame(pd.read_csv(result_file)) if result_file.exists() else []
    except pd.errors.EmptyDataError:
        results = []
    return {
        "metadata": metadata,
        "count": len(results),
        "results": results,
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
    run_id = create_run_id()
    created_at = utc_now()
    config = build_config(request.config)
    config_dict = default_config_dict() | config.__dict__
    tickers = build_ticker_list(
        explicit_tickers=request.tickers,
        universe=request.universe,
        limit=request.limit,
    )
    report = run_screen_report(
        tickers,
        config=config,
        use_cache=request.use_cache,
        cache_max_age_hours=request.cache_max_age_hours,
        force_refresh=request.force_refresh,
    )
    df = report.results
    failures = [failure.__dict__ for failure in report.failures]
    completed_at = utc_now()
    metadata = {
        "run_id": run_id,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": "completed",
        "git_commit": current_git_commit(),
        "request": {
            "explicit_tickers": request.tickers,
            "universe": request.universe,
            "limit": request.limit,
            "config": request.config,
            "use_cache": request.use_cache,
            "cache_max_age_hours": request.cache_max_age_hours,
            "force_refresh": request.force_refresh,
        },
        "resolved_tickers": tickers,
        "config": config_dict,
        "result_count": len(df),
        "failure_count": len(failures),
        "failures": failures,
        "cache": asdict(report.cache_stats),
        "storage_dir": str(STORAGE_DIR),
        "results_file": str(results_path(run_id)),
    }
    write_run_artifacts(run_id, metadata, df)
    return {
        "run": metadata,
        "count": len(df),
        "tickers": tickers,
        "failures": failures,
        "cache": asdict(report.cache_stats),
        "config": config_dict,
        "results": serialize_frame(df),
    }


@app.get("/api/history/{ticker}")
def history(ticker: str, max_bars: int = 180) -> dict[str, Any]:
    symbol = ticker.strip().upper()
    monthly = fetch_monthly_history(symbol)
    if monthly is None or monthly.empty:
        raise HTTPException(status_code=404, detail="No monthly history found.")

    features = compute_features(symbol, monthly)
    trimmed = monthly.tail(max_bars).copy()

    bars = []
    for idx, row in trimmed.iterrows():
        bars.append(
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": clean_value(row["Open"]),
                "high": clean_value(row["High"]),
                "low": clean_value(row["Low"]),
                "close": clean_value(row["Close"]),
                "volume": clean_value(row["Volume"]) if "Volume" in row else None,
            }
        )

    return {
        "ticker": symbol,
        "bars": bars,
        "features": features.__dict__ if features else None,
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
