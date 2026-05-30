"""Endpoint contract + history-cache + saved-run loader tests.

Network (yfinance) is never hit: ``fetch_monthly_history`` is monkeypatched and
the history disk cache is redirected to a tmp dir per test.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
import history_cache
from screen_monthly import compute_features


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    """Redirect the history disk cache to an isolated tmp dir."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(history_cache, "CACHE_DIR", cache_dir)
    return cache_dir


def _synthetic_monthly(rows: int = 130) -> pd.DataFrame:
    """A flat coil: enough bars (>=120) for compute_features to return a result."""
    idx = pd.date_range("2010-01-01", periods=rows, freq="MS")
    base = 100.0
    high = [base + 5 + (i % 3) for i in range(rows)]
    low = [base - 5 - (i % 3) for i in range(rows)]
    close = [base + (i % 2) for i in range(rows)]
    return pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": [1_000_000 + i for i in range(rows)],
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Simple endpoint contracts
# --------------------------------------------------------------------------- #
def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_default_tickers(client):
    resp = client.get("/api/default-tickers")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["tickers"], list)
    assert len(body["tickers"]) > 0
    assert all(isinstance(t, str) for t in body["tickers"])


def test_saved_runs_lists_csvs_newest_first(client):
    resp = client.get("/api/saved-runs")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert isinstance(runs, list)
    assert len(runs) > 0
    # contract: each run has a name + size, and the curated demo run is present
    names = [r["name"] for r in runs]
    assert all(n.endswith(".csv") for n in names)
    assert all(isinstance(r["size_bytes"], int) for r in runs)
    assert "demo_curated_coils_results.csv" in names
    # newest-first: the curated demo CSV (most recently written) is first
    assert names[0] == "demo_curated_coils_results.csv"


# --------------------------------------------------------------------------- #
# Saved-run loader: success + 404 + 400 (path traversal) branches
# --------------------------------------------------------------------------- #
def test_saved_run_success(client):
    resp = client.get("/api/saved-runs/demo_curated_coils_results.csv")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "demo_curated_coils_results.csv"
    assert body["count"] == len(body["results"])
    assert body["count"] > 0
    assert "ticker" in body["results"][0]


def test_saved_run_missing_returns_404(client):
    resp = client.get("/api/saved-runs/does_not_exist.csv")
    assert resp.status_code == 404


def test_saved_run_non_csv_returns_404(client):
    resp = client.get("/api/saved-runs/README.txt")
    assert resp.status_code == 404


def test_saved_run_forward_slash_guard_returns_400():
    # A "/" in the filename must hit the explicit guard (400), never read outside
    # the project root. Starlette won't route %2F as a single path param, so we
    # exercise the handler's guard branch directly.
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        app_module.saved_run("../../etc/passwd")
    assert exc.value.status_code == 400


def test_saved_run_backslash_traversal_returns_400(client):
    # %5C decodes to "\\" which DOES reach the handler -> 400 guard branch
    resp = client.get("/api/saved-runs/..%5Csecret.csv")
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# History endpoint: cache miss (live fetch) then cache hit (no fetch)
# --------------------------------------------------------------------------- #
def test_history_cache_miss_then_hit(client, monkeypatch, tmp_cache):
    calls = {"n": 0}

    def fake_fetch(symbol: str):
        calls["n"] += 1
        return _synthetic_monthly()

    monkeypatch.setattr(app_module, "fetch_monthly_history", fake_fetch)

    # MISS: triggers live fetch, writes cache, returns full contract shape
    r1 = client.get("/api/history/TEST")
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["ticker"] == "TEST"
    assert set(body1.keys()) == {"ticker", "bars", "features"}
    assert len(body1["bars"]) > 0
    bar = body1["bars"][0]
    assert set(bar.keys()) == {"date", "open", "high", "low", "close", "volume"}
    assert body1["features"] is not None
    assert calls["n"] == 1
    # cache file written
    assert (tmp_cache / "TEST.json").exists()

    # HIT: served from disk, fetch NOT called again, identical body
    r2 = client.get("/api/history/TEST")
    assert r2.status_code == 200
    assert r2.json() == body1
    assert calls["n"] == 1


def test_history_max_bars_trims_response(client, monkeypatch):
    monkeypatch.setattr(app_module, "fetch_monthly_history", lambda s: _synthetic_monthly(130))
    resp = client.get("/api/history/TRIM", params={"max_bars": 24})
    assert resp.status_code == 200
    bars = resp.json()["bars"]
    assert len(bars) == 24
    # trimming keeps the most-recent bars
    assert bars[-1]["date"] == "2020-10-01"


def test_history_uses_cache_without_fetch(client, monkeypatch, tmp_cache):
    """A pre-seeded cache file is served without any live fetch."""
    tmp_cache.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": "SEED",
        "bars": [
            {"date": "2024-01-01", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}
        ],
        "features": None,
    }
    (tmp_cache / "SEED.json").write_text(json.dumps(payload), encoding="utf-8")

    def boom(symbol: str):
        raise AssertionError("live fetch must not be called on a cache hit")

    monkeypatch.setattr(app_module, "fetch_monthly_history", boom)

    resp = client.get("/api/history/SEED")
    assert resp.status_code == 200
    assert resp.json() == payload


def test_history_404_when_no_data(client, monkeypatch):
    monkeypatch.setattr(app_module, "fetch_monthly_history", lambda s: None)
    resp = client.get("/api/history/NODATA")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# /api/screen contract (run_screen monkeypatched -> no network)
# --------------------------------------------------------------------------- #
def test_screen_contract(client, monkeypatch):
    df = pd.DataFrame([compute_features("TEST", _synthetic_monthly()).__dict__])
    monkeypatch.setattr(app_module, "run_screen", lambda tickers: df)

    resp = client.post("/api/screen", json={"tickers": ["TEST"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["tickers"] == ["TEST"]
    assert body["results"][0]["ticker"] == "TEST"
