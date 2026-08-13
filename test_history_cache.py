from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import history_cache


NOW = datetime(2026, 7, 11, 6, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    writable = tmp_path / "cache"
    seed = tmp_path / "seed"
    seed.mkdir()
    monkeypatch.setattr(history_cache, "CACHE_DIR", writable)
    monkeypatch.setattr(history_cache, "SEED_DIR", seed)
    monkeypatch.setattr(history_cache, "_utcnow", lambda: NOW)
    return writable, seed


def frame(
    dates: list[str],
    closes: list[float],
    *,
    adjustment_mode: str = "split_adjusted",
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "Open": closes,
            "High": [v + 1 for v in closes],
            "Low": [v - 1 for v in closes],
            "Close": closes,
            "Volume": [1000.0] * len(dates),
        },
        index=pd.to_datetime(dates),
    )
    result.attrs["adjustment_mode"] = adjustment_mode
    if adjustment_mode == "split_adjusted":
        result.attrs["adjustment_source"] = "yfinance_stock_splits"
        result.attrs["source_interval"] = "1d"
        result.attrs["adjustment_transform_version"] = (
            "yfinance-stock-splits-v1"
        )
    return result


def feature(symbol: str, monthly: pd.DataFrame):
    return SimpleNamespace(ticker=symbol, last_close=float(monthly["Close"].iloc[-1]))


def test_fresh_writable_cache_is_reused_for_24_hours(isolated_cache):
    history_cache.write_cache(
        "FRESH",
        history_cache._bars_from_frame(frame(["2026-06-01"], [10.0])),
        {"ticker": "FRESH"},
        fetched_at=NOW - timedelta(hours=23, minutes=59),
        adjustment_mode="split_adjusted",
        adjustment_source="yfinance_stock_splits",
        source_interval="1d",
        adjustment_transform_version="yfinance-stock-splits-v1",
    )

    def unexpected(_: str):
        raise AssertionError("fresh cache must not refresh")

    payload = history_cache.get_history_payload("FRESH", unexpected, feature)
    assert payload["freshness"]["status"] == "fresh"
    assert payload["freshness"]["origin"] == "writable_cache"


def test_legacy_unknown_cache_is_not_mixed_into_split_adjusted_refresh(
    isolated_cache,
):
    writable, _ = isolated_cache
    writable.mkdir()
    legacy = {
        "ticker": "MERGE",
        "bars": history_cache._bars_from_frame(
            frame(["2020-01-01", "2020-02-01"], [10.0, 20.0])
        ),
        "features": {"old": True},
    }
    (writable / "MERGE.json").write_text(json.dumps(legacy), encoding="utf-8")
    live = frame(["2020-02-28", "2020-03-31"], [22.0, 30.0])

    payload = history_cache.get_history_payload("MERGE", lambda _: live, feature)

    assert [b["date"] for b in payload["bars"]] == [
        "2020-02-01",
        "2020-03-01",
    ]
    assert [b["close"] for b in payload["bars"]] == [22.0, 30.0]
    assert payload["features"]["last_close"] == 30.0
    assert payload["cache_metadata"] == {
        "schema_version": history_cache.CACHE_SCHEMA_VERSION,
        "fetched_at": "2026-07-11T06:00:00.000Z",
            "last_bar_date": "2020-03-01",
            "source": "yfinance",
            "adjustment_mode": "split_adjusted",
            "adjustment_source": "yfinance_stock_splits",
            "source_interval": "1d",
            "adjustment_transform_version": "yfinance-stock-splits-v1",
    }
    assert not (writable / "MERGE.json.tmp").exists()


def test_seed_refresh_failure_returns_stale_fallback(isolated_cache):
    _, seed = isolated_cache
    payload = {
        "ticker": "SEED",
        "bars": history_cache._bars_from_frame(frame(["2010-01-01"], [8.0])),
        "features": None,
    }
    (seed / "SEED.json").write_text(json.dumps(payload), encoding="utf-8")

    def offline(_: str):
        raise RuntimeError("offline")

    result = history_cache.get_history_payload("SEED", offline, feature)
    assert result["bars"] == payload["bars"]
    assert result["freshness"] == {
        "status": "stale_fallback",
        "schema_version": None,
        "fetched_at": None,
        "last_bar_date": "2010-01-01",
            "source": "seed_cache",
            "adjustment_mode": "unknown",
            "adjustment_source": None,
            "source_interval": None,
            "adjustment_transform_version": None,
        "origin": "seed_cache",
        "refresh_error": "offline",
    }


def test_force_refresh_bypasses_fresh_cache_and_merges(isolated_cache):
    calls = {"count": 0}
    history_cache.write_cache(
        "FORCE",
        history_cache._bars_from_frame(frame(["2025-01-01"], [10.0])),
        None,
        fetched_at=NOW,
        adjustment_mode="split_adjusted",
        adjustment_source="yfinance_stock_splits",
        source_interval="1d",
        adjustment_transform_version="yfinance-stock-splits-v1",
    )

    def fetch(_: str):
        calls["count"] += 1
        return frame(["2025-02-01"], [11.0])

    result = history_cache.get_history_payload(
        "FORCE", fetch, feature, force_refresh=True
    )
    assert calls["count"] == 1
    assert len(result["bars"]) == 1
    assert result["bars"][0]["close"] == 11.0
    assert result["freshness"]["status"] == "fresh"


def test_missing_and_failed_refresh_has_no_usable_payload():
    assert history_cache.get_history_payload("NONE", lambda _: None, feature) is None
