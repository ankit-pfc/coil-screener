"""Disk cache for monthly history responses.

`/api/history/{ticker}` calls live yfinance on every request, which is slow and
rate-limit flaky. This module caches the *full* set of monthly bars plus the
computed features as JSON under ``cache/<TICKER>.json`` so requests are served
cache-first and only hit yfinance on a cache miss.

Reads fall back to a tracked ``seed_cache/<TICKER>.json`` when the writable
``cache/`` has no entry. ``cache/`` is gitignored and ephemeral on hosts like
Railway, so without the seed every cold start would re-hit live yfinance; the
seed makes the curated demo universe instant and offline-safe. Writes still go
to ``cache/`` (the seed is read-only), so live tickers outside the demo set are
fetched and cached normally.

The cache stores the complete bar history; per-request ``max_bars`` trimming
happens at response-build time, so the API response shape stays byte-identical
to the uncached path regardless of how many bars are requested.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
# Tracked, read-only fallback shipped in the repo (see module docstring).
SEED_DIR = PROJECT_ROOT / "seed_cache"


def _clean(value: Any) -> Any:
    import math

    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.json"


def _bars_from_frame(monthly: pd.DataFrame) -> list[dict[str, Any]]:
    bars: list[dict[str, Any]] = []
    for idx, row in monthly.iterrows():
        bars.append(
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": _clean(float(row["Open"])),
                "high": _clean(float(row["High"])),
                "low": _clean(float(row["Low"])),
                "close": _clean(float(row["Close"])),
                "volume": _clean(float(row["Volume"])) if "Volume" in row else None,
            }
        )
    return bars


def _read_payload(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or "bars" not in payload:
        return None
    return payload


def read_cache(symbol: str) -> Optional[dict[str, Any]]:
    """Cached payload for ``symbol``: writable ``cache/`` first, then the tracked
    ``seed_cache/`` fallback. ``None`` on miss/corruption in both."""
    return _read_payload(cache_path(symbol)) or _read_payload(
        SEED_DIR / f"{symbol}.json"
    )


def write_cache(symbol: str, bars: list[dict[str, Any]], features: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Persist the full bar set + features for ``symbol`` and return the payload."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"ticker": symbol, "bars": bars, "features": features}
    tmp = cache_path(symbol).with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(cache_path(symbol))
    return payload


def get_history_payload(
    symbol: str,
    fetch_monthly: Callable[[str], Optional[pd.DataFrame]],
    compute_features: Callable[[str, pd.DataFrame], Any],
) -> Optional[dict[str, Any]]:
    """Cache-first history payload (full bars + features) for ``symbol``.

    Returns ``None`` when there is no cached entry and the live fetch yields no
    monthly history (the caller maps this to a 404).
    """
    cached = read_cache(symbol)
    if cached is not None:
        return cached

    monthly = fetch_monthly(symbol)
    if monthly is None or monthly.empty:
        return None

    monthly = monthly.sort_index()
    features = compute_features(symbol, monthly)
    bars = _bars_from_frame(monthly)
    features_dict = features.__dict__ if features else None
    return write_cache(symbol, bars, features_dict)
