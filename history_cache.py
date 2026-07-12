"""Freshness-aware disk cache for complete monthly price histories.

Writable entries are valid for 24 hours.  The tracked ``seed_cache`` and old
payloads without metadata are intentionally stale: they are useful offline,
but should never prevent a live refresh.  Refreshes merge by month (live bars
win), recompute features over the complete history, and replace the writable
file atomically.  If refresh fails, any usable cached/seed bars are returned
with an explicit ``stale_fallback`` freshness status.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
SEED_DIR = PROJECT_ROOT / "seed_cache"
CACHE_SCHEMA_VERSION = 2
CACHE_TTL = timedelta(hours=24)


def _clean(value: Any) -> Any:
    import math

    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.json"


def _bars_from_frame(monthly: pd.DataFrame) -> list[dict[str, Any]]:
    bars: list[dict[str, Any]] = []
    for idx, row in monthly.sort_index().iterrows():
        bars.append(
            {
                "date": pd.Timestamp(idx).strftime("%Y-%m-%d"),
                "open": _clean(float(row["Open"])),
                "high": _clean(float(row["High"])),
                "low": _clean(float(row["Low"])),
                "close": _clean(float(row["Close"])),
                "volume": _clean(float(row["Volume"]))
                if "Volume" in row and pd.notna(row["Volume"])
                else None,
            }
        )
    return bars


def _frame_from_bars(bars: list[dict[str, Any]]) -> Optional[pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    dates: list[pd.Timestamp] = []
    for bar in bars:
        try:
            date = pd.Timestamp(bar["date"])
            row = {
                "Open": float(bar["open"]),
                "High": float(bar["high"]),
                "Low": float(bar["low"]),
                "Close": float(bar["close"]),
                "Volume": float(bar["volume"])
                if bar.get("volume") is not None
                else float("nan"),
            }
        except (KeyError, TypeError, ValueError):
            continue
        dates.append(date)
        rows.append(row)
    if not rows:
        return None
    monthly_index = pd.DatetimeIndex(dates).to_period("M").to_timestamp()
    frame = pd.DataFrame(rows, index=monthly_index)
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def _read_payload(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("bars"), list)
        or not payload["bars"]
    ):
        return None
    return payload


def _read_cache_entry(symbol: str) -> tuple[Optional[dict[str, Any]], str]:
    writable = _read_payload(cache_path(symbol))
    if writable is not None:
        return writable, "writable_cache"
    seed = _read_payload(SEED_DIR / f"{symbol}.json")
    if seed is not None:
        return seed, "seed_cache"
    return None, "missing"


def read_cache(symbol: str) -> Optional[dict[str, Any]]:
    """Return the raw writable payload, falling back to the tracked seed."""
    return _read_cache_entry(symbol)[0]


def _metadata_is_fresh(payload: dict[str, Any], now: datetime) -> bool:
    metadata = payload.get("cache_metadata")
    if not isinstance(metadata, dict):
        return False
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False
    fetched_at = _parse_time(metadata.get("fetched_at"))
    if fetched_at is None:
        return False
    age = now - fetched_at
    return timedelta(0) <= age < CACHE_TTL


def _freshness(
    payload: dict[str, Any],
    status: str,
    origin: str,
    refresh_error: Optional[str] = None,
) -> dict[str, Any]:
    metadata = payload.get("cache_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    bars = payload.get("bars") or []
    return {
        "status": status,
        "schema_version": metadata.get("schema_version"),
        "fetched_at": metadata.get("fetched_at"),
        "last_bar_date": metadata.get("last_bar_date")
        or (bars[-1].get("date") if bars else None),
        "source": metadata.get("source") or origin,
        "origin": origin,
        "refresh_error": refresh_error,
    }


def _with_freshness(
    payload: dict[str, Any],
    status: str,
    origin: str,
    refresh_error: Optional[str] = None,
) -> dict[str, Any]:
    response = dict(payload)
    response["freshness"] = _freshness(
        payload, status, origin, refresh_error=refresh_error
    )
    return response


def write_cache(
    symbol: str,
    bars: list[dict[str, Any]],
    features: Optional[dict[str, Any]],
    *,
    fetched_at: Optional[datetime] = None,
    source: str = "yfinance",
) -> dict[str, Any]:
    """Atomically persist complete bars, features, and cache metadata."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = fetched_at or _utcnow()
    payload = {
        "ticker": symbol,
        "bars": bars,
        "features": features,
        "cache_metadata": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fetched_at": _iso_utc(timestamp),
            "last_bar_date": bars[-1]["date"] if bars else None,
            "source": source,
        },
    }
    target = cache_path(symbol)
    tmp = target.with_suffix(f"{target.suffix}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
            fh.flush()
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return payload


def _merge_history(
    cached: Optional[dict[str, Any]], live: pd.DataFrame
) -> pd.DataFrame:
    live = live.copy().sort_index()
    live.index = pd.DatetimeIndex(live.index).to_period("M").to_timestamp()
    live = live[~live.index.duplicated(keep="last")]
    cached_frame = _frame_from_bars(cached.get("bars", [])) if cached else None
    if cached_frame is None:
        return live[~live.index.duplicated(keep="last")].sort_index()
    # Live rows win for overlapping months while old listing history survives a
    # provider response that contains only the most recent window.
    combined = pd.concat([cached_frame, live])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def get_history_payload(
    symbol: str,
    fetch_monthly: Callable[[str], Optional[pd.DataFrame]],
    compute_features: Callable[[str, pd.DataFrame], Any],
    *,
    force_refresh: bool = False,
    source: str = "yfinance",
) -> Optional[dict[str, Any]]:
    """Return full history, refreshing stale data and preserving fallback bars."""
    symbol = symbol.strip().upper()
    cached, origin = _read_cache_entry(symbol)
    now = _utcnow()
    if (
        cached is not None
        and origin == "writable_cache"
        and not force_refresh
        and _metadata_is_fresh(cached, now)
    ):
        return _with_freshness(cached, "fresh", origin)

    refresh_error: Optional[str] = None
    try:
        live = fetch_monthly(symbol)
    except Exception as exc:
        refresh_error = str(exc) or type(exc).__name__
        live = None

    if live is None or not isinstance(live, pd.DataFrame) or live.empty:
        if cached is None:
            return None
        return _with_freshness(
            cached,
            "stale_fallback",
            origin,
            refresh_error=refresh_error or "refresh returned no monthly history",
        )

    try:
        merged = _merge_history(cached, live)
        if merged.empty:
            raise ValueError("refresh returned no usable monthly history")
        features = compute_features(symbol, merged)
        features_dict = features.__dict__ if features else None
        bars = _bars_from_frame(merged)
        written = write_cache(
            symbol,
            bars,
            features_dict,
            fetched_at=now,
            source=source,
        )
    except Exception as exc:
        if cached is None:
            return None
        return _with_freshness(
            cached,
            "stale_fallback",
            origin,
            refresh_error=str(exc) or type(exc).__name__,
        )
    return _with_freshness(written, "fresh", "writable_cache")
