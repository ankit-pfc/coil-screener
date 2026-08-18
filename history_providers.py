"""Provider-neutral daily-history inputs for research snapshots.

The production default remains yfinance in :mod:`screen_monthly`.  This module
adds explicit, auditable inputs for long-history research so detector work is
not coupled to a vendor SDK or to a live quote path.  Every provider returns raw
daily OHLCV plus split factors; split-only adjustment remains CoilingView's own
canonical transform.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import requests


HistoryProviderName = Literal["yfinance", "eodhd", "frozen_csv"]
PROVIDER_ENV = "COILINGVIEW_HISTORY_PROVIDER"
EODHD_TOKEN_ENV = "COILINGVIEW_EODHD_API_TOKEN"
EODHD_SYMBOL_MAP_ENV = "COILINGVIEW_EODHD_SYMBOL_MAP"
FROZEN_HISTORY_DIR_ENV = "COILINGVIEW_FROZEN_HISTORY_DIR"
EODHD_BASE_URL = "https://eodhd.com/api"
_SAFE_SYMBOL = re.compile(r"^[A-Z0-9._-]+$")


class HistoryProviderError(ValueError):
    """Raised when a provider cannot prove a canonical history input."""


def configured_provider_name() -> HistoryProviderName:
    value = os.getenv(PROVIDER_ENV, "yfinance").strip().lower()
    if value not in {"yfinance", "eodhd", "frozen_csv"}:
        raise HistoryProviderError(
            f"{PROVIDER_ENV} must be yfinance, eodhd, or frozen_csv"
        )
    return value  # type: ignore[return-value]


def load_symbol_map(path: str | Path | None = None) -> dict[str, str]:
    configured = path or os.getenv(EODHD_SYMBOL_MAP_ENV)
    if not configured:
        return {}
    source = Path(configured).expanduser()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoryProviderError(f"cannot read EODHD symbol map: {source}") from exc
    if not isinstance(payload, dict):
        raise HistoryProviderError("EODHD symbol map must be a JSON object")
    result: dict[str, str] = {}
    for ticker, provider_symbol in payload.items():
        local = str(ticker).strip().upper()
        remote = str(provider_symbol).strip().upper()
        if not _SAFE_SYMBOL.fullmatch(local) or not _SAFE_SYMBOL.fullmatch(remote):
            raise HistoryProviderError("symbol map contains an unsafe ticker")
        result[local] = remote
    return result


def resolve_eodhd_symbol(
    ticker: str,
    *,
    symbol_map: dict[str, str] | None = None,
) -> str:
    local = ticker.strip().upper()
    if not _SAFE_SYMBOL.fullmatch(local):
        raise HistoryProviderError("ticker contains unsupported characters")
    mapped = (symbol_map or {}).get(local)
    if mapped:
        return mapped
    if "." not in local:
        return f"{local}.US"
    raise HistoryProviderError(
        f"{local} needs an explicit EODHD symbol mapping; exchange suffixes "
        "are never guessed"
    )


def _response_json(response: Any, label: str) -> Any:
    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise HistoryProviderError(f"EODHD {label} request failed") from exc
    if isinstance(payload, dict) and (payload.get("error") or payload.get("message")):
        raise HistoryProviderError(f"EODHD {label} returned an error")
    return payload


def _split_factor(value: Any) -> float:
    if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
        return float(value)
    if not isinstance(value, str):
        raise HistoryProviderError("EODHD split factor is missing")
    normalized = value.strip()
    try:
        if "/" in normalized:
            numerator, denominator = normalized.split("/", 1)
            factor = float(numerator) / float(denominator)
        else:
            factor = float(normalized)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise HistoryProviderError("EODHD split factor is invalid") from exc
    if not np.isfinite(factor) or factor <= 0:
        raise HistoryProviderError("EODHD split factor must be positive")
    return factor


def fetch_eodhd_daily_history(
    ticker: str,
    *,
    api_token: str | None = None,
    provider_symbol: str | None = None,
    symbol_map: dict[str, str] | None = None,
    date_from: str = "1900-01-01",
    request_get: Callable[..., Any] = requests.get,
) -> pd.DataFrame:
    """Fetch raw EOD bars and split events without persisting the API token."""
    token = (api_token or os.getenv(EODHD_TOKEN_ENV, "")).strip()
    if not token:
        raise HistoryProviderError(f"{EODHD_TOKEN_ENV} is required for eodhd")
    remote = (
        provider_symbol.strip().upper()
        if provider_symbol
        else resolve_eodhd_symbol(ticker, symbol_map=symbol_map or load_symbol_map())
    )
    if not _SAFE_SYMBOL.fullmatch(remote):
        raise HistoryProviderError("provider symbol contains unsupported characters")
    common = {
        "api_token": token,
        "fmt": "json",
        "from": date_from,
        "order": "a",
    }
    bars_payload = _response_json(
        request_get(f"{EODHD_BASE_URL}/eod/{remote}", params=common, timeout=45),
        "EOD history",
    )
    splits_payload = _response_json(
        request_get(
            f"{EODHD_BASE_URL}/splits/{remote}",
            params=common,
            timeout=45,
        ),
        "split history",
    )
    if not isinstance(bars_payload, list) or not bars_payload:
        raise HistoryProviderError("EODHD returned no daily history")
    if not isinstance(splits_payload, list):
        raise HistoryProviderError("EODHD split history is not a list")

    rows: list[dict[str, Any]] = []
    for item in bars_payload:
        if not isinstance(item, dict):
            raise HistoryProviderError("EODHD EOD history contains a non-object row")
        try:
            rows.append(
                {
                    "Date": pd.Timestamp(str(item["date"])),
                    "Open": float(item["open"]),
                    "High": float(item["high"]),
                    "Low": float(item["low"]),
                    "Close": float(item["close"]),
                    "Volume": float(item["volume"]),
                    "Stock Splits": 0.0,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HistoryProviderError("EODHD EOD history has an invalid row") from exc
    frame = pd.DataFrame(rows).set_index("Date").sort_index()
    if frame.index.has_duplicates:
        raise HistoryProviderError("EODHD EOD history contains duplicate dates")
    out_of_range_split_events: list[dict[str, Any]] = []
    for item in splits_payload:
        if not isinstance(item, dict) or not item.get("date"):
            raise HistoryProviderError("EODHD split history has an invalid row")
        split_date = pd.Timestamp(str(item["date"]))
        raw_factor = item.get("split") or item.get("split_factor")
        factor = _split_factor(raw_factor)
        if split_date < frame.index[0] or split_date > frame.index[-1]:
            out_of_range_split_events.append(
                {
                    "date": split_date.strftime("%Y-%m-%d"),
                    "factor": factor,
                }
            )
            continue
        if split_date not in frame.index:
            raise HistoryProviderError(
                f"EODHD split date {split_date.date()} has no matching trading bar"
            )
        frame.loc[split_date, "Stock Splits"] = factor
    frame.attrs.update(
        {
            "source": "eodhd",
            "provider_symbol": remote,
            "adjustment_mode": "raw_with_split_events",
            "adjustment_source": "eodhd_splits_api",
            "source_interval": "1d",
            "requested_history_start": date_from,
            "provider_history_start": frame.index[0].strftime("%Y-%m-%d"),
            "provider_history_end": frame.index[-1].strftime("%Y-%m-%d"),
            "split_events_outside_provider_history": out_of_range_split_events,
        }
    )
    return frame


def load_frozen_daily_history(
    ticker: str,
    *,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Load a vendor-neutral, dated CSV with raw OHLCV and split events."""
    symbol = ticker.strip().upper()
    if not _SAFE_SYMBOL.fullmatch(symbol):
        raise HistoryProviderError("ticker contains unsupported characters")
    configured = root or os.getenv(FROZEN_HISTORY_DIR_ENV)
    if not configured:
        raise HistoryProviderError(f"{FROZEN_HISTORY_DIR_ENV} is required")
    source = Path(configured).expanduser().resolve() / f"{symbol}.csv"
    try:
        frame = pd.read_csv(source)
    except OSError as exc:
        raise HistoryProviderError(f"cannot read frozen history: {source}") from exc
    aliases = {column.strip().lower(): column for column in frame.columns}
    required = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "stock splits": "Stock Splits",
    }
    missing = [name for name in required if name not in aliases]
    if missing:
        raise HistoryProviderError(
            "frozen history cannot prove raw OHLC plus splits; missing "
            + ", ".join(missing)
        )
    frame = frame.rename(columns={aliases[key]: value for key, value in required.items()})
    try:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise HistoryProviderError("frozen history contains an invalid date") from exc
    frame = frame.set_index("Date").sort_index()
    if frame.empty or frame.index.has_duplicates:
        raise HistoryProviderError("frozen history is empty or has duplicate dates")
    frame.attrs.update(
        {
            "source": "frozen_csv",
            "provider_symbol": symbol,
            "source_file": source.name,
            "adjustment_mode": "raw_with_split_events",
            "adjustment_source": "frozen_csv_stock_splits",
            "source_interval": "1d",
            "provider_history_start": frame.index[0].strftime("%Y-%m-%d"),
            "provider_history_end": frame.index[-1].strftime("%Y-%m-%d"),
        }
    )
    return frame[["Open", "High", "Low", "Close", "Volume", "Stock Splits"]]
