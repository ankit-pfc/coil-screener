#!/usr/bin/env python3
"""Pull a restartable, label-free EODHD price archive for the v2.4 universe.

The archive contains raw daily OHLCV and separate split events. It is
acquisition evidence, not a verified benchmark corpus; provider history bounds
must still be checked against listing dates before benchmark promotion.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "review_snapshots"
    / "benchmark_2026-08-13_v24_72"
    / "benchmark-spec.json"
)
EODHD_BASE_URL = "https://eodhd.com/api"
TOKEN_ENV = "COILINGVIEW_EODHD_API_TOKEN"

EXCHANGE_SUFFIXES = {
    "AS": "AS",
    "AX": "AU",
    "DE": "XETRA",
    "HK": "HK",
    "L": "LSE",
    "PA": "PA",
    "SW": "SW",
    "TO": "TO",
    "TW": "TW",
}
UNSUPPORTED_EXCHANGES = {
    "NS": "EODHD EOD exchange list does not expose NSE",
    "T": "EODHD EOD exchange list does not expose Tokyo/TSE",
}


class PullError(RuntimeError):
    """A provider response could not be safely archived."""


def resolve_provider_symbol(ticker: str) -> tuple[str | None, str | None]:
    symbol = ticker.strip().upper()
    if "." not in symbol:
        return f"{symbol}.US", None
    code, suffix = symbol.rsplit(".", 1)
    if suffix in UNSUPPORTED_EXCHANGES:
        return None, UNSUPPORTED_EXCHANGES[suffix]
    exchange = EXCHANGE_SUFFIXES.get(suffix)
    if not exchange:
        return None, f"no reviewed EODHD mapping for .{suffix}"
    return f"{code}.{exchange}", None


def load_tickers(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        markets = payload["markets"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PullError(f"cannot read benchmark spec: {path}") from exc
    tickers = [
        str(item["ticker"]).strip().upper()
        for rows in markets.values()
        for item in rows
    ]
    if len(tickers) != 72 or len(set(tickers)) != 72:
        raise PullError("benchmark spec must contain 72 unique tickers")
    return tickers


def summarize_daily_rows(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        raise PullError("EOD history is empty or not a list")
    dates: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("date"):
            raise PullError("EOD history contains an invalid row")
        dates.append(str(row["date"])[:10])
        for key in ("open", "high", "low", "close", "volume"):
            try:
                float(row[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise PullError(f"EOD history contains an invalid {key}") from exc
    if len(set(dates)) != len(dates):
        raise PullError("EOD history contains duplicate dates")
    ordered = sorted(dates)
    return {
        "daily_bar_count": len(rows),
        "provider_history_start": ordered[0],
        "provider_history_end": ordered[-1],
        "response_order": "ascending" if dates == ordered else "not_ascending",
    }


def _response_json(
    request_get: Callable[..., Any],
    path: str,
    *,
    token: str,
    params: dict[str, Any] | None = None,
) -> Any:
    query = {"api_token": token, "fmt": "json", **(params or {})}
    response = request_get(f"{EODHD_BASE_URL}/{path}", params=query, timeout=60)
    status = int(getattr(response, "status_code", 200))
    if status >= 400:
        raise PullError(f"{path.split('/', 1)[0]} endpoint returned HTTP {status}")
    try:
        return response.json()
    except (TypeError, ValueError) as exc:
        raise PullError(f"{path.split('/', 1)[0]} endpoint returned invalid JSON") from exc


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_gzip_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.GzipFile(
        filename=temporary,
        mode="wb",
        compresslevel=9,
        mtime=0,
    ) as handle:
        handle.write(_json_bytes(value))
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resume_item_valid(item: Any, output: Path) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("status") == "unsupported_exchange":
        return True
    if item.get("status") != "downloaded":
        return False
    for file_key, hash_key in (
        ("raw_daily_file", "raw_daily_sha256"),
        ("raw_splits_file", "raw_splits_sha256"),
    ):
        relative = item.get(file_key)
        expected = item.get(hash_key)
        if not isinstance(relative, str) or not isinstance(expected, str):
            return False
        candidate = (output / relative).resolve()
        if output != candidate and output not in candidate.parents:
            return False
        if not candidate.is_file():
            return False
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            return False
    return True


def pull_one(
    ticker: str,
    provider_symbol: str,
    output: Path,
    *,
    token: str,
    request_get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    common = {"from": "1900-01-01", "order": "a"}
    daily = _response_json(
        request_get,
        f"eod/{provider_symbol}",
        token=token,
        params=common,
    )
    splits = _response_json(
        request_get,
        f"splits/{provider_symbol}",
        token=token,
        params=common,
    )
    summary = summarize_daily_rows(daily)
    if not isinstance(splits, list):
        raise PullError("split history is not a list")
    daily_path = Path("raw") / f"{ticker}.daily.json.gz"
    splits_path = Path("raw") / f"{ticker}.splits.json.gz"
    daily_sha256 = _write_gzip_json(output / daily_path, daily)
    splits_sha256 = _write_gzip_json(output / splits_path, splits)
    return {
        "ticker": ticker,
        "provider_symbol": provider_symbol,
        "status": "downloaded",
        **summary,
        "split_event_count": len(splits),
        "raw_daily_file": daily_path.as_posix(),
        "raw_daily_sha256": daily_sha256,
        "raw_splits_file": splits_path.as_posix(),
        "raw_splits_sha256": splits_sha256,
        "coverage_disposition": "provider_history_unverified_against_listing_date",
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--delay-seconds", type=float, default=0.15)
    return value


def main() -> int:
    args = parser().parse_args()
    token = os.getenv(TOKEN_ENV, "").strip()
    if not token:
        raise SystemExit(f"{TOKEN_ENV} is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    items_dir = output / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    tickers = load_tickers(args.spec.resolve())
    for index, ticker in enumerate(tickers, start=1):
        sidecar = items_dir / f"{ticker}.json"
        if sidecar.exists():
            try:
                prior = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior = {}
            if _resume_item_valid(prior, output):
                print(f"[{index:02d}/72] {ticker}: resume-skip", flush=True)
                continue
        provider_symbol, unsupported_reason = resolve_provider_symbol(ticker)
        if unsupported_reason:
            item = {
                "ticker": ticker,
                "provider_symbol": None,
                "status": "unsupported_exchange",
                "reason": unsupported_reason,
                "coverage_disposition": "not_downloaded",
            }
        else:
            try:
                item = pull_one(
                    ticker,
                    str(provider_symbol),
                    output,
                    token=token,
                )
            except PullError as exc:
                item = {
                    "ticker": ticker,
                    "provider_symbol": provider_symbol,
                    "status": "failed",
                    "reason": str(exc),
                    "coverage_disposition": "not_downloaded",
                }
            except requests.RequestException:
                item = {
                    "ticker": ticker,
                    "provider_symbol": provider_symbol,
                    "status": "failed",
                    "reason": "provider request failed",
                    "coverage_disposition": "not_downloaded",
                }
        _write_json(sidecar, item)
        print(f"[{index:02d}/72] {ticker}: {item['status']}", flush=True)
        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    items = [
        json.loads((items_dir / f"{ticker}.json").read_text(encoding="utf-8"))
        for ticker in tickers
    ]
    counts: dict[str, int] = {}
    for item in items:
        status = str(item["status"])
        counts[status] = counts.get(status, 0) + 1
    manifest = {
        "schema_version": 1,
        "kind": "coilingview.eodhd-raw-history-archive",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provider": "eodhd",
        "data_scope": "raw_daily_ohlcv_and_split_events",
        "requested_from": "1900-01-01",
        "requested_ticker_count": len(tickers),
        "holdout_labels_included": False,
        "canonical_benchmark_eligible": False,
        "coverage_note": (
            "Provider maximum only; every downloaded item requires listing-date "
            "and corporate-action verification before promotion."
        ),
        "counts": counts,
        "items": items,
    }
    _write_json(output / "manifest.json", manifest)
    print(output / "manifest.json", flush=True)
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
