#!/usr/bin/env python3
"""Build a development-only, frozen long-history corpus.

This pilot is deliberately separate from the sealed 72-sample v2.4 benchmark.
It is for coverage and detector refinement only and contains no holdout labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bar_integrity import inspect_monthly_bars  # noqa: E402
from benchmark_v24 import canonical_json  # noqa: E402
from history_cache import _bars_from_frame  # noqa: E402
from history_providers import (  # noqa: E402
    fetch_eodhd_daily_history,
    load_frozen_daily_history,
)
from screen_monthly import _split_adjust_and_aggregate_monthly  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _last_completed_quarter_end(today: date) -> date:
    quarter = (today.month - 1) // 3
    if quarter == 0:
        return date(today.year - 1, 12, 31)
    month = quarter * 3
    return date(today.year, month, 31 if month in {3, 12} else 30)


def _load_spec(path: Path) -> list[dict[str, str | None]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read pilot spec: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise SystemExit("pilot spec must contain a symbols list")
    rows: list[dict[str, str | None]] = []
    for raw in payload["symbols"]:
        if not isinstance(raw, dict) or not raw.get("ticker"):
            raise SystemExit("each pilot symbol needs a ticker")
        rows.append(
            {
                "ticker": str(raw["ticker"]).strip().upper(),
                "provider_symbol": (
                    str(raw["provider_symbol"]).strip().upper()
                    if raw.get("provider_symbol")
                    else None
                ),
                "listed_since": (
                    str(raw["listed_since"])[:10] if raw.get("listed_since") else None
                ),
            }
        )
    if len({row["ticker"] for row in rows}) != len(rows):
        raise SystemExit("pilot spec contains duplicate tickers")
    return rows


def _coverage(
    first_date: str,
    last_date: str,
    listed_since: str | None,
) -> dict[str, Any]:
    lag_days = None
    status = "listing_date_not_supplied"
    if listed_since:
        first = date.fromisoformat(first_date)
        listing = date.fromisoformat(listed_since)
        lag_days = (first - listing).days
        status = "plausible_inception_coverage" if lag_days <= 366 else "truncated"
    return {
        "first_bar_date": first_date,
        "last_bar_date": last_date,
        "listed_since": listed_since,
        "first_bar_lag_days": lag_days,
        "status": status,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--spec", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument(
        "--provider",
        choices=("eodhd", "frozen_csv"),
        default="eodhd",
    )
    value.add_argument("--frozen-root", type=Path)
    value.add_argument("--as-of", type=date.fromisoformat)
    return value


def main() -> int:
    args = parser().parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output must be absent or empty: {output}")
    items = _load_spec(args.spec.resolve())
    today = datetime.now(timezone.utc).date()
    cutoff = args.as_of or _last_completed_quarter_end(today)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_items: list[dict[str, Any]] = []

    for item in items:
        ticker = str(item["ticker"])
        if args.provider == "eodhd":
            daily = fetch_eodhd_daily_history(
                ticker,
                provider_symbol=item["provider_symbol"],
            )
        else:
            daily = load_frozen_daily_history(ticker, root=args.frozen_root)
        monthly = _split_adjust_and_aggregate_monthly(daily)
        raw_bars = _bars_from_frame(monthly)
        inspected = inspect_monthly_bars(
            raw_bars,
            as_of=cutoff.isoformat(),
            adjustment_mode="split_adjusted",
            today=today,
        )
        if inspected.report.get("blocked") or not inspected.bars:
            raise SystemExit(
                f"{ticker} failed strict integrity: {canonical_json(inspected.report)}"
            )
        bars = inspected.bars
        coverage = _coverage(
            bars[0]["date"],
            bars[-1]["date"],
            item["listed_since"],
        )
        bars_sha256 = hashlib.sha256(canonical_json(bars).encode()).hexdigest()
        snapshot = {
            "schema_version": 1,
            "kind": "coilingview.long-history-research-snapshot",
            "ticker": ticker,
            "as_of": cutoff.isoformat(),
            "generated_at": generated_at,
            "provider": {
                "name": args.provider,
                "symbol": daily.attrs.get("provider_symbol"),
                "source_interval": "1d",
                "adjustment_mode": "split_adjusted",
                "adjustment_source": monthly.attrs.get("adjustment_source"),
                "adjustment_transform_version": monthly.attrs.get(
                    "adjustment_transform_version"
                ),
            },
            "coverage": coverage,
            "data_quality": inspected.report,
            "bars_sha256": bars_sha256,
            "monthly_bars": bars,
        }
        snapshot_path = output / f"{ticker}.json"
        _write_json(snapshot_path, snapshot)
        manifest_items.append(
            {
                "ticker": ticker,
                "snapshot_file": snapshot_path.name,
                "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                "bars_sha256": bars_sha256,
                "bar_count": len(bars),
                "coverage": coverage,
                "data_quality_status": inspected.report.get("status"),
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "coilingview.long-history-research-manifest",
        "generated_at": generated_at,
        "provider": args.provider,
        "as_of": cutoff.isoformat(),
        "purpose": "development-only history coverage and detector refinement",
        "holdout_labels_included": False,
        "items": manifest_items,
        "counts": {
            "total": len(manifest_items),
            "plausible_inception_coverage": sum(
                item["coverage"]["status"] == "plausible_inception_coverage"
                for item in manifest_items
            ),
            "truncated": sum(
                item["coverage"]["status"] == "truncated"
                for item in manifest_items
            ),
            "listing_date_not_supplied": sum(
                item["coverage"]["status"] == "listing_date_not_supplied"
                for item in manifest_items
            ),
        },
    }
    _write_json(output / "manifest.json", manifest)
    print(output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
