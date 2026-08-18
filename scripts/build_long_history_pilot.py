#!/usr/bin/env python3
"""Build a development-only, frozen long-history corpus.

This pilot is deliberately separate from the sealed 72-sample v2.4 benchmark.
It is for coverage and detector refinement only and contains no holdout labels.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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
    path.write_bytes(_json_bytes(value))


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _last_completed_quarter_end(today: date) -> date:
    quarter = (today.month - 1) // 3
    if quarter == 0:
        return date(today.year - 1, 12, 31)
    month = quarter * 3
    return date(today.year, month, 31 if month in {3, 12} else 30)


def _require_quarter_end(value: date, label: str) -> None:
    if value.month not in {3, 6, 9, 12} or value.day != calendar.monthrange(
        value.year, value.month
    )[1]:
        raise SystemExit(f"{label} must be a completed calendar quarter-end")


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
        company_name = str(raw.get("company_name") or "").strip()
        listed_since = str(raw.get("listed_since") or "")[:10]
        listing_date_source = str(raw.get("listing_date_source") or "").strip()
        if not company_name:
            raise SystemExit(f"{raw['ticker']} needs a full company_name")
        if company_name.upper() == str(raw["ticker"]).strip().upper():
            raise SystemExit(f"{raw['ticker']} company_name must not equal its ticker")
        if not listed_since:
            raise SystemExit(f"{raw['ticker']} needs a trusted listed_since date")
        try:
            date.fromisoformat(listed_since)
        except ValueError as exc:
            raise SystemExit(
                f"{raw['ticker']} has an invalid listed_since date"
            ) from exc
        if not listing_date_source.startswith(("https://", "http://")):
            raise SystemExit(
                f"{raw['ticker']} needs an HTTP(S) listing_date_source"
            )
        rows.append(
            {
                "ticker": str(raw["ticker"]).strip().upper(),
                "company_name": company_name,
                "provider_symbol": (
                    str(raw["provider_symbol"]).strip().upper()
                    if raw.get("provider_symbol")
                    else None
                ),
                "listed_since": listed_since,
                "listing_date_source": listing_date_source,
                "as_of": (
                    str(raw["as_of"])[:10] if raw.get("as_of") else None
                ),
            }
        )
        if rows[-1]["as_of"]:
            try:
                parsed_as_of = date.fromisoformat(str(rows[-1]["as_of"]))
            except ValueError as exc:
                raise SystemExit(f"{raw['ticker']} has an invalid as_of date") from exc
            _require_quarter_end(parsed_as_of, f"{raw['ticker']} as_of")
    if len({row["ticker"] for row in rows}) != len(rows):
        raise SystemExit("pilot spec contains duplicate tickers")
    return rows


def _coverage(
    first_date: str,
    last_date: str,
    listed_since: str,
    listing_date_source: str,
    as_of: date,
    data_quality: dict[str, Any],
    admitted_daily_start: str,
    admitted_daily_end: str,
) -> dict[str, Any]:
    first = date.fromisoformat(first_date)
    last = date.fromisoformat(last_date)
    listing = date.fromisoformat(listed_since)
    first_month = first.replace(day=1)
    listing_month = listing.replace(day=1)
    last_month = last.replace(day=1)
    expected_last_month = as_of.replace(day=1)

    issue_codes = {
        str(issue.get("code"))
        for issue in data_quality.get("issues", [])
        if isinstance(issue, dict)
    }
    if "missing_periods" in issue_codes:
        status = "internal_month_gaps"
    elif "corporate_action_like_discontinuity" in issue_codes:
        status = "adjustment_unreconciled"
    elif first_month > listing_month:
        status = "truncated_start"
    elif first_month < listing_month:
        status = "prelisting_history_unreconciled"
    elif last_month < expected_last_month:
        status = "stale_end"
    elif last_month > expected_last_month:
        status = "future_data_unreconciled"
    else:
        status = "verified_listing_quarter_to_date"
    return {
        "granularity": "completed_calendar_months_aggregated_for_quarterly_review",
        "admitted_daily_start": admitted_daily_start,
        "admitted_daily_end": admitted_daily_end,
        "first_bar_date": first_date,
        "last_bar_date": last_date,
        "listed_since": listed_since,
        "listing_date_source": listing_date_source,
        "review_as_of": as_of.isoformat(),
        "first_bar_lag_days": (first - listing).days,
        "start_month_gap": (first_month.year - listing_month.year) * 12
        + first_month.month
        - listing_month.month,
        "end_month_gap": (expected_last_month.year - last_month.year) * 12
        + expected_last_month.month
        - last_month.month,
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
    value.add_argument(
        "--allow-unverified",
        action="store_true",
        help="write a diagnostic-only corpus when listing-quarter coverage fails",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output must be absent or empty: {output}")
    items = _load_spec(args.spec.resolve())
    today = datetime.now(timezone.utc).date()
    default_cutoff = args.as_of or _last_completed_quarter_end(today)
    _require_quarter_end(default_cutoff, "default as_of")
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_items: list[dict[str, Any]] = []
    pending_snapshots: list[tuple[Path, bytes]] = []

    for item in items:
        ticker = str(item["ticker"])
        cutoff = (
            date.fromisoformat(str(item["as_of"]))
            if item.get("as_of")
            else default_cutoff
        )
        if args.provider == "eodhd":
            daily = fetch_eodhd_daily_history(
                ticker,
                provider_symbol=item["provider_symbol"],
            )
        else:
            daily = load_frozen_daily_history(ticker, root=args.frozen_root)
        provider_attrs = dict(daily.attrs)
        admitted_daily = daily.loc[daily.index <= pd.Timestamp(cutoff)].copy()
        admitted_daily.attrs.update(provider_attrs)
        if admitted_daily.empty:
            raise SystemExit(f"{ticker} has no daily history at {cutoff.isoformat()}")
        admitted_daily_start = admitted_daily.index[0].strftime("%Y-%m-%d")
        admitted_daily_end = admitted_daily.index[-1].strftime("%Y-%m-%d")
        admitted_daily.attrs["admitted_daily_start"] = admitted_daily_start
        admitted_daily.attrs["admitted_daily_end"] = admitted_daily_end
        daily = admitted_daily
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
            str(item["listed_since"]),
            str(item["listing_date_source"]),
            cutoff,
            inspected.report,
            admitted_daily_start,
            admitted_daily_end,
        )
        bars_sha256 = hashlib.sha256(canonical_json(bars).encode()).hexdigest()
        security = {
            "ticker": ticker,
            "company_name": item["company_name"],
            "provider_symbol": daily.attrs.get("provider_symbol"),
            "listed_since": item["listed_since"],
            "listing_date_source": item["listing_date_source"],
        }
        security_identity_sha256 = hashlib.sha256(
            canonical_json(security).encode()
        ).hexdigest()
        coverage_sha256 = hashlib.sha256(
            canonical_json(coverage).encode()
        ).hexdigest()
        snapshot = {
            "schema_version": 1,
            "kind": "coilingview.long-history-research-snapshot",
            "ticker": ticker,
            "company_name": item["company_name"],
            "security": security,
            "security_identity_sha256": security_identity_sha256,
            "as_of": cutoff.isoformat(),
            "generated_at": generated_at,
            "provider": {
                "name": args.provider,
                "symbol": daily.attrs.get("provider_symbol"),
                "source_interval": "1d",
                "provider_history_start": provider_attrs.get(
                    "provider_history_start"
                ),
                "provider_history_end": provider_attrs.get("provider_history_end"),
                "admitted_daily_start": admitted_daily_start,
                "admitted_daily_end": admitted_daily_end,
                "adjustment_mode": "split_adjusted",
                "adjustment_source": monthly.attrs.get("adjustment_source"),
                "adjustment_transform_version": monthly.attrs.get(
                    "adjustment_transform_version"
                ),
            },
            "coverage": coverage,
            "coverage_sha256": coverage_sha256,
            "data_quality": inspected.report,
            "bars_sha256": bars_sha256,
            "monthly_bars": bars,
        }
        snapshot_path = output / f"{ticker}.json"
        snapshot_bytes = _json_bytes(snapshot)
        pending_snapshots.append((snapshot_path, snapshot_bytes))
        manifest_items.append(
            {
                "ticker": ticker,
                "company_name": item["company_name"],
                "security_identity_sha256": security_identity_sha256,
                "coverage_sha256": coverage_sha256,
                "snapshot_file": snapshot_path.name,
                "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
                "bars_sha256": bars_sha256,
                "bar_count": len(bars),
                "coverage": coverage,
                "data_quality_status": inspected.report.get("status"),
            }
        )

    unverified = [
        {
            "ticker": item["ticker"],
            "status": item["coverage"]["status"],
        }
        for item in manifest_items
        if item["coverage"]["status"] != "verified_listing_quarter_to_date"
    ]
    if unverified and not args.allow_unverified:
        raise SystemExit(
            "listing-quarter coverage is not verified; no corpus was written: "
            + canonical_json(unverified)
        )

    manifest = {
        "schema_version": 1,
        "kind": "coilingview.long-history-research-manifest",
        "generated_at": generated_at,
        "provider": args.provider,
        "as_of_policy": "per_symbol_or_default",
        "default_as_of": default_cutoff.isoformat(),
        "distinct_as_of": sorted(
            {str(item["coverage"]["review_as_of"]) for item in manifest_items}
        ),
        "purpose": "development-only history coverage and detector refinement",
        "release_status": (
            "diagnostic_unverified" if unverified else "verified_listing_quarter_to_date"
        ),
        "holdout_labels_included": False,
        "items": manifest_items,
        "counts": {
            "total": len(manifest_items),
            "verified_listing_quarter_to_date": sum(
                item["coverage"]["status"] == "verified_listing_quarter_to_date"
                for item in manifest_items
            ),
            "truncated_start": sum(
                item["coverage"]["status"] == "truncated_start"
                for item in manifest_items
            ),
            "other_unverified": sum(
                item["coverage"]["status"]
                not in {"verified_listing_quarter_to_date", "truncated_start"}
                for item in manifest_items
            ),
        },
    }
    for snapshot_path, snapshot_bytes in pending_snapshots:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(snapshot_bytes)
    _write_json(output / "manifest.json", manifest)
    print(output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
