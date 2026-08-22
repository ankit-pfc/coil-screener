"""Freeze EODHD monthly OHLCV for detector-only benchmark episodes.

The adapter is intentionally separate from policy evaluation and detector
execution. It fetches one provider response, validates the documented monthly
EOD shape, records token-free request provenance, and seals both the provider
rows and normalized detector bars with SHA-256 identities.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from gold_labels import canonical_json, sha256_json


SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_KIND = "coilingview.eodhd-monthly-ohlcv-snapshot"
PROVIDER = "EODHD"
DEFAULT_BASE_URL = "https://eodhd.com/api"
TOKEN_ENVIRONMENT_VARIABLE = "EODHD_API_TOKEN"
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=^-]{0,63}$")
_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class EodhdIngestionError(ValueError):
    """Provider data or frozen snapshot failed an integrity check."""


def _iso_date(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise EodhdIngestionError(f"{field} must use ISO YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise EodhdIngestionError(
            f"{field} must use ISO YYYY-MM-DD"
        ) from exc
    if parsed.isoformat() != value:
        raise EodhdIngestionError(f"{field} must use ISO YYYY-MM-DD")
    return value


def _utc_timestamp(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EodhdIngestionError(f"{field} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EodhdIngestionError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EodhdIngestionError(f"{field} must be UTC")
    return value


def _finite_number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise EodhdIngestionError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EodhdIngestionError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise EodhdIngestionError(f"{field} must be finite")
    if positive and number <= 0:
        raise EodhdIngestionError(f"{field} must be positive")
    return number


def _normalized_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if not _SYMBOL_RE.fullmatch(symbol):
        raise EodhdIngestionError("EODHD symbol is missing or invalid")
    return symbol


def _validated_range(date_from: Any, date_to: Any) -> tuple[str, str]:
    start = _iso_date(date_from, field="requested_from")
    end = _iso_date(date_to, field="requested_to")
    if start > end:
        raise EodhdIngestionError("requested_from must not follow requested_to")
    return start, end


def normalize_monthly_rows(
    provider_rows: Sequence[Mapping[str, Any]],
    *,
    date_from: str,
    date_to: str,
) -> list[dict[str, Any]]:
    """Validate EODHD's monthly response and return detector-safe OHLCV bars."""
    start, end = _validated_range(date_from, date_to)
    if not isinstance(provider_rows, Sequence) or isinstance(
        provider_rows, (str, bytes, bytearray)
    ):
        raise EodhdIngestionError("EODHD response must be a JSON array")
    if not provider_rows:
        raise EodhdIngestionError("EODHD returned no monthly bars")

    normalized: list[dict[str, Any]] = []
    previous_date: str | None = None
    seen_months: set[str] = set()
    for index, raw in enumerate(provider_rows):
        if not isinstance(raw, Mapping):
            raise EodhdIngestionError(f"provider row {index} must be an object")
        bar_date = _iso_date(raw.get("date"), field=f"provider row {index} date")
        if not start <= bar_date <= end:
            raise EodhdIngestionError(
                f"provider row {index} date falls outside the requested range"
            )
        if previous_date is not None and bar_date <= previous_date:
            raise EodhdIngestionError(
                "EODHD monthly rows must be unique and strictly ascending"
            )
        month = bar_date[:7]
        if month in seen_months:
            raise EodhdIngestionError(
                "EODHD monthly rows must contain at most one bar per month"
            )
        previous_date = bar_date
        seen_months.add(month)

        open_price = _finite_number(
            raw.get("open"), field=f"provider row {index} open", positive=True
        )
        high = _finite_number(
            raw.get("high"), field=f"provider row {index} high", positive=True
        )
        low = _finite_number(
            raw.get("low"), field=f"provider row {index} low", positive=True
        )
        close = _finite_number(
            raw.get("close"), field=f"provider row {index} close", positive=True
        )
        if high < max(open_price, low, close) or low > min(open_price, high, close):
            raise EodhdIngestionError(
                f"provider row {index} violates OHLC containment"
            )
        volume = _finite_number(
            raw.get("volume"), field=f"provider row {index} volume"
        )
        if volume < 0:
            raise EodhdIngestionError(
                f"provider row {index} volume must be nonnegative"
            )
        normalized.append(
            {
                "date": bar_date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return normalized


class EodhdClient:
    """Small authenticated client for the documented historical EOD endpoint."""

    def __init__(
        self,
        api_token: str,
        *,
        session: requests.Session | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        token = str(api_token or "").strip()
        if not token:
            raise EodhdIngestionError("EODHD API token is required")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise EodhdIngestionError("timeout_seconds must be positive")
        self._api_token = token
        self._session = session or requests.Session()
        self._base_url = str(base_url).rstrip("/")
        self._timeout_seconds = float(timeout_seconds)

    def fetch_monthly_rows(
        self, symbol: str, *, date_from: str, date_to: str
    ) -> list[dict[str, Any]]:
        normalized_symbol = _normalized_symbol(symbol)
        start, end = _validated_range(date_from, date_to)
        endpoint = f"{self._base_url}/eod/{normalized_symbol}"
        params = {
            "api_token": self._api_token,
            "fmt": "json",
            "period": "m",
            "order": "a",
            "from": start,
            "to": end,
        }
        try:
            response = self._session.get(
                endpoint,
                params=params,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            suffix = f" (HTTP {status})" if status is not None else ""
            raise EodhdIngestionError(f"EODHD request failed{suffix}") from None
        try:
            payload = response.json()
        except ValueError:
            raise EodhdIngestionError("EODHD returned invalid JSON") from None
        if not isinstance(payload, list):
            raise EodhdIngestionError("EODHD response must be a JSON array")
        rows = [dict(row) if isinstance(row, Mapping) else row for row in payload]
        normalize_monthly_rows(rows, date_from=start, date_to=end)
        return rows


def build_frozen_snapshot(
    *,
    symbol: str,
    date_from: str,
    date_to: str,
    provider_rows: Sequence[Mapping[str, Any]],
    fetched_at: str,
    code_sha: str,
) -> dict[str, Any]:
    """Build and hash the replayable provider/normalization envelope."""
    normalized_symbol = _normalized_symbol(symbol)
    start, end = _validated_range(date_from, date_to)
    timestamp = _utc_timestamp(fetched_at, field="fetched_at")
    normalized_sha = str(code_sha or "").strip().lower()
    if not _GIT_SHA_RE.fullmatch(normalized_sha):
        raise EodhdIngestionError("code_sha must be a full Git SHA")

    rows = [dict(row) if isinstance(row, Mapping) else row for row in provider_rows]
    try:
        canonical_json(rows)
    except (TypeError, ValueError) as exc:
        raise EodhdIngestionError(
            "provider rows must be finite canonical JSON"
        ) from exc
    monthly_bars = normalize_monthly_rows(rows, date_from=start, date_to=end)
    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "provider": PROVIDER,
        "symbol": normalized_symbol,
        "fetched_at": timestamp,
        "code_sha": normalized_sha,
        "request": {
            "endpoint": f"{DEFAULT_BASE_URL}/eod/{normalized_symbol}",
            "params": {
                "fmt": "json",
                "period": "m",
                "order": "a",
                "from": start,
                "to": end,
            },
        },
        "provider_semantics": {
            "ohlc_adjustment": "raw_unadjusted",
            "adjusted_close_used": False,
            "volume_adjustment": "provider_split_adjusted",
        },
        "provider_rows": rows,
        "provider_rows_sha256": sha256_json(rows),
        "monthly_bars": monthly_bars,
        "monthly_bars_sha256": sha256_json(monthly_bars),
    }
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    return snapshot


def validate_frozen_snapshot(raw: Any) -> dict[str, Any]:
    """Reproduce and verify every derived field in an EODHD snapshot."""
    if not isinstance(raw, dict):
        raise EodhdIngestionError("EODHD snapshot must be a JSON object")
    if raw.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise EodhdIngestionError("unsupported EODHD snapshot schema")
    if raw.get("kind") != SNAPSHOT_KIND or raw.get("provider") != PROVIDER:
        raise EodhdIngestionError("unexpected EODHD snapshot identity")
    request = raw.get("request")
    params = request.get("params") if isinstance(request, dict) else None
    if not isinstance(params, dict):
        raise EodhdIngestionError("EODHD snapshot request metadata is missing")
    if "api_token" in params:
        raise EodhdIngestionError("EODHD snapshot must not persist an API token")
    if set(params) != {"fmt", "period", "order", "from", "to"}:
        raise EodhdIngestionError("EODHD snapshot request parameters are invalid")
    if params.get("fmt") != "json" or params.get("period") != "m":
        raise EodhdIngestionError("EODHD snapshot request format is invalid")
    if params.get("order") != "a":
        raise EodhdIngestionError("EODHD snapshot rows must be requested ascending")

    symbol = _normalized_symbol(raw.get("symbol"))
    endpoint = f"{DEFAULT_BASE_URL}/eod/{symbol}"
    if request.get("endpoint") != endpoint:
        raise EodhdIngestionError("EODHD snapshot endpoint does not match symbol")
    if raw.get("provider_semantics") != {
        "ohlc_adjustment": "raw_unadjusted",
        "adjusted_close_used": False,
        "volume_adjustment": "provider_split_adjusted",
    }:
        raise EodhdIngestionError("EODHD provider semantics are invalid")
    start, end = _validated_range(params.get("from"), params.get("to"))
    _utc_timestamp(raw.get("fetched_at"), field="fetched_at")
    code_sha = str(raw.get("code_sha") or "").strip().lower()
    if not _GIT_SHA_RE.fullmatch(code_sha):
        raise EodhdIngestionError("EODHD snapshot code_sha is invalid")

    rows = raw.get("provider_rows")
    if not isinstance(rows, list):
        raise EodhdIngestionError("EODHD snapshot provider_rows are missing")
    try:
        rows_sha = sha256_json(rows)
    except (TypeError, ValueError) as exc:
        raise EodhdIngestionError(
            "EODHD snapshot provider_rows are not canonical JSON"
        ) from exc
    if raw.get("provider_rows_sha256") != rows_sha:
        raise EodhdIngestionError("EODHD provider-row hash mismatch")
    reproduced = normalize_monthly_rows(rows, date_from=start, date_to=end)
    if canonical_json(raw.get("monthly_bars")) != canonical_json(reproduced):
        raise EodhdIngestionError("EODHD normalized monthly bars do not reproduce")
    if raw.get("monthly_bars_sha256") != sha256_json(reproduced):
        raise EodhdIngestionError("EODHD monthly-bar hash mismatch")

    without_snapshot_hash = dict(raw)
    claimed = without_snapshot_hash.pop("snapshot_sha256", None)
    if claimed != sha256_json(without_snapshot_hash):
        raise EodhdIngestionError("EODHD snapshot hash mismatch")
    return raw


def write_frozen_snapshot(
    snapshot: dict[str, Any], path: str | Path, *, overwrite: bool = False
) -> Path:
    """Validate and atomically persist one snapshot without silent replacement."""
    validate_frozen_snapshot(snapshot)
    target = Path(path)
    if target.exists() and not overwrite:
        raise EodhdIngestionError(f"snapshot already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_frozen_snapshot(path: str | Path) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EodhdIngestionError("EODHD snapshot is missing or unreadable") from exc
    return validate_frozen_snapshot(raw)


def repository_identity(repository: str | Path) -> str:
    """Return HEAD only for a clean commit known to an origin tracking ref."""
    root = Path(repository)

    def git(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EodhdIngestionError("Git repository identity check failed") from exc
        return completed.stdout.strip()

    code_sha = git("rev-parse", "HEAD").lower()
    if not _GIT_SHA_RE.fullmatch(code_sha):
        raise EodhdIngestionError("Git returned an invalid full SHA")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise EodhdIngestionError(
            "authoritative EODHD ingestion requires a clean worktree"
        )
    containing = git(
        "branch",
        "-r",
        "--contains",
        code_sha,
        "--format=%(refname:short)",
    ).splitlines()
    if not any(ref.startswith("origin/") for ref in containing):
        raise EodhdIngestionError(
            "authoritative EODHD ingestion requires an exact pushed SHA"
        )
    return code_sha


def ingest_symbol(
    client: EodhdClient,
    *,
    symbol: str,
    date_from: str,
    date_to: str,
    code_sha: str,
    fetched_at: str | None = None,
) -> dict[str, Any]:
    rows = client.fetch_monthly_rows(
        symbol,
        date_from=date_from,
        date_to=date_to,
    )
    timestamp = fetched_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    return build_frozen_snapshot(
        symbol=symbol,
        date_from=date_from,
        date_to=date_to,
        provider_rows=rows,
        fetched_at=timestamp,
        code_sha=code_sha,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze EODHD monthly OHLCV for detector-only evaluation."
    )
    parser.add_argument(
        "--symbol",
        action="append",
        required=True,
        help="EODHD symbol including exchange suffix; repeat for more symbols.",
    )
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    token = os.environ.get(TOKEN_ENVIRONMENT_VARIABLE)
    if not token:
        raise SystemExit(
            f"{TOKEN_ENVIRONMENT_VARIABLE} must contain the EODHD API token"
        )
    code_sha = repository_identity(Path(__file__).resolve().parent)
    client = EodhdClient(token, timeout_seconds=args.timeout_seconds)
    for raw_symbol in args.symbol:
        symbol = _normalized_symbol(raw_symbol)
        snapshot = ingest_symbol(
            client,
            symbol=symbol,
            date_from=args.date_from,
            date_to=args.date_to,
            code_sha=code_sha,
        )
        target = write_frozen_snapshot(
            snapshot,
            args.output_dir / f"{symbol}.json",
        )
        print(f"{target}\t{snapshot['snapshot_sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
