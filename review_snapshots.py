"""Immutable saved-run inputs for protected human-review sessions.

Fresh review sessions never read the mutable history cache.  A saved CSV run
maps to ``review_snapshots/<run-stem>/<TICKER>.json`` and each file contains
the exact monthly bars and screener row used for that run.  This module
validates those files, derives content identities, and serves the algorithm
snapshot captured with that run without consulting human overrides.
"""
from __future__ import annotations

import calendar
import hashlib
import hmac
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any

from coil_analysis import (
    ALGORITHM_VERSION,
    ANALYSIS_MODE_ALGORITHM_ONLY,
    ANALYSIS_VARIANT_V2_3_1,
    ANALYSIS_VARIANT_V2_4_VALIDATION,
    _aggregate_quarterly_display_bars,
    analyze_coil,
)

PROJECT_ROOT = Path(__file__).resolve().parent
REVIEW_SNAPSHOT_ROOT = PROJECT_ROOT / "review_snapshots"
SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_KIND = "coilingview.saved-run-review-snapshot"
REVIEW_CORPUS_MANIFEST_KIND = "coilingview.saved-run-review-corpus-manifest"
REVIEW_LEGACY_MANIFEST_KIND = "coilingview.saved-run-review-manifest"
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.^=_-]{0,31}$")


class ReviewSnapshotError(ValueError):
    """A saved-run snapshot is missing, unsafe, or internally inconsistent."""


def canonical_json(value: Any) -> str:
    """One stable JSON representation used by all review content hashes."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _saved_run_stem(source: str) -> str:
    name = str(source or "").strip()
    if not name or Path(name).name != name or not name.lower().endswith(".csv"):
        raise ReviewSnapshotError(
            "fresh review source must be a saved-run CSV filename"
        )
    stem = Path(name).stem
    if not stem or stem in {".", ".."}:
        raise ReviewSnapshotError("fresh review source is invalid")
    return stem


def _normalized_ticker(ticker: str) -> str:
    symbol = str(ticker or "").strip().upper()
    if not _TICKER_RE.fullmatch(symbol):
        raise ReviewSnapshotError("snapshot ticker contains unsupported characters")
    return symbol


def snapshot_path(source: str, ticker: str) -> Path:
    """Resolve a snapshot path without allowing source/ticker traversal."""
    stem = _saved_run_stem(source)
    symbol = _normalized_ticker(ticker)
    root = REVIEW_SNAPSHOT_ROOT.resolve()
    candidate = (root / stem / f"{symbol}.json").resolve()
    if candidate.parent != (root / stem).resolve():
        raise ReviewSnapshotError("snapshot path escapes the snapshot root")
    return candidate


def load_review_manifest(source: str) -> dict[str, Any]:
    """Load the immutable corpus manifest for a saved run."""
    stem = _saved_run_stem(source)
    path = (REVIEW_SNAPSHOT_ROOT / stem / "manifest.json").resolve()
    expected_parent = (REVIEW_SNAPSHOT_ROOT.resolve() / stem).resolve()
    if path.parent != expected_parent or not path.is_file():
        raise ReviewSnapshotError(
            f"frozen review manifest is missing for {source}"
        )
    try:
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewSnapshotError("frozen review manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise ReviewSnapshotError("frozen review manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ReviewSnapshotError("unsupported frozen review manifest schema")
    if manifest.get("kind") not in {
        REVIEW_CORPUS_MANIFEST_KIND,
        REVIEW_LEGACY_MANIFEST_KIND,
    }:
        raise ReviewSnapshotError("unexpected frozen review manifest kind")
    manifest_source = manifest.get("source")
    if manifest_source is None and isinstance(manifest.get("source_run"), dict):
        manifest_source = manifest["source_run"].get("filename")
    if manifest_source is not None and str(manifest_source) != source:
        raise ReviewSnapshotError("manifest source does not match the session source")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ReviewSnapshotError("frozen review manifest has no items")
    manifest["_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest


def _finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReviewSnapshotError(f"snapshot bar {field} must be numeric") from exc
    if not math.isfinite(number):
        raise ReviewSnapshotError(f"snapshot bar {field} must be finite")
    return number


def _inspect_bars(
    raw_bars: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Preserve raw bars while quarantining, never repairing, bad OHLC data."""
    if not isinstance(raw_bars, list) or not raw_bars:
        raise ReviewSnapshotError("snapshot must contain monthly bars")
    raw_copy: list[dict[str, Any]] = []
    analysis_bars: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    previous_date: str | None = None
    for index, raw in enumerate(raw_bars):
        if not isinstance(raw, dict):
            raise ReviewSnapshotError("snapshot bars must be JSON objects")
        raw_copy.append(dict(raw))
        bar_date = str(raw.get("date", ""))
        try:
            parsed_date = date.fromisoformat(bar_date)
            date_valid = parsed_date.isoformat() == bar_date
        except (TypeError, ValueError):
            date_valid = False
        if not date_valid:
            issues.append(
                {
                    "index": index,
                    "date": bar_date or None,
                    "code": "invalid_date",
                    "message": "bar date is not ISO YYYY-MM-DD",
                }
            )
        if bar_date in seen_dates:
            issues.append(
                {
                    "index": index,
                    "date": bar_date,
                    "code": "duplicate_date",
                    "message": "bar date is duplicated",
                }
            )
        if previous_date is not None and bar_date <= previous_date:
            issues.append(
                {
                    "index": index,
                    "date": bar_date,
                    "code": "non_chronological_date",
                    "message": "bars are not strictly chronological",
                }
            )
        seen_dates.add(bar_date)
        previous_date = bar_date
        values: dict[str, float] = {}
        for field in ("open", "high", "low", "close"):
            try:
                values[field] = _finite_number(raw.get(field), field=field)
            except ReviewSnapshotError:
                issues.append(
                    {
                        "index": index,
                        "date": bar_date or None,
                        "code": f"invalid_{field}",
                        "message": f"{field} is missing or non-finite",
                    }
                )
        if len(values) != 4:
            continue
        nonpositive = [field for field, value in values.items() if value <= 0]
        if nonpositive:
            issues.append(
                {
                    "index": index,
                    "date": bar_date,
                    "code": "nonpositive_ohlc",
                    "fields": nonpositive,
                    "message": "OHLC prices must be positive for chart evidence",
                }
            )
        if (
            values["low"]
            > min(values["open"], values["close"], values["high"])
            or values["high"]
            < max(values["open"], values["close"], values["low"])
        ):
            issues.append(
                {
                    "index": index,
                    "date": bar_date,
                    "code": "impossible_ohlc_range",
                    "message": "high/low do not contain the candle body",
                }
            )
        bar: dict[str, Any] = {
            "date": bar_date,
            **values,
        }
        volume = raw.get("volume")
        if volume is not None:
            try:
                bar["volume"] = _finite_number(volume, field="volume")
            except ReviewSnapshotError:
                issues.append(
                    {
                        "index": index,
                        "date": bar_date,
                        "code": "invalid_volume",
                        "message": "volume is non-finite",
                    }
                )
        analysis_bars.append(bar)
    reviewable = not issues and len(analysis_bars) == len(raw_copy)
    quality = {
        "reviewable": reviewable,
        "status": "accepted" if reviewable else "quarantined",
        "blocking_issue_count": len(issues),
        "blocking_issues": issues,
        "repair_applied": False,
    }
    return raw_copy, analysis_bars, quality


def _identity_payload(
    *,
    source: str,
    ticker: str,
    run: dict[str, Any],
    screen_snapshot: dict[str, Any],
    bars: list[dict[str, Any]],
    snapshot_sha256: str,
    algorithm_version: str,
) -> tuple[str, str]:
    bars_hash = sha256_json(
        {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "interval": "1M",
            "ticker": ticker,
            "bars": bars,
        }
    )
    sample_id = sha256_json(
        {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "source": source,
            "ticker": ticker,
            "algorithm_version": algorithm_version,
            "run": run,
            "screen_snapshot": screen_snapshot,
            "bars_hash": bars_hash,
            "snapshot_sha256": snapshot_sha256,
        }
    )
    return bars_hash, sample_id


def load_review_snapshot(source: str, ticker: str) -> dict[str, Any]:
    """Load and validate one frozen snapshot, without running the analyzer."""
    symbol = _normalized_ticker(ticker)
    path = snapshot_path(source, symbol)
    if not path.is_file():
        raise ReviewSnapshotError(
            f"frozen review snapshot is missing for {symbol} in {source}"
        )
    try:
        snapshot_bytes = path.read_bytes()
        raw = json.loads(snapshot_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewSnapshotError(
            f"frozen review snapshot is unreadable for {symbol}"
        ) from exc
    if not isinstance(raw, dict):
        raise ReviewSnapshotError("snapshot root must be a JSON object")
    if raw.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ReviewSnapshotError("unsupported frozen review snapshot schema")
    if raw.get("kind") != SNAPSHOT_KIND:
        raise ReviewSnapshotError("unexpected frozen review snapshot kind")
    if str(raw.get("source", "")) != str(source):
        raise ReviewSnapshotError("snapshot source does not match the session source")
    if str(raw.get("ticker", "")).upper() != symbol:
        raise ReviewSnapshotError("snapshot ticker does not match the requested ticker")

    run = raw.get("run")
    screen_snapshot = raw.get("screen_snapshot")
    if not isinstance(run, dict) or not isinstance(screen_snapshot, dict):
        raise ReviewSnapshotError("snapshot run and screen metadata are required")
    run_algorithm_version = str(run.get("algorithm_version", "")).strip()
    if not run_algorithm_version:
        raise ReviewSnapshotError("snapshot run algorithm version is required")
    if str(screen_snapshot.get("ticker", "")).strip().upper() != symbol:
        raise ReviewSnapshotError("screen snapshot ticker does not match")
    bars, analysis_bars, server_quality = _inspect_bars(raw.get("monthly_bars"))
    frozen_as_of = raw.get("as_of")
    if frozen_as_of is not None:
        try:
            parsed_as_of = date.fromisoformat(str(frozen_as_of))
        except ValueError as exc:
            raise ReviewSnapshotError("snapshot as_of must be an ISO date") from exc
        if (
            parsed_as_of.month not in {3, 6, 9, 12}
            or parsed_as_of.day
            != calendar.monthrange(parsed_as_of.year, parsed_as_of.month)[1]
        ):
            raise ReviewSnapshotError("snapshot as_of must be a completed quarter end")
        last_bar = date.fromisoformat(str(bars[-1]["date"])[:10])
        if (last_bar.year, last_bar.month) != (parsed_as_of.year, parsed_as_of.month):
            raise ReviewSnapshotError("snapshot bars must end in the as_of month")
    corpus_quality = raw.get("data_quality", raw.get("data_quality_snapshot"))
    if corpus_quality is not None and not isinstance(corpus_quality, dict):
        raise ReviewSnapshotError("snapshot data_quality must be a JSON object")
    corpus_reviewable = (
        bool(corpus_quality.get("reviewable"))
        if isinstance(corpus_quality, dict)
        else True
    )
    reviewable = bool(server_quality["reviewable"] and corpus_reviewable)
    data_quality = corpus_quality or server_quality
    data_quality_validation = {
        "reviewable": reviewable,
        "status": "accepted" if reviewable else "quarantined",
        "server_checks": server_quality,
        "corpus_report_present": corpus_quality is not None,
    }
    frozen_analysis = raw.get("algorithm_analysis")
    if frozen_analysis is not None and not isinstance(frozen_analysis, dict):
        raise ReviewSnapshotError("frozen algorithm analysis must be a JSON object")
    if frozen_analysis is not None and str(
        frozen_analysis.get("algorithm_version", "")
    ) != run_algorithm_version:
        raise ReviewSnapshotError(
            "frozen analysis algorithm version does not match its run"
        )
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    bars_hash, sample_id = _identity_payload(
        source=source,
        ticker=symbol,
        run=run,
        screen_snapshot=screen_snapshot,
        bars=bars,
        snapshot_sha256=snapshot_sha256,
        algorithm_version=run_algorithm_version,
    )
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "source": source,
        "ticker": symbol,
        "as_of": str(frozen_as_of) if frozen_as_of is not None else None,
        "run": run,
        "screen_snapshot": screen_snapshot,
        "corpus_labels": raw.get("corpus_labels") or {},
        "provenance": raw.get("provenance") or {},
        "source_features": raw.get("source_features"),
        "source_cache_metadata": raw.get("source_cache_metadata"),
        "monthly_bars": bars,
        "_analysis_bars": analysis_bars,
        "_frozen_analysis": frozen_analysis,
        "data_quality": data_quality,
        "data_quality_validation": data_quality_validation,
        "reviewable": reviewable,
        "bars_hash": bars_hash,
        "snapshot_sha256": snapshot_sha256,
        "screen_snapshot_sha256": sha256_json(screen_snapshot),
        "sample_id": sample_id,
    }


def review_snapshot_identity(source: str, ticker: str) -> dict[str, Any]:
    """Small immutable identity used while creating a fresh session."""
    snapshot = load_review_snapshot(source, ticker)
    return {
        "source": snapshot["source"],
        "ticker": snapshot["ticker"],
        "as_of": snapshot.get("as_of"),
        "run": snapshot["run"],
        "screen_snapshot": snapshot["screen_snapshot"],
        "corpus_labels": snapshot["corpus_labels"],
        "provenance": snapshot["provenance"],
        "bars_hash": snapshot["bars_hash"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "screen_snapshot_sha256": snapshot["screen_snapshot_sha256"],
        "sample_id": snapshot["sample_id"],
        "data_date": snapshot["monthly_bars"][-1]["date"],
        "data_quality": snapshot["data_quality"],
        "data_quality_validation": snapshot["data_quality_validation"],
        "reviewable": snapshot["reviewable"],
    }


def verify_manifest_identity(
    identity: dict[str, Any], manifest_item: dict[str, Any]
) -> None:
    """Fail closed when a snapshot differs from its corpus manifest."""
    ticker = identity["ticker"]
    expected_ticker = str(manifest_item.get("ticker", "")).strip().upper()
    if expected_ticker != ticker:
        raise ReviewSnapshotError(f"manifest ticker mismatch for {ticker}")
    comparisons = (
        (
            "backend_bars_identity_sha256",
            identity["bars_hash"],
            "bars identity",
        ),
        (
            "snapshot_sha256",
            identity["snapshot_sha256"],
            "snapshot file",
        ),
        (
            "screen_snapshot_sha256",
            identity["screen_snapshot_sha256"],
            "screen snapshot",
        ),
    )
    for key, actual, label in comparisons:
        expected = manifest_item.get(key)
        if expected is not None and not hmac.compare_digest(
            str(expected), str(actual)
        ):
            raise ReviewSnapshotError(
                f"{ticker} {label} does not match the frozen manifest"
            )
    manifest_quality = manifest_item.get("data_quality")
    if isinstance(manifest_quality, dict) and (
        bool(manifest_quality.get("reviewable")) != bool(identity["reviewable"])
    ):
        raise ReviewSnapshotError(
            f"{ticker} reviewability does not match the frozen manifest"
        )


def load_blind_review_context(source: str, ticker: str) -> dict[str, Any]:
    """Frozen price evidence only; deliberately excludes every model output."""
    snapshot = load_review_snapshot(source, ticker)
    return {
        "schema_version": snapshot["schema_version"],
        "kind": snapshot["kind"],
        "source": snapshot["source"],
        "ticker": snapshot["ticker"],
        "as_of": snapshot.get("as_of"),
        "interval": "3M",
        "data_date": snapshot["monthly_bars"][-1]["date"],
        "monthly_bars": snapshot["monthly_bars"],
        "bars_hash": snapshot["bars_hash"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "sample_id": snapshot["sample_id"],
        "data_quality": snapshot["data_quality"],
        "data_quality_validation": snapshot["data_quality_validation"],
        "reviewable": snapshot["reviewable"],
        "model_revealed": False,
    }


def load_review_context(
    source: str,
    ticker: str,
    *,
    validation_config: Any | None = None,
) -> dict[str, Any]:
    """Frozen inputs plus the algorithm-only analysis captured for the run."""
    snapshot = load_review_snapshot(source, ticker)
    analysis = snapshot.get("_frozen_analysis")
    run_algorithm_version = str(snapshot["run"]["algorithm_version"])
    quarterly_bars: list[dict[str, Any]] = []
    detector_outputs: dict[str, Any] = {}
    analysis_status = "quarantined_unavailable"
    if snapshot["reviewable"]:
        frozen_as_of = snapshot.get("as_of")
        adjustment_mode = str(
            (snapshot.get("source_cache_metadata") or {}).get("adjustment_mode")
            or "unknown"
        )
        if analysis is None:
            if run_algorithm_version == ALGORITHM_VERSION:
                analysis = analyze_coil(
                    snapshot["_analysis_bars"],
                    ticker=snapshot["ticker"],
                    as_of=frozen_as_of,
                    review_override=None,
                    mode=ANALYSIS_MODE_ALGORITHM_ONLY,
                    adjustment_mode=adjustment_mode,
                )
                analysis_status = "frozen_algorithm_only"
            else:
                # The price evidence remains valid across releases. If an
                # older corpus did not persist its model output, do not
                # silently reinterpret it with the current analyzer.
                analysis_status = "frozen_model_unavailable"
        else:
            analysis_status = "frozen_algorithm_only"
        quarterly_bars = [
            {key: value for key, value in bar.items() if not key.startswith("_")}
            for bar in _aggregate_quarterly_display_bars(
                snapshot["_analysis_bars"]
            )
        ]
        detector_outputs = {
            ANALYSIS_VARIANT_V2_3_1: analyze_coil(
                snapshot["_analysis_bars"],
                ticker=snapshot["ticker"],
                as_of=frozen_as_of,
                variant=ANALYSIS_VARIANT_V2_3_1,
                mode=ANALYSIS_MODE_ALGORITHM_ONLY,
                adjustment_mode=adjustment_mode,
            ),
            ANALYSIS_VARIANT_V2_4_VALIDATION: analyze_coil(
                snapshot["_analysis_bars"],
                ticker=snapshot["ticker"],
                as_of=frozen_as_of,
                variant=ANALYSIS_VARIANT_V2_4_VALIDATION,
                mode=ANALYSIS_MODE_ALGORITHM_ONLY,
                adjustment_mode=adjustment_mode,
                validation_config=validation_config,
            ),
        }
    snapshot.pop("_analysis_bars", None)
    snapshot.pop("_frozen_analysis", None)
    return {
        **snapshot,
        "interval": "3M",
        "quarterly_bars": quarterly_bars,
        "analysis": analysis,
        "analysis_status": analysis_status,
        "detector_outputs": detector_outputs,
        "model_snapshot": {
            "screen_snapshot": snapshot["screen_snapshot"],
            "analysis": analysis,
            "algorithm_version": run_algorithm_version,
            "review_override_applied": False,
            "frozen": True,
        },
        "model_revealed": True,
    }
