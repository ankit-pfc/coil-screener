"""Reproducible development benchmark for lifetime reference geometry.

The benchmark keeps three boundaries explicit:

* detectors receive only a physically truncated OHLCV prefix;
* outcome-revealed Amrut geometry is added only after prediction is locked;
* the audited blind queue, including duplicate ticker identities in other
  corpora, is never executed or rendered.

This is deliberately separate from the production screener.  It compares the
current detector with :mod:`lifetime_structure`, measures both against
development references where available, and records where a future hybrid
boundary-to-lifecycle adapter is still missing.
"""

from __future__ import annotations

import calendar
import csv
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Callable, Iterable, Optional

from coil_analysis import (
    ALGORITHM_VERSION as CURRENT_ALGORITHM_VERSION,
    _aggregate_quarterly_display_bars,
    _completed_quarters,
    analyze_coil,
)
from lifetime_structure import (
    ALGORITHM_VERSION as LIFETIME_ALGORITHM_VERSION,
    DEFAULT_LIFETIME_CONFIG,
    analyze_lifetime_references,
)
from review_snapshots import (
    REVIEW_SNAPSHOT_ROOT,
    load_review_manifest,
    load_review_snapshot,
    verify_manifest_identity,
)


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_VERSION = "lifetime-reference-benchmark-v1"
BENCHMARK_KIND = "coilingview.lifetime-reference-benchmark"

EXACT_SOURCE = "amrut_reviewed_exemplars_2026-08-18.csv"
PORTFOLIO_SOURCE = "amrut_portfolio_exemplars_2026-08-21.csv"
SHADOW_SOURCE = "screen_2026-07-27_v2.2.0.csv"
SEALED_SOURCE = "blinded_boundary_negative_2026-08-18_v1.csv"
SEALED_CORPUS_ID = "blinded_boundary_negative_2026-08-18_v1"

ALLOWED_CORPUS_IDS = {
    "amrut_reviewed_exemplars_2026-08-18",
    "amrut_portfolio_exemplars_2026-08-21",
    "screen_2026-07-27_v2.2.0",
}

TOP_TOLERANCE_QUARTERS = 1
PROJECTED_LINE_TOLERANCE_PCT = 5.0
DESCRIPTIVE_RMS_TOLERANCE_PCT = 10.0
SLOPE_TOLERANCE_PCT_PER_YEAR = 1.0
DIRECTION_THRESHOLD_PCT_PER_YEAR = 0.5


class BenchmarkSafetyError(ValueError):
    """The requested corpus or snapshot would violate benchmark isolation."""


@dataclass(frozen=True)
class BenchmarkCase:
    cohort: str
    geometry_quality: Optional[str]
    corpus_id: str
    source: str
    ticker: str
    manifest_sha256: str
    manifest_item: dict[str, Any]
    snapshot: dict[str, Any]
    decision_as_of: str


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    candidate = str(value).strip()[:10]
    try:
        return date.fromisoformat(candidate).isoformat()
    except ValueError:
        return None


def quarter_ordinal(value: str) -> int:
    parsed = date.fromisoformat(str(value)[:10])
    return parsed.year * 4 + (parsed.month - 1) // 3


def _quarter_end_from_ordinal(ordinal: int) -> str:
    year = ordinal // 4
    quarter = ordinal % 4
    month = (quarter + 1) * 3
    day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-{day:02d}"


def _direction(slope_pct_per_year: Optional[float]) -> Optional[str]:
    if slope_pct_per_year is None:
        return None
    if slope_pct_per_year > DIRECTION_THRESHOLD_PCT_PER_YEAR:
        return "rising"
    if slope_pct_per_year < -DIRECTION_THRESHOLD_PCT_PER_YEAR:
        return "falling"
    return "flat"


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _snapshot_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": snapshot["ticker"],
        "bars_hash": snapshot["bars_hash"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "screen_snapshot_sha256": snapshot["screen_snapshot_sha256"],
        "reference_setup_sha256": snapshot.get("reference_setup_sha256"),
        "source_image_sha256": snapshot.get("source_image_sha256"),
        "source_evidence_sha256": snapshot.get("source_evidence_sha256"),
        "reviewable": snapshot["reviewable"],
    }


def _assert_allowed_manifest(manifest: dict[str, Any], source: str) -> None:
    corpus_id = str(manifest.get("corpus_id") or "")
    policy = manifest.get("review_policy") or {}
    if (
        corpus_id == SEALED_CORPUS_ID
        or source == SEALED_SOURCE
        or policy.get("detector_outputs_hidden") is True
    ):
        raise BenchmarkSafetyError(
            "the audited blind corpus is sealed and cannot be benchmarked"
        )
    if corpus_id not in ALLOWED_CORPUS_IDS:
        raise BenchmarkSafetyError(
            f"corpus {corpus_id or source!r} is not on the benchmark allowlist"
        )


def _assert_snapshot_not_blind(snapshot: dict[str, Any]) -> None:
    run = snapshot.get("run") or {}
    screen = snapshot.get("screen_snapshot") or {}
    if (
        run.get("review_policy") == "detector_outputs_hidden"
        or screen.get("review_mode") == "blinded_boundary_negative"
        or snapshot.get("source") == SEALED_SOURCE
    ):
        raise BenchmarkSafetyError("blind snapshot execution is prohibited")


def _sealed_ticker_identities() -> set[str]:
    """Read only the blind manifest roster; never open a blind snapshot."""
    manifest = load_review_manifest(SEALED_SOURCE)
    if (
        manifest.get("corpus_id") != SEALED_CORPUS_ID
        or (manifest.get("review_policy") or {}).get("detector_outputs_hidden")
        is not True
    ):
        raise BenchmarkSafetyError("blind roster did not satisfy its sealed policy")
    universe = manifest.get("ordered_universe")
    if not isinstance(universe, list):
        raise BenchmarkSafetyError("blind roster is missing its ordered universe")
    return {str(ticker).strip().upper() for ticker in universe}


def snapshot_decision_as_of(snapshot: dict[str, Any]) -> str:
    provenance = snapshot.get("provenance") or {}
    cache_metadata = snapshot.get("source_cache_metadata") or {}
    screen = snapshot.get("screen_snapshot") or {}
    candidates = (
        provenance.get("future_data_excluded_after"),
        provenance.get("market_data_fetched_at"),
        cache_metadata.get("fetched_at"),
        screen.get("fetched_at"),
    )
    for candidate in candidates:
        parsed = _iso_date(candidate)
        if parsed:
            return parsed
    last_date = _iso_date(snapshot["monthly_bars"][-1]["date"])
    if last_date is None:
        raise BenchmarkSafetyError("snapshot has no usable point-in-time cutoff")
    return last_date


def sanitize_ohlcv_prefix(
    bars: Iterable[dict[str, Any]], decision_as_of: str
) -> list[dict[str, Any]]:
    """Physically truncate and strip input before either detector sees it."""
    cutoff = date.fromisoformat(decision_as_of)
    output: list[dict[str, Any]] = []
    for raw in bars:
        bar_date = date.fromisoformat(str(raw["date"])[:10])
        if bar_date > cutoff:
            continue
        output.append(
            {
                "date": bar_date.isoformat(),
                "open": float(raw["open"]),
                "high": float(raw["high"]),
                "low": float(raw["low"]),
                "close": float(raw["close"]),
                "volume": (
                    float(raw["volume"])
                    if raw.get("volume") is not None
                    else None
                ),
            }
        )
    output.sort(key=lambda item: item["date"])
    if not output:
        raise BenchmarkSafetyError("physical OHLCV prefix is empty")
    return output


def _load_cases(
    source: str,
    *,
    cohort: str,
    geometry_quality: Optional[str],
    select: Callable[[dict[str, Any]], bool],
    exclude_tickers: Optional[set[str]] = None,
) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    manifest = load_review_manifest(source)
    _assert_allowed_manifest(manifest, source)
    items = [item for item in manifest["items"] if select(item)]
    position_key = "universe_position" if cohort == "shadow" else "position"
    items.sort(key=lambda item: int(item.get(position_key, 0)))
    excluded = exclude_tickers or set()
    cases: list[BenchmarkCase] = []
    for item in items:
        ticker = str(item.get("ticker") or "").strip().upper()
        if ticker in excluded:
            continue
        snapshot = load_review_snapshot(source, ticker)
        _assert_snapshot_not_blind(snapshot)
        verify_manifest_identity(_snapshot_identity(snapshot), item)
        if snapshot.get("reviewable") is not True:
            raise BenchmarkSafetyError(f"selected snapshot {ticker} is not reviewable")
        cases.append(
            BenchmarkCase(
                cohort=cohort,
                geometry_quality=geometry_quality,
                corpus_id=str(manifest["corpus_id"]),
                source=source,
                ticker=ticker,
                manifest_sha256=str(manifest["_manifest_sha256"]),
                manifest_item=item,
                snapshot=snapshot,
                decision_as_of=snapshot_decision_as_of(snapshot),
            )
        )
    return cases, manifest


def load_benchmark_cases() -> dict[str, Any]:
    sealed = _sealed_ticker_identities()
    exact, exact_manifest = _load_cases(
        EXACT_SOURCE,
        cohort="exact_reference",
        geometry_quality="exact_clicks",
        select=lambda item: bool(
            item.get("geometry_status") == "exact"
            and (item.get("data_quality") or {}).get("reviewable") is True
        ),
    )
    portfolio, portfolio_manifest = _load_cases(
        PORTFOLIO_SOURCE,
        cohort="portfolio_reference",
        geometry_quality="estimated_source",
        select=lambda item: bool(
            (item.get("data_quality") or {}).get("reviewable") is True
        ),
    )
    shadow_all_reviewable = [
        item
        for item in load_review_manifest(SHADOW_SOURCE)["items"]
        if (item.get("data_quality") or {}).get("reviewable") is True
    ]
    shadow, shadow_manifest = _load_cases(
        SHADOW_SOURCE,
        cohort="shadow",
        geometry_quality=None,
        select=lambda item: bool(
            (item.get("data_quality") or {}).get("reviewable") is True
        ),
        exclude_tickers=sealed,
    )
    return {
        "exact": exact,
        "portfolio": portfolio,
        "shadow": shadow,
        "manifests": {
            "exact": exact_manifest,
            "portfolio": portfolio_manifest,
            "shadow": shadow_manifest,
        },
        "counts": {
            "exact_selected": len(exact),
            "portfolio_selected": len(portfolio),
            "shadow_quality_accepted": len(shadow_all_reviewable),
            "shadow_executed_safe": len(shadow),
            "shadow_withheld_blind_overlap": len(shadow_all_reviewable) - len(shadow),
            "blind_snapshots_executed": 0,
        },
    }


def _run_status(exc: Exception) -> str:
    message = str(exc).lower()
    if "insufficient" in message:
        return "insufficient_history"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "invalid_data"
    return "error"


def _safe_current_run(
    bars: list[dict[str, Any]], decision_as_of: str
) -> dict[str, Any]:
    try:
        analysis = analyze_coil(
            bars,
            as_of=decision_as_of,
            review_override=None,
        )
        review = analysis.get("review") or {}
        if review.get("reviewed") is not False or review.get("effective") != "algorithm":
            raise BenchmarkSafetyError("current detector became human-effective")
        if analysis.get("bar_count") != len(bars):
            raise BenchmarkSafetyError("current detector did not use the physical prefix")
        boundary = (analysis.get("pattern_anatomy") or {}).get("boundary") or {}
        if boundary.get("family") == "human_review":
            raise BenchmarkSafetyError("human review geometry leaked into current run")
        return {
            "run_status": "ok",
            "error": None,
            "analysis": analysis,
            "raw_output_sha256": sha256_json(analysis),
        }
    except BenchmarkSafetyError:
        raise
    except Exception as exc:  # benchmark must record abstentions/errors, not stop the batch
        return {
            "run_status": _run_status(exc),
            "error": f"{type(exc).__name__}: {exc}",
            "analysis": None,
            "raw_output_sha256": None,
        }


def _safe_lifetime_run(
    bars: list[dict[str, Any]], decision_as_of: str
) -> dict[str, Any]:
    try:
        analysis = analyze_lifetime_references(bars, as_of=decision_as_of)
        if analysis.get("source") != "timeseries":
            raise BenchmarkSafetyError("lifetime detector provenance is not timeseries")
        return {
            "run_status": "ok",
            "error": None,
            "analysis": analysis,
            "raw_output_sha256": sha256_json(analysis),
        }
    except BenchmarkSafetyError:
        raise
    except Exception as exc:
        return {
            "run_status": _run_status(exc),
            "error": f"{type(exc).__name__}: {exc}",
            "analysis": None,
            "raw_output_sha256": None,
        }


def run_detector_pair(
    bars: Iterable[dict[str, Any]], decision_as_of: str
) -> dict[str, Any]:
    prefix = sanitize_ohlcv_prefix(bars, decision_as_of)
    current = _safe_current_run(prefix, decision_as_of)
    lifetime = _safe_lifetime_run(prefix, decision_as_of)
    return {"bars": prefix, "current": current, "lifetime": lifetime}


def _ols_line(
    points: Iterable[dict[str, Any]],
    *,
    cutoff_date: str,
    line_id: str,
    role: str,
    boundary_kind: str,
    selection: str,
    relationship: str,
    touch_dates: Optional[list[str]] = None,
) -> Optional[dict[str, Any]]:
    normalized: list[tuple[int, str, float]] = []
    for point in points:
        point_date = _iso_date(point.get("date"))
        price = _safe_float(point.get("price"))
        if point_date is None or price is None or price <= 0:
            continue
        normalized.append((quarter_ordinal(point_date), point_date, price))
    by_ordinal: dict[int, tuple[int, str, float]] = {}
    for item in normalized:
        by_ordinal[item[0]] = item
    ordered = [by_ordinal[key] for key in sorted(by_ordinal)]
    if len(ordered) < 2:
        return None
    xs = [float(item[0]) for item in ordered]
    ys = [item[2] for item in ordered]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator <= 0:
        return None
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(xs, ys)
    ) / denominator
    intercept = mean_y - slope * mean_x
    cutoff_ordinal = quarter_ordinal(cutoff_date)
    value_at_cutoff = intercept + slope * cutoff_ordinal
    if value_at_cutoff <= 0:
        return None
    slope_pct = slope * 4.0 / value_at_cutoff * 100.0
    return {
        "id": line_id,
        "role": role,
        "boundary_kind": boundary_kind,
        "selection": selection,
        "relationship": relationship,
        "anchor_dates": [item[1] for item in ordered],
        "anchor_prices": [round(item[2], 6) for item in ordered],
        "anchor_points": [
            {"date": item[1], "price": round(item[2], 6)} for item in ordered
        ],
        "touch_dates": sorted(set(touch_dates or [])),
        "slope_per_quarter": round(slope, 9),
        "intercept": round(intercept, 9),
        "value_at_cutoff": round(value_at_cutoff, 6),
        "slope_pct_per_year": round(slope_pct, 6),
        "direction": _direction(slope_pct),
    }


def _line_value(line: dict[str, Any], point_date: str) -> float:
    return float(line["intercept"]) + float(line["slope_per_quarter"]) * quarter_ordinal(
        point_date
    )


def normalize_current_structures(
    analysis: Optional[dict[str, Any]], cutoff_date: str
) -> list[dict[str, Any]]:
    if not analysis:
        return []
    active = analysis.get("active_lid")
    if not isinstance(active, dict):
        return []
    line = _ols_line(
        active.get("anchors") or [active.get("from"), active.get("to")],
        cutoff_date=cutoff_date,
        line_id="current-active-boundary",
        role="primary_lid",
        boundary_kind="line",
        selection="primary",
        relationship="standalone",
        touch_dates=[
            str(item.get("date"))[:10]
            for item in active.get("touches") or []
            if item.get("date")
        ],
    )
    if line:
        line["geometry_start_ordinal"] = min(
            quarter_ordinal(item["date"])
            for item in line.get("anchor_points") or []
        )
        line["geometry_end_ordinal"] = quarter_ordinal(cutoff_date)
        line["normalization_source"] = "construction_anchor_ols"
    return [line] if line else []


def normalize_lifetime_structures(
    analysis: Optional[dict[str, Any]], cutoff_date: str
) -> list[dict[str, Any]]:
    if not analysis:
        return []
    episodes = {item["id"]: item for item in analysis.get("top_episodes") or []}
    output: list[dict[str, Any]] = []
    for index, structure in enumerate(analysis.get("structures") or []):
        anchors = [
            {
                "date": episodes[episode_id]["date"],
                "price": episodes[episode_id]["price"],
            }
            for episode_id in structure.get("construction_anchor_ids") or []
            if episode_id in episodes
        ]
        touch_dates = [
            str(episodes[episode_id]["date"])[:10]
            for episode_id in structure.get("supporting_touch_ids") or []
            if episode_id in episodes
        ]
        normalized = _ols_line(
            anchors,
            cutoff_date=cutoff_date,
            line_id=str(structure.get("id") or f"lifetime-{index}"),
            role="primary_lid" if index == 0 else "secondary_lid",
            boundary_kind=(
                "resistance_band"
                if structure.get("kind") == "resistance_band"
                else "line"
            ),
            selection="primary" if index == 0 else "alternate",
            relationship=str(
                structure.get("relationship")
                or ("outer_reference" if index == 0 else "nested_below_outer")
            ),
            touch_dates=touch_dates,
        )
        if normalized:
            detector_line = structure.get("line") or {}
            line_from = detector_line.get("from") or {}
            try:
                geometry_start_ordinal = int(
                    line_from.get("time_ordinal")
                    if line_from.get("time_ordinal") is not None
                    else quarter_ordinal(str(line_from["date"])[:10])
                )
            except (KeyError, TypeError, ValueError):
                geometry_start_ordinal = min(
                    quarter_ordinal(item["date"]) for item in anchors
                )

            # A retained resistance band is emitted as a travelling horizontal
            # median level.  Its construction episodes establish and confirm
            # the band; their individual prices are not endpoints of a new
            # sloped line.  Preserve the detector's emitted centreline instead
            # of silently rotating the band with an OLS refit.
            if structure.get("kind") == "resistance_band":
                origin_price = _safe_float(line_from.get("price"))
                emitted_slope = _safe_float(detector_line.get("slope_per_quarter"))
                if origin_price is None or emitted_slope is None:
                    continue
                intercept = origin_price - emitted_slope * geometry_start_ordinal
                value_at_cutoff = (
                    intercept + emitted_slope * quarter_ordinal(cutoff_date)
                )
                if value_at_cutoff <= 0:
                    continue
                slope_pct = emitted_slope * 4.0 / value_at_cutoff * 100.0
                normalized.update(
                    {
                        "slope_per_quarter": round(emitted_slope, 9),
                        "intercept": round(intercept, 9),
                        "value_at_cutoff": round(value_at_cutoff, 6),
                        "slope_pct_per_year": round(slope_pct, 6),
                        "direction": _direction(slope_pct),
                        "normalization_source": "detector_emitted_band_centerline",
                    }
                )
            else:
                normalized["normalization_source"] = "construction_anchor_ols"

            normalized["geometry_start_ordinal"] = geometry_start_ordinal
            normalized["geometry_end_ordinal"] = quarter_ordinal(cutoff_date)
            normalized["band"] = structure.get("band")
            normalized["detector_status"] = structure.get("status")
            normalized["fit_error_pct"] = (structure.get("fit") or {}).get(
                "rms_error_pct"
            )
            output.append(normalized)
    return output


def normalize_reference_structures(
    reference_setup: dict[str, Any], cutoff_date: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    lines = reference_setup.get("lines") or []
    for index, reference in enumerate(lines):
        role = str(reference.get("role") or ("primary_lid" if index == 0 else "other"))
        anchors = reference.get("constructionAnchors") or []
        if len(anchors) < 2:
            anchors = [reference.get("from"), reference.get("to")]
        normalized = _ols_line(
            [item for item in anchors if isinstance(item, dict)],
            cutoff_date=cutoff_date,
            line_id=str(reference.get("id") or f"reference-{index}"),
            role=role,
            boundary_kind=str(
                reference.get("boundary_kind")
                or reference.get("boundaryKind")
                or reference.get("kind")
                or "unspecified"
            ),
            selection="primary" if role == "primary_lid" else "alternate",
            relationship=str(reference.get("relationship") or "unspecified"),
            touch_dates=[],
        )
        if normalized:
            normalized["geometry_start_ordinal"] = min(
                quarter_ordinal(item["date"])
                for item in normalized.get("anchor_points") or []
            )
            normalized["geometry_end_ordinal"] = quarter_ordinal(cutoff_date)
            normalized["confidence"] = reference.get("confidence")
            output.append(normalized)
    return output


def _primary(structures: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    return next(
        (item for item in structures if item.get("selection") == "primary"),
        structures[0] if structures else None,
    )


def match_dates(
    expected_dates: Iterable[str],
    predicted_dates: Iterable[str],
    *,
    tolerance: int = TOP_TOLERANCE_QUARTERS,
) -> dict[str, Any]:
    expected = sorted(set(str(value)[:10] for value in expected_dates))
    predicted = sorted(set(str(value)[:10] for value in predicted_dates))
    expected_ordinals = [quarter_ordinal(value) for value in expected]
    predicted_ordinals = [quarter_ordinal(value) for value in predicted]

    @lru_cache(maxsize=None)
    def solve(
        expected_idx: int, predicted_idx: int
    ) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
        if expected_idx >= len(expected) or predicted_idx >= len(predicted):
            return (0, 0, ())
        options = [
            solve(expected_idx + 1, predicted_idx),
            solve(expected_idx, predicted_idx + 1),
        ]
        distance = abs(
            expected_ordinals[expected_idx] - predicted_ordinals[predicted_idx]
        )
        if distance <= tolerance:
            matched, total_error, suffix = solve(
                expected_idx + 1, predicted_idx + 1
            )
            options.append(
                (
                    matched + 1,
                    total_error + distance,
                    ((expected_idx, predicted_idx, distance),) + suffix,
                )
            )
        return min(options, key=lambda item: (-item[0], item[1], item[2]))

    _, _, solved_pairs = solve(0, 0)
    pairs = list(solved_pairs)
    used_expected = {item[0] for item in pairs}
    used_predicted = {item[1] for item in pairs}
    return {
        "tp": len(pairs),
        "fp": len(predicted) - len(pairs),
        "fn": len(expected) - len(pairs),
        "pairs": [
            {
                "expected": expected[expected_idx],
                "predicted": predicted[predicted_idx],
                "quarter_error": distance,
            }
            for expected_idx, predicted_idx, distance in sorted(pairs)
        ],
        "unmatched_expected": [
            value for index, value in enumerate(expected) if index not in used_expected
        ],
        "unmatched_predicted": [
            value for index, value in enumerate(predicted) if index not in used_predicted
        ],
    }


def compare_lines(
    expected: Optional[dict[str, Any]], predicted: Optional[dict[str, Any]]
) -> dict[str, Any]:
    if expected is None or predicted is None:
        return {
            "direction_match": None,
            "projected_line_error_pct": None,
            "slope_error_pct_per_year": None,
            "reference_anchor_rms_error_pct": None,
            "anchor_matching": None,
            "primary_geometry_match": False,
            "descriptive_rms_within_10pct": False,
        }
    expected_value = float(expected["value_at_cutoff"])
    predicted_value = float(predicted["value_at_cutoff"])
    projected_error = abs(predicted_value - expected_value) / expected_value * 100.0
    slope_error = abs(
        float(predicted["slope_pct_per_year"])
        - float(expected["slope_pct_per_year"])
    )
    errors = [
        (
            (_line_value(predicted, point["date"]) - float(point["price"]))
            / float(point["price"])
            * 100.0
        )
        for point in expected.get("anchor_points") or []
        if float(point["price"]) > 0
    ]
    rms = math.sqrt(sum(value * value for value in errors) / len(errors)) if errors else None
    anchor_matching = match_dates(
        expected.get("anchor_dates") or [],
        predicted.get("anchor_dates") or [],
    )
    required = min(2, len(expected.get("anchor_dates") or []))
    direction_match = expected.get("direction") == predicted.get("direction")
    geometry_match = bool(
        required > 0
        and direction_match
        and projected_error <= PROJECTED_LINE_TOLERANCE_PCT
        and slope_error <= SLOPE_TOLERANCE_PCT_PER_YEAR
        and anchor_matching["tp"] >= required
        and anchor_matching["fp"] == 0
    )
    return {
        "direction_match": direction_match,
        "projected_line_error_pct": round(projected_error, 6),
        "slope_error_pct_per_year": round(slope_error, 6),
        "reference_anchor_rms_error_pct": round(rms, 6) if rms is not None else None,
        "anchor_matching": anchor_matching,
        "primary_geometry_match": geometry_match,
        "descriptive_rms_within_10pct": bool(
            direction_match and rms is not None and rms <= DESCRIPTIVE_RMS_TOLERANCE_PCT
        ),
    }


def compare_structure_sets(
    expected: list[dict[str, Any]], predicted: list[dict[str, Any]]
) -> dict[str, Any]:
    candidates: dict[tuple[int, int], tuple[float, dict[str, Any]]] = {}
    for expected_idx, expected_line in enumerate(expected):
        for predicted_idx, predicted_line in enumerate(predicted):
            if not _structures_semantically_compatible(expected_line, predicted_line):
                continue
            comparison = compare_lines(expected_line, predicted_line)
            rms = comparison["reference_anchor_rms_error_pct"]
            if (
                comparison["direction_match"] is True
                and rms is not None
                and float(rms) <= DESCRIPTIVE_RMS_TOLERANCE_PCT
            ):
                candidates[(expected_idx, predicted_idx)] = (
                    float(rms),
                    comparison,
                )

    @lru_cache(maxsize=None)
    def solve(
        expected_idx: int, used_predicted_mask: int
    ) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if expected_idx >= len(expected):
            return (0, 0.0, ())
        options = [solve(expected_idx + 1, used_predicted_mask)]
        for predicted_idx in range(len(predicted)):
            if used_predicted_mask & (1 << predicted_idx):
                continue
            candidate = candidates.get((expected_idx, predicted_idx))
            if candidate is None:
                continue
            rms, _ = candidate
            count, total_rms, suffix = solve(
                expected_idx + 1,
                used_predicted_mask | (1 << predicted_idx),
            )
            options.append(
                (
                    count + 1,
                    total_rms + rms,
                    ((expected_idx, predicted_idx),) + suffix,
                )
            )
        return min(options, key=lambda item: (-item[0], item[1], item[2]))

    _, _, matched_indices = solve(0, 0)
    pairs: list[dict[str, Any]] = []
    for expected_idx, predicted_idx in matched_indices:
        _, comparison = candidates[(expected_idx, predicted_idx)]
        pairs.append(
            {
                "expected_id": expected[expected_idx]["id"],
                "predicted_id": predicted[predicted_idx]["id"],
                **comparison,
            }
        )
    strict_tp = sum(bool(pair["primary_geometry_match"]) for pair in pairs)
    return {
        "reference_count": len(expected),
        "predicted_count": len(predicted),
        "within_10pct_tp": len(pairs),
        "within_10pct_fp": len(predicted) - len(pairs),
        "within_10pct_fn": len(expected) - len(pairs),
        "strict_5pct_tp": strict_tp,
        "pairs": pairs,
    }


_UNSPECIFIED_STRUCTURE_SEMANTICS = {"", "other", "unknown", "unspecified"}


def _structures_semantically_compatible(
    expected: dict[str, Any], predicted: dict[str, Any]
) -> bool:
    """Return whether two boundaries are eligible for geometric matching.

    Human references in the development corpus predate explicit kind and
    relationship fields, so an absent/unspecified value is intentionally a
    wildcard.  Whenever both sides do state a semantic value, role, boundary
    kind, and relationship must all agree before RMS can pair the structures.
    """

    for field in ("role", "boundary_kind", "relationship"):
        expected_value = str(expected.get(field) or "").strip().lower()
        predicted_value = str(predicted.get(field) or "").strip().lower()
        if (
            expected_value in _UNSPECIFIED_STRUCTURE_SEMANTICS
            or predicted_value in _UNSPECIFIED_STRUCTURE_SEMANTICS
        ):
            continue
        if expected_value != predicted_value:
            return False
    return True


def _current_summary(
    run: dict[str, Any], normalized: list[dict[str, Any]]
) -> dict[str, Any]:
    analysis = run.get("analysis") or {}
    anatomy = analysis.get("pattern_anatomy") or {}
    primary = _primary(normalized)
    return {
        "run_status": run["run_status"],
        "error": run["error"],
        "raw_output_sha256": run["raw_output_sha256"],
        "algorithm_version": analysis.get("algorithm_version"),
        "line_present": primary is not None,
        "structure_count": len(normalized),
        "normalized_structures": normalized,
        "lifecycle": analysis.get("lifecycle"),
        "status": analysis.get("status"),
        "grade": analysis.get("grade"),
        "boundary_family": (analysis.get("active_lid") or {}).get(
            "boundary_family"
        ),
        "major_high_count": len(analysis.get("major_highs") or []),
        "maturity": anatomy.get("maturity"),
        "congestion": anatomy.get("congestion"),
        "compression": anatomy.get("compression"),
        "breakout": anatomy.get("breakout") or analysis.get("breakout"),
        "proximity_pct": (analysis.get("metrics") or {}).get("proximity_pct"),
    }


def _lifetime_summary(
    run: dict[str, Any], normalized: list[dict[str, Any]]
) -> dict[str, Any]:
    analysis = run.get("analysis") or {}
    episodes = analysis.get("top_episodes") or []
    return {
        "run_status": run["run_status"],
        "error": run["error"],
        "raw_output_sha256": run["raw_output_sha256"],
        "algorithm_version": analysis.get("algorithm_version"),
        "line_present": bool(normalized),
        "structure_count": len(normalized),
        "normalized_structures": normalized,
        "confirmed_top_count": sum(
            item.get("status") == "confirmed_rejection" for item in episodes
        ),
        "tracking_high_count": sum(
            item.get("status") == "tracking_only" for item in episodes
        ),
        "demoted_singleton_count": len(
            (analysis.get("diagnostics") or {}).get("demoted_singleton_ids") or []
        ),
        "reference_ladder": analysis.get("reference_ladder") or [],
        "history": analysis.get("history"),
        "semantic_status": "geometry_only",
    }


def compare_flows(
    current_structures: list[dict[str, Any]],
    lifetime_structures: list[dict[str, Any]],
) -> dict[str, Any]:
    current = _primary(current_structures)
    lifetime = _primary(lifetime_structures)
    if current is None and lifetime is None:
        return {"agreement_class": "neither"}
    if current is not None and lifetime is None:
        return {"agreement_class": "current_only"}
    if current is None and lifetime is not None:
        return {"agreement_class": "lifetime_only"}
    assert current is not None and lifetime is not None
    current_value = float(current["value_at_cutoff"])
    lifetime_value = float(lifetime["value_at_cutoff"])
    symmetric_error = (
        200.0
        * abs(current_value - lifetime_value)
        / max(abs(current_value) + abs(lifetime_value), 1e-9)
    )
    slope_difference = abs(
        float(current["slope_pct_per_year"])
        - float(lifetime["slope_pct_per_year"])
    )
    anchors = match_dates(current["anchor_dates"], lifetime["anchor_dates"])
    required = min(2, len(current["anchor_dates"]), len(lifetime["anchor_dates"]))
    direction_match = current["direction"] == lifetime["direction"]
    aligned = bool(
        required > 0
        and direction_match
        and symmetric_error <= PROJECTED_LINE_TOLERANCE_PCT
        and slope_difference <= SLOPE_TOLERANCE_PCT_PER_YEAR
        and anchors["tp"] >= required
    )
    return {
        "agreement_class": "both_aligned" if aligned else "both_different",
        "direction_match": direction_match,
        "projected_value_symmetric_error_pct": round(symmetric_error, 6),
        "slope_abs_diff_pct_per_year": round(slope_difference, 6),
        "anchor_matching": anchors,
    }


def _reference_primary(
    reference_structures: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    return next(
        (item for item in reference_structures if item.get("role") == "primary_lid"),
        reference_structures[0] if reference_structures else None,
    )


def _reference_assessment(
    expected_pattern: str,
    reference_primary: Optional[dict[str, Any]],
    prediction_primary: Optional[dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    if reference_primary is None:
        return "no_line_correct" if prediction_primary is None else "false_line"
    if prediction_primary is None:
        return "missed_line"
    if comparison.get("primary_geometry_match"):
        return "geometry_aligned"
    if expected_pattern == "not_coil":
        return "false_line"
    return "different_geometry"


def _top_coverage(
    reference_setup: dict[str, Any], lifetime_analysis: Optional[dict[str, Any]]
) -> dict[str, Any]:
    expected = [
        str(item["date"])[:10]
        for item in reference_setup.get("expertTops") or []
        if item.get("date")
    ]
    episodes = (lifetime_analysis or {}).get("top_episodes") or []
    confirmed = [
        str(item["date"])[:10]
        for item in episodes
        if item.get("status") == "confirmed_rejection"
    ]
    observed = [str(item["date"])[:10] for item in episodes if item.get("date")]
    return {
        "reference_top_count": len(expected),
        "confirmed_episode_matching": match_dates(expected, confirmed),
        "observed_episode_matching": match_dates(expected, observed),
        "note": "Reference top annotations may be selective; unmatched detector episodes are not precision false positives.",
    }


def _observation_identity(case: BenchmarkCase, bars: list[dict[str, Any]]) -> dict[str, str]:
    bars_hash = sha256_json({"interval": "1M", "bars": bars})
    observation_id = hashlib.sha256(
        f"v1|{case.ticker}|{bars_hash}|{case.decision_as_of}".encode("utf-8")
    ).hexdigest()[:20]
    return {
        "setup_id": f"setup:{case.ticker}",
        "issuer_cluster_id": f"issuer:{case.ticker}",
        "observation_id": observation_id,
        "bars_sha256": bars_hash,
    }


def _input_summary(case: BenchmarkCase, bars: list[dict[str, Any]]) -> dict[str, Any]:
    quarters = _aggregate_quarterly_display_bars(bars)
    prior_review = bool(
        (case.manifest_item.get("prior_review") or {}).get("flag") is True
    )
    tuning_anchor = bool(
        (case.manifest_item.get("tuning_anchor") or {}).get("flag") is True
    )
    source_cohort_role = case.manifest_item.get("cohort_role")
    independent_review_candidate = bool(
        case.cohort == "shadow"
        and source_cohort_role == "prospective_international"
        and not prior_review
        and not tuning_anchor
    )
    history_years = (
        (quarter_ordinal(bars[-1]["date"]) - quarter_ordinal(bars[0]["date"])) / 4.0
        if len(bars) >= 2
        else 0.0
    )
    return {
        "corpus_id": case.corpus_id,
        "source": case.source,
        "snapshot_sha256": case.snapshot["snapshot_sha256"],
        "source_bars_hash": case.snapshot["bars_hash"],
        "cutoff_date": bars[-1]["date"],
        "decision_as_of": case.decision_as_of,
        "bar_count": len(bars),
        "completed_quarter_count": len(
            _completed_quarters(quarters, as_of=case.decision_as_of)
        ),
        "history_start_date": bars[0]["date"],
        "history_years": round(history_years, 2),
        "data_quality_status": (case.snapshot.get("data_quality") or {}).get("status"),
        "reviewable": case.snapshot.get("reviewable") is True,
        "source_cohort_role": source_cohort_role,
        "source_prior_review": prior_review,
        "source_tuning_anchor": tuning_anchor,
        "independent_review_candidate": independent_review_candidate,
    }


def _checkpoint_specs(
    reference_setup: dict[str, Any], full_cutoff: str
) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    recognizable = _iso_date(reference_setup.get("firstRecognizableDate"))
    if recognizable:
        specs.append(
            {
                "role": "pre_recognizable",
                "target_date": _quarter_end_from_ordinal(quarter_ordinal(recognizable) - 1),
            }
        )
    for field, role in (
        ("firstRecognizableDate", "first_recognizable"),
        ("firstWatchDate", "first_watch"),
        ("firstActionableDate", "first_actionable"),
    ):
        parsed = _iso_date(reference_setup.get(field))
        if parsed:
            specs.append({"role": role, "target_date": parsed})
    specs.append({"role": "full_cutoff", "target_date": full_cutoff})

    milestone_roles = {"first_recognizable", "first_watch", "first_actionable"}
    for spec in specs:
        label_date = spec["target_date"]
        data_as_of = min(label_date, full_cutoff)
        # Historical milestone dates are calendar labels, while each stored
        # monthly bar is a finalized candle labelled with the month's first day.
        # At a month-start milestone that candle did not yet exist, so the
        # replay may use data only through the preceding calendar day.
        if (
            spec["role"] in milestone_roles
            and data_as_of == label_date
            and date.fromisoformat(label_date).day == 1
        ):
            data_as_of = (date.fromisoformat(label_date) - timedelta(days=1)).isoformat()
        spec["decision_as_of"] = data_as_of
    return specs


def _trim_reference_for_replay(
    reference_setup: dict[str, Any],
    cutoff_date: str,
    *,
    reference_line_expected: bool = True,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    if not reference_line_expected:
        return [], "reference_line_not_expected_at_checkpoint"
    trimmed = {**reference_setup, "lines": []}
    for raw in reference_setup.get("lines") or []:
        available = [
            item
            for item in raw.get("constructionAnchors") or []
            if item.get("date")
            and quarter_ordinal(str(item["date"])) <= quarter_ordinal(cutoff_date)
        ]
        if len(available) < 2:
            continue
        trimmed["lines"].append({**raw, "constructionAnchors": available})
    normalized = normalize_reference_structures(trimmed, cutoff_date)
    if not normalized:
        return [], "fewer_than_two_reference_anchors_available"
    return normalized, None


def _reference_line_expected(
    reference_setup: dict[str, Any], checkpoint_role: str, target_date: str
) -> bool:
    has_line = any(
        len(line.get("constructionAnchors") or []) >= 2
        or (isinstance(line.get("from"), dict) and isinstance(line.get("to"), dict))
        for line in reference_setup.get("lines") or []
        if isinstance(line, dict)
    )
    if not has_line:
        return False
    recognizable = _iso_date(reference_setup.get("firstRecognizableDate"))
    if recognizable:
        return date.fromisoformat(target_date) >= date.fromisoformat(recognizable)
    return checkpoint_role == "full_cutoff"


def _replay_case(
    case: BenchmarkCase,
    reference_setup: dict[str, Any],
    full_decision_as_of: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    prior_signatures: dict[str, Optional[tuple[str, ...]]] = {
        "current": None,
        "lifetime": None,
    }
    prior_presence: dict[str, Optional[bool]] = {
        "current": None,
        "lifetime": None,
    }
    has_prior = {"current": False, "lifetime": False}
    for spec in _checkpoint_specs(reference_setup, full_decision_as_of):
        data_as_of = spec["decision_as_of"]
        pair = run_detector_pair(case.snapshot["monthly_bars"], data_as_of)
        bars = pair["bars"]
        cutoff = bars[-1]["date"]
        current_structures = normalize_current_structures(
            pair["current"]["analysis"], cutoff
        )
        lifetime_structures = normalize_lifetime_structures(
            pair["lifetime"]["analysis"], cutoff
        )
        reference_line_expected = _reference_line_expected(
            reference_setup, spec["role"], spec["target_date"]
        )
        references, reference_reason = _trim_reference_for_replay(
            reference_setup,
            cutoff,
            reference_line_expected=reference_line_expected,
        )
        reference_primary = _reference_primary(references)
        current_primary = _primary(current_structures)
        lifetime_primary = _primary(lifetime_structures)
        current_comparison = compare_lines(reference_primary, current_primary)
        lifetime_comparison = compare_lines(reference_primary, lifetime_primary)
        record: dict[str, Any] = {
            "ticker": case.ticker,
            "setup_id": f"setup:{case.ticker}",
            "cohort": case.cohort,
            "geometry_quality": case.geometry_quality,
            "checkpoint_role": spec["role"],
            "checkpoint_target_date": spec["target_date"],
            "cutoff_date": cutoff,
            "decision_as_of": data_as_of,
            "bar_count": len(bars),
            "bars_sha256": sha256_json({"interval": "1M", "bars": bars}),
            "reference_line_expected": reference_line_expected,
            "reference_replay_evaluable": reference_primary is not None,
            "reference_replay_reason": reference_reason,
            "reference": reference_primary,
            "current": {
                "run_status": pair["current"]["run_status"],
                "primary": current_primary,
                "structure_count": len(current_structures),
                "comparison": current_comparison,
            },
            "lifetime": {
                "run_status": pair["lifetime"]["run_status"],
                "primary": lifetime_primary,
                "structure_count": len(lifetime_structures),
                "comparison": lifetime_comparison,
            },
        }
        for detector, primary in (
            ("current", current_primary),
            ("lifetime", lifetime_primary),
        ):
            signature = tuple(primary["anchor_dates"]) if primary else None
            previous = prior_signatures[detector]
            present = primary is not None
            record[detector]["presence_assessment"] = (
                "false_early"
                if present and not reference_line_expected
                else "correctly_absent"
                if not present and not reference_line_expected
                else "line_present_when_expected"
                if present
                else "missed_expected_line"
            )
            record[detector]["geometry_scored"] = bool(
                reference_line_expected and reference_primary is not None and present
            )
            record[detector]["presence_changed_from_prior"] = bool(
                has_prior[detector] and present != prior_presence[detector]
            )
            record[detector]["anchor_signature_changed_from_prior"] = bool(
                has_prior[detector]
                and previous is not None
                and signature is not None
                and signature != previous
            )
            prior_signatures[detector] = signature
            prior_presence[detector] = present
            has_prior[detector] = True
        rows.append(record)
        runtime.append(pair)
    return rows, runtime


def _labelled_record(case: BenchmarkCase) -> tuple[dict[str, Any], dict[str, Any]]:
    # Lock both predictions before reading any reference coordinates.
    pair = run_detector_pair(case.snapshot["monthly_bars"], case.decision_as_of)
    bars = pair["bars"]
    identity = _observation_identity(case, bars)
    cutoff = bars[-1]["date"]
    current_structures = normalize_current_structures(pair["current"]["analysis"], cutoff)
    lifetime_structures = normalize_lifetime_structures(
        pair["lifetime"]["analysis"], cutoff
    )
    current_summary = _current_summary(pair["current"], current_structures)
    lifetime_summary = _lifetime_summary(pair["lifetime"], lifetime_structures)

    reference_setup = (case.snapshot.get("corpus_labels") or {}).get(
        "reference_setup"
    ) or {}
    reference_structures = normalize_reference_structures(reference_setup, cutoff)
    reference_primary = _reference_primary(reference_structures)
    current_primary = _primary(current_structures)
    lifetime_primary = _primary(lifetime_structures)
    expected_pattern = str(reference_setup.get("expectedPatternLabel") or "coil")
    current_comparison = compare_lines(reference_primary, current_primary)
    lifetime_comparison = compare_lines(reference_primary, lifetime_primary)
    replay, replay_runtime = _replay_case(
        case, reference_setup, case.decision_as_of
    )

    record = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "kind": "coilingview.lifetime-reference-labelled-observation",
        "identity": {**identity, "ticker": case.ticker},
        "input": _input_summary(case, bars),
        "reference": {
            "availability": "outcome_revealed_development_reference",
            "geometry_quality": case.geometry_quality,
            "known_outcome": reference_setup.get("knownOutcome") is True,
            "expected_pattern_label": expected_pattern,
            "reference_id": reference_setup.get("referenceId"),
            "reference_setup_sha256": case.snapshot.get("reference_setup_sha256"),
            "normalized_structures": reference_structures,
            "milestones": {
                "first_recognizable_date": reference_setup.get("firstRecognizableDate"),
                "first_watch_date": reference_setup.get("firstWatchDate"),
                "first_actionable_date": reference_setup.get("firstActionableDate"),
            },
        },
        "predictions": {
            "current": current_summary,
            "lifetime": lifetime_summary,
            "hybrid": {
                "run_status": "not_available",
                "reason": "no detector-only external-boundary downstream adapter exists",
            },
            "gold_boundary_downstream": {
                "run_status": "not_available",
                "reason": "gold geometry is comparison-only and is never injected as detector output",
            },
        },
        "comparisons": {
            "current_vs_lifetime": compare_flows(
                current_structures, lifetime_structures
            ),
            "current_vs_reference": {
                **current_comparison,
                "assessment": _reference_assessment(
                    expected_pattern,
                    reference_primary,
                    current_primary,
                    current_comparison,
                ),
                "structure_set": compare_structure_sets(
                    reference_structures, current_structures
                ),
            },
            "lifetime_vs_reference": {
                **lifetime_comparison,
                "assessment": _reference_assessment(
                    expected_pattern,
                    reference_primary,
                    lifetime_primary,
                    lifetime_comparison,
                ),
                "structure_set": compare_structure_sets(
                    reference_structures, lifetime_structures
                ),
                "top_coverage": _top_coverage(
                    reference_setup, pair["lifetime"]["analysis"]
                ),
            },
        },
        "replay": replay,
        "artifacts": {"chart_path": None, "setup_json_path": None},
    }
    runtime = {
        "case": case,
        "pair": pair,
        "replay_pairs": replay_runtime,
        "reference_setup": reference_setup,
    }
    return record, runtime


def _shadow_record(case: BenchmarkCase) -> tuple[dict[str, Any], dict[str, Any]]:
    pair = run_detector_pair(case.snapshot["monthly_bars"], case.decision_as_of)
    bars = pair["bars"]
    identity = _observation_identity(case, bars)
    cutoff = bars[-1]["date"]
    current_structures = normalize_current_structures(pair["current"]["analysis"], cutoff)
    lifetime_structures = normalize_lifetime_structures(
        pair["lifetime"]["analysis"], cutoff
    )
    record = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "kind": "coilingview.lifetime-reference-shadow-observation",
        "identity": {**identity, "ticker": case.ticker},
        "input": _input_summary(case, bars),
        "reference": {
            "availability": "no_exact_geometry_joined",
            "outcome_visible": False,
            "note": (
                "No exact line geometry is joined. Source-screen review and "
                "tuning context is retained explicitly; shadow agreement is "
                "not accuracy."
            ),
        },
        "predictions": {
            "current": _current_summary(pair["current"], current_structures),
            "lifetime": _lifetime_summary(pair["lifetime"], lifetime_structures),
            "hybrid": {
                "run_status": "not_available",
                "reason": "no detector-only external-boundary downstream adapter exists",
            },
        },
        "comparisons": {
            "current_vs_lifetime": compare_flows(
                current_structures, lifetime_structures
            )
        },
        "replay": [],
        "artifacts": {"chart_path": None, "setup_json_path": None},
    }
    return record, {"case": case, "pair": pair, "reference_setup": None}


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6) if denominator else None,
    }


def _error_summary(values: Iterable[Optional[float]]) -> dict[str, Any]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    p90_index = min(len(clean) - 1, math.ceil(len(clean) * 0.90) - 1)
    return {
        "count": len(clean),
        "mean": round(statistics.mean(clean), 6),
        "median": round(statistics.median(clean), 6),
        "p90": round(clean[p90_index], 6),
        "max": round(clean[-1], 6),
    }


def _labelled_summary(
    records: list[dict[str, Any]], geometry_quality: str
) -> dict[str, Any]:
    selected = [
        item
        for item in records
        if item["reference"]["geometry_quality"] == geometry_quality
    ]
    positives = [
        item
        for item in selected
        if bool(item["reference"]["normalized_structures"])
    ]
    negatives = [item for item in selected if item not in positives]
    output: dict[str, Any] = {
        "setup_count": len(selected),
        "positive_line_reference_count": len(positives),
        "no_line_reference_count": len(negatives),
    }
    for detector in ("current", "lifetime"):
        comparison_key = f"{detector}_vs_reference"
        line_found = sum(
            item["predictions"][detector]["line_present"] for item in positives
        )
        false_lines = sum(
            item["predictions"][detector]["line_present"] for item in negatives
        )
        comparable = [
            item["comparisons"][comparison_key]
            for item in positives
            if item["predictions"][detector]["line_present"]
        ]
        strict = sum(item["primary_geometry_match"] for item in comparable)
        descriptive = sum(item["descriptive_rms_within_10pct"] for item in comparable)
        direction = sum(item["direction_match"] is True for item in comparable)
        output[detector] = {
            "positive_line_detection": _rate(line_found, len(positives)),
            "false_line_on_no_line_reference": _rate(false_lines, len(negatives)),
            "direction_match_when_emitted": _rate(direction, len(comparable)),
            "strict_primary_geometry_match": _rate(strict, len(positives)),
            "descriptive_primary_rms_within_10pct": _rate(
                descriptive, len(positives)
            ),
            "reference_anchor_rms_error_pct": _error_summary(
                item["reference_anchor_rms_error_pct"] for item in comparable
            ),
            "projected_line_error_pct": _error_summary(
                item["projected_line_error_pct"] for item in comparable
            ),
        }
    if geometry_quality == "exact_clicks":
        coverage = [
            item["comparisons"]["lifetime_vs_reference"]["top_coverage"]
            for item in selected
        ]
        expected_tops = sum(item["reference_top_count"] for item in coverage)
        confirmed_matches = sum(
            item["confirmed_episode_matching"]["tp"] for item in coverage
        )
        observed_matches = sum(
            item["observed_episode_matching"]["tp"] for item in coverage
        )
        output["lifetime_top_coverage"] = {
            "confirmed_episodes": _rate(confirmed_matches, expected_tops),
            "including_tracking_observations": _rate(observed_matches, expected_tops),
            "precision_not_reported": True,
        }
    return output


def _replay_detector_summary(
    replay_rows: list[dict[str, Any]], detector: str
) -> dict[str, Any]:
    not_expected = [row for row in replay_rows if not row["reference_line_expected"]]
    expected = [row for row in replay_rows if row["reference_line_expected"]]
    first_recognizable = [
        row
        for row in expected
        if row["checkpoint_role"] == "first_recognizable"
    ]
    geometry_scored = [row for row in expected if row[detector]["geometry_scored"]]

    by_setup: dict[str, list[dict[str, Any]]] = {}
    for row in expected:
        by_setup.setdefault(row["setup_id"], []).append(row)
    lags: list[float] = []
    missed_setups = 0
    for setup_rows in by_setup.values():
        ordered = sorted(
            setup_rows,
            key=lambda row: (
                quarter_ordinal(row["checkpoint_target_date"]),
                row["checkpoint_role"],
            ),
        )
        baseline = quarter_ordinal(ordered[0]["checkpoint_target_date"])
        first_detected = next(
            (row for row in ordered if row[detector]["primary"] is not None),
            None,
        )
        if first_detected is None:
            missed_setups += 1
        else:
            lags.append(
                float(
                    max(
                        0,
                        quarter_ordinal(first_detected["checkpoint_target_date"])
                        - baseline,
                    )
                )
            )

    return {
        "false_early_line": _rate(
            sum(row[detector]["primary"] is not None for row in not_expected),
            len(not_expected),
        ),
        "line_present_when_expected": _rate(
            sum(row[detector]["primary"] is not None for row in expected),
            len(expected),
        ),
        "line_present_at_first_recognizable": _rate(
            sum(row[detector]["primary"] is not None for row in first_recognizable),
            len(first_recognizable),
        ),
        "strict_geometry_when_scored": _rate(
            sum(
                row[detector]["comparison"]["primary_geometry_match"]
                for row in geometry_scored
            ),
            len(geometry_scored),
        ),
        "presence_transition_count": sum(
            row[detector]["presence_changed_from_prior"] for row in replay_rows
        ),
        "anchor_churn_while_present_count": sum(
            row[detector]["anchor_signature_changed_from_prior"]
            for row in replay_rows
        ),
        "recognition_lag_quarters": _error_summary(lags),
        "recognition_missed_setup_count": missed_setups,
    }


def summarize_benchmark(
    labelled: list[dict[str, Any]],
    shadow: list[dict[str, Any]],
    selection_counts: dict[str, int],
) -> dict[str, Any]:
    shadow_classes = Counter(
        item["comparisons"]["current_vs_lifetime"]["agreement_class"]
        for item in shadow
    )
    shadow_roles = Counter(item["input"]["source_cohort_role"] for item in shadow)
    independent_shadow = [
        item for item in shadow if item["input"]["independent_review_candidate"]
    ]
    error_taxonomy = Counter()
    for item in labelled:
        comparison = item["comparisons"]["lifetime_vs_reference"]
        assessment = str(comparison["assessment"])
        if assessment not in {"geometry_aligned", "no_line_correct"}:
            error_taxonomy[assessment] += 1
        if (
            assessment == "different_geometry"
            and comparison.get("direction_match") is False
        ):
            error_taxonomy["direction_mismatch"] += 1
        if (
            comparison.get("reference_anchor_rms_error_pct") is not None
            and comparison["reference_anchor_rms_error_pct"]
            > DESCRIPTIVE_RMS_TOLERANCE_PCT
        ):
            error_taxonomy["reference_anchor_rms_over_10pct"] += 1
    replay_rows = [row for item in labelled for row in item["replay"]]
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "kind": "coilingview.lifetime-reference-benchmark-summary",
        "trust": {
            "authoritative": False,
            "outcome_revealed_development_references": True,
            "shadow_is_accuracy": False,
            "blind_queue_executed": False,
        },
        "counts": {
            **selection_counts,
            "labelled_executed": len(labelled),
            "shadow_executed": len(shadow),
            "replay_rows": len(replay_rows),
        },
        "shadow": {
            "agreement_class_counts": dict(sorted(shadow_classes.items())),
            "disagreement_count": sum(
                count
                for key, count in shadow_classes.items()
                if key in {"current_only", "lifetime_only", "both_different"}
            ),
            "source_context": {
                "exact_geometry_joined": False,
                "cohort_role_counts": dict(sorted(shadow_roles.items())),
                "prior_review_count": sum(
                    item["input"]["source_prior_review"] for item in shadow
                ),
                "tuning_anchor_count": sum(
                    item["input"]["source_tuning_anchor"] for item in shadow
                ),
                "independent_review_candidate_count": len(independent_shadow),
                "independent_review_disagreement_count": sum(
                    item["comparisons"]["current_vs_lifetime"]["agreement_class"]
                    in {"current_only", "lifetime_only", "both_different"}
                    for item in independent_shadow
                ),
                "independent_review_definition": (
                    "prospective_international and not prior_review and not tuning_anchor"
                ),
            },
        },
        "labelled": {
            "exact_clicks": _labelled_summary(labelled, "exact_clicks"),
            "estimated_source": _labelled_summary(labelled, "estimated_source"),
        },
        "replay": {
            "row_count": len(replay_rows),
            "reference_line_expected_rows": sum(
                row["reference_line_expected"] for row in replay_rows
            ),
            "reference_geometry_evaluable_rows": sum(
                row["reference_replay_evaluable"] for row in replay_rows
            ),
            "current": _replay_detector_summary(replay_rows, "current"),
            "lifetime": _replay_detector_summary(replay_rows, "lifetime"),
        },
        "error_taxonomy": dict(sorted(error_taxonomy.items())),
        "flow_availability": {
            "current": "executed",
            "lifetime_geometry": "executed",
            "hybrid_lifetime_plus_existing_semantics": "not_available_no_supported_adapter",
            "gold_boundary_plus_existing_semantics": "not_run_comparison_only",
        },
        "limitations": [
            "The 33 labelled cases are outcome-revealed development examples, not a blind population estimate.",
            "The 43 shadow cases have no exact geometry joined; some carry prior-review or tuning context, and agreement/disagreement is not accuracy.",
            "Reference top lists may be selective, so detector top precision is not reported.",
            "The lifetime detector is geometry-only; lifecycle and actionability require a supported algorithmic boundary adapter.",
            "The audited 24-chart blind queue and all duplicate ticker identities were withheld.",
        ],
    }


def execute_benchmark() -> dict[str, Any]:
    loaded = load_benchmark_cases()
    labelled_records: list[dict[str, Any]] = []
    shadow_records: list[dict[str, Any]] = []
    runtime: dict[str, dict[str, Any]] = {}
    for case in [*loaded["exact"], *loaded["portfolio"]]:
        record, record_runtime = _labelled_record(case)
        labelled_records.append(record)
        runtime[record["identity"]["observation_id"]] = record_runtime
    for case in loaded["shadow"]:
        record, record_runtime = _shadow_record(case)
        shadow_records.append(record)
        runtime[record["identity"]["observation_id"]] = record_runtime
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "kind": BENCHMARK_KIND,
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection": loaded["counts"],
        "labelled": labelled_records,
        "shadow": shadow_records,
        "summary": summarize_benchmark(
            labelled_records, shadow_records, loaded["counts"]
        ),
        "manifests": loaded["manifests"],
        "_runtime": runtime,
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return canonical_json(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _flatten_labelled(record: dict[str, Any]) -> dict[str, Any]:
    reference = record["reference"]
    reference_primary = _reference_primary(reference["normalized_structures"])
    row: dict[str, Any] = {
        "observation_id": record["identity"]["observation_id"],
        "setup_id": record["identity"]["setup_id"],
        "ticker": record["identity"]["ticker"],
        "source_corpus": record["input"]["corpus_id"],
        "cutoff_date": record["input"]["cutoff_date"],
        "decision_as_of": record["input"]["decision_as_of"],
        "bars_sha256": record["identity"]["bars_sha256"],
        "reference_id": reference["reference_id"],
        "reference_geometry_quality": reference["geometry_quality"],
        "reference_expected_pattern": reference["expected_pattern_label"],
        "reference_line_present": reference_primary is not None,
        "reference_primary_anchor_dates": (
            reference_primary["anchor_dates"] if reference_primary else []
        ),
        "reference_primary_direction": (
            reference_primary["direction"] if reference_primary else None
        ),
        "reference_structure_count": len(reference["normalized_structures"]),
    }
    for detector in ("current", "lifetime"):
        prediction = record["predictions"][detector]
        primary = _primary(prediction["normalized_structures"])
        comparison = record["comparisons"][f"{detector}_vs_reference"]
        row.update(
            {
                f"{detector}_run_status": prediction["run_status"],
                f"{detector}_line_present": prediction["line_present"],
                f"{detector}_primary_anchor_dates": primary["anchor_dates"] if primary else [],
                f"{detector}_primary_direction": primary["direction"] if primary else None,
                f"{detector}_primary_value_at_cutoff": primary["value_at_cutoff"] if primary else None,
                f"{detector}_primary_slope_pct_per_year": primary["slope_pct_per_year"] if primary else None,
                f"{detector}_structure_count": prediction["structure_count"],
                f"{detector}_assessment": comparison["assessment"],
                f"{detector}_direction_match": comparison["direction_match"],
                f"{detector}_projected_line_error_pct": comparison["projected_line_error_pct"],
                f"{detector}_slope_error_pct_per_year": comparison["slope_error_pct_per_year"],
                f"{detector}_reference_anchor_rms_error_pct": comparison["reference_anchor_rms_error_pct"],
                f"{detector}_strict_primary_geometry_match": comparison["primary_geometry_match"],
                f"{detector}_descriptive_rms_within_10pct": comparison["descriptive_rms_within_10pct"],
                f"{detector}_structure_set": comparison["structure_set"],
            }
        )
    row["setup_json_path"] = record["artifacts"]["setup_json_path"]
    row["chart_path"] = record["artifacts"]["chart_path"]
    return row


def _flatten_shadow(record: dict[str, Any]) -> dict[str, Any]:
    flow = record["comparisons"]["current_vs_lifetime"]
    row: dict[str, Any] = {
        "observation_id": record["identity"]["observation_id"],
        "setup_id": record["identity"]["setup_id"],
        "ticker": record["identity"]["ticker"],
        "source_corpus": record["input"]["corpus_id"],
        "cutoff_date": record["input"]["cutoff_date"],
        "decision_as_of": record["input"]["decision_as_of"],
        "bar_count": record["input"]["bar_count"],
        "history_years": record["input"]["history_years"],
        "data_quality_status": record["input"]["data_quality_status"],
        "label_availability": "no_exact_geometry_joined",
        "source_cohort_role": record["input"]["source_cohort_role"],
        "source_prior_review": record["input"]["source_prior_review"],
        "source_tuning_anchor": record["input"]["source_tuning_anchor"],
        "independent_review_candidate": record["input"][
            "independent_review_candidate"
        ],
    }
    for detector in ("current", "lifetime"):
        prediction = record["predictions"][detector]
        primary = _primary(prediction["normalized_structures"])
        row.update(
            {
                f"{detector}_run_status": prediction["run_status"],
                f"{detector}_line_present": prediction["line_present"],
                f"{detector}_primary_anchor_dates": primary["anchor_dates"] if primary else [],
                f"{detector}_primary_direction": primary["direction"] if primary else None,
                f"{detector}_primary_value_at_cutoff": primary["value_at_cutoff"] if primary else None,
                f"{detector}_primary_slope_pct_per_year": primary["slope_pct_per_year"] if primary else None,
                f"{detector}_structure_count": prediction["structure_count"],
            }
        )
    row.update(
        {
            "current_lifecycle": record["predictions"]["current"]["lifecycle"],
            "current_status": record["predictions"]["current"]["status"],
            "current_grade": record["predictions"]["current"]["grade"],
            "lifetime_demoted_singleton_count": record["predictions"]["lifetime"]["demoted_singleton_count"],
            "lifetime_confirmed_top_count": record["predictions"]["lifetime"]["confirmed_top_count"],
            "flow_agreement_class": flow["agreement_class"],
            "flow_direction_match": flow.get("direction_match"),
            "flow_projected_value_symmetric_error_pct": flow.get("projected_value_symmetric_error_pct"),
            "flow_slope_abs_diff_pct_per_year": flow.get("slope_abs_diff_pct_per_year"),
            "setup_json_path": record["artifacts"]["setup_json_path"],
            "chart_path": record["artifacts"]["chart_path"],
        }
    )
    return row


def _flatten_replay(record: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "observation_id": record["identity"]["observation_id"],
        "setup_id": record["identity"]["setup_id"],
        "ticker": record["identity"]["ticker"],
        "source_corpus": record["input"]["corpus_id"],
        "reference_geometry_quality": record["reference"]["geometry_quality"],
        "checkpoint_role": replay["checkpoint_role"],
        "checkpoint_target_date": replay["checkpoint_target_date"],
        "cutoff_date": replay["cutoff_date"],
        "decision_as_of": replay["decision_as_of"],
        "bar_count": replay["bar_count"],
        "reference_line_expected": replay["reference_line_expected"],
        "reference_replay_evaluable": replay["reference_replay_evaluable"],
        "reference_replay_reason": replay["reference_replay_reason"],
    }
    for detector in ("current", "lifetime"):
        prediction = replay[detector]
        primary = prediction["primary"]
        comparison = prediction["comparison"]
        row.update(
            {
                f"{detector}_run_status": prediction["run_status"],
                f"{detector}_line_present": primary is not None,
                f"{detector}_primary_anchor_dates": primary["anchor_dates"] if primary else [],
                f"{detector}_primary_direction": primary["direction"] if primary else None,
                f"{detector}_structure_count": prediction["structure_count"],
                f"{detector}_presence_assessment": prediction[
                    "presence_assessment"
                ],
                f"{detector}_geometry_scored": prediction["geometry_scored"],
                f"{detector}_primary_geometry_match": comparison["primary_geometry_match"],
                f"{detector}_direction_match": comparison["direction_match"],
                f"{detector}_reference_anchor_rms_error_pct": comparison["reference_anchor_rms_error_pct"],
                f"{detector}_anchor_signature_changed_from_prior": prediction["anchor_signature_changed_from_prior"],
                f"{detector}_presence_changed_from_prior": prediction[
                    "presence_changed_from_prior"
                ],
            }
        )
    return row


def _git_value(args: list[str], project_root: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _file_inventory(output_dir: Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    files = [
        {"path": relative, "sha256": sha256_file(output_dir / relative)}
        for relative in sorted(set(relative_paths))
    ]
    return {
        "file_count": len(files),
        "tree_sha256": sha256_json(files),
        "files": files,
    }


def _report_markdown(summary: dict[str, Any]) -> str:
    exact = summary["labelled"]["exact_clicks"]
    estimated = summary["labelled"]["estimated_source"]
    shadow = summary["shadow"]["agreement_class_counts"]
    shadow_context = summary["shadow"]["source_context"]
    replay = summary["replay"]

    def pct(metric: dict[str, Any]) -> str:
        value = metric.get("value")
        return "n/a" if value is None else f"{value * 100:.1f}%"

    lines = [
        "# Lifetime reference benchmark — v1",
        "",
        "This is an outcome-revealed development benchmark, not a blind accuracy estimate and not an investment signal. Both detectors receive only frozen OHLCV prefixes. Human reference geometry is introduced afterward for comparison.",
        "",
        "## What ran",
        "",
        f"- 24 exact reviewed examples and 9 source-estimated portfolio examples ({summary['counts']['labelled_executed']} labelled development charts).",
        f"- {summary['counts']['shadow_executed']} July shadow charts with no exact geometry joined.",
        f"- {summary['counts']['shadow_withheld_blind_overlap']} otherwise-reviewable July charts withheld because their ticker identities overlap the sealed 24-chart blind queue.",
        "- Production `analyze_coil()` and experimental lifetime geometry ran independently; production code was not changed.",
        "- Hybrid lifetime-boundary lifecycle scoring did not run because there is no supported detector-only boundary-injection interface yet.",
        "",
        "## Exact clicked geometry (23 line-positive, 1 explicit no-line)",
        "",
        "| Measure | Current detector | Lifetime geometry |",
        "|---|---:|---:|",
        f"| Line found on positive examples | {pct(exact['current']['positive_line_detection'])} | {pct(exact['lifetime']['positive_line_detection'])} |",
        f"| Direction match when a line was emitted | {pct(exact['current']['direction_match_when_emitted'])} | {pct(exact['lifetime']['direction_match_when_emitted'])} |",
        f"| Primary line within 10% RMS at clicked anchors | {pct(exact['current']['descriptive_primary_rms_within_10pct'])} | {pct(exact['lifetime']['descriptive_primary_rms_within_10pct'])} |",
        f"| Strict 5%/slope/anchor geometry match | {pct(exact['current']['strict_primary_geometry_match'])} | {pct(exact['lifetime']['strict_primary_geometry_match'])} |",
        f"| Line drawn on the explicit no-line example | {pct(exact['current']['false_line_on_no_line_reference'])} | {pct(exact['lifetime']['false_line_on_no_line_reference'])} |",
        "",
        "The 10% RMS line is a descriptive review aid. The stricter row additionally requires projected-value, slope, direction, and clicked-anchor timing agreement. Neither is population accuracy because these examples are outcome-visible development material.",
        "",
        "## Source-estimated portfolio geometry",
        "",
        "| Measure | Current detector | Lifetime geometry |",
        "|---|---:|---:|",
        f"| Line found | {pct(estimated['current']['positive_line_detection'])} | {pct(estimated['lifetime']['positive_line_detection'])} |",
        f"| Primary line within 10% RMS | {pct(estimated['current']['descriptive_primary_rms_within_10pct'])} | {pct(estimated['lifetime']['descriptive_primary_rms_within_10pct'])} |",
        "",
        "## Shadow comparison (no exact geometry joined)",
        "",
        "| Relationship | Charts |",
        "|---|---:|",
        f"| Both aligned | {shadow.get('both_aligned', 0)} |",
        f"| Both drew different geometry | {shadow.get('both_different', 0)} |",
        f"| Current only | {shadow.get('current_only', 0)} |",
        f"| Lifetime only | {shadow.get('lifetime_only', 0)} |",
        f"| Neither | {shadow.get('neither', 0)} |",
        "",
        "These are disagreement buckets, not wins or losses. Source-screen context is not hidden: "
        f"{shadow_context['prior_review_count']} cases had prior review context and "
        f"{shadow_context['tuning_anchor_count']} were tuning anchors. "
        f"Only {shadow_context['independent_review_candidate_count']} cases are prospective, "
        "not previously reviewed, and not tuning anchors; "
        f"{shadow_context['independent_review_disagreement_count']} of those are detector disagreements and form the independent review queue.",
        "",
        "## Point-in-time recognition replay",
        "",
        "Geometry is expected only from Amrut's first-recognizable milestone onward. Earlier detector lines are counted as false-early presence; geometry is scored only where the contemporaneously available reference has at least two anchors.",
        "",
        "| Measure | Current detector | Lifetime geometry |",
        "|---|---:|---:|",
        f"| False-early line before recognition | {pct(replay['current']['false_early_line'])} | {pct(replay['lifetime']['false_early_line'])} |",
        f"| Line present at first-recognizable checkpoint | {pct(replay['current']['line_present_at_first_recognizable'])} | {pct(replay['lifetime']['line_present_at_first_recognizable'])} |",
        f"| Presence transitions | {replay['current']['presence_transition_count']} | {replay['lifetime']['presence_transition_count']} |",
        f"| Anchor churn while continuously present | {replay['current']['anchor_churn_while_present_count']} | {replay['lifetime']['anchor_churn_while_present_count']} |",
        "",
        "`False-early` means the geometry existed before the whole setup's reference recognition milestone; it does not by itself make the resistance line invalid. Recognition lag is recorded separately in `summary.json`; replay checkpoints are sparse milestones, not a quarter-by-quarter latency study.",
        "",
        "## Files",
        "",
        "- [`labelled_comparison.csv`](labelled_comparison.csv): one row per outcome-revealed development chart.",
        "- [`shadow_comparison.csv`](shadow_comparison.csv): one row per safe shadow chart, including prior-review, tuning-anchor, and cohort-role context.",
        "- [`replay.csv`](replay.csv): point-in-time runs at available Amrut milestones, with future reference anchors withheld.",
        "- [`summary.json`](summary.json): aggregate counts and error taxonomy.",
        "- [`manifest.json`](manifest.json): source hashes, detector versions, protocol, and blind exclusions.",
        "- [`chart_index.md`](chart_index.md): ticker-to-chart index for every labelled failure and safe shadow disagreement.",
        "- `setups/`: compact per-observation JSON records; raw price bars are not duplicated.",
        "",
        "## Interpretation boundary",
        "",
        "A mathematically credible resistance line is not automatically a coil. ADM and short-history cases show why congestion, maturity, compression, and lifecycle must remain separate downstream decisions. The next engineering step is a supported algorithmic-boundary adapter with parity tests against the unchanged current detector.",
        "",
    ]
    return "\n".join(lines)


def _chart_index_markdown(
    labelled: list[dict[str, Any]], shadow: list[dict[str, Any]]
) -> str:
    lines = [
        "# Benchmark chart index",
        "",
        "Every chart below is a development or no-exact-geometry shadow comparison. Purple reference geometry, where present, is added only after both detector runs are locked.",
        "",
        "## Labelled failures and strict disagreements",
        "",
        "| Ticker | Reference quality | Lifetime assessment | Anchor RMS | Chart |",
        "|---|---|---|---:|---|",
    ]
    for record in labelled:
        chart_path = record["artifacts"]["chart_path"]
        if not chart_path:
            continue
        comparison = record["comparisons"]["lifetime_vs_reference"]
        rms = comparison.get("reference_anchor_rms_error_pct")
        rms_text = f"{rms:.2f}%" if rms is not None else "n/a"
        lines.append(
            f"| {record['identity']['ticker']} | {record['reference']['geometry_quality']} | "
            f"{comparison['assessment']} | {rms_text} | [open chart]({chart_path}) |"
        )
    lines.extend(
        [
            "",
            "A `different_geometry` row can still be visually close: the strict assessment also checks projected value, annualized slope, and construction-anchor timing. Use `labelled_comparison.csv` for the separate 10% RMS flag.",
            "",
            "## Independent shadow review queue",
            "",
            "Prospective cases with no prior-review or tuning-anchor context are isolated here.",
            "",
            "| Ticker | Flow relationship | Chart |",
            "|---|---|---|",
        ]
    )
    independent = [
        record
        for record in shadow
        if record["input"]["independent_review_candidate"]
    ]
    contextual = [
        record
        for record in shadow
        if not record["input"]["independent_review_candidate"]
    ]
    for record in independent:
        chart_path = record["artifacts"]["chart_path"]
        if not chart_path:
            continue
        flow_class = record["comparisons"]["current_vs_lifetime"][
            "agreement_class"
        ]
        lines.append(
            f"| {record['identity']['ticker']} | {flow_class} | [open chart]({chart_path}) |"
        )
    lines.extend(
        [
            "",
            "## Contextual shadow disagreements",
            "",
            "These rows carried prior-review, tuning-anchor, or control-reference context in the source screen and are kept separate from independent review.",
            "",
            "| Ticker | Source context | Flow relationship | Chart |",
            "|---|---|---|---|",
        ]
    )
    for record in contextual:
        chart_path = record["artifacts"]["chart_path"]
        if not chart_path:
            continue
        flow_class = record["comparisons"]["current_vs_lifetime"][
            "agreement_class"
        ]
        context = []
        if record["input"]["source_prior_review"]:
            context.append("prior review")
        if record["input"]["source_tuning_anchor"]:
            context.append("tuning anchor")
        if record["input"]["source_cohort_role"] == "control_reference":
            context.append("control reference")
        lines.append(
            f"| {record['identity']['ticker']} | {', '.join(context) or 'source context'} | "
            f"{flow_class} | [open chart]({chart_path}) |"
        )
    lines.extend(
        [
            "",
            "The sealed 24-chart blind queue and duplicate ticker identities are absent from this index.",
            "",
        ]
    )
    return "\n".join(lines)


def write_benchmark_artifacts(execution: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from lifetime_benchmark_charts import render_benchmark_chart

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = execution["_runtime"]
    labelled = execution["labelled"]
    shadow = execution["shadow"]

    for record in labelled:
        observation_id = record["identity"]["observation_id"]
        setup_path = Path("setups") / f"{observation_id}.json"
        record["artifacts"]["setup_json_path"] = setup_path.as_posix()
        lifetime_comparison = record["comparisons"]["lifetime_vs_reference"]
        should_render = lifetime_comparison["assessment"] not in {
            "geometry_aligned",
            "no_line_correct",
        }
        if should_render:
            chart_path = Path("charts") / "labelled-failures" / f"{observation_id}.png"
            record["artifacts"]["chart_path"] = chart_path.as_posix()
            payload = runtime[observation_id]
            render_benchmark_chart(
                record["identity"]["ticker"],
                payload["pair"]["bars"],
                payload["pair"]["current"]["analysis"] or {},
                payload["pair"]["lifetime"]["analysis"] or {},
                output_dir / chart_path,
                reference_setup=payload["reference_setup"],
                subtitle=(
                    f"{record['reference']['geometry_quality']} | "
                    f"lifetime: {lifetime_comparison['assessment']}"
                ),
            )

    for record in shadow:
        observation_id = record["identity"]["observation_id"]
        setup_path = Path("setups") / f"{observation_id}.json"
        record["artifacts"]["setup_json_path"] = setup_path.as_posix()
        flow_class = record["comparisons"]["current_vs_lifetime"]["agreement_class"]
        if flow_class in {"current_only", "lifetime_only", "both_different"}:
            chart_path = Path("charts") / "shadow-disagreements" / f"{observation_id}.png"
            record["artifacts"]["chart_path"] = chart_path.as_posix()
            payload = runtime[observation_id]
            render_benchmark_chart(
                record["identity"]["ticker"],
                payload["pair"]["bars"],
                payload["pair"]["current"]["analysis"] or {},
                payload["pair"]["lifetime"]["analysis"] or {},
                output_dir / chart_path,
                subtitle=f"no exact geometry joined | {flow_class}",
            )

    for record in [*labelled, *shadow]:
        setup_relative = Path(record["artifacts"]["setup_json_path"])
        _write_json(output_dir / setup_relative, record)

    expected_setup_paths = {
        str(record["artifacts"]["setup_json_path"])
        for record in [*labelled, *shadow]
    }
    expected_chart_paths = {
        str(record["artifacts"]["chart_path"])
        for record in [*labelled, *shadow]
        if record["artifacts"]["chart_path"]
    }
    actual_setup_paths = {
        path.relative_to(output_dir).as_posix()
        for path in (output_dir / "setups").glob("*.json")
    }
    actual_chart_paths = {
        path.relative_to(output_dir).as_posix()
        for path in (output_dir / "charts").glob("**/*.png")
    }
    if actual_setup_paths != expected_setup_paths:
        raise BenchmarkSafetyError(
            "output directory contains stale or missing setup JSON files; use a fresh version directory"
        )
    if actual_chart_paths != expected_chart_paths:
        raise BenchmarkSafetyError(
            "output directory contains stale or missing charts; use a fresh version directory"
        )

    labelled_rows = [_flatten_labelled(record) for record in labelled]
    shadow_rows = [_flatten_shadow(record) for record in shadow]
    replay_rows = [
        _flatten_replay(record, replay)
        for record in labelled
        for replay in record["replay"]
    ]
    _write_csv(output_dir / "labelled_comparison.csv", labelled_rows)
    _write_csv(output_dir / "shadow_comparison.csv", shadow_rows)
    _write_csv(output_dir / "replay.csv", replay_rows)

    summary = summarize_benchmark(labelled, shadow, execution["selection"])
    summary["artifacts"] = {
        "labelled_failure_charts": sum(
            bool(record["artifacts"]["chart_path"]) for record in labelled
        ),
        "shadow_disagreement_charts": sum(
            bool(record["artifacts"]["chart_path"]) for record in shadow
        ),
        "setup_json_files": len(labelled) + len(shadow),
    }
    summary_payload = {**summary, "summary_sha256": sha256_json(summary)}
    _write_json(output_dir / "summary.json", summary_payload)
    (output_dir / "README.md").write_text(
        _report_markdown(summary_payload), encoding="utf-8"
    )
    (output_dir / "chart_index.md").write_text(
        _chart_index_markdown(labelled, shadow), encoding="utf-8"
    )

    project_root = Path(__file__).resolve().parent
    inputs = []
    for role, manifest in execution["manifests"].items():
        source = manifest.get("source") or (manifest.get("source_run") or {}).get(
            "filename"
        )
        inputs.append(
            {
                "role": role,
                "corpus_id": manifest.get("corpus_id"),
                "source": source,
                "manifest_sha256": manifest.get("_manifest_sha256"),
            }
        )
    sealed_manifest = load_review_manifest(SEALED_SOURCE)
    inputs.append(
        {
            "role": "sealed_exclusion_roster_only",
            "corpus_id": sealed_manifest.get("corpus_id"),
            "source": SEALED_SOURCE,
            "manifest_sha256": sealed_manifest.get("_manifest_sha256"),
            "snapshots_opened": 0,
            "detector_runs": 0,
        }
    )
    artifact_files = [
        "README.md",
        "chart_index.md",
        "labelled_comparison.csv",
        "shadow_comparison.csv",
        "replay.csv",
        "summary.json",
    ]
    manifest_payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "kind": BENCHMARK_KIND,
        "run_id": BENCHMARK_VERSION,
        "generated_at": execution["generated_at"],
        "authoritative": False,
        "code": {
            "git_branch": _git_value(["branch", "--show-current"], project_root),
            "git_commit": _git_value(["rev-parse", "HEAD"], project_root),
            "worktree_clean": not bool(
                _git_value(["status", "--porcelain"], project_root)
            ),
            "lifetime_structure_sha256": sha256_file(
                project_root / "lifetime_structure.py"
            ),
            "coil_analysis_sha256": sha256_file(project_root / "coil_analysis.py"),
            "runner_sha256": sha256_file(project_root / "lifetime_reference_benchmark.py"),
            "renderer_sha256": sha256_file(project_root / "lifetime_benchmark_charts.py"),
        },
        "detectors": {
            "current": {
                "entrypoint": "coil_analysis.analyze_coil",
                "algorithm_version": CURRENT_ALGORITHM_VERSION,
            },
            "lifetime": {
                "entrypoint": "lifetime_structure.analyze_lifetime_references",
                "algorithm_version": LIFETIME_ALGORITHM_VERSION,
                "config": asdict(DEFAULT_LIFETIME_CONFIG),
            },
        },
        "inputs": inputs,
        "protocol": {
            "interval": "3M",
            "price_field": "high",
            "detector_input": "physically_truncated_ohlcv_only",
            "partial_quarter_policy": "detector_native_incomplete_quarter_handling",
            "quarter_ordinal": "year*4+(month-1)//3",
            "direction_threshold_pct_per_year": DIRECTION_THRESHOLD_PCT_PER_YEAR,
            "anchor_tolerance_quarters": TOP_TOLERANCE_QUARTERS,
            "projected_line_tolerance_pct": PROJECTED_LINE_TOLERANCE_PCT,
            "slope_tolerance_pct_per_year": SLOPE_TOLERANCE_PCT_PER_YEAR,
            "descriptive_rms_tolerance_pct": DESCRIPTIVE_RMS_TOLERANCE_PCT,
            "blind_identity_exclusion": True,
        },
        "counts": summary_payload["counts"],
        "artifacts": {
            name: {
                "path": name,
                "sha256": sha256_file(output_dir / name),
            }
            for name in artifact_files
        },
        "artifact_collections": {
            "setups": _file_inventory(output_dir, expected_setup_paths),
            "charts": _file_inventory(output_dir, expected_chart_paths),
        },
        "limitations": summary_payload["limitations"],
    }
    manifest_payload["manifest_sha256"] = sha256_json(manifest_payload)
    _write_json(output_dir / "manifest.json", manifest_payload)
    return {
        "output_dir": str(output_dir),
        "summary": summary_payload,
        "manifest": manifest_payload,
    }


def verify_benchmark_artifacts(output_dir: Path) -> dict[str, Any]:
    """Fail closed when a generated benchmark bundle is incomplete or stale."""
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    expected_manifest_hash = manifest.pop("manifest_sha256", None)
    if expected_manifest_hash != sha256_json(manifest):
        raise BenchmarkSafetyError("manifest self-hash does not verify")
    expected_summary_hash = summary.pop("summary_sha256", None)
    if expected_summary_hash != sha256_json(summary):
        raise BenchmarkSafetyError("summary self-hash does not verify")

    for artifact in manifest.get("artifacts", {}).values():
        relative = Path(str(artifact["path"]))
        path = (output_dir / relative).resolve()
        if output_dir.resolve() not in path.parents:
            raise BenchmarkSafetyError("artifact path escapes the benchmark bundle")
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise BenchmarkSafetyError(f"artifact hash mismatch: {relative}")

    for collection_name, collection in manifest.get(
        "artifact_collections", {}
    ).items():
        files = collection.get("files") or []
        if collection.get("file_count") != len(files):
            raise BenchmarkSafetyError(
                f"{collection_name} inventory count does not verify"
            )
        if collection.get("tree_sha256") != sha256_json(files):
            raise BenchmarkSafetyError(
                f"{collection_name} inventory tree hash does not verify"
            )
        expected_paths = set()
        for artifact in files:
            relative = Path(str(artifact["path"]))
            path = (output_dir / relative).resolve()
            if output_dir.resolve() not in path.parents:
                raise BenchmarkSafetyError(
                    f"{collection_name} path escapes the benchmark bundle"
                )
            if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
                raise BenchmarkSafetyError(
                    f"{collection_name} artifact hash mismatch: {relative}"
                )
            expected_paths.add(relative.as_posix())
        pattern = "setups/*.json" if collection_name == "setups" else "charts/**/*.png"
        actual_paths = {
            path.relative_to(output_dir).as_posix()
            for path in output_dir.glob(pattern)
        }
        if actual_paths != expected_paths:
            raise BenchmarkSafetyError(
                f"{collection_name} contains files outside its manifest inventory"
            )

    with (output_dir / "labelled_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        labelled_rows = list(csv.DictReader(handle))
    with (output_dir / "shadow_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        shadow_rows = list(csv.DictReader(handle))
    with (output_dir / "replay.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        replay_rows = list(csv.DictReader(handle))

    counts = summary["counts"]
    if len(labelled_rows) != int(counts["labelled_executed"]):
        raise BenchmarkSafetyError("labelled CSV row count does not match summary")
    if len(shadow_rows) != int(counts["shadow_executed"]):
        raise BenchmarkSafetyError("shadow CSV row count does not match summary")
    if len(replay_rows) != int(counts["replay_rows"]):
        raise BenchmarkSafetyError("replay CSV row count does not match summary")

    sealed = _sealed_ticker_identities()
    shadow_tickers = {str(row["ticker"]).upper() for row in shadow_rows}
    if shadow_tickers & sealed:
        raise BenchmarkSafetyError("sealed ticker identity leaked into shadow output")
    if counts.get("blind_snapshots_executed") != 0:
        raise BenchmarkSafetyError("summary claims a blind snapshot was executed")

    setup_files = sorted((output_dir / "setups").glob("*.json"))
    if len(setup_files) != int(summary["artifacts"]["setup_json_files"]):
        raise BenchmarkSafetyError("setup JSON count does not match summary")
    observation_ids: set[str] = set()
    for setup_path in setup_files:
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        if "monthly_bars" in canonical_json(setup):
            raise BenchmarkSafetyError("raw monthly bars were duplicated into setup JSON")
        observation_id = str(setup["identity"]["observation_id"])
        if observation_id in observation_ids:
            raise BenchmarkSafetyError("duplicate observation id in setup JSON")
        observation_ids.add(observation_id)
        for artifact_path in (setup.get("artifacts") or {}).values():
            if not artifact_path:
                continue
            resolved = (output_dir / str(artifact_path)).resolve()
            if output_dir.resolve() not in resolved.parents or not resolved.is_file():
                raise BenchmarkSafetyError(
                    f"setup artifact is missing or unsafe: {artifact_path}"
                )

    return {
        "verified": True,
        "labelled_rows": len(labelled_rows),
        "shadow_rows": len(shadow_rows),
        "replay_rows": len(replay_rows),
        "setup_json_files": len(setup_files),
        "sealed_ticker_overlap": 0,
    }
