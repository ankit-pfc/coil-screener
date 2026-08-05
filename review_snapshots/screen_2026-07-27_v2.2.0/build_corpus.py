#!/usr/bin/env python3
"""Build the immutable July 27, 2026 v2.2.0 review evidence corpus.

This generator deliberately copies source values without correction.  Derived
quality findings are annotations only; they never alter a source bar.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CACHE_ROOT = PROJECT_ROOT / "cache"
SOURCE_CSV_NAME = "screen_2026-07-27_v2.2.0.csv"
SOURCE_CSV = PROJECT_ROOT / SOURCE_CSV_NAME
SOURCE_CSV_COPY = HERE / "source_run.csv"
MANIFEST_PATH = HERE / "manifest.json"
PRIOR_REVIEW_PATH = PROJECT_ROOT / "validation" / "major_high_feedback.json"

SCHEMA_VERSION = 1
SNAPSHOT_KIND = "coilingview.saved-run-review-snapshot"
MANIFEST_KIND = "coilingview.saved-run-review-corpus-manifest"
ALGORITHM_VERSION = "2.2.0"
CODE_SHA = "65a04cf6d13d4e31a4e9b15cbab54fa150a935ba"

CONTROL_REFERENCE_TICKERS = {
    "REG",
    "SPG",
    "BG",
    "VTR",
    "UNP",
    "LH",
    "MSCI",
    "FCX",
    "NUE",
    "CF",
    "KN",
    "EAT",
    "MSFT",
    "COF",
    "CSX",
    "AAPL",
    "NSC",
    "UEC",
}

# These are not inferred gold labels. They record names explicitly used in
# v2.2.0 comments to pin or illustrate detector thresholds at CODE_SHA.
TUNING_ANCHOR_EVIDENCE: dict[str, list[dict[str, Any]]] = {
    "KN": [
        {
            "file": "coil_analysis.py",
            "lines": [154, 159],
            "purpose": "pins the exclusion side of the zone prominence gate",
        },
        {
            "file": "coil_analysis.py",
            "lines": [175, 178],
            "purpose": "pins the breakout peak-extension distinction",
        },
        {
            "file": "coil_analysis.py",
            "lines": [200, 203],
            "purpose": "illustrates the pressed-at-lid compression exception",
        },
    ],
    "CNR.TO": [
        {
            "file": "coil_analysis.py",
            "lines": [154, 159],
            "purpose": "pins the inclusion side of the zone prominence gate",
        }
    ],
    "FCX": [
        {
            "file": "coil_analysis.py",
            "lines": [175, 178],
            "purpose": "contrasts a normal rising-lid continuation",
        },
        {
            "file": "coil_analysis.py",
            "lines": [210, 214],
            "purpose": "pins the C-grade slope side of calibration",
        },
    ],
    "EAT": [
        {
            "file": "coil_analysis.py",
            "lines": [200, 203],
            "purpose": "illustrates the pressed-at-lid compression exception",
        }
    ],
    "REG": [
        {
            "file": "coil_analysis.py",
            "lines": [210, 214],
            "purpose": "pins the A-grade slope side of calibration",
        }
    ],
    "SPG": [
        {
            "file": "coil_analysis.py",
            "lines": [210, 214],
            "purpose": "pins the A-grade slope side of calibration",
        }
    ],
    "LH": [
        {
            "file": "coil_analysis.py",
            "lines": [210, 214],
            "purpose": "pins the A-grade slope side of calibration",
        }
    ],
    "CF": [
        {
            "file": "coil_analysis.py",
            "lines": [210, 214],
            "purpose": "pins the A-grade slope side of calibration",
        }
    ],
}

PRICE_FIELDS = ("open", "high", "low", "close")
WICK_MIN_RATIO = 0.50
WICK_BASELINE_MULTIPLIER = 6.0
WICK_BASELINE_WINDOW = 24
WICK_MIN_BASELINE_OBSERVATIONS = 12
DISCONTINUITY_SCALE_RATIO = 4.0


def canonical_json_bytes(value: Any) -> bytes:
    """Return the corpus-wide deterministic JSON identity representation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def pretty_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def rounded(value: float) -> float:
    return round(value, 12)


def finite_price(bar: dict[str, Any], field: str) -> float:
    value = bar.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite")
    return number


def analyze_data_quality(bars: list[dict[str, Any]]) -> dict[str, Any]:
    """Annotate exact bars using explicit, intentionally conservative checks."""
    invariant_failures: list[dict[str, Any]] = []
    nonpositive_fields: list[dict[str, Any]] = []
    numeric_failures: list[dict[str, Any]] = []
    extreme_wicks: list[dict[str, Any]] = []
    discontinuities: list[dict[str, Any]] = []
    date_failures: list[dict[str, Any]] = []
    range_ratios: list[float] = []
    seen_dates: set[str] = set()
    previous_date = ""

    for index, bar in enumerate(bars):
        date = str(bar.get("date", ""))
        date_violations: list[str] = []
        if len(date) != 10 or date[4:5] != "-" or date[7:8] != "-":
            date_violations.append("not_iso_yyyy_mm_dd")
        if date in seen_dates:
            date_violations.append("duplicate")
        if previous_date and date <= previous_date:
            date_violations.append("not_strictly_chronological")
        if date_violations:
            date_failures.append(
                {
                    "bar_index": index,
                    "date": date,
                    "violations": date_violations,
                }
            )
        seen_dates.add(date)
        previous_date = date

        try:
            open_price, high, low, close = (
                finite_price(bar, field) for field in PRICE_FIELDS
            )
        except ValueError as exc:
            numeric_failures.append(
                {"bar_index": index, "date": date, "reason": str(exc)}
            )
            range_ratios.append(float("nan"))
            continue

        violations: list[str] = []
        if high < low:
            violations.append("high_below_low")
        if high < open_price:
            violations.append("high_below_open")
        if high < close:
            violations.append("high_below_close")
        if low > open_price:
            violations.append("low_above_open")
        if low > close:
            violations.append("low_above_close")
        if violations:
            invariant_failures.append(
                {
                    "bar_index": index,
                    "date": date,
                    "violations": violations,
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                }
            )

        for field, value in zip(
            PRICE_FIELDS, (open_price, high, low, close), strict=True
        ):
            if value <= 0:
                nonpositive_fields.append(
                    {
                        "bar_index": index,
                        "date": date,
                        "field": field,
                        "value": bar[field],
                    }
                )

        body_reference = statistics.median((abs(open_price), abs(close)))
        range_ratio = (
            (high - low) / body_reference
            if body_reference > 0
            else float("inf")
        )
        baseline_values = [
            value
            for value in range_ratios[-WICK_BASELINE_WINDOW:]
            if math.isfinite(value) and value >= 0
        ]
        baseline = (
            statistics.median(baseline_values)
            if len(baseline_values) >= WICK_MIN_BASELINE_OBSERVATIONS
            else None
        )
        upper_wick = max(0.0, high - max(open_price, close))
        lower_wick = max(0.0, min(open_price, close) - low)
        upper_ratio = (
            upper_wick / body_reference if body_reference > 0 else float("inf")
        )
        lower_ratio = (
            lower_wick / body_reference if body_reference > 0 else float("inf")
        )
        maximum_wick_ratio = max(upper_ratio, lower_ratio)
        if (
            baseline is not None
            and maximum_wick_ratio >= WICK_MIN_RATIO
            and maximum_wick_ratio >= WICK_BASELINE_MULTIPLIER * baseline
        ):
            if upper_ratio == lower_ratio:
                direction = "both"
            elif upper_ratio > lower_ratio:
                direction = "upper"
            else:
                direction = "lower"
            extreme_wicks.append(
                {
                    "bar_index": index,
                    "date": date,
                    "direction": direction,
                    "upper_wick_ratio": rounded(upper_ratio),
                    "lower_wick_ratio": rounded(lower_ratio),
                    "maximum_wick_ratio": rounded(maximum_wick_ratio),
                    "trailing_median_range_ratio": rounded(baseline),
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                }
            )
        range_ratios.append(range_ratio)

        if index:
            previous = bars[index - 1]
            previous_close_value = previous.get("close")
            if (
                isinstance(previous_close_value, (int, float))
                and not isinstance(previous_close_value, bool)
                and math.isfinite(float(previous_close_value))
                and float(previous_close_value) > 0
                and body_reference > 0
            ):
                previous_close = float(previous_close_value)
                scale_ratio = max(
                    body_reference / previous_close,
                    previous_close / body_reference,
                )
                if scale_ratio >= DISCONTINUITY_SCALE_RATIO:
                    discontinuities.append(
                        {
                            "bar_index": index,
                            "date": date,
                            "previous_date": str(previous.get("date", "")),
                            "previous_close": previous["close"],
                            "current_body_midpoint": rounded(body_reference),
                            "scale_ratio": rounded(scale_ratio),
                            "direction": (
                                "up"
                                if body_reference > previous_close
                                else "down"
                            ),
                        }
                    )

    hard_failure_count = (
        len(date_failures)
        + len(numeric_failures)
        + len(invariant_failures)
        + len(nonpositive_fields)
    )
    heuristic_flag_count = len(extreme_wicks) + len(discontinuities)
    if hard_failure_count:
        status = "quarantined"
    elif heuristic_flag_count:
        status = "flagged_for_review"
    else:
        status = "clear_by_defined_checks"
    return {
        "status": status,
        "reviewable": hard_failure_count == 0,
        "hard_failure_count": hard_failure_count,
        "heuristic_flag_count": heuristic_flag_count,
        "date_order_failures": date_failures,
        "numeric_price_failures": numeric_failures,
        "ohlc_invariant_failures": invariant_failures,
        "nonpositive_price_fields": nonpositive_fields,
        "extreme_wicks": extreme_wicks,
        "extreme_discontinuities": discontinuities,
    }


def prior_review_summary(
    ticker: str, prior_reviews: dict[str, Any]
) -> dict[str, Any]:
    review = prior_reviews.get(ticker)
    if not isinstance(review, dict):
        return {
            "flag": False,
            "source": "validation/major_high_feedback.json",
        }
    return {
        "flag": True,
        "source": "validation/major_high_feedback.json",
        "human_grade": review.get("human_grade"),
        "model_grade_at_review": review.get("model_grade_at_review"),
        "model_status_at_review": review.get("model_status_at_review"),
        "algorithm_version_at_review": review.get(
            "algorithm_version_at_review"
        ),
    }


def tuning_anchor_summary(ticker: str) -> dict[str, Any]:
    evidence = TUNING_ANCHOR_EVIDENCE.get(ticker, [])
    return {
        "flag": bool(evidence),
        "source_code_sha": CODE_SHA,
        "evidence": evidence,
    }


def quality_counts(quality: dict[str, Any]) -> dict[str, int]:
    return {
        "date_order_failures": len(quality["date_order_failures"]),
        "numeric_price_failures": len(quality["numeric_price_failures"]),
        "ohlc_invariant_failures": len(
            quality["ohlc_invariant_failures"]
        ),
        "nonpositive_price_fields": len(
            quality["nonpositive_price_fields"]
        ),
        "extreme_wicks": len(quality["extreme_wicks"]),
        "extreme_discontinuities": len(
            quality["extreme_discontinuities"]
        ),
    }


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames or not rows:
        raise RuntimeError("source run CSV is empty")
    tickers = [row.get("ticker", "") for row in rows]
    if any(not ticker for ticker in tickers):
        raise RuntimeError("every source run row must have a ticker")
    if len(tickers) != len(set(tickers)):
        raise RuntimeError("source run contains duplicate tickers")
    return fieldnames, rows


def build() -> dict[str, Any]:
    if not SOURCE_CSV.is_file():
        raise RuntimeError(f"missing source run: {SOURCE_CSV}")
    source_csv_bytes = SOURCE_CSV.read_bytes()
    source_csv_sha = sha256_bytes(source_csv_bytes)
    shutil.copyfile(SOURCE_CSV, SOURCE_CSV_COPY)

    fieldnames, rows = load_csv_rows(SOURCE_CSV)
    prior_payload = json.loads(PRIOR_REVIEW_PATH.read_text(encoding="utf-8"))
    prior_reviews = prior_payload.get("reviews", {})
    if not isinstance(prior_reviews, dict):
        raise RuntimeError("prior-review source has no reviews mapping")

    items: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    cohort_counts: Counter[str] = Counter()
    aggregate_quality: Counter[str] = Counter()
    fetched_times: list[str] = []
    all_dates: list[str] = []

    for position, row in enumerate(rows, start=1):
        ticker = row["ticker"]
        cache_path = CACHE_ROOT / f"{ticker}.json"
        if not cache_path.is_file():
            raise RuntimeError(f"missing exact cache for {ticker}: {cache_path}")
        cache_bytes = cache_path.read_bytes()
        cache_payload = json.loads(cache_bytes)
        if cache_payload.get("ticker") != ticker:
            raise RuntimeError(f"cache ticker mismatch for {ticker}")
        bars = cache_payload.get("bars")
        if not isinstance(bars, list) or not bars:
            raise RuntimeError(f"cache bars missing for {ticker}")

        cache_metadata = cache_payload.get("cache_metadata")
        if not isinstance(cache_metadata, dict):
            raise RuntimeError(f"cache metadata missing for {ticker}")
        if row.get("data_date") != str(bars[-1].get("date")):
            raise RuntimeError(f"CSV/cache data_date mismatch for {ticker}")
        if row.get("fetched_at") != str(cache_metadata.get("fetched_at")):
            raise RuntimeError(f"CSV/cache fetched_at mismatch for {ticker}")
        if float(row["last_close"]) != float(bars[-1]["close"]):
            raise RuntimeError(f"CSV/cache last_close mismatch for {ticker}")

        quality = analyze_data_quality(bars)
        cohort_role = (
            "control_reference"
            if ticker in CONTROL_REFERENCE_TICKERS
            else "prospective_international"
        )
        prior_review = prior_review_summary(ticker, prior_reviews)
        tuning_anchor = tuning_anchor_summary(ticker)
        canonical_bars_hash = sha256_json(bars)
        backend_bars_identity_hash = sha256_json(
            {
                "schema_version": SCHEMA_VERSION,
                "interval": "1M",
                "ticker": ticker,
                "bars": bars,
            }
        )
        screen_snapshot_hash = sha256_json(row)
        source_cache_hash = sha256_bytes(cache_bytes)
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "kind": SNAPSHOT_KIND,
            "source": SOURCE_CSV_NAME,
            "ticker": ticker,
            "run": {
                "algorithm_version": ALGORITHM_VERSION,
                "code_sha": CODE_SHA,
                "source_csv_sha256": source_csv_sha,
                "universe_position": position,
            },
            "screen_snapshot": row,
            "monthly_bars": bars,
            "source_cache_metadata": cache_metadata,
            "provenance": {
                "source_cache_file": f"cache/{ticker}.json",
                "source_cache_sha256": source_cache_hash,
                "canonical_monthly_bars_sha256": canonical_bars_hash,
                "backend_bars_identity_sha256": backend_bars_identity_hash,
                "screen_snapshot_sha256": screen_snapshot_hash,
            },
            "corpus_labels": {
                "cohort_role": cohort_role,
                "prior_review": prior_review,
                "tuning_anchor": tuning_anchor,
            },
            "data_quality": quality,
        }
        snapshot_path = HERE / f"{ticker}.json"
        snapshot_path.write_text(pretty_json(snapshot), encoding="utf-8")
        snapshot_hash = sha256_file(snapshot_path)
        counts = quality_counts(quality)
        for name, count in counts.items():
            aggregate_quality[name] += count
        status_counts[quality["status"]] += 1
        cohort_counts[cohort_role] += 1
        fetched_times.append(row["fetched_at"])
        all_dates.extend((str(bars[0]["date"]), str(bars[-1]["date"])))
        items.append(
            {
                "universe_position": position,
                "ticker": ticker,
                "snapshot_file": f"{ticker}.json",
                "snapshot_sha256": snapshot_hash,
                "source_cache_file": f"cache/{ticker}.json",
                "source_cache_sha256": source_cache_hash,
                "canonical_monthly_bars_sha256": canonical_bars_hash,
                "backend_bars_identity_sha256": backend_bars_identity_hash,
                "screen_snapshot_sha256": screen_snapshot_hash,
                "bar_count": len(bars),
                "first_data_date": str(bars[0]["date"]),
                "last_data_date": str(bars[-1]["date"]),
                "fetched_at": row["fetched_at"],
                "data_source": row["data_source"],
                "cohort_role": cohort_role,
                "prior_review": prior_review,
                "tuning_anchor": tuning_anchor,
                "data_quality": {
                    "status": quality["status"],
                    "reviewable": quality["reviewable"],
                    "hard_failure_count": quality["hard_failure_count"],
                    "heuristic_flag_count": quality[
                        "heuristic_flag_count"
                    ],
                    "counts": counts,
                },
            }
        )

    expected_tickers = {row["ticker"] for row in rows}
    if not CONTROL_REFERENCE_TICKERS <= expected_tickers:
        missing = sorted(CONTROL_REFERENCE_TICKERS - expected_tickers)
        raise RuntimeError(f"control/reference tickers missing: {missing}")

    prior_in_universe = sorted(expected_tickers & set(prior_reviews))
    tuning_in_universe = sorted(expected_tickers & set(TUNING_ANCHOR_EVIDENCE))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "corpus_id": "screen_2026-07-27_v2.2.0",
        "purpose": (
            "Immutable evidence for blinded, human review of the July 27 "
            "v2.2.0 screened-stock run."
        ),
        "trust_status": "review_evidence_not_validated_training_truth",
        "source_run": {
            "filename": SOURCE_CSV_NAME,
            "embedded_exact_copy": SOURCE_CSV_COPY.name,
            "sha256": source_csv_sha,
            "byte_count": len(source_csv_bytes),
            "row_count": len(rows),
            "column_count": len(fieldnames),
            "columns": fieldnames,
            "algorithm_version": ALGORITHM_VERSION,
            "code_sha": CODE_SHA,
            "fetched_at_min": min(fetched_times),
            "fetched_at_max": max(fetched_times),
        },
        "supporting_sources": {
            "prior_review": {
                "file": "validation/major_high_feedback.json",
                "sha256": sha256_file(PRIOR_REVIEW_PATH),
                "schema_version": prior_payload.get("schema_version"),
                "in_universe_count": len(prior_in_universe),
                "in_universe_tickers": prior_in_universe,
            },
            "tuning_anchor_comments": {
                "file": "coil_analysis.py",
                "code_sha": CODE_SHA,
                "in_universe_count": len(tuning_in_universe),
                "in_universe_tickers": tuning_in_universe,
            },
        },
        "canonicalization": {
            "encoding": "UTF-8",
            "json_identity": (
                "json.dumps(sort_keys=True,separators=(',',':'),"
                "ensure_ascii=False,allow_nan=False)"
            ),
            "snapshot_storage": "sorted keys, two-space indentation, LF newline",
            "source_cache_hash": "SHA-256 of exact source cache file bytes",
            "canonical_monthly_bars_hash": (
                "SHA-256 of the canonical JSON monthly_bars array"
            ),
            "backend_bars_identity_hash": (
                "SHA-256 of canonical JSON containing schema_version=1, "
                "interval=1M, ticker, and bars"
            ),
        },
        "quality_checks": {
            "nature": (
                "Mechanical flags only; no finding repairs source evidence "
                "or establishes a market-data error."
            ),
            "quarantine_conditions": [
                "date is missing, duplicated, or non-chronological",
                "OHLC price is non-numeric or non-finite",
                "high/low violates OHLC containment",
                "OHLC price is zero or negative",
            ],
            "extreme_wick": {
                "definition": (
                    "Maximum wick divided by median absolute open/close is "
                    "at least 0.50 and at least 6x the trailing median full-"
                    "range ratio."
                ),
                "minimum_ratio": WICK_MIN_RATIO,
                "trailing_baseline_multiplier": WICK_BASELINE_MULTIPLIER,
                "trailing_window_bars": WICK_BASELINE_WINDOW,
                "minimum_baseline_observations": (
                    WICK_MIN_BASELINE_OBSERVATIONS
                ),
            },
            "extreme_discontinuity": {
                "definition": (
                    "Median absolute open/close differs from the prior close "
                    "by a symmetric scale ratio of at least 4x."
                ),
                "minimum_symmetric_scale_ratio": (
                    DISCONTINUITY_SCALE_RATIO
                ),
            },
        },
        "cohorts": {
            "control_reference": {
                "definition": (
                    "The 18 non-international names deliberately appended as "
                    "calibration/control references."
                ),
                "count": cohort_counts["control_reference"],
                "tickers": [
                    row["ticker"]
                    for row in rows
                    if row["ticker"] in CONTROL_REFERENCE_TICKERS
                ],
            },
            "prospective_international": {
                "definition": (
                    "Cross-market review universe; prior-review and tuning-"
                    "anchor flags must still be consulted per item."
                ),
                "count": cohort_counts["prospective_international"],
                "tickers": [
                    row["ticker"]
                    for row in rows
                    if row["ticker"] not in CONTROL_REFERENCE_TICKERS
                ],
            },
        },
        "ordered_universe": [row["ticker"] for row in rows],
        "summary": {
            "ticker_count": len(items),
            "monthly_bar_count": sum(item["bar_count"] for item in items),
            "first_data_date": min(all_dates),
            "last_data_date": max(all_dates),
            "cohort_counts": dict(sorted(cohort_counts.items())),
            "prior_review_ticker_count": len(prior_in_universe),
            "tuning_anchor_ticker_count": len(tuning_in_universe),
            "data_quality_status_counts": dict(
                sorted(status_counts.items())
            ),
            "data_quality_finding_counts": dict(
                sorted(aggregate_quality.items())
            ),
        },
        "generator": {
            "file": Path(__file__).name,
            "sha256": sha256_file(Path(__file__)),
        },
        "items": items,
    }
    MANIFEST_PATH.write_text(pretty_json(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build()
    summary = manifest["summary"]
    print(
        f"built {summary['ticker_count']} snapshots with "
        f"{summary['monthly_bar_count']} exact monthly bars"
    )
    print(f"source CSV SHA-256: {manifest['source_run']['sha256']}")
    print(f"manifest SHA-256: {sha256_file(MANIFEST_PATH)}")
    print(
        "quality status counts: "
        f"{summary['data_quality_status_counts']}"
    )


if __name__ == "__main__":
    main()
