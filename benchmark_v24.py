"""Fail-closed tooling for the lean v2.4 72-sample blind benchmark.

The benchmark is intentionally separate from screening and production review
state.  It consumes immutable split-adjusted samples, runs both detectors in
``algorithm_only`` mode, and refuses to expose holdout labels until a matching
configuration-freeze artifact exists.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import statistics
import calendar
from itertools import product
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from coil_analysis import (
    ALGORITHM_VERSION as V23_ALGORITHM_VERSION,
    ANALYSIS_MODE_ALGORITHM_ONLY,
    ANALYSIS_VARIANT_V2_3_1,
    analyze_coil,
    replay_completed_quarter_prefixes,
)
from coil_validation_v24 import (
    ALGORITHM_VERSION as V24_ALGORITHM_VERSION,
    DEFAULT_CONFIG as V24_DEFAULT_CONFIG,
    ValidationConfig,
    analyze_coil_v24,
    config_fingerprint,
)

BENCHMARK_ID = "coilingview-v24-72"
MANIFEST_SCHEMA_VERSION = 1
LABEL_SCHEMA_VERSION = 1
FREEZE_SCHEMA_VERSION = 1
DECISION_SCHEMA_VERSION = 1
PARTITIONS = ("development", "validation", "holdout")
PARTITION_COUNTS = {"development": 36, "validation": 18, "holdout": 18}
MARKETS = ("india", "united_states", "global_ex_us")
COHORTS = (
    "expert_positive",
    "clear_negative",
    "disagreement_exception",
    "point_in_time_lifecycle",
)
COHORT_COUNT = 18
INTERVIEW_LEAKAGE_TICKERS = {
    "NATR",
    "BHARTIARTL.NS",
    "PRESTIGE.NS",
    "VOLTAMP.NS",
    "UEC",
    "WBS",
    "AWI",
    "DBX",
    "TXN",
    "ENB.TO",
}

GATES = {
    "label_pattern_agreement": 0.80,
    "label_top_f1": 0.75,
    "label_band_agreement": 0.75,
    "major_top_precision": 0.80,
    "major_top_recall": 0.65,
    "band_agreement": 0.70,
    "pattern_precision": 0.80,
    "automated_positive_coverage": 0.40,
    "clear_negative_false_positive_rate_max": 0.15,
    "point_in_time_violations_max": 0,
    "accepted_hard_invalid_max": 0,
    "review_time_reduction": 0.30,
}


class BenchmarkError(ValueError):
    """Benchmark input is unsafe, incomplete, or inconsistent."""


class HoldoutSealedError(PermissionError):
    """Holdout labels were requested without a valid configuration freeze."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def file_sha256(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"{label} is missing or unreadable") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} must be a JSON object")
    return value


def stable_sample_id(
    ticker: str,
    as_of: str,
    bars_sha256: str,
) -> str:
    digest = sha256_json(
        {
            "benchmark_id": BENCHMARK_ID,
            "ticker": ticker.strip().upper(),
            "as_of": as_of,
            "bars_sha256": bars_sha256,
        }
    )
    return f"sample_{digest[:20]}"


def stable_task_id(sample_id: str, attempt: int, queue_salt: str) -> str:
    digest = sha256_json(
        {
            "benchmark_id": BENCHMARK_ID,
            "sample_id": sample_id,
            "attempt": attempt,
            "queue_salt": queue_salt,
        }
    )
    return f"task_{digest[:20]}"


def _validate_date(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise BenchmarkError(f"{field} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise BenchmarkError(f"{field} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise BenchmarkError(f"{field} must be an ISO date")
    return value


def _ticker_family(item: Mapping[str, Any]) -> str:
    family = str(item.get("ticker_family") or item.get("ticker") or "").strip().upper()
    if not family:
        raise BenchmarkError("every sample requires ticker_family")
    return family


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all immutable corpus and split invariants."""
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("unsupported manifest schema")
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        errors.append("unexpected benchmark id")
    generated_at = manifest.get("generated_at")
    generated_date: date | None = None
    if not isinstance(generated_at, str):
        errors.append("manifest requires generated_at")
    else:
        try:
            generated_date = datetime.fromisoformat(
                generated_at.replace("Z", "+00:00")
            ).date()
        except ValueError:
            errors.append("manifest generated_at must be an ISO timestamp")
    latest_completed_quarter: date | None = None
    if generated_date is not None:
        quarter_start_month = ((generated_date.month - 1) // 3) * 3 + 1
        if quarter_start_month == 1:
            latest_completed_quarter = date(generated_date.year - 1, 12, 31)
        else:
            previous_end_month = quarter_start_month - 1
            latest_completed_quarter = date(
                generated_date.year,
                previous_end_month,
                calendar.monthrange(generated_date.year, previous_end_month)[1],
            )
    items = manifest.get("items")
    if not isinstance(items, list):
        raise BenchmarkError("manifest items must be a list")
    if len(items) != 72:
        errors.append(f"expected 72 samples, found {len(items)}")

    ids: list[str] = []
    identities: list[tuple[str, str]] = []
    partitions: Counter[str] = Counter()
    markets: Counter[str] = Counter()
    cohorts: Counter[str] = Counter()
    split_tickers: dict[str, set[str]] = defaultdict(set)
    split_families: dict[str, set[str]] = defaultdict(set)
    canonical_count = 0
    reviewable_count = 0
    new_or_remediated_count = 0
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            errors.append(f"item {index} is not an object")
            continue
        sample_id = str(raw.get("sample_id") or "")
        ticker = str(raw.get("ticker") or "").strip().upper()
        as_of = str(raw.get("as_of") or "")
        ids.append(sample_id)
        identities.append((ticker, as_of))
        partition = str(raw.get("partition") or "")
        market = str(raw.get("market") or "")
        cohort = str(raw.get("cohort") or "")
        partitions[partition] += 1
        markets[market] += 1
        cohorts[cohort] += 1
        split_tickers[partition].add(ticker)
        try:
            split_families[partition].add(_ticker_family(raw))
            _validate_date(as_of, f"item {index} as_of")
            parsed_as_of = date.fromisoformat(as_of)
            if parsed_as_of.month not in {3, 6, 9, 12} or parsed_as_of.day != calendar.monthrange(
                parsed_as_of.year, parsed_as_of.month
            )[1]:
                errors.append(f"{ticker or index} as_of is not a calendar quarter-end")
            if latest_completed_quarter is not None and parsed_as_of > latest_completed_quarter:
                errors.append(f"{ticker or index} as_of is not a completed quarter")
        except BenchmarkError as exc:
            errors.append(str(exc))
        if partition not in PARTITIONS:
            errors.append(f"item {index} has invalid partition {partition!r}")
        if market not in MARKETS:
            errors.append(f"item {index} has invalid market {market!r}")
        if cohort not in COHORTS:
            errors.append(f"item {index} has invalid cohort {cohort!r}")
        if partition != "development" and ticker in INTERVIEW_LEAKAGE_TICKERS:
            errors.append(f"interview example {ticker} leaked into {partition}")
        if raw.get("adjustment_mode") == "split_adjusted":
            canonical_count += 1
        else:
            errors.append(f"{ticker or index} is not split_adjusted")
        quality = raw.get("data_quality")
        if isinstance(quality, dict) and quality.get("reviewable") is True:
            reviewable_count += 1
        elif not (isinstance(quality, dict) and quality.get("expected_hard_invalid") is True):
            errors.append(f"{ticker or index} lacks a valid quality disposition")
        provenance = raw.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(f"{ticker or index} lacks provenance")
        else:
            for key in ("bars_sha256", "sample_file_sha256", "source_fingerprint"):
                value = provenance.get(key)
                if not isinstance(value, str) or len(value.removeprefix("sha256:")) != 64:
                    errors.append(f"{ticker or index} lacks valid {key}")
        if raw.get("new_or_remediated") is True and (
            isinstance(quality, dict) and quality.get("reviewable") is True
        ):
            new_or_remediated_count += 1

    if len(set(ids)) != len(ids) or any(not value for value in ids):
        errors.append("sample ids must be non-empty and unique")
    if len(set(identities)) != len(identities):
        errors.append("ticker/as_of identities must be unique")
    if len({ticker for ticker, _ in identities if ticker}) < 54:
        errors.append("benchmark must contain at least 54 unique tickers")
    if partitions != Counter(PARTITION_COUNTS):
        errors.append(f"partition counts are {dict(partitions)}")
    if markets != Counter({market: 24 for market in MARKETS}):
        errors.append(f"market counts are {dict(markets)}")
    if cohorts != Counter({cohort: COHORT_COUNT for cohort in COHORTS}):
        errors.append(f"cohort counts are {dict(cohorts)}")
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            overlap = split_tickers[left] & split_tickers[right]
            if overlap:
                errors.append(f"ticker leakage between {left} and {right}: {sorted(overlap)}")
            family_overlap = split_families[left] & split_families[right]
            if family_overlap:
                errors.append(
                    f"ticker-family leakage between {left} and {right}: {sorted(family_overlap)}"
                )
    if new_or_remediated_count < 5:
        errors.append("at least five clean new/remediated samples are required")

    return {
        "valid": not errors,
        "errors": errors,
        "sample_count": len(items),
        "unique_ticker_count": len({ticker for ticker, _ in identities if ticker}),
        "partition_counts": dict(partitions),
        "market_counts": dict(markets),
        "cohort_counts": dict(cohorts),
        "canonical_count": canonical_count,
        "reviewable_count": reviewable_count,
        "new_or_remediated_count": new_or_remediated_count,
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = _load_json(path, label="benchmark manifest")
    validation = validate_manifest(manifest)
    if not validation["valid"]:
        raise BenchmarkError("invalid benchmark manifest: " + "; ".join(validation["errors"]))
    manifest["_manifest_sha256"] = file_sha256(path)
    return manifest


def load_sample(root: str | Path, item: Mapping[str, Any]) -> dict[str, Any]:
    sample_path = (Path(root) / str(item.get("sample_file") or "")).resolve()
    root_path = Path(root).resolve()
    if root_path not in sample_path.parents:
        raise BenchmarkError("sample path escapes benchmark root")
    expected_file_hash = str((item.get("provenance") or {}).get("sample_file_sha256") or "")
    if not hmac.compare_digest(file_sha256(sample_path), expected_file_hash.removeprefix("sha256:")):
        raise BenchmarkError(f"sample file hash mismatch for {item.get('ticker')}")
    sample = _load_json(sample_path, label=f"sample {item.get('ticker')}")
    bars = sample.get("monthly_bars")
    if not isinstance(bars, list) or not bars:
        raise BenchmarkError(f"sample {item.get('ticker')} has no bars")
    bars_hash = sha256_json(bars)
    expected_bars_hash = str((item.get("provenance") or {}).get("bars_sha256") or "")
    if not hmac.compare_digest(bars_hash, expected_bars_hash.removeprefix("sha256:")):
        raise BenchmarkError(f"bar hash mismatch for {item.get('ticker')}")
    if sample.get("sample_id") != item.get("sample_id"):
        raise BenchmarkError(f"sample id mismatch for {item.get('ticker')}")
    if sample.get("ticker") != item.get("ticker") or sample.get("as_of") != item.get("as_of"):
        raise BenchmarkError(f"sample identity mismatch for {item.get('ticker')}")
    return sample


def make_freeze(
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    code_commit: str,
    evaluation_command: str,
    selection_report_sha256: str,
    config: ValidationConfig = V24_DEFAULT_CONFIG,
) -> dict[str, Any]:
    if len(code_commit) != 40 or any(ch not in "0123456789abcdef" for ch in code_commit.lower()):
        raise BenchmarkError("code_commit must be a full 40-character Git commit")
    if not evaluation_command.strip():
        raise BenchmarkError("evaluation_command is required")
    manifest = load_manifest(manifest_path)
    protocol = _load_json(protocol_path, label="benchmark protocol")
    if protocol.get("benchmark_id") != BENCHMARK_ID:
        raise BenchmarkError("protocol benchmark id mismatch")
    if evaluation_command.strip() != protocol.get("evaluation_command"):
        raise BenchmarkError("evaluation command does not match the frozen protocol")
    if len(selection_report_sha256.removeprefix("sha256:")) != 64:
        raise BenchmarkError("selection report hash is required before freeze")
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "kind": "coilingview.v24-benchmark-configuration-freeze",
        "benchmark_id": BENCHMARK_ID,
        "manifest_sha256": manifest["_manifest_sha256"],
        "protocol_sha256": file_sha256(protocol_path),
        "code_commit": code_commit.lower(),
        "selection_report_sha256": selection_report_sha256.removeprefix("sha256:"),
        "detectors": {
            "v2_3_1": {
                "algorithm_version": V23_ALGORITHM_VERSION,
                "mode": ANALYSIS_MODE_ALGORITHM_ONLY,
            },
            "v2_4_validation": {
                "algorithm_version": V24_ALGORITHM_VERSION,
                "mode": ANALYSIS_MODE_ALGORITHM_ONLY,
                "config": asdict(config),
                "config_fingerprint": config_fingerprint(config),
            },
        },
        "evaluation_command": evaluation_command.strip(),
        "holdout_labels_sha256": None,
        "holdout_revealed_at": None,
    }


def register_holdout_labels(
    freeze: Mapping[str, Any], labels_path: str | Path, *, revealed_at: str
) -> dict[str, Any]:
    if freeze.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise BenchmarkError("unsupported configuration freeze")
    if freeze.get("holdout_labels_sha256"):
        raise BenchmarkError("holdout labels are already registered")
    value = dict(freeze)
    value["holdout_labels_sha256"] = file_sha256(labels_path)
    value["holdout_revealed_at"] = revealed_at
    return value


def verify_freeze(
    freeze: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    selection_report_path: str | Path,
    current_code_commit: str,
    labels_path: str | Path | None = None,
    config: ValidationConfig = V24_DEFAULT_CONFIG,
    recompute_selection_metrics: bool = True,
) -> None:
    if freeze.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise BenchmarkError("unsupported configuration freeze")
    comparisons = (
        ("manifest_sha256", file_sha256(manifest_path)),
        ("protocol_sha256", file_sha256(protocol_path)),
    )
    for key, actual in comparisons:
        if not hmac.compare_digest(str(freeze.get(key) or ""), actual):
            raise BenchmarkError(f"configuration freeze {key} mismatch")
    if len(current_code_commit) != 40 or freeze.get("code_commit") != current_code_commit.lower():
        raise BenchmarkError("configuration freeze code commit mismatch")
    if not hmac.compare_digest(
        str(freeze.get("selection_report_sha256") or ""),
        file_sha256(selection_report_path),
    ):
        raise BenchmarkError("configuration freeze selection report mismatch")
    selection = _load_json(selection_report_path, label="validation-selection report")
    selected_config = validate_selection_report(
        selection,
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        current_code_commit=current_code_commit,
        recompute_metrics=recompute_selection_metrics,
    )
    if selected_config != config:
        raise BenchmarkError("configuration freeze does not use the validated winner")
    detector = (freeze.get("detectors") or {}).get("v2_4_validation") or {}
    if detector.get("algorithm_version") != V24_ALGORITHM_VERSION:
        raise BenchmarkError("unregistered v2.4 algorithm version")
    if detector.get("mode") != ANALYSIS_MODE_ALGORITHM_ONLY:
        raise BenchmarkError("v2.4 benchmark mode must be algorithm_only")
    if detector.get("config_fingerprint") != config_fingerprint(config):
        raise BenchmarkError("unregistered v2.4 configuration")
    if detector.get("config") != asdict(config):
        raise BenchmarkError("registered v2.4 raw configuration mismatch")
    baseline = (freeze.get("detectors") or {}).get("v2_3_1") or {}
    if baseline.get("algorithm_version") != V23_ALGORITHM_VERSION:
        raise BenchmarkError("unregistered v2.3.1 algorithm version")
    if baseline.get("mode") != ANALYSIS_MODE_ALGORITHM_ONLY:
        raise BenchmarkError("v2.3.1 benchmark mode must be algorithm_only")
    if labels_path is not None:
        expected = str(freeze.get("holdout_labels_sha256") or "")
        if not expected:
            raise HoldoutSealedError("holdout labels remain sealed until registered after freeze")
        if not hmac.compare_digest(expected, file_sha256(labels_path)):
            raise BenchmarkError("holdout labels do not match the registered seal")


def load_labels(
    path: str | Path,
    *,
    partition: str,
    manifest_path: str | Path,
    protocol_path: str | Path,
    freeze_path: str | Path | None = None,
    selection_report_path: str | Path | None = None,
    current_code_commit: str | None = None,
    config: ValidationConfig = V24_DEFAULT_CONFIG,
) -> dict[str, Any]:
    if partition not in PARTITIONS:
        raise BenchmarkError(f"unsupported label partition {partition!r}")
    if partition == "holdout":
        if freeze_path is None or selection_report_path is None or current_code_commit is None:
            raise HoldoutSealedError("holdout labels cannot be loaded before configuration freeze")
        freeze = _load_json(freeze_path, label="configuration freeze")
        verify_freeze(
            freeze,
            manifest_path=manifest_path,
            protocol_path=protocol_path,
            selection_report_path=selection_report_path,
            current_code_commit=current_code_commit,
            labels_path=path,
            config=config,
        )
    labels = _load_json(path, label=f"{partition} labels")
    if labels.get("schema_version") != LABEL_SCHEMA_VERSION:
        raise BenchmarkError("unsupported label schema")
    if labels.get("benchmark_id") != BENCHMARK_ID or labels.get("partition") != partition:
        raise BenchmarkError("label identity mismatch")
    if labels.get("manifest_sha256") != file_sha256(manifest_path):
        raise BenchmarkError("labels were produced for a different manifest")
    rows = labels.get("labels")
    if not isinstance(rows, list):
        raise BenchmarkError("labels must be a list")
    manifest = load_manifest(manifest_path)
    task_manifest_path = Path(manifest_path).parent / "review-task-manifest.internal.json"
    task_manifest = _load_json(task_manifest_path, label="benchmark task manifest")
    if task_manifest.get("benchmark_id") != BENCHMARK_ID:
        raise BenchmarkError("benchmark task manifest identity mismatch")
    expected_tasks = {
        str(task["task_id"]): task
        for task in task_manifest.get("tasks") or []
        if task.get("partition") in {partition, f"{partition}_repeat"}
    }
    if not expected_tasks:
        raise BenchmarkError(f"benchmark task manifest lacks {partition} tasks")
    expected_items = {
        str(item["sample_id"]): item
        for item in manifest["items"]
        if item.get("partition") == partition
    }
    primary_ids: list[str] = []
    repeat_ids: list[str] = []
    attempts: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BenchmarkError(f"label row {index} must be an object")
        sample_id = str(row.get("sample_id") or "")
        item = expected_items.get(sample_id)
        if item is None:
            raise BenchmarkError(f"label row {index} is not a {partition} sample")
        if str(row.get("ticker") or "").upper() != item["ticker"]:
            raise BenchmarkError(f"label row {index} ticker does not match its sample")
        role = row.get("attempt_role")
        if role not in {"primary", "repeat"}:
            raise BenchmarkError(f"label row {index} has invalid attempt_role")
        if partition != "development" and role == "repeat":
            raise BenchmarkError("repeat labels belong to development only")
        attempt_id = str(row.get("attempt_id") or "")
        if not attempt_id or attempt_id in attempts:
            raise BenchmarkError("label attempt ids must be non-empty and unique")
        attempts.add(attempt_id)
        source_task_id = str(row.get("source_task_id") or "")
        if source_task_id != attempt_id or source_task_id not in expected_tasks:
            raise BenchmarkError("label attempt does not match a frozen task id")
        task = expected_tasks[source_task_id]
        expected_role = "primary" if int(task.get("attempt", 0)) == 1 else "repeat"
        if (
            str(task.get("sample_id") or "") != sample_id
            or str(task.get("ticker") or "").upper() != item["ticker"]
            or role != expected_role
        ):
            raise BenchmarkError("label attempt does not match its frozen task identity")
        if row.get("pattern_label") not in {"coil", "not_coil", "uncertain"}:
            raise BenchmarkError(f"label row {index} has invalid pattern_label")
        if not isinstance(row.get("tops"), list):
            raise BenchmarkError(f"label row {index} tops must be a list")
        (primary_ids if role == "primary" else repeat_ids).append(sample_id)
    if set(primary_ids) != set(expected_items) or len(primary_ids) != len(expected_items):
        raise BenchmarkError(f"{partition} labels require exactly one primary per sample")
    if len(repeat_ids) != len(set(repeat_ids)):
        raise BenchmarkError("development repeats require at most one repeat per sample")
    if partition == "development" and len(repeat_ids) != 12:
        raise BenchmarkError("development labels require exactly 12 hidden repeats")
    if attempts != set(expected_tasks):
        raise BenchmarkError(f"{partition} labels do not cover the frozen task set exactly")
    return labels


def labels_from_review_exports(
    exports: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    partition: str,
    manifest_sha256: str,
    task_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract ground truth exclusively from pre-reveal blind assessments."""
    if partition not in PARTITIONS:
        raise BenchmarkError(f"unsupported label partition {partition!r}")
    items = {
        str(item["sample_id"]): item
        for item in manifest["items"]
        if item.get("partition") == partition
    }
    tasks = [
        task
        for task in task_manifest.get("tasks") or []
        if task.get("partition") in {partition, f"{partition}_repeat"}
    ]
    expected_tasks = {
        (str(task.get("source")), str(task.get("ticker")).upper()): task
        for task in tasks
    }
    if len(expected_tasks) != len(tasks):
        raise BenchmarkError("benchmark task source/ticker identities must be unique")
    expected_primary = {
        str(task["sample_id"])
        for task in tasks
        if int(task.get("attempt", 0)) == 1
    }
    if expected_primary != set(items):
        raise BenchmarkError("task manifest does not cover the partition exactly")
    rows_by_task: dict[str, dict[str, Any]] = {}
    seen_events: set[str] = set()
    for export in exports:
        if export.get("schema_version") != 5:
            raise BenchmarkError("benchmark labels require schema-v5 review exports")
        session = export.get("session")
        source = str((session or {}).get("source") or "")
        if not source:
            raise BenchmarkError("review export lacks its server-owned session source")
        reviewer = str((export.get("reviewer") or {}).get("name") or "").strip()
        if not reviewer:
            raise BenchmarkError("review export lacks its assigned reviewer")
        for wrapper in export.get("records") or []:
            if not isinstance(wrapper, dict) or not isinstance(wrapper.get("record"), dict):
                continue
            record = wrapper["record"]
            ticker = str(wrapper.get("ticker") or record.get("ticker") or "").strip().upper()
            task = expected_tasks.get((source, ticker))
            if task is None:
                raise BenchmarkError(f"unexpected review task {source}/{ticker}")
            event_id = str(wrapper.get("event_id") or record.get("eventId") or wrapper.get("id") or "")
            if not event_id:
                raise BenchmarkError(f"review event for {ticker} lacks an immutable id")
            if event_id in seen_events:
                continue
            seen_events.add(event_id)
            task_id = str(task["task_id"])
            if task_id in rows_by_task:
                raise BenchmarkError(f"task {task_id} has multiple finalized review events")
            provenance = record.get("provenance")
            if not isinstance(provenance, dict) or provenance.get("frozen") is not True:
                raise BenchmarkError(f"review event for {ticker} lacks server-owned provenance")
            required_provenance = {
                "source": source,
                "sampleId": task.get("workbench_sample_id"),
                "barsHash": task.get("workbench_bars_hash"),
                "dataDate": task.get("data_date"),
                "reviewOverrideApplied": False,
            }
            for key, expected in required_provenance.items():
                if provenance.get(key) != expected:
                    raise BenchmarkError(f"review event for {ticker} has mismatched {key}")
            if str(record.get("asOf") or "") != str(task.get("data_date") or ""):
                raise BenchmarkError(f"review event for {ticker} has mismatched asOf")
            blind = record.get("blindAssessment")
            if not isinstance(blind, dict):
                raise BenchmarkError(f"review event for {ticker} lacks blindAssessment")
            pattern = blind.get("patternLabel")
            if pattern not in {"coil", "not_coil", "uncertain"}:
                raise BenchmarkError(f"review event for {ticker} has invalid blind pattern label")
            tops = blind.get("humanTops")
            if not isinstance(tops, list):
                raise BenchmarkError(f"review event for {ticker} lacks blind human tops")
            normalized_tops = []
            for top in tops:
                if not isinstance(top, dict) or not top.get("date"):
                    raise BenchmarkError(f"review event for {ticker} has invalid blind top")
                normalized_tops.append(
                    {
                        "peak_date": _validate_date(str(top["date"])[:10], "blind human top"),
                        "price": float(top["price"]),
                        "role": top.get("role", "major_top"),
                        "lid_member": top.get("lidMember"),
                    }
                )
            band = blind.get("resistanceBand")
            normalized_band = None
            if isinstance(band, dict):
                normalized_band = {
                    "lower": float(band["lower"]),
                    "upper": float(band["upper"]),
                    "confidence": record.get("confidence", "low"),
                }
            timing = (record.get("detectorReview") or {}).get("timing") or {}
            item = items[str(task["sample_id"])]
            rows_by_task[task_id] = {
                    "sample_id": task["sample_id"],
                    "ticker": ticker,
                    "pattern_label": pattern,
                    "lifecycle_label": blind.get("lifecycleLabel"),
                    "tops": normalized_tops,
                    "band": normalized_band,
                    "blind_active_seconds": timing.get("blindActiveSeconds"),
                    "assisted_active_seconds": timing.get("assistedActiveSeconds"),
                    "timing_order": timing.get("reviewOrder"),
                    "source_event_id": event_id,
                    "source_session": source,
                    "source_task_id": task_id,
                    "reviewer": reviewer,
                    "source_created_at": wrapper.get("created_at") or record.get("createdAt"),
                }

    expected_task_ids = {str(task["task_id"]) for task in tasks}
    if set(rows_by_task) != expected_task_ids:
        missing = sorted(expected_task_ids - set(rows_by_task))
        raise BenchmarkError(f"label partition is incomplete: missing tasks={missing}")
    labels: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda value: str(value["task_id"])):
        row = rows_by_task[str(task["task_id"])]
        attempt = int(task["attempt"])
        row["attempt_id"] = str(task["task_id"])
        row["attempt_role"] = "primary" if attempt == 1 else "repeat"
        planned_order = task.get("timing_order")
        if attempt == 2 and row.get("timing_order") != planned_order:
            raise BenchmarkError(
                f"{task['task_id']} reviewOrder does not match the counterbalanced task plan"
            )
        labels.append(row)
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "kind": "coilingview.v24-benchmark-blind-labels",
        "benchmark_id": BENCHMARK_ID,
        "partition": partition,
        "manifest_sha256": manifest_sha256,
        "labels": labels,
    }


def _quarter_ordinal(value: str) -> int:
    parsed = date.fromisoformat(value[:10])
    return parsed.year * 4 + (parsed.month - 1) // 3


def maximum_top_match(
    human_dates: Sequence[str], detector_dates: Sequence[str], *, tolerance_quarters: int = 1
) -> dict[str, Any]:
    """Maximum one-to-one date match using a polynomial augmenting-path matcher."""
    human = [_validate_date(value[:10], "human top") for value in human_dates]
    detector = [_validate_date(value[:10], "detector top") for value in detector_dates]
    adjacency = [
        [
            index
            for index, candidate in enumerate(detector)
            if abs(_quarter_ordinal(reference) - _quarter_ordinal(candidate))
            <= tolerance_quarters
        ]
        for reference in human
    ]
    matched_human_for_detector: dict[int, int] = {}

    def augment(human_index: int, seen: set[int]) -> bool:
        candidates = sorted(
            adjacency[human_index],
            key=lambda index: (
                abs(
                    _quarter_ordinal(human[human_index])
                    - _quarter_ordinal(detector[index])
                ),
                detector[index],
            ),
        )
        for detector_index in candidates:
            if detector_index in seen:
                continue
            seen.add(detector_index)
            owner = matched_human_for_detector.get(detector_index)
            if owner is None or augment(owner, seen):
                matched_human_for_detector[detector_index] = human_index
                return True
        return False

    for human_index in range(len(human)):
        augment(human_index, set())
    pairs = sorted(
        (
            {"human": human[human_index], "detector": detector[detector_index]}
            for detector_index, human_index in matched_human_for_detector.items()
        ),
        key=lambda item: (item["human"], item["detector"]),
    )
    tp = len(pairs)
    fp = len(detector) - tp
    fn = len(human) - tp
    precision = tp / (tp + fp) if tp + fp else (1.0 if not human else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pairs": pairs,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def band_agrees(human: Mapping[str, Any], detector: Mapping[str, Any]) -> bool:
    try:
        h_lower, h_upper = float(human["lower"]), float(human["upper"])
        d_lower, d_upper = float(detector["lower"]), float(detector["upper"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (h_lower, h_upper, d_lower, d_upper)):
        return False
    if h_lower <= 0 or d_lower <= 0 or h_upper < h_lower or d_upper < d_lower:
        return False
    intersection = max(0.0, min(h_upper, d_upper) - max(h_lower, d_lower))
    union = max(h_upper, d_upper) - min(h_lower, d_lower)
    iou = intersection / union if union else 1.0
    h_centre = (h_lower + h_upper) / 2.0
    d_centre = (d_lower + d_upper) / 2.0
    centre_delta = abs(h_centre - d_centre) / h_centre
    return iou >= 0.50 and centre_delta <= 0.05


def _detector_top_dates(result: Mapping[str, Any], variant: str) -> list[str]:
    if variant == "v2_4_validation":
        candidates = result.get("top_candidates") or []
        return [
            str(item.get("peak_date") or item.get("date"))[:10]
            for item in candidates
            if isinstance(item, dict) and item.get("strict_major") is True
        ]
    return [
        str(item.get("peak_date") or item.get("date"))[:10]
        for item in result.get("major_highs") or []
        if isinstance(item, dict)
        and item.get("role") == "major_top"
        and (item.get("peak_date") or item.get("date"))
    ]


def _v24_positive(result: Mapping[str, Any]) -> bool:
    assessment = result.get("pattern_assessment") or {}
    return bool(
        assessment.get("structure_state") == "qualified"
        and assessment.get("confidence") == "high"
        and assessment.get("abstained") is False
    )


def _v23_positive(result: Mapping[str, Any]) -> bool:
    return bool(
        result.get("grade") is not None
        and result.get("lifecycle")
        in {"forming", "pre_breakout", "breaking_out", "post_breakout"}
        and not (result.get("analysis_metadata") or {}).get("classification_blocked")
    )


def _prediction_band(result: Mapping[str, Any], variant: str) -> dict[str, float] | None:
    if variant == "v2_4_validation":
        band = result.get("resistance_band")
        return dict(band) if isinstance(band, dict) else None
    resistance = result.get("resistance")
    if not isinstance(resistance, dict):
        return None
    centre = resistance.get("value_at_last_bar")
    if centre is None:
        return None
    centre = float(centre)
    return {"lower": centre * 0.965, "upper": centre * 1.035, "centre": centre}


def run_detectors(
    manifest: Mapping[str, Any], root: str | Path, *, config: ValidationConfig = V24_DEFAULT_CONFIG
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for item in manifest["items"]:
        sample = load_sample(root, item)
        bars = sample["monthly_bars"]
        ticker = str(sample["ticker"])
        as_of = str(sample["as_of"])
        outputs[str(sample["sample_id"])] = {
            "v2_3_1": analyze_coil(
                bars,
                ticker=ticker,
                as_of=as_of,
                variant=ANALYSIS_VARIANT_V2_3_1,
                mode=ANALYSIS_MODE_ALGORITHM_ONLY,
                adjustment_mode="split_adjusted",
            ),
            "v2_4_validation": analyze_coil_v24(
                bars,
                ticker=ticker,
                as_of=as_of,
                adjustment_mode="split_adjusted",
                config=config,
                mode=ANALYSIS_MODE_ALGORITHM_ONLY,
            ),
        }
    return outputs


def prediction_summary(
    manifest: Mapping[str, Any], outputs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Report paired operating behavior without treating sampling strata as truth."""
    result: dict[str, Any] = {}
    positives: dict[str, set[str]] = {}
    for variant in ("v2_3_1", "v2_4_validation"):
        positive_ids: set[str] = set()
        abstained = blocked = top_count = band_count = 0
        by_partition: dict[str, Counter[str]] = defaultdict(Counter)
        for item in manifest["items"]:
            sample_id = str(item["sample_id"])
            output = outputs[sample_id][variant]
            is_positive = (
                _v24_positive(output)
                if variant == "v2_4_validation"
                else _v23_positive(output)
            )
            if is_positive:
                positive_ids.add(sample_id)
            if output.get("abstained") is True:
                abstained += 1
            if (output.get("analysis_metadata") or {}).get("classification_blocked"):
                blocked += 1
            top_count += len(_detector_top_dates(output, variant))
            if _prediction_band(output, variant) is not None:
                band_count += 1
            by_partition[str(item["partition"])]["sample_count"] += 1
            by_partition[str(item["partition"])][
                "positive_count" if is_positive else "non_positive_count"
            ] += 1
        positives[variant] = positive_ids
        result[variant] = {
            "sample_count": len(manifest["items"]),
            "positive_count": len(positive_ids),
            "positive_rate": len(positive_ids) / len(manifest["items"]),
            "abstention_count": abstained,
            "classification_blocked_count": blocked,
            "accepted_top_count": top_count,
            "band_count": band_count,
            "by_partition": {key: dict(value) for key, value in by_partition.items()},
        }
    result["paired"] = {
        "both_positive": len(positives["v2_3_1"] & positives["v2_4_validation"]),
        "v2_3_1_only_positive": len(positives["v2_3_1"] - positives["v2_4_validation"]),
        "v2_4_only_positive": len(positives["v2_4_validation"] - positives["v2_3_1"]),
        "both_non_positive": len(manifest["items"])
        - len(positives["v2_3_1"] | positives["v2_4_validation"]),
    }
    result["warning"] = "Unlabeled operating behavior is not precision, recall, or ground truth."
    return result


def point_in_time_audit(
    manifest: Mapping[str, Any],
    root: str | Path,
    outputs: Mapping[str, Any],
    *,
    config: ValidationConfig = V24_DEFAULT_CONFIG,
) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    prefixes_checked = 0
    prefixes_checked_by_variant = {"v2_3_1": 0, "v2_4_validation": 0}
    for item in manifest["items"]:
        sample_id = str(item["sample_id"])
        sample = load_sample(root, item)
        as_of = str(sample["as_of"])
        for variant, result in outputs[sample_id].items():
            points = list(result.get("top_candidates") or []) + list(result.get("major_highs") or [])
            for point in points:
                peak_date = str(point.get("peak_date") or point.get("date") or "")[:10]
                confirmed_at = point.get("confirmed_at")
                if confirmed_at and str(confirmed_at)[:10] > as_of:
                    violations.append(
                        {"sample_id": sample_id, "variant": variant, "reason": "confirmation_after_as_of"}
                    )
                if confirmed_at and peak_date and str(confirmed_at)[:10] < peak_date:
                    violations.append(
                        {"sample_id": sample_id, "variant": variant, "reason": "confirmation_before_peak"}
                    )
        if item.get("cohort") == "point_in_time_lifecycle":
            replay = replay_completed_quarter_prefixes(
                sample["monthly_bars"], adjustment_mode="split_adjusted"
            )
            prefixes_checked += len(replay["snapshots"])
            prefixes_checked_by_variant["v2_3_1"] += len(replay["snapshots"])
            for snapshot in replay["snapshots"]:
                cutoff = snapshot["as_of"]
                for point in snapshot["analysis"].get("major_highs") or []:
                    confirmed = point.get("confirmed_at")
                    if confirmed is None or str(confirmed)[:10] > cutoff:
                        violations.append(
                            {"sample_id": sample_id, "variant": "v2_3_1", "reason": "prefix_evidence_leak"}
                        )
            bars = sample["monthly_bars"]
            ticker = str(sample["ticker"])
            for bar_index, bar in enumerate(bars):
                parsed = date.fromisoformat(str(bar["date"])[:10])
                if parsed.month not in {3, 6, 9, 12}:
                    continue
                cutoff = date(
                    parsed.year,
                    parsed.month,
                    calendar.monthrange(parsed.year, parsed.month)[1],
                ).isoformat()
                if cutoff > str(sample["as_of"]):
                    continue
                direct = analyze_coil_v24(
                    bars[: bar_index + 1],
                    ticker=ticker,
                    as_of=cutoff,
                    adjustment_mode="split_adjusted",
                    config=config,
                )
                replayed = analyze_coil_v24(
                    bars,
                    ticker=ticker,
                    as_of=cutoff,
                    adjustment_mode="split_adjusted",
                    config=config,
                )
                prefixes_checked += 1
                prefixes_checked_by_variant["v2_4_validation"] += 1
                stable_fields = (
                    "top_candidates",
                    "lid_hypotheses",
                    "resistance_band",
                    "pattern_assessment",
                    "structure_validity",
                    "readiness",
                    "abstained",
                )
                if any(direct.get(key) != replayed.get(key) for key in stable_fields):
                    violations.append(
                        {
                            "sample_id": sample_id,
                            "variant": "v2_4_validation",
                            "reason": "prefix_replay_mismatch",
                            "as_of": cutoff,
                        }
                    )
                for top in replayed.get("top_candidates") or []:
                    confirmed = top.get("confirmed_at")
                    if confirmed and str(confirmed)[:10] > cutoff:
                        violations.append(
                            {
                                "sample_id": sample_id,
                                "variant": "v2_4_validation",
                                "reason": "prefix_evidence_leak",
                                "as_of": cutoff,
                            }
                        )
    return {
        "violations": violations,
        "violation_count": len(violations),
        "prefixes_checked": prefixes_checked,
        "prefixes_checked_by_variant": prefixes_checked_by_variant,
    }


def label_stability(labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for label in labels:
        by_sample[str(label.get("sample_id") or "")].append(label)
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for attempts in by_sample.values():
        primary = [row for row in attempts if row.get("attempt_role") == "primary"]
        repeat = [row for row in attempts if row.get("attempt_role") == "repeat"]
        if len(primary) == 1 and len(repeat) == 1:
            pairs.append((primary[0], repeat[0]))
    pattern_agreements: list[bool] = []
    top_scores: list[float] = []
    band_agreements: list[bool] = []
    timing_reductions: list[float] = []
    timing_orders: Counter[str] = Counter()
    for first, second in pairs:
        pattern_agreements.append(first.get("pattern_label") == second.get("pattern_label"))
        top_scores.append(
            maximum_top_match(
                [
                    str(item["peak_date"])
                    for item in first.get("tops") or []
                    if item.get("role", "major_top") == "major_top"
                ],
                [
                    str(item["peak_date"])
                    for item in second.get("tops") or []
                    if item.get("role", "major_top") == "major_top"
                ],
            )["f1"]
        )
        first_band, second_band = first.get("band"), second.get("band")
        if (
            isinstance(first_band, dict)
            and isinstance(second_band, dict)
            and first_band.get("confidence") == "high"
            and second_band.get("confidence") == "high"
        ):
            band_agreements.append(band_agrees(first_band, second_band))
        blind = second.get("blind_active_seconds")
        assisted = second.get("assisted_active_seconds")
        timing_order = second.get("timing_order")
        if (
            isinstance(blind, (int, float))
            and isinstance(assisted, (int, float))
            and blind > 0
            and timing_order in {"blind_first", "assisted_first"}
        ):
            timing_reductions.append((float(blind) - float(assisted)) / float(blind))
            timing_orders[str(timing_order)] += 1
    result = {
        "repeat_pair_count": len(pairs),
        "pattern_agreement": sum(pattern_agreements) / len(pattern_agreements) if pattern_agreements else None,
        "matched_top_f1": statistics.mean(top_scores) if top_scores else None,
        "high_confidence_band_pair_count": len(band_agreements),
        "band_agreement": sum(band_agreements) / len(band_agreements) if band_agreements else None,
        "timed_repeat_count": len(timing_reductions),
        "timing_order_counts": dict(timing_orders),
        "counterbalanced": timing_orders == Counter({"blind_first": 6, "assisted_first": 6}),
        "median_review_time_reduction": (
            statistics.median(timing_reductions) if timing_reductions else None
        ),
    }
    result["passed"] = bool(
        len(pairs) >= 12
        and result["pattern_agreement"] is not None
        and result["pattern_agreement"] >= GATES["label_pattern_agreement"]
        and result["matched_top_f1"] is not None
        and result["matched_top_f1"] >= GATES["label_top_f1"]
        and result["band_agreement"] is not None
        and result["band_agreement"] >= GATES["label_band_agreement"]
    )
    return result


def registered_validation_configs() -> list[ValidationConfig]:
    """The only 54 configurations admitted to the lean development sweep."""
    return [
        ValidationConfig(
            zone_candidate_prominence_pct=prominence,
            zone_similarity_pct=similarity,
            touch_tolerance_pct=tolerance,
            max_qualifying_lid_slope_pct_per_year=slope,
        )
        for prominence, similarity, tolerance, slope in product(
            (15.0, 18.5, 22.5),
            (3.5, 5.0, 7.5),
            (2.5, 3.5, 5.0),
            (6.5, 12.0),
        )
    ]


def select_development_configuration(
    scored: Sequence[tuple[ValidationConfig, Mapping[str, Any]]]
) -> tuple[ValidationConfig, Mapping[str, Any]]:
    """Select by the frozen lexicographic objective, with a stable hash tie-break."""
    if not scored:
        raise BenchmarkError("development configuration sweep produced no scores")

    def value(metrics: Mapping[str, Any], name: str) -> float:
        raw = metrics.get(name)
        return float(raw) if isinstance(raw, (int, float)) and math.isfinite(raw) else -1.0

    return max(
        scored,
        key=lambda row: (
            value(row[1], "major_top_precision"),
            value(row[1], "pattern_precision"),
            value(row[1], "band_agreement"),
            value(row[1], "major_top_recall"),
            # Smaller fingerprint wins an otherwise exact tie.
            tuple(-ord(char) for char in config_fingerprint(row[0])),
        ),
    )


def validate_selection_report(
    report: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    current_code_commit: str,
    recompute_metrics: bool = True,
) -> ValidationConfig:
    """Validate the immutable 54-config sweep and single validation receipt.

    Freeze creation and final evaluation use the default full recomputation.
    Runtime holdout-session creation may set ``recompute_metrics=False`` only
    after the freeze has hash-bound this already-validated receipt, avoiding a
    minute-long synchronous detector sweep on every API retry.
    """
    if report.get("schema_version") != 1 or report.get("kind") != (
        "coilingview.v24-development-selection"
    ):
        raise BenchmarkError("invalid validation-selection receipt")
    if report.get("benchmark_id") != BENCHMARK_ID:
        raise BenchmarkError("validation-selection benchmark mismatch")
    if report.get("manifest_sha256") != file_sha256(manifest_path):
        raise BenchmarkError("validation-selection manifest mismatch")
    if report.get("protocol_sha256") != file_sha256(protocol_path):
        raise BenchmarkError("validation-selection protocol mismatch")
    if (
        len(current_code_commit) != 40
        or report.get("code_commit") != current_code_commit.lower()
    ):
        raise BenchmarkError("validation-selection code commit mismatch")
    stability = report.get("label_stability") or {}
    if stability.get("repeat_pair_count") != 12 or stability.get("passed") is not True:
        raise BenchmarkError("validation-selection label stability did not pass")
    registered = registered_validation_configs()
    expected_fingerprints = {config_fingerprint(config) for config in registered}
    sweep = report.get("development_sweep")
    if not isinstance(sweep, list) or len(sweep) != len(registered):
        raise BenchmarkError("validation-selection must contain all 54 registered configs")
    seen: dict[str, Mapping[str, Any]] = {}
    for row in sweep:
        if not isinstance(row, dict) or not isinstance(row.get("config"), dict):
            raise BenchmarkError("validation-selection sweep row is invalid")
        config = ValidationConfig(**row["config"])
        fingerprint = config_fingerprint(config)
        if row.get("config_fingerprint") != fingerprint or fingerprint in seen:
            raise BenchmarkError("validation-selection sweep fingerprints are invalid")
        if not isinstance(row.get("development_metrics"), dict):
            raise BenchmarkError("validation-selection sweep metrics are missing")
        seen[fingerprint] = row
    if set(seen) != expected_fingerprints:
        raise BenchmarkError("validation-selection sweep is not the registered grid")
    expected_config, expected_metrics = select_development_configuration(
        [
            (ValidationConfig(**row["config"]), row["development_metrics"])
            for row in sweep
        ]
    )
    selected = report.get("selected") or {}
    if not isinstance(selected.get("config"), dict):
        raise BenchmarkError("validation-selection selected config is missing")
    config = ValidationConfig(**selected["config"])
    fingerprint = config_fingerprint(config)
    if selected.get("config_fingerprint") != fingerprint:
        raise BenchmarkError("validation-selection selected fingerprint mismatch")
    if selected.get("development_metrics") != seen[fingerprint]["development_metrics"]:
        raise BenchmarkError("validation-selection selected metrics mismatch")
    if config != expected_config or selected.get("development_metrics") != expected_metrics:
        raise BenchmarkError("validation-selection did not select the frozen lexicographic winner")
    validation_metrics = selected.get("single_use_validation_metrics") or {}
    if validation_metrics.get("labeled_sample_count") != 18:
        raise BenchmarkError("validation-selection lacks the 18-sample validation audit")
    report_root = Path(manifest_path).parent
    label_artifacts = (
        ("development_labels_sha256", report_root / "development-labels.json"),
        ("validation_labels_sha256", report_root / "validation-labels.json"),
    )
    for key, label_path in label_artifacts:
        if not label_path.is_file() or not hmac.compare_digest(
            str(report.get(key) or ""), file_sha256(label_path)
        ):
            raise BenchmarkError(f"validation-selection {key} mismatch")

    if not recompute_metrics:
        return config

    # Never trust reported sweep/validation metrics. Recompute every registered
    # development row and the selected validation audit from the exact label
    # artifacts that are hash-bound above.
    manifest = load_manifest(manifest_path)
    development_labels = load_labels(
        report_root / "development-labels.json",
        partition="development",
        manifest_path=manifest_path,
        protocol_path=protocol_path,
    )
    validation_labels = load_labels(
        report_root / "validation-labels.json",
        partition="validation",
        manifest_path=manifest_path,
        protocol_path=protocol_path,
    )
    development_manifest = {
        "items": [
            item for item in manifest["items"] if item["partition"] == "development"
        ]
    }
    recomputed_scores: list[tuple[ValidationConfig, Mapping[str, Any]]] = []
    for fingerprint_value, row in seen.items():
        row_config = ValidationConfig(**row["config"])
        outputs = run_detectors(development_manifest, report_root, config=row_config)
        recomputed = detector_metrics(
            development_labels["labels"],
            outputs,
            variant="v2_4_validation",
            manifest=development_manifest,
        )
        if recomputed != row["development_metrics"]:
            raise BenchmarkError(
                f"validation-selection development metrics mismatch for {fingerprint_value}"
            )
        recomputed_scores.append((row_config, recomputed))
    recomputed_winner, recomputed_winner_metrics = select_development_configuration(
        recomputed_scores
    )
    if config != recomputed_winner or selected.get(
        "development_metrics"
    ) != recomputed_winner_metrics:
        raise BenchmarkError("validation-selection recomputed winner mismatch")
    validation_manifest = {
        "items": [
            item for item in manifest["items"] if item["partition"] == "validation"
        ]
    }
    validation_outputs = run_detectors(
        validation_manifest, report_root, config=recomputed_winner
    )
    recomputed_validation = detector_metrics(
        validation_labels["labels"],
        validation_outputs,
        variant="v2_4_validation",
        manifest=validation_manifest,
    )
    if recomputed_validation != validation_metrics:
        raise BenchmarkError("validation-selection validation metrics mismatch")
    return config


def detector_metrics(
    labels: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any]],
    *,
    variant: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    top_tp = top_fp = top_fn = 0
    definite_positive = definite_negative = predicted_positive_count = false_positive_count = 0
    clear_negative_count = clear_negative_false_positive_count = 0
    positive_predictions_on_human_positive = 0
    band_denominator = band_numerator = 0
    paired_times: list[tuple[float, float]] = []
    labeled_samples = 0
    manifest_items = {
        str(item["sample_id"]): item for item in manifest.get("items") or []
    }
    for label in labels:
        if label.get("attempt_role", "primary") != "primary":
            continue
        sample_id = str(label.get("sample_id") or "")
        if sample_id not in outputs:
            raise BenchmarkError(f"missing detector output for {sample_id}")
        item = manifest_items.get(sample_id)
        if item is None:
            raise BenchmarkError(f"missing manifest item for {sample_id}")
        result = outputs[sample_id][variant]
        labeled_samples += 1
        match = maximum_top_match(
            [
                str(top["peak_date"])
                for top in label.get("tops") or []
                if top.get("role", "major_top") == "major_top"
            ],
            _detector_top_dates(result, variant),
        )
        top_tp += match["true_positive"]
        top_fp += match["false_positive"]
        top_fn += match["false_negative"]
        human_pattern = label.get("pattern_label")
        predicted_positive = _v24_positive(result) if variant == "v2_4_validation" else _v23_positive(result)
        if predicted_positive:
            predicted_positive_count += 1
        if human_pattern == "coil":
            definite_positive += 1
            if predicted_positive:
                positive_predictions_on_human_positive += 1
            human_band = label.get("band")
            if isinstance(human_band, dict) and human_band.get("confidence") == "high":
                band_denominator += 1
                detector_band = _prediction_band(result, variant)
                if detector_band is not None and band_agrees(human_band, detector_band):
                    band_numerator += 1
        elif human_pattern == "not_coil":
            definite_negative += 1
            if predicted_positive:
                false_positive_count += 1
            if item.get("cohort") == "clear_negative":
                clear_negative_count += 1
                if predicted_positive:
                    clear_negative_false_positive_count += 1
        blind = label.get("blind_active_seconds")
        assisted = label.get("assisted_active_seconds")
        if isinstance(blind, (int, float)) and isinstance(assisted, (int, float)) and blind > 0:
            paired_times.append((float(blind), float(assisted)))

    precision_denominator = positive_predictions_on_human_positive + false_positive_count
    pattern_precision = (
        positive_predictions_on_human_positive / precision_denominator
        if precision_denominator
        else None
    )
    reductions = [(blind - assisted) / blind for blind, assisted in paired_times]
    return {
        "labeled_sample_count": labeled_samples,
        "major_top_precision": top_tp / (top_tp + top_fp) if top_tp + top_fp else None,
        "major_top_recall": top_tp / (top_tp + top_fn) if top_tp + top_fn else None,
        "major_top_f1": (2 * top_tp / (2 * top_tp + top_fp + top_fn)) if (2 * top_tp + top_fp + top_fn) else None,
        "band_agreement": band_numerator / band_denominator if band_denominator else None,
        "band_denominator": band_denominator,
        "pattern_precision": pattern_precision,
        "automated_positive_coverage": positive_predictions_on_human_positive / definite_positive if definite_positive else None,
        "clear_negative_false_positive_rate": (
            clear_negative_false_positive_count / clear_negative_count
            if clear_negative_count
            else None
        ),
        "clear_negative_count": clear_negative_count,
        "predicted_positive_count": predicted_positive_count,
        "definite_positive_count": definite_positive,
        "definite_negative_count": definite_negative,
        "timed_pair_count": len(paired_times),
        "median_review_time_reduction": statistics.median(reductions) if reductions else None,
    }


def gate_decision(
    *,
    manifest_validation: Mapping[str, Any],
    stability: Mapping[str, Any] | None,
    v24_metrics: Mapping[str, Any] | None,
    point_in_time: Mapping[str, Any],
    accepted_hard_invalid: int,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def minimum(name: str, value: Any, threshold: float) -> None:
        checks[name] = {
            "value": value,
            "threshold": threshold,
            "operator": ">=",
            "passed": value is not None and value >= threshold,
        }

    def maximum(name: str, value: Any, threshold: float) -> None:
        checks[name] = {
            "value": value,
            "threshold": threshold,
            "operator": "<=",
            "passed": value is not None and value <= threshold,
        }

    checks["manifest_valid"] = {
        "value": bool(manifest_validation.get("valid")),
        "threshold": True,
        "operator": "==",
        "passed": bool(manifest_validation.get("valid")),
    }
    stability_ready = stability is not None and stability.get("repeat_pair_count", 0) >= 12
    checks["label_stability_ready"] = {
        "value": stability.get("repeat_pair_count") if stability else 0,
        "threshold": 12,
        "operator": ">=",
        "passed": stability_ready,
    }
    if stability is not None:
        minimum("label_pattern_agreement", stability.get("pattern_agreement"), GATES["label_pattern_agreement"])
        minimum("label_top_f1", stability.get("matched_top_f1"), GATES["label_top_f1"])
        minimum("label_band_agreement", stability.get("band_agreement"), GATES["label_band_agreement"])
    metrics_ready = v24_metrics is not None and v24_metrics.get("labeled_sample_count") == 18
    checks["holdout_metrics_ready"] = {
        "value": v24_metrics.get("labeled_sample_count") if v24_metrics else 0,
        "threshold": 18,
        "operator": "==",
        "passed": metrics_ready,
    }
    if v24_metrics is not None:
        checks["clear_negative_labels_ready"] = {
            "value": v24_metrics.get("clear_negative_count"),
            "threshold": 3,
            "operator": ">=",
            "passed": (v24_metrics.get("clear_negative_count") or 0) >= 3,
        }
        minimum("major_top_precision", v24_metrics.get("major_top_precision"), GATES["major_top_precision"])
        minimum("major_top_recall", v24_metrics.get("major_top_recall"), GATES["major_top_recall"])
        minimum("band_agreement", v24_metrics.get("band_agreement"), GATES["band_agreement"])
        minimum("pattern_precision", v24_metrics.get("pattern_precision"), GATES["pattern_precision"])
        minimum("automated_positive_coverage", v24_metrics.get("automated_positive_coverage"), GATES["automated_positive_coverage"])
        maximum(
            "clear_negative_false_positive_rate",
            v24_metrics.get("clear_negative_false_positive_rate"),
            GATES["clear_negative_false_positive_rate_max"],
        )
    if stability is not None:
        checks["review_time_counterbalanced"] = {
            "value": bool(stability.get("counterbalanced")),
            "threshold": True,
            "operator": "==",
            "passed": bool(stability.get("counterbalanced")),
        }
        minimum(
            "review_time_reduction",
            stability.get("median_review_time_reduction"),
            GATES["review_time_reduction"],
        )
    maximum("point_in_time_violations", point_in_time.get("violation_count"), 0)
    maximum("accepted_hard_invalid", accepted_hard_invalid, 0)

    if not manifest_validation.get("valid") or point_in_time.get("violation_count", 0) > 0 or accepted_hard_invalid > 0:
        outcome = "no_go"
        reason = "hard_integrity_gate_failed"
    elif not stability_ready or not metrics_ready:
        outcome = "inconclusive"
        reason = "required_blind_labels_or_repeats_are_not_complete"
    elif stability is not None and not stability.get("passed"):
        outcome = "inconclusive"
        reason = "label_quality_gate_failed"
    elif all(check["passed"] for check in checks.values()):
        outcome = "go"
        reason = "all_frozen_holdout_gates_passed"
    else:
        outcome = "no_go"
        reason = "one_or_more_performance_gates_failed"
    return {
        "outcome": outcome,
        "promotion_authorized": outcome == "go",
        "reason": reason,
        "checks": checks,
    }
