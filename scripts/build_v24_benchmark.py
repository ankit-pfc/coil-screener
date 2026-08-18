#!/usr/bin/env python3
"""Freeze the 72 canonical v2.4 samples from verified listing history."""
from __future__ import annotations

import argparse
import copy
import csv
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
from benchmark_v24 import (  # noqa: E402
    BENCHMARK_ID,
    COHORTS,
    MANIFEST_SCHEMA_VERSION,
    canonical_json,
    history_coverage_audit,
    sha256_json,
    stable_sample_id,
    stable_task_id,
    validate_manifest,
)
from coil_analysis import ALGORITHM_VERSION  # noqa: E402
from review_snapshots import (  # noqa: E402
    REVIEW_CORPUS_MANIFEST_KIND,
    review_snapshot_identity,
)

SOURCE = "benchmark_2026-08-13_v24_72.csv"
CORPUS_DIR = PROJECT_ROOT / "review_snapshots" / Path(SOURCE).stem
SPEC_PATH = CORPUS_DIR / "benchmark-spec.json"
PROTOCOL_PATH = CORPUS_DIR / "protocol.json"
QUEUE_SALT = "coilingview-v24-repeat-queue-v1"
SESSION_SOURCES = {
    "development": "benchmark_2026-08-13_v24_72_batch_a.csv",
    "repeat": "benchmark_2026-08-13_v24_72_batch_b.csv",
    "validation": "benchmark_2026-08-13_v24_72_batch_c.csv",
    "holdout": "benchmark_2026-08-13_v24_72_batch_d.csv",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _load_long_history_corpus(
    root: Path,
    planned: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"long-history manifest is unreadable: {manifest_path}") from exc
    if manifest.get("kind") != "coilingview.long-history-research-manifest":
        raise SystemExit("long-history manifest kind is invalid")
    manifest_items = {
        str(item.get("ticker") or "").strip().upper(): item
        for item in manifest.get("items") or []
        if isinstance(item, dict)
    }
    expected = {str(item["ticker"]) for item in planned}
    if set(manifest_items) != expected:
        missing = sorted(expected - set(manifest_items))
        extra = sorted(set(manifest_items) - expected)
        raise SystemExit(
            "long-history ticker set mismatch: "
            + canonical_json({"missing": missing, "extra": extra})
        )
    snapshots: dict[str, dict[str, Any]] = {}
    failures: dict[str, list[str]] = {}
    for item in planned:
        ticker = str(item["ticker"])
        manifest_item = manifest_items[ticker]
        snapshot_path = (root / str(manifest_item.get("snapshot_file") or "")).resolve()
        if root not in snapshot_path.parents:
            failures[ticker] = ["snapshot path escapes the long-history root"]
            continue
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures[ticker] = ["snapshot is missing or unreadable"]
            continue
        expected_snapshot_hash = str(manifest_item.get("snapshot_sha256") or "")
        if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != expected_snapshot_hash:
            failures[ticker] = ["snapshot hash does not match the long-history manifest"]
            continue
        bars = payload.get("monthly_bars")
        if (
            payload.get("kind") != "coilingview.long-history-research-snapshot"
            or payload.get("ticker") != ticker
            or payload.get("as_of") != item["as_of"]
            or not isinstance(bars, list)
            or sha256_json(bars) != payload.get("bars_sha256")
        ):
            failures[ticker] = ["snapshot identity, cutoff, or bars hash is invalid"]
            continue
        audit_item = {
            "ticker": ticker,
            "as_of": item["as_of"],
            "company_name": payload.get("company_name"),
            "security": payload.get("security"),
            "security_identity_sha256": payload.get("security_identity_sha256"),
            "coverage": payload.get("coverage"),
            "coverage_sha256": payload.get("coverage_sha256"),
        }
        audit = history_coverage_audit({"items": [audit_item]})
        if not audit["valid"]:
            failures[ticker] = list(audit["errors"])
            continue
        snapshots[ticker] = payload
    if failures:
        raise SystemExit(
            "long-history coverage failures: " + canonical_json(failures)
        )
    return snapshots


def _items_from_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    partition_by_index = ["development"] * 12 + ["validation"] * 6 + ["holdout"] * 6
    lifecycle_cutoffs = [
        "2022-09-30",
        "2023-03-31",
        "2023-09-30",
        "2024-03-31",
        "2024-09-30",
        "2025-03-31",
    ]
    for market, tickers in spec["markets"].items():
        if len(tickers) != 24:
            raise SystemExit(f"{market} must define exactly 24 tickers")
        lifecycle_index = 0
        for index, raw in enumerate(tickers):
            ticker = str(raw["ticker"]).strip().upper()
            cohort = COHORTS[index % len(COHORTS)]
            as_of = "2026-06-30"
            if cohort == "point_in_time_lifecycle":
                as_of = lifecycle_cutoffs[lifecycle_index]
                lifecycle_index += 1
            items.append(
                {
                    "ticker": ticker,
                    "ticker_family": str(raw.get("ticker_family") or ticker).strip().upper(),
                    "market": market,
                    "partition": partition_by_index[index],
                    "cohort": cohort,
                    "as_of": as_of,
                    "prior_review_or_tuning": bool(raw.get("prior_review_or_tuning", False)),
                }
            )
    return items


def _write_workbench_source(
    *,
    source: str,
    partition: str,
    items: list[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
    fetched_at: str,
    attempt: int,
) -> list[dict[str, Any]]:
    """Materialize one unique-ticker session so partitions never share exports."""
    root = PROJECT_ROOT / "review_snapshots" / Path(source).stem
    session_items: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for position, item in enumerate(items, start=1):
        ticker = item["ticker"]
        snapshot = copy.deepcopy(snapshots[ticker])
        snapshot["source"] = source
        snapshot["run"] = {
            "algorithm_version": ALGORITHM_VERSION,
            "code_sha": None,
            "source_csv_sha256": None,
            "universe_position": position,
        }
        task_id = stable_task_id(item["sample_id"], attempt, QUEUE_SALT)
        snapshot["corpus_labels"].update(
            {
                "benchmark_task_id": task_id,
                "benchmark_sample_id": item["sample_id"],
                "benchmark_attempt": attempt,
                "benchmark_partition": partition,
            }
        )
        path = root / f"{ticker}.json"
        _write_json(path, snapshot)
        snapshot_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        session_item = {
            **item,
            "snapshot_sha256": snapshot_hash,
            "screen_snapshot_sha256": sha256_json(snapshot["screen_snapshot"]),
            "backend_bars_identity_sha256": item["provenance"][
                "backend_bars_identity_sha256"
            ],
            "benchmark_sample_id": item["sample_id"],
            "benchmark_task_id": task_id,
            "benchmark_attempt": attempt,
            "sample_file": path.name,
        }
        session_items.append(session_item)
        task_rows.append(
            {
                "task_id": task_id,
                "source": source,
                "ticker": ticker,
                "as_of": item["as_of"],
                "attempt": attempt,
                "sample_id": item["sample_id"],
                "bars_sha256": item["provenance"]["bars_sha256"],
                "partition": partition,
            }
        )
    session_manifest = {
        "schema_version": 1,
        "kind": REVIEW_CORPUS_MANIFEST_KIND,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_partition": partition,
        "benchmark_canonical_source": SOURCE,
        "source": source,
        "generated_at": fetched_at,
        "source_run": {
            "filename": source,
            "algorithm_version": ALGORITHM_VERSION,
            "fetched_at": fetched_at,
            "code_sha": None,
        },
        "canonicalization": {
            "adjustment_mode": "split_adjusted",
            "adjustment_source": "per_snapshot_split_events",
            "source_interval": "1d",
            "adjustment_transform_version": "per_snapshot",
            "as_of_policy": "completed quarter only",
            "history_policy": "verified_listing_quarter_to_date",
        },
        "items": session_items,
    }
    _write_json(root / "manifest.json", session_manifest)
    rows = [
        {
            "ticker": item["ticker"],
            "data_date": item["last_data_date"],
            "partition": partition,
            "cohort": item["cohort"],
            "task_id": item["benchmark_task_id"],
        }
        for item in session_items
    ]
    with (root / "source_run.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for task in task_rows:
        identity = review_snapshot_identity(source, task["ticker"])
        task["workbench_sample_id"] = identity["sample_id"]
        task["workbench_bars_hash"] = identity["bars_hash"]
        task["data_date"] = identity["data_date"]
    return task_rows


def _materialize_workbench(
    manifest_items: list[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
    *,
    fetched_at: str,
) -> None:
    tasks: list[dict[str, Any]] = []
    for partition in ("development", "validation", "holdout"):
        tasks.extend(
            _write_workbench_source(
                source=SESSION_SOURCES[partition],
                partition=partition,
                items=[item for item in manifest_items if item["partition"] == partition],
                snapshots=snapshots,
                fetched_at=fetched_at,
                attempt=1,
            )
        )
    repeat_sources = [
        item for item in manifest_items if item["partition"] == "development"
    ][:12]
    tasks.extend(
        _write_workbench_source(
            source=SESSION_SOURCES["repeat"],
            partition="development_repeat",
            items=repeat_sources,
            snapshots=snapshots,
            fetched_at=fetched_at,
            attempt=2,
        )
    )
    tasks.sort(key=lambda row: (row["source"], row["task_id"]))
    _write_json(
        CORPUS_DIR / "review-task-manifest.internal.json",
        {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "task_count": 84,
            "repeat_count": 12,
            "reviewer_payload_must_omit": [
                "attempt",
                "sample_id",
                "repeat_of",
                "timing_order",
            ],
            "tasks": tasks,
        },
    )
    reviewer_queue = [
        {key: task[key] for key in ("task_id", "source", "ticker", "as_of")}
        for task in tasks
    ]
    _write_json(
        CORPUS_DIR / "review-queue.json",
        {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "task_count": 84,
            "tasks": reviewer_queue,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long-history-root",
        type=Path,
        help="72-ticker output from build_long_history_pilot.py, with each ticker frozen at its benchmark cutoff",
    )
    parser.add_argument(
        "--workbench-only",
        action="store_true",
        help="Rebuild the four review sources from the frozen canonical samples without downloading data.",
    )
    parser.add_argument("--fetched-at", default=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    old_manifest_path = CORPUS_DIR / "manifest.json"
    old_manifest = (
        json.loads(old_manifest_path.read_text(encoding="utf-8"))
        if old_manifest_path.exists()
        else None
    )
    if spec.get("benchmark_id") != BENCHMARK_ID or protocol.get("benchmark_id") != BENCHMARK_ID:
        raise SystemExit("benchmark spec/protocol identity mismatch")
    if args.workbench_only:
        if not isinstance(old_manifest, dict):
            raise SystemExit("workbench-only rebuild requires the canonical manifest")
        validation = validate_manifest(old_manifest)
        if not validation["valid"]:
            raise SystemExit("canonical manifest validation failed: " + "; ".join(validation["errors"]))
        coverage = history_coverage_audit(old_manifest)
        if not coverage["valid"]:
            raise SystemExit(
                "workbench-only rebuild requires verified listing-quarter history"
            )
        manifest_items = list(old_manifest["items"])
        snapshots = {
            item["ticker"]: json.loads(
                (CORPUS_DIR / item["sample_file"]).read_text(encoding="utf-8")
            )
            for item in manifest_items
        }
        _materialize_workbench(
            manifest_items,
            snapshots,
            fetched_at=str(old_manifest.get("generated_at") or args.fetched_at),
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    planned = _items_from_spec(spec)
    holdout_leaks = [item["ticker"] for item in planned if item["partition"] == "holdout" and item["prior_review_or_tuning"]]
    if holdout_leaks:
        raise SystemExit(f"prior review/tuning names leaked into holdout: {holdout_leaks}")

    if args.long_history_root is None:
        raise SystemExit(
            "--long-history-root is required; the benchmark cannot use an unverified provider maximum"
        )
    long_history = _load_long_history_corpus(args.long_history_root, planned)

    prepared: dict[
        str, tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]
    ] = {}
    quality_failures: dict[str, Any] = {}
    for item in planned:
        ticker = item["ticker"]
        history = long_history[ticker]
        bars = list(history["monthly_bars"])
        inspected = inspect_monthly_bars(
            bars,
            as_of=item["as_of"],
            adjustment_mode="split_adjusted",
            today=date(2026, 8, 13),
        )
        quality = inspected.report
        if quality.get("blocked"):
            quality_failures[ticker] = quality
            continue
        bars = inspected.bars
        if not bars or bars[-1]["date"][:7] != item["as_of"][:7]:
            quality_failures[ticker] = {
                "reason": "missing_completed_cutoff_month",
                "expected": item["as_of"][:7],
                "actual": bars[-1]["date"][:7] if bars else None,
            }
            continue
        prepared[ticker] = (bars, quality, history)
    if quality_failures:
        raise SystemExit("strict OHLC/cutoff failures: " + canonical_json(quality_failures))

    manifest_items: list[dict[str, Any]] = []
    source_rows: list[dict[str, str]] = []
    for position, item in enumerate(planned, start=1):
        ticker = item["ticker"]
        bars, quality, history = prepared[ticker]
        bars_hash = sha256_json(bars)
        sample_id = stable_sample_id(ticker, item["as_of"], bars_hash)
        source_fingerprint = sha256_json(
            {
                "provider": history.get("provider"),
                "ticker": ticker,
                "fetched_at": args.fetched_at,
                "adjustment_mode": "split_adjusted",
                "security_identity_sha256": history["security_identity_sha256"],
                "coverage_sha256": history["coverage_sha256"],
                "bars_sha256": bars_hash,
            }
        )
        screen_snapshot = {
            "ticker": ticker,
            "data_date": bars[-1]["date"],
            "fetched_at": args.fetched_at,
            "data_source": str((history.get("provider") or {}).get("name") or "frozen_long_history"),
            "freshness": "frozen_benchmark",
            "status": "blind_unlabeled",
            "grade": None,
        }
        snapshot = {
            "schema_version": 1,
            "kind": "coilingview.saved-run-review-snapshot",
            "source": SOURCE,
            "ticker": ticker,
            "company_name": history["company_name"],
            "sample_id": sample_id,
            "as_of": item["as_of"],
            "run": {
                "algorithm_version": ALGORITHM_VERSION,
                "code_sha": None,
                "source_csv_sha256": None,
                "universe_position": position,
            },
            "screen_snapshot": screen_snapshot,
            "corpus_labels": {
                "benchmark_id": BENCHMARK_ID,
                "partition": item["partition"],
                "cohort": item["cohort"],
                "cohort_status": "pending_blind_confirmation",
                "ground_truth": None,
            },
            "data_quality": {
                "reviewable": True,
                "status": quality["status"],
                "expected_hard_invalid": False,
                "strict_report": quality,
            },
            "security": history["security"],
            "security_identity_sha256": history["security_identity_sha256"],
            "coverage": history["coverage"],
            "coverage_sha256": history["coverage_sha256"],
            "provenance": {
                "bars_sha256": bars_hash,
                "source_fingerprint": source_fingerprint,
            },
            "source_cache_metadata": {
                "schema_version": 4,
                "fetched_at": args.fetched_at,
                "last_bar_date": bars[-1]["date"],
                "source": str((history.get("provider") or {}).get("name") or "frozen_long_history"),
                "adjustment_mode": "split_adjusted",
                "adjustment_source": (history.get("provider") or {}).get("adjustment_source"),
                "source_interval": (history.get("provider") or {}).get("source_interval"),
                "adjustment_transform_version": (history.get("provider") or {}).get("adjustment_transform_version"),
            },
            "monthly_bars": bars,
        }
        path = CORPUS_DIR / f"{ticker}.json"
        _write_json(path, snapshot)
        sample_file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        backend_bars_hash = sha256_json(
            {"schema_version": 1, "interval": "1M", "ticker": ticker, "bars": bars}
        )
        manifest_items.append(
            {
                **item,
                "company_name": history["company_name"],
                "security": history["security"],
                "security_identity_sha256": history["security_identity_sha256"],
                "coverage": history["coverage"],
                "coverage_sha256": history["coverage_sha256"],
                "sample_id": sample_id,
                "sample_file": path.name,
                "bar_count": len(bars),
                "first_data_date": bars[0]["date"],
                "last_data_date": bars[-1]["date"],
                "adjustment_mode": "split_adjusted",
                "new_or_remediated": True,
                "history_start_policy": {"kind": "verified_listing_quarter_to_date"},
                "data_quality": {
                    "reviewable": True,
                    "status": quality["status"],
                    "expected_hard_invalid": False,
                    "issue_count": len(quality.get("issues") or []),
                },
                "provenance": {
                    "bars_sha256": bars_hash,
                    "sample_file_sha256": sample_file_hash,
                    "source_fingerprint": source_fingerprint,
                    "backend_bars_identity_sha256": backend_bars_hash,
                },
                "backend_bars_identity_sha256": backend_bars_hash,
                "snapshot_sha256": sample_file_hash,
                "screen_snapshot_sha256": sha256_json(screen_snapshot),
            }
        )
        source_rows.append(
            {
                "ticker": ticker,
                "data_date": bars[-1]["date"],
                "partition": item["partition"],
                "cohort": item["cohort"],
                "sample_id": sample_id,
                "company_name": history["company_name"],
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "coilingview.v24-benchmark-canonical-manifest",
        "benchmark_id": BENCHMARK_ID,
        "source": SOURCE,
        "generated_at": args.fetched_at,
        "source_run": {
            "filename": SOURCE,
            "algorithm_version": ALGORITHM_VERSION,
            "fetched_at": args.fetched_at,
            "code_sha": None,
        },
        "canonicalization": {
            "adjustment_mode": "split_adjusted",
            "adjustment_source": "per_snapshot_split_events",
            "source_interval": "1d",
            "adjustment_transform_version": "per_snapshot",
            "as_of_policy": "completed quarter only",
            "history_policy": "verified_listing_quarter_to_date",
        },
        "items": manifest_items,
    }
    validation = validate_manifest(manifest)
    if not validation["valid"]:
        raise SystemExit("manifest validation failed: " + "; ".join(validation["errors"]))
    coverage = history_coverage_audit(manifest)
    if not coverage["valid"]:
        raise SystemExit(
            "listing-history validation failed: " + "; ".join(coverage["errors"])
        )
    _write_json(CORPUS_DIR / "manifest.json", manifest)
    if isinstance(old_manifest, dict) and old_manifest.get("benchmark_id") == BENCHMARK_ID:
        current_files = {str(item["sample_file"]) for item in manifest_items}
        stale_files = {
            str(item.get("sample_file"))
            for item in old_manifest.get("items") or []
            if isinstance(item, dict) and item.get("sample_file")
        } - current_files
        for stale_name in sorted(stale_files):
            stale_path = (CORPUS_DIR / stale_name).resolve()
            if stale_path.parent != CORPUS_DIR.resolve():
                raise SystemExit("old manifest contains an unsafe sample path")
            stale_path.unlink(missing_ok=True)
    with (CORPUS_DIR / "source_run.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(source_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(source_rows)

    snapshots = {
        item["ticker"]: json.loads((CORPUS_DIR / item["sample_file"]).read_text(encoding="utf-8"))
        for item in manifest_items
    }
    _materialize_workbench(
        manifest_items,
        snapshots,
        fetched_at=args.fetched_at,
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
