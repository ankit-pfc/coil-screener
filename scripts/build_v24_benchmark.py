#!/usr/bin/env python3
"""Download and freeze the 72 canonical v2.4 benchmark samples."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    sha256_json,
    stable_sample_id,
    stable_task_id,
    validate_manifest,
)
from coil_analysis import ALGORITHM_VERSION  # noqa: E402
from history_cache import _bars_from_frame  # noqa: E402
from review_snapshots import (  # noqa: E402
    REVIEW_CORPUS_MANIFEST_KIND,
    review_snapshot_identity,
)
from screen_monthly import fetch_monthly_history  # noqa: E402

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


def _download(ticker: str, attempts: int = 3):
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            frame = fetch_monthly_history(ticker)
            if frame is not None and not frame.empty:
                return frame
        except Exception as exc:  # provider errors vary by curl backend
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(1 + attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError("provider returned no history")


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
                    "history_start": raw.get("history_start"),
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
        timing_order = None
        if attempt == 2:
            timing_order = "blind_first" if position <= 6 else "assisted_first"
        snapshot["corpus_labels"].update(
            {
                "benchmark_task_id": task_id,
                "benchmark_sample_id": item["sample_id"],
                "benchmark_attempt": attempt,
                "benchmark_partition": partition,
                "benchmark_timing_order": timing_order,
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
                **({"timing_order": timing_order} if timing_order else {}),
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
            "adjustment_source": "yfinance_stock_splits",
            "source_interval": "1d",
            "adjustment_transform_version": "yfinance-stock-splits-v1",
            "as_of_policy": "completed quarter only",
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
    parser.add_argument("--workers", type=int, default=6)
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

    frames: dict[str, Any] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_download, item["ticker"]): item["ticker"] for item in planned}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                frames[ticker] = future.result()
                print(f"downloaded {ticker}: {len(frames[ticker])} months", flush=True)
            except Exception as exc:
                failures[ticker] = str(exc)
                print(f"failed {ticker}: {exc}", file=sys.stderr, flush=True)
    if failures:
        raise SystemExit("provider failures: " + canonical_json(failures))

    prepared: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    quality_failures: dict[str, Any] = {}
    for item in planned:
        ticker = item["ticker"]
        frame = frames[ticker]
        if item.get("history_start"):
            start_month = date.fromisoformat(str(item["history_start"])).replace(day=1)
            frame = frame[frame.index.date >= start_month]
        cutoff_month = date.fromisoformat(item["as_of"]).replace(day=1)
        frame = frame[frame.index.date <= cutoff_month]
        bars = _bars_from_frame(frame)
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
        prepared[ticker] = (bars, quality)
    if quality_failures:
        raise SystemExit("strict OHLC/cutoff failures: " + canonical_json(quality_failures))

    manifest_items: list[dict[str, Any]] = []
    source_rows: list[dict[str, str]] = []
    for position, item in enumerate(planned, start=1):
        ticker = item["ticker"]
        bars, quality = prepared[ticker]
        bars_hash = sha256_json(bars)
        sample_id = stable_sample_id(ticker, item["as_of"], bars_hash)
        source_fingerprint = sha256_json(
            {
                "provider": "yfinance",
                "ticker": ticker,
                "fetched_at": args.fetched_at,
                "adjustment_mode": "split_adjusted",
                "adjustment_source": "yfinance_stock_splits",
                "source_interval": "1d",
                "adjustment_transform_version": "yfinance-stock-splits-v1",
                "history_start": item.get("history_start"),
                "bars_sha256": bars_hash,
            }
        )
        screen_snapshot = {
            "ticker": ticker,
            "data_date": bars[-1]["date"],
            "fetched_at": args.fetched_at,
            "data_source": "yfinance",
            "freshness": "frozen_benchmark",
            "status": "blind_unlabeled",
            "grade": None,
        }
        snapshot = {
            "schema_version": 1,
            "kind": "coilingview.saved-run-review-snapshot",
            "source": SOURCE,
            "ticker": ticker,
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
            "provenance": {
                "bars_sha256": bars_hash,
                "source_fingerprint": source_fingerprint,
            },
            "source_cache_metadata": {
                "schema_version": 4,
                "fetched_at": args.fetched_at,
                "last_bar_date": bars[-1]["date"],
                "source": "yfinance",
                "adjustment_mode": "split_adjusted",
                "adjustment_source": "yfinance_stock_splits",
                "source_interval": "1d",
                "adjustment_transform_version": "yfinance-stock-splits-v1",
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
                "sample_id": sample_id,
                "sample_file": path.name,
                "bar_count": len(bars),
                "first_data_date": bars[0]["date"],
                "last_data_date": bars[-1]["date"],
                "adjustment_mode": "split_adjusted",
                "new_or_remediated": True,
                "history_start_policy": (
                    {
                        "kind": "exclude_pre_cutoff_provider_rows",
                        "first_included_month": item["history_start"],
                        "reason": "predeclared strict-OHLC remediation before label exposure",
                    }
                    if item.get("history_start")
                    else {"kind": "full_provider_history"}
                ),
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
            "adjustment_source": "yfinance_stock_splits",
            "source_interval": "1d",
            "adjustment_transform_version": "yfinance-stock-splits-v1",
            "as_of_policy": "completed quarter only",
        },
        "items": manifest_items,
    }
    validation = validate_manifest(manifest)
    if not validation["valid"]:
        raise SystemExit("manifest validation failed: " + "; ".join(validation["errors"]))
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
