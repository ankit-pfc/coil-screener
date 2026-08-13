#!/usr/bin/env python3
"""Build a versioned major-top screen and its immutable review corpus.

The input is an existing frozen review corpus. Price bars are copied exactly;
only the deterministic analyzer output and candidate selection are refreshed.
This makes algorithm releases comparable without a live-data confound.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from coil_analysis import (
    ALGORITHM_VERSION,
    ANALYSIS_MODE_ALGORITHM_ONLY,
    analyze_coil,
)
from screen_monthly import _lifecycle_row, _screen_sort_key


SNAPSHOT_KIND = "coilingview.saved-run-review-snapshot"
MANIFEST_KIND = "coilingview.saved-run-review-corpus-manifest"
SCHEMA_VERSION = 1

TEXT_FIELDS = {
    "ticker",
    "lifecycle",
    "status",
    "grade",
    "lid_grade",
    "current_price_position",
    "review_status",
    "review_algorithm_version",
    "review_effective",
    "data_date",
    "freshness",
    "fetched_at",
    "data_source",
}
BOOLEAN_FIELDS = {"reviewed", "review_stale"}


def canonical_json_bytes(value: Any) -> bytes:
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


def typed_value(key: str, value: Any) -> Any:
    if value is None or value == "":
        return None
    if key in TEXT_FIELDS:
        return str(value)
    if key in BOOLEAN_FIELDS:
        return str(value).strip().lower() == "true"
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def typed_screen_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: typed_value(key, value) for key, value in raw.items()}


def load_source(source_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    order = manifest.get("ordered_universe") or [
        item["ticker"] for item in manifest["items"]
    ]
    snapshots: list[dict[str, Any]] = []
    for ticker in order:
        path = source_dir / f"{ticker}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["_source_snapshot_file"] = path.name
        raw["_source_snapshot_sha256"] = sha256_file(path)
        snapshots.append(raw)
    manifest["_manifest_sha256"] = sha256_file(manifest_path)
    return manifest, snapshots


def candidate_rows(
    snapshots: list[dict[str, Any]], screened_at: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    analyses: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        quality = snapshot.get("data_quality") or {}
        if not quality.get("reviewable", True):
            continue
        ticker = str(snapshot["ticker"]).strip().upper()
        bars = snapshot["monthly_bars"]
        analysis = analyze_coil(
            bars,
            review_override=None,
            mode=ANALYSIS_MODE_ALGORITHM_ONLY,
            adjustment_mode=str(
                (snapshot.get("source_cache_metadata") or {}).get(
                    "adjustment_mode"
                )
                or "unknown"
            ),
        )
        if analysis.get("grade") is None:
            continue
        metadata = snapshot.get("source_cache_metadata") or {}
        payload = {
            "bars": bars,
            "features": typed_screen_snapshot(snapshot["screen_snapshot"]),
            "freshness": {
                "status": "frozen_rerun",
                "last_bar_date": bars[-1]["date"],
                "fetched_at": metadata.get("fetched_at"),
                "source": metadata.get("source") or "frozen_snapshot",
            },
        }
        row = _lifecycle_row(ticker, payload, analysis)
        row.update(
            {
                "algorithm_version": ALGORITHM_VERSION,
                "screened_at": screened_at,
                "selection_policy": "grade_is_not_null",
            }
        )
        rows.append(row)
        analyses[ticker] = analysis
    rows.sort(key=_screen_sort_key)
    return rows, analyses


def write_csv(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not rows:
        raise RuntimeError("the released screen has no graded candidates")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_corpus(
    *,
    source_dir: Path,
    source_manifest: dict[str, Any],
    source_snapshots: list[dict[str, Any]],
    output_csv: Path,
    output_dir: Path,
    csv_rows: list[dict[str, str]],
    analyses: dict[str, dict[str, Any]],
    code_sha: str,
    screened_at: str,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output corpus already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_bytes = output_csv.read_bytes()
    csv_sha = sha256_bytes(csv_bytes)
    shutil.copyfile(output_csv, output_dir / "source_run.csv")
    source_by_ticker = {
        str(snapshot["ticker"]).strip().upper(): snapshot
        for snapshot in source_snapshots
    }
    items: list[dict[str, Any]] = []
    for position, row in enumerate(csv_rows, start=1):
        ticker = row["ticker"]
        source = source_by_ticker[ticker]
        bars = source["monthly_bars"]
        bars_hash = sha256_json(
            {
                "schema_version": SCHEMA_VERSION,
                "interval": "1M",
                "ticker": ticker,
                "bars": bars,
            }
        )
        screen_hash = sha256_json(row)
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "kind": SNAPSHOT_KIND,
            "source": output_csv.name,
            "ticker": ticker,
            "run": {
                "algorithm_version": ALGORITHM_VERSION,
                "code_sha": code_sha,
                "source_csv_sha256": csv_sha,
                "screened_at": screened_at,
                "selection_policy": "grade_is_not_null",
                "universe_position": position,
            },
            "screen_snapshot": row,
            "algorithm_analysis": analyses[ticker],
            "monthly_bars": bars,
            "source_cache_metadata": source.get("source_cache_metadata") or {},
            "provenance": {
                "derived_from_corpus": source_dir.name,
                "derived_from_manifest_sha256": source_manifest["_manifest_sha256"],
                "derived_from_snapshot_file": source["_source_snapshot_file"],
                "derived_from_snapshot_sha256": source["_source_snapshot_sha256"],
                "canonical_monthly_bars_sha256": sha256_json(bars),
                "backend_bars_identity_sha256": bars_hash,
                "screen_snapshot_sha256": screen_hash,
            },
            "corpus_labels": source.get("corpus_labels") or {},
            "data_quality": source.get("data_quality") or {},
        }
        snapshot_path = output_dir / f"{ticker}.json"
        snapshot_path.write_text(pretty_json(snapshot), encoding="utf-8")
        quality = snapshot["data_quality"]
        items.append(
            {
                "universe_position": position,
                "ticker": ticker,
                "snapshot_file": snapshot_path.name,
                "snapshot_sha256": sha256_file(snapshot_path),
                "backend_bars_identity_sha256": bars_hash,
                "screen_snapshot_sha256": screen_hash,
                "bar_count": len(bars),
                "first_data_date": bars[0]["date"],
                "last_data_date": bars[-1]["date"],
                "data_quality": {
                    "status": quality.get("status"),
                    "reviewable": bool(quality.get("reviewable", True)),
                },
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "corpus_id": output_csv.stem,
        "purpose": "Immutable blind-review evidence for the v2.3 graded major-top candidates.",
        "trust_status": "review_evidence_not_validated_training_truth",
        "source": output_csv.name,
        "source_run": {
            "filename": output_csv.name,
            "embedded_exact_copy": "source_run.csv",
            "sha256": csv_sha,
            "byte_count": len(csv_bytes),
            "row_count": len(csv_rows),
            "algorithm_version": ALGORITHM_VERSION,
            "code_sha": code_sha,
            "screened_at": screened_at,
            "selection_policy": "grade_is_not_null",
            "input_corpus": source_dir.name,
            "input_manifest_sha256": source_manifest["_manifest_sha256"],
        },
        "ordered_universe": [row["ticker"] for row in csv_rows],
        "summary": {
            "ticker_count": len(items),
            "monthly_bar_count": sum(item["bar_count"] for item in items),
            "grade_counts": {
                str(key): int(value)
                for key, value in pd.Series(
                    [row["grade"] for row in csv_rows]
                ).value_counts().items()
            },
            "lifecycle_counts": {
                str(key): int(value)
                for key, value in pd.Series(
                    [row["lifecycle"] for row in csv_rows]
                ).value_counts().items()
            },
        },
        "generator": {
            "file": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(Path(__file__)),
        },
        "items": items,
    }
    (output_dir / "manifest.json").write_text(
        pretty_json(manifest), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# v2.3 major-top review cohort\n\n"
        "This corpus re-runs the exact frozen monthly bars from `"
        f"{source_dir.name}` through algorithm `{ALGORITHM_VERSION}`. "
        "Only rows with a non-null algorithm grade are included. The prior "
        "corpus remains immutable historical evidence.\n",
        encoding="utf-8",
    )
    return manifest


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = (PROJECT_ROOT / args.source_corpus).resolve()
    output_csv = (PROJECT_ROOT / args.output_csv).resolve()
    output_dir = (PROJECT_ROOT / args.output_corpus).resolve()
    source_manifest, snapshots = load_source(source_dir)
    rows, analyses = candidate_rows(snapshots, args.screened_at)
    csv_rows = write_csv(output_csv, rows)
    return build_corpus(
        source_dir=source_dir,
        source_manifest=source_manifest,
        source_snapshots=snapshots,
        output_csv=output_csv,
        output_dir=output_dir,
        csv_rows=csv_rows,
        analyses=analyses,
        code_sha=args.code_sha,
        screened_at=args.screened_at,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-corpus", required=True)
    result.add_argument("--output-csv", required=True)
    result.add_argument("--output-corpus", required=True)
    result.add_argument("--screened-at", required=True)
    result.add_argument("--code-sha", required=True)
    return result


def main() -> None:
    manifest = build(parser().parse_args())
    print(
        f"built {manifest['summary']['ticker_count']} candidates: "
        f"{', '.join(manifest['ordered_universe'])}"
    )


if __name__ == "__main__":
    main()
