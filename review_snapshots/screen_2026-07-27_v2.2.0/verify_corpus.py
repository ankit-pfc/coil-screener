#!/usr/bin/env python3
"""Verify provenance, exact evidence, identities, and quality annotations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_corpus as corpus


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify() -> dict[str, Any]:
    require(corpus.MANIFEST_PATH.is_file(), "manifest.json is missing")
    require(corpus.SOURCE_CSV_COPY.is_file(), "source_run.csv is missing")
    manifest = load_json(corpus.MANIFEST_PATH)
    require(
        manifest.get("schema_version") == corpus.SCHEMA_VERSION,
        "manifest schema version mismatch",
    )
    require(
        manifest.get("kind") == corpus.MANIFEST_KIND,
        "manifest kind mismatch",
    )
    require(
        manifest.get("trust_status")
        == "review_evidence_not_validated_training_truth",
        "manifest trust boundary is missing",
    )
    require(
        manifest["generator"]["sha256"]
        == corpus.sha256_file(Path(corpus.__file__)),
        "generator hash mismatch",
    )

    copied_csv_bytes = corpus.SOURCE_CSV_COPY.read_bytes()
    source_csv_hash = corpus.sha256_bytes(copied_csv_bytes)
    require(
        source_csv_hash == manifest["source_run"]["sha256"],
        "embedded source CSV hash mismatch",
    )
    require(
        len(copied_csv_bytes) == manifest["source_run"]["byte_count"],
        "embedded source CSV byte count mismatch",
    )
    if corpus.SOURCE_CSV.is_file():
        require(
            corpus.SOURCE_CSV.read_bytes() == copied_csv_bytes,
            "embedded source CSV is not an exact byte copy",
        )
    fieldnames, rows = corpus.load_csv_rows(corpus.SOURCE_CSV_COPY)
    require(
        fieldnames == manifest["source_run"]["columns"],
        "source CSV column order mismatch",
    )
    require(
        len(rows) == manifest["source_run"]["row_count"],
        "source CSV row count mismatch",
    )

    items = manifest.get("items")
    require(isinstance(items, list), "manifest items must be a list")
    require(len(items) == len(rows), "manifest item count mismatch")
    ordered_tickers = [row["ticker"] for row in rows]
    require(
        manifest["ordered_universe"] == ordered_tickers,
        "ordered universe differs from source CSV",
    )
    expected_snapshot_names = {f"{ticker}.json" for ticker in ordered_tickers}
    actual_snapshot_names = {
        path.name
        for path in corpus.HERE.glob("*.json")
        if path.name != corpus.MANIFEST_PATH.name
    }
    require(
        actual_snapshot_names == expected_snapshot_names,
        "snapshot file set differs from ordered universe",
    )

    aggregate_quality: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    cohort_counts: Counter[str] = Counter()
    bar_count = 0
    prior_count = 0
    tuning_count = 0

    for position, (row, item) in enumerate(
        zip(rows, items, strict=True), start=1
    ):
        ticker = row["ticker"]
        require(
            item["universe_position"] == position,
            f"{ticker}: universe position mismatch",
        )
        require(item["ticker"] == ticker, f"{ticker}: manifest ticker mismatch")
        snapshot_path = corpus.HERE / item["snapshot_file"]
        require(snapshot_path.name == f"{ticker}.json", f"{ticker}: bad path")
        require(
            corpus.sha256_file(snapshot_path) == item["snapshot_sha256"],
            f"{ticker}: snapshot file hash mismatch",
        )
        snapshot = load_json(snapshot_path)
        require(
            snapshot.get("schema_version") == corpus.SCHEMA_VERSION,
            f"{ticker}: snapshot schema mismatch",
        )
        require(
            snapshot.get("kind") == corpus.SNAPSHOT_KIND,
            f"{ticker}: snapshot kind mismatch",
        )
        require(
            snapshot.get("source") == corpus.SOURCE_CSV_NAME,
            f"{ticker}: snapshot source mismatch",
        )
        require(snapshot.get("ticker") == ticker, f"{ticker}: ticker mismatch")
        require(
            snapshot["run"]["algorithm_version"]
            == corpus.ALGORITHM_VERSION,
            f"{ticker}: algorithm version mismatch",
        )
        require(
            snapshot["run"]["code_sha"] == corpus.CODE_SHA,
            f"{ticker}: code SHA mismatch",
        )
        require(
            snapshot["run"]["source_csv_sha256"] == source_csv_hash,
            f"{ticker}: source CSV identity mismatch",
        )
        require(
            snapshot["run"]["universe_position"] == position,
            f"{ticker}: snapshot position mismatch",
        )
        require(
            snapshot["screen_snapshot"] == row,
            f"{ticker}: screen row is not exact",
        )
        require(
            snapshot["screen_snapshot"]["ticker"] == ticker,
            f"{ticker}: screen ticker mismatch",
        )

        bars = snapshot["monthly_bars"]
        require(isinstance(bars, list) and bars, f"{ticker}: bars missing")
        dates = [str(bar.get("date", "")) for bar in bars]
        require(
            dates == sorted(dates) and len(dates) == len(set(dates)),
            f"{ticker}: bars are not unique and chronological",
        )
        require(
            corpus.sha256_json(bars)
            == item["canonical_monthly_bars_sha256"],
            f"{ticker}: canonical bars hash mismatch",
        )
        backend_identity_hash = corpus.sha256_json(
            {
                "schema_version": corpus.SCHEMA_VERSION,
                "interval": "1M",
                "ticker": ticker,
                "bars": bars,
            }
        )
        require(
            backend_identity_hash == item["backend_bars_identity_sha256"],
            f"{ticker}: backend bars identity mismatch",
        )
        require(
            snapshot["provenance"]["canonical_monthly_bars_sha256"]
            == item["canonical_monthly_bars_sha256"],
            f"{ticker}: snapshot bars provenance mismatch",
        )
        require(
            snapshot["provenance"]["backend_bars_identity_sha256"]
            == item["backend_bars_identity_sha256"],
            f"{ticker}: backend identity provenance mismatch",
        )
        require(
            corpus.sha256_json(row) == item["screen_snapshot_sha256"],
            f"{ticker}: screen snapshot hash mismatch",
        )

        cache_path = corpus.PROJECT_ROOT / item["source_cache_file"]
        if cache_path.is_file():
            cache_bytes = cache_path.read_bytes()
            cache_payload = json.loads(cache_bytes)
            require(
                corpus.sha256_bytes(cache_bytes)
                == item["source_cache_sha256"],
                f"{ticker}: exact source cache hash mismatch",
            )
            require(
                cache_payload["bars"] == bars,
                f"{ticker}: bars differ from exact source cache",
            )
            require(
                cache_payload.get("cache_metadata")
                == snapshot["source_cache_metadata"],
                f"{ticker}: source metadata differs from cache",
            )
        recomputed_quality = corpus.analyze_data_quality(bars)
        require(
            snapshot["data_quality"] == recomputed_quality,
            f"{ticker}: quality findings mismatch",
        )
        counts = corpus.quality_counts(recomputed_quality)
        require(
            item["data_quality"]["counts"] == counts,
            f"{ticker}: quality counts mismatch",
        )
        require(
            item["data_quality"]["status"] == recomputed_quality["status"],
            f"{ticker}: quality status mismatch",
        )
        require(
            item["data_quality"]["reviewable"]
            == recomputed_quality["reviewable"],
            f"{ticker}: reviewability mismatch",
        )
        require(
            snapshot["corpus_labels"]["cohort_role"]
            == item["cohort_role"],
            f"{ticker}: cohort mismatch",
        )
        require(
            snapshot["corpus_labels"]["prior_review"]
            == item["prior_review"],
            f"{ticker}: prior-review flag mismatch",
        )
        require(
            snapshot["corpus_labels"]["tuning_anchor"]
            == item["tuning_anchor"],
            f"{ticker}: tuning-anchor flag mismatch",
        )

        for name, count in counts.items():
            aggregate_quality[name] += count
        status_counts[recomputed_quality["status"]] += 1
        cohort_counts[item["cohort_role"]] += 1
        bar_count += len(bars)
        prior_count += int(bool(item["prior_review"]["flag"]))
        tuning_count += int(bool(item["tuning_anchor"]["flag"]))

    summary = manifest["summary"]
    require(summary["ticker_count"] == len(items), "summary ticker count mismatch")
    require(summary["monthly_bar_count"] == bar_count, "summary bar count mismatch")
    require(
        summary["cohort_counts"] == dict(sorted(cohort_counts.items())),
        "summary cohort counts mismatch",
    )
    require(
        summary["prior_review_ticker_count"] == prior_count,
        "summary prior-review count mismatch",
    )
    require(
        summary["tuning_anchor_ticker_count"] == tuning_count,
        "summary tuning-anchor count mismatch",
    )
    require(
        summary["data_quality_status_counts"]
        == dict(sorted(status_counts.items())),
        "summary quality status counts mismatch",
    )
    require(
        summary["data_quality_finding_counts"]
        == dict(sorted(aggregate_quality.items())),
        "summary quality finding counts mismatch",
    )
    require(
        manifest["supporting_sources"]["prior_review"]["sha256"]
        == corpus.sha256_file(corpus.PRIOR_REVIEW_PATH),
        "prior-review source hash mismatch",
    )
    return {
        "ticker_count": len(items),
        "monthly_bar_count": bar_count,
        "source_csv_sha256": source_csv_hash,
        "manifest_sha256": corpus.sha256_file(corpus.MANIFEST_PATH),
        "status_counts": dict(sorted(status_counts.items())),
        "quality_counts": dict(sorted(aggregate_quality.items())),
    }


def main() -> None:
    result = verify()
    print(
        f"verified {result['ticker_count']} snapshots and "
        f"{result['monthly_bar_count']} exact monthly bars"
    )
    print(f"source CSV SHA-256: {result['source_csv_sha256']}")
    print(f"manifest SHA-256: {result['manifest_sha256']}")
    print(f"quality status counts: {result['status_counts']}")
    print(f"quality finding counts: {result['quality_counts']}")


if __name__ == "__main__":
    main()
