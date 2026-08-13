"""Build an auditable candidate corpus from CoilingView session exports.

This module deliberately stops before detector tuning or golden-fixture
promotion. It preserves raw events, deduplicates them by stable event identity,
derives the latest label for each immutable reviewed sample, and quarantines
records that are ambiguous, low-confidence, or lack reproducible bar identity.

Usage:

    python -m review_corpus build export-1.md export-2.json -o candidate.json
    python -m review_corpus report export-1.md export-2.json
    python -m review_corpus gate export.json --min-labeled-samples 20 \
        --min-top-exact-match-rate 0.8 --min-coil-candidate-precision 0.8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LEGACY_EXPORT_KIND = "coilingview.review-session-feedback"
FRESH_EXPORT_KIND = "coilingview.fresh-review-session-feedback"
EXPORT_KINDS = {LEGACY_EXPORT_KIND, FRESH_EXPORT_KIND}
CORPUS_KIND = "coilingview.review-corpus-candidate"
JSON_FENCE = re.compile(r"````json\s*\n(.*?)\n````", re.DOTALL)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def load_export(path: Path | str) -> dict[str, Any]:
    """Load a raw JSON export or the canonical JSON block in a Markdown export."""
    export_path = Path(path)
    text = export_path.read_text(encoding="utf-8")
    if export_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        match = JSON_FENCE.search(text)
        if match is None:
            raise ValueError(f"{export_path}: canonical JSON block not found")
        payload = json.loads(match.group(1))
    if payload.get("kind") not in EXPORT_KINDS:
        raise ValueError(f"{export_path}: unsupported export kind")
    if payload.get("schema_version") not in (1, 2, 3, 4, 5):
        raise ValueError(f"{export_path}: unsupported export schema version")
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"{export_path}: records must be a list")
    return payload


def _field(record: dict[str, Any], camel: str, snake: str | None = None) -> Any:
    if camel in record:
        return record[camel]
    return record.get(snake or camel)


def _event_id(entry: dict[str, Any]) -> str:
    record = entry.get("record") or {}
    explicit = entry.get("event_id") or _field(record, "eventId", "event_id")
    if explicit:
        return str(explicit)
    envelope = {
        "ticker": entry.get("ticker"),
        "interval": entry.get("interval"),
        "created_at": entry.get("created_at"),
        "record": record,
    }
    return f"legacy-sha256:{_sha256(envelope)}"


def _sample_id(entry: dict[str, Any]) -> str:
    record = entry.get("record") or {}
    provenance = record.get("provenance") or {}
    identity = {
        "ticker": str(entry.get("ticker", "")).upper(),
        "interval": entry.get("interval"),
        "as_of": _field(record, "asOf", "as_of"),
        "algorithm_version": _field(
            record, "algorithmVersion", "algorithm_version"
        ),
        "bar_data_hash": _bar_data_hash(provenance),
    }
    return f"sample-sha256:{_sha256(identity)}"


def _bar_data_hash(provenance: dict[str, Any]) -> Any:
    """Read the immutable bars identity across legacy and capture-v5 names."""
    for key in ("barsHash", "bars_hash", "barDataHash", "bar_data_hash"):
        value = provenance.get(key)
        if value:
            return value
    return None


def _quarantine_reasons(entry: dict[str, Any]) -> list[str]:
    record = entry.get("record") or {}
    provenance = record.get("provenance") or {}
    reasons: list[str] = []
    schema = _field(record, "schemaVersion", "schema_version")
    label_policy = _field(record, "labelPolicyVersion", "label_policy_version")
    if (schema, label_policy) not in {(4, 1), (5, 2)}:
        reasons.append("unsupported_label_contract")
    coil_label = _field(record, "coilLabel", "coil_label")
    if coil_label not in {"coil", "not_coil", "uncertain"}:
        reasons.append("missing_coil_label")
    if coil_label == "uncertain":
        reasons.append("uncertain_label")
    if _field(record, "confidence", "confidence") == "low":
        reasons.append("low_confidence")
    if coil_label == "coil" and _field(record, "humanGrade", "human_grade") not in {
        "A",
        "B",
        "C",
    }:
        reasons.append("missing_human_grade")
    if not _field(record, "asOf", "as_of"):
        reasons.append("missing_as_of")
    if not _field(record, "algorithmVersion", "algorithm_version"):
        reasons.append("missing_algorithm_version")
    if not _bar_data_hash(provenance):
        reasons.append("missing_bar_data_hash")
    if (
        _field(record, "decision", "decision") == "corrected"
        or coil_label == "not_coil"
    ) and not str(_field(record, "note", "note") or "").strip():
        reasons.append("missing_rationale")
    return reasons


def _point_dates(points: Any) -> set[str]:
    if not isinstance(points, list):
        return set()
    return {
        str(point["date"])
        for point in points
        if isinstance(point, dict) and point.get("date")
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _report(
    samples: list[dict[str, Any]],
    events_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    top_verdicts: Counter[str] = Counter()
    coil_labels: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    grades: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    statuses: Counter[str] = Counter()
    top_exact_matches = 0
    top_decisions = 0
    point_true_positives = 0
    algorithm_point_count = 0
    human_point_count = 0
    coil_candidates = 0
    definite_coil_labels = 0

    for sample in samples:
        statuses[sample["status"]] += 1
        trusted = sample["status"] == "candidate"
        entry = events_by_id[sample["current_event_id"]]
        record = entry["record"]
        decision = _field(record, "decision", "decision")
        coil_label = _field(record, "coilLabel", "coil_label")
        human_grade = _field(record, "humanGrade", "human_grade")
        certainty = _field(record, "confidence", "confidence")
        if decision:
            top_verdicts[str(decision)] += 1
        if trusted and decision in {"approved", "corrected"}:
            top_decisions += 1
            if decision == "approved":
                top_exact_matches += 1
        if coil_label:
            coil_labels[str(coil_label)] += 1
        if trusted and coil_label in {"coil", "not_coil"}:
            definite_coil_labels += 1
            if coil_label == "coil":
                coil_candidates += 1
        if certainty:
            confidence[str(certainty)] += 1
        if human_grade:
            grades[str(human_grade)] += 1
        algorithm = record.get("algorithm") or {}
        model_grade = _field(algorithm, "grade", "grade")
        if human_grade and model_grade:
            confusion[str(human_grade)][str(model_grade)] += 1

        algorithm_dates = _point_dates(
            _field(algorithm, "majorHighs", "major_highs")
        )
        human_dates = (
            algorithm_dates
            if decision == "approved"
            else _point_dates(_field(record, "reviewedHighs", "reviewed_highs"))
        )
        if trusted and decision in {"approved", "corrected"}:
            point_true_positives += len(algorithm_dates & human_dates)
            algorithm_point_count += len(algorithm_dates)
            human_point_count += len(human_dates)

    return {
        "sample_count": len(samples),
        "event_count": len(events_by_id),
        "status_counts": dict(sorted(statuses.items())),
        "top_verdict_counts": dict(sorted(top_verdicts.items())),
        "coil_label_counts": dict(sorted(coil_labels.items())),
        "confidence_counts": dict(sorted(confidence.items())),
        "human_grade_counts": dict(sorted(grades.items())),
        "human_vs_model_grade": {
            human: dict(sorted(models.items()))
            for human, models in sorted(confusion.items())
        },
        "accuracy": {
            "top_exact_matches": top_exact_matches,
            "top_decisions": top_decisions,
            "top_exact_match_rate": _ratio(top_exact_matches, top_decisions),
            "top_point_precision": _ratio(
                point_true_positives, algorithm_point_count
            ),
            "top_point_recall": _ratio(point_true_positives, human_point_count),
            "coil_candidates": coil_candidates,
            "definite_coil_labels": definite_coil_labels,
            "coil_candidate_precision": _ratio(
                coil_candidates, definite_coil_labels
            ),
        },
    }


def evaluate_promotion_gate(
    report: dict[str, Any],
    *,
    min_labeled_samples: int,
    min_top_exact_match_rate: float,
    min_coil_candidate_precision: float,
) -> dict[str, Any]:
    """Evaluate explicit release thresholds against human-review evidence."""
    if min_labeled_samples < 1:
        raise ValueError("min_labeled_samples must be at least 1")
    for name, value in (
        ("min_top_exact_match_rate", min_top_exact_match_rate),
        ("min_coil_candidate_precision", min_coil_candidate_precision),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    accuracy = report.get("accuracy") or {}
    labeled = int(accuracy.get("top_decisions") or 0)
    top_rate = accuracy.get("top_exact_match_rate")
    coil_precision = accuracy.get("coil_candidate_precision")
    failures: list[str] = []
    if labeled < min_labeled_samples:
        failures.append(
            f"only {labeled} labeled samples; requires {min_labeled_samples}"
        )
    if top_rate is None or float(top_rate) < min_top_exact_match_rate:
        failures.append(
            f"top exact-match rate {top_rate!r} is below {min_top_exact_match_rate:.3f}"
        )
    if (
        coil_precision is None
        or float(coil_precision) < min_coil_candidate_precision
    ):
        failures.append(
            "coil candidate precision "
            f"{coil_precision!r} is below {min_coil_candidate_precision:.3f}"
        )
    return {
        "passed": not failures,
        "thresholds": {
            "min_labeled_samples": min_labeled_samples,
            "min_top_exact_match_rate": min_top_exact_match_rate,
            "min_coil_candidate_precision": min_coil_candidate_precision,
        },
        "failures": failures,
    }


def build_corpus(
    exports: Iterable[dict[str, Any]],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Combine exports into a lossless candidate corpus and derived label view."""
    events_by_id: dict[str, dict[str, Any]] = {}
    source_exports: list[dict[str, Any]] = []

    for payload in exports:
        session = payload.get("session") or {}
        source_exports.append(
            {
                "session_id": session.get("id"),
                "session_fingerprint": session.get("fingerprint"),
                "source": session.get("source"),
                "exported_at": payload.get("exported_at"),
                "schema_version": payload.get("schema_version"),
            }
        )
        for raw_entry in payload.get("records") or []:
            entry = dict(raw_entry)
            event_id = _event_id(entry)
            entry["event_id"] = event_id
            existing = events_by_id.get(event_id)
            if existing is not None and _canonical(existing) != _canonical(entry):
                raise ValueError(f"conflicting payloads for event {event_id}")
            events_by_id[event_id] = entry

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in events_by_id.values():
        grouped[_sample_id(entry)].append(entry)

    samples: list[dict[str, Any]] = []
    for sample_id, entries in sorted(grouped.items()):
        entries.sort(
            key=lambda entry: (
                str(entry.get("created_at") or ""),
                str(entry["event_id"]),
            )
        )
        event_ids = [str(entry["event_id"]) for entry in entries]
        current = entries[-1]
        reasons = _quarantine_reasons(current)
        samples.append(
            {
                "sample_id": sample_id,
                "ticker": str(current.get("ticker", "")).upper(),
                "interval": current.get("interval"),
                "event_ids": event_ids,
                "current_event_id": event_ids[-1],
                "supersedes_event_id": event_ids[-2] if len(event_ids) > 1 else None,
                "status": "quarantine" if reasons else "candidate",
                "quarantine_reasons": reasons,
            }
        )

    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema_version": 1,
        "kind": CORPUS_KIND,
        "generated_at": timestamp,
        "promotion_policy": "candidate_only_manual_curator_promotion_required",
        "source_exports": source_exports,
        "events": sorted(
            events_by_id.values(),
            key=lambda entry: (
                str(entry.get("created_at") or ""),
                str(entry["event_id"]),
            ),
        ),
        "samples": samples,
        "report": _report(samples, events_by_id),
    }


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        f"Samples: {report['sample_count']}",
        f"Events: {report['event_count']}",
        f"Status: {json.dumps(report['status_counts'], sort_keys=True)}",
        f"Top verdicts: {json.dumps(report['top_verdict_counts'], sort_keys=True)}",
        f"Coil labels: {json.dumps(report['coil_label_counts'], sort_keys=True)}",
        f"Confidence: {json.dumps(report['confidence_counts'], sort_keys=True)}",
        f"Human grades: {json.dumps(report['human_grade_counts'], sort_keys=True)}",
        f"Accuracy: {json.dumps(report['accuracy'], sort_keys=True)}",
        "Human vs model grade:",
        json.dumps(report["human_vs_model_grade"], indent=2, sort_keys=True),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "report", "gate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("exports", nargs="+", type=Path)
        if command == "build":
            subparser.add_argument("-o", "--output", type=Path, required=True)
        if command == "gate":
            subparser.add_argument("--min-labeled-samples", type=int, required=True)
            subparser.add_argument(
                "--min-top-exact-match-rate", type=float, required=True
            )
            subparser.add_argument(
                "--min-coil-candidate-precision", type=float, required=True
            )
    args = parser.parse_args(argv)

    corpus = build_corpus(load_export(path) for path in args.exports)
    if args.command == "report":
        print(_render_report(corpus["report"]))
        return 0
    if args.command == "gate":
        result = evaluate_promotion_gate(
            corpus["report"],
            min_labeled_samples=args.min_labeled_samples,
            min_top_exact_match_rate=args.min_top_exact_match_rate,
            min_coil_candidate_precision=args.min_coil_candidate_precision,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    args.output.write_text(
        json.dumps(corpus, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {corpus['report']['sample_count']} samples and "
        f"{corpus['report']['event_count']} events to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
