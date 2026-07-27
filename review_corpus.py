"""Build an auditable candidate corpus from CoilingView session exports.

This module deliberately stops before detector tuning or golden-fixture
promotion. It preserves raw events, deduplicates them by stable event identity,
derives the latest label for each immutable reviewed sample, and quarantines
records that are ambiguous, low-confidence, or lack reproducible bar identity.

Usage:

    python -m review_corpus build export-1.md export-2.json -o candidate.json
    python -m review_corpus report export-1.md export-2.json
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

EXPORT_KIND = "coilingview.review-session-feedback"
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
    if payload.get("kind") != EXPORT_KIND:
        raise ValueError(f"{export_path}: unsupported export kind")
    if payload.get("schema_version") not in (1, 2):
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
        "bar_data_hash": _field(
            provenance, "bar_data_hash", "barDataHash"
        ),
    }
    return f"sample-sha256:{_sha256(identity)}"


def _quarantine_reasons(entry: dict[str, Any]) -> list[str]:
    record = entry.get("record") or {}
    provenance = record.get("provenance") or {}
    reasons: list[str] = []
    if _field(record, "schemaVersion", "schema_version") != 4:
        reasons.append("legacy_label_schema")
    if _field(record, "labelPolicyVersion", "label_policy_version") != 1:
        reasons.append("missing_label_policy")
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
    if not _field(provenance, "bar_data_hash", "barDataHash"):
        reasons.append("missing_bar_data_hash")
    if (
        _field(record, "decision", "decision") == "corrected"
        or coil_label == "not_coil"
    ) and not str(_field(record, "note", "note") or "").strip():
        reasons.append("missing_rationale")
    return reasons


def _report(samples: list[dict[str, Any]], events_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    top_verdicts: Counter[str] = Counter()
    coil_labels: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    grades: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    statuses: Counter[str] = Counter()

    for sample in samples:
        statuses[sample["status"]] += 1
        entry = events_by_id[sample["current_event_id"]]
        record = entry["record"]
        decision = _field(record, "decision", "decision")
        coil_label = _field(record, "coilLabel", "coil_label")
        human_grade = _field(record, "humanGrade", "human_grade")
        certainty = _field(record, "confidence", "confidence")
        if decision:
            top_verdicts[str(decision)] += 1
        if coil_label:
            coil_labels[str(coil_label)] += 1
        if certainty:
            confidence[str(certainty)] += 1
        if human_grade:
            grades[str(human_grade)] += 1
        algorithm = record.get("algorithm") or {}
        model_grade = _field(algorithm, "grade", "grade")
        if human_grade and model_grade:
            confusion[str(human_grade)][str(model_grade)] += 1

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
        "Human vs model grade:",
        json.dumps(report["human_vs_model_grade"], indent=2, sort_keys=True),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "report"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("exports", nargs="+", type=Path)
        if command == "build":
            subparser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    corpus = build_corpus(load_export(path) for path in args.exports)
    if args.command == "report":
        print(_render_report(corpus["report"]))
        return 0
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
