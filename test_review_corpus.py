from __future__ import annotations

import json

import pytest

from review_corpus import build_corpus, load_export


def _record(
    *,
    event_id: str = "event-1",
    created_at: str = "2026-07-25T10:00:00Z",
    decision: str = "approved",
    coil_label: str = "coil",
    confidence: str = "high",
    bar_hash: str | None = "sha256:abc",
) -> dict:
    record = {
        "schemaVersion": 4,
        "labelPolicyVersion": 1,
        "eventId": event_id,
        "ticker": "AAA",
        "interval": "3M",
        "asOf": "2026-06-01",
        "algorithmVersion": "2.2.0",
        "decision": decision,
        "coilLabel": coil_label,
        "humanGrade": "B" if coil_label == "coil" else None,
        "confidence": confidence,
        "note": "Reviewed against the full history.",
        "algorithm": {"grade": "A"},
        "provenance": {"bar_data_hash": bar_hash},
        "reviewedHighs": [],
    }
    return {
        "id": 1,
        "event_id": event_id,
        "ticker": "AAA",
        "interval": "3M",
        "created_at": created_at,
        "record": record,
    }


def _export(*records: dict, session_id: int = 1) -> dict:
    return {
        "schema_version": 2,
        "kind": "coilingview.review-session-feedback",
        "exported_at": "2026-07-25T12:00:00Z",
        "session": {
            "id": session_id,
            "fingerprint": f"fp-{session_id}",
            "source": "review.csv",
        },
        "records": list(records),
    }


def test_load_export_reads_canonical_markdown(tmp_path):
    payload = _export(_record())
    path = tmp_path / "feedback.md"
    path.write_text(
        "# Feedback\n\n````json\n"
        + json.dumps(payload, indent=2)
        + "\n````\n",
        encoding="utf-8",
    )
    assert load_export(path) == payload


def test_build_deduplicates_cross_session_event_and_reports_labels():
    event = _record()
    corpus = build_corpus(
        [_export(event, session_id=1), _export(event, session_id=2)],
        generated_at="2026-07-25T12:00:00Z",
    )
    assert len(corpus["events"]) == 1
    assert corpus["samples"][0]["status"] == "candidate"
    assert corpus["report"] == {
        "sample_count": 1,
        "event_count": 1,
        "status_counts": {"candidate": 1},
        "top_verdict_counts": {"approved": 1},
        "coil_label_counts": {"coil": 1},
        "confidence_counts": {"high": 1},
        "human_grade_counts": {"B": 1},
        "human_vs_model_grade": {"B": {"A": 1}},
    }


def test_build_preserves_revisions_and_marks_superseded_event():
    first = _record(event_id="event-1")
    second = _record(
        event_id="event-2",
        created_at="2026-07-25T11:00:00Z",
        decision="corrected",
    )
    second["record"]["reviewedHighs"] = [
        {"date": "2020-03-01", "price": 100.0},
        {"date": "2024-03-01", "price": 103.0},
    ]
    corpus = build_corpus([_export(first, second)])
    sample = corpus["samples"][0]
    assert sample["event_ids"] == ["event-1", "event-2"]
    assert sample["current_event_id"] == "event-2"
    assert sample["supersedes_event_id"] == "event-1"
    assert len(corpus["events"]) == 2


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (_record(confidence="low"), "low_confidence"),
        (_record(coil_label="uncertain"), "uncertain_label"),
        (_record(bar_hash=None), "missing_bar_data_hash"),
    ],
)
def test_build_quarantines_ambiguous_or_non_reproducible_labels(entry, reason):
    corpus = build_corpus([_export(entry)])
    sample = corpus["samples"][0]
    assert sample["status"] == "quarantine"
    assert reason in sample["quarantine_reasons"]


def test_conflicting_payload_for_same_event_id_is_rejected():
    first = _record()
    second = _record()
    second["record"]["note"] = "Different payload."
    with pytest.raises(ValueError, match="conflicting payloads"):
        build_corpus([_export(first), _export(second, session_id=2)])
