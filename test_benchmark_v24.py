import json
import shlex
from copy import deepcopy
from pathlib import Path

import pytest
import benchmark_v24 as benchmark_module

from benchmark_v24 import (
    BENCHMARK_ID,
    BenchmarkError,
    HoldoutSealedError,
    band_agrees,
    gate_decision,
    label_stability,
    labels_from_review_exports,
    load_labels,
    make_freeze,
    maximum_top_match,
    prediction_summary,
    register_holdout_labels,
    registered_validation_configs,
    select_development_configuration,
    validate_manifest,
    validate_selection_report,
    verify_freeze,
    config_fingerprint,
)
from review_snapshots import (
    REVIEW_CORPUS_MANIFEST_KIND,
    ReviewSnapshotError,
    load_blind_review_context,
    load_review_context,
    load_review_manifest,
    review_snapshot_identity,
    verify_manifest_identity,
)
from coil_validation_v24 import DEFAULT_CONFIG, ValidationConfig
from scripts import build_v24_benchmark


CORPUS_SOURCE = "benchmark_2026-08-13_v24_72.csv"
DEVELOPMENT_SOURCE = "benchmark_2026-08-13_v24_72_batch_a.csv"
CORPUS_ROOT = Path(__file__).parent / "review_snapshots" / Path(CORPUS_SOURCE).stem


def _manifest():
    partitions = ["development"] * 36 + ["validation"] * 18 + ["holdout"] * 18
    markets = ["india"] * 24 + ["united_states"] * 24 + ["global_ex_us"] * 24
    cohorts = [
        "expert_positive",
        "clear_negative",
        "disagreement_exception",
        "point_in_time_lifecycle",
    ] * 18
    items = []
    for index in range(72):
        digest = f"{index:064x}"
        items.append(
            {
                "sample_id": f"sample-{index}",
                "ticker": f"T{index}",
                "ticker_family": f"F{index}",
                "as_of": "2026-06-30",
                "partition": partitions[index],
                "market": markets[index],
                "cohort": cohorts[index],
                "adjustment_mode": "split_adjusted",
                "new_or_remediated": index < 5,
                "data_quality": {"reviewable": True},
                "provenance": {
                    "bars_sha256": digest,
                    "sample_file_sha256": digest,
                    "source_fingerprint": digest,
                },
            }
        )
    return {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "generated_at": "2026-08-13T00:00:00Z",
        "items": items,
    }


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _partition_labels(manifest, partition):
    return [
        {
            "sample_id": item["sample_id"],
            "ticker": item["ticker"],
            "attempt_id": f"task-primary-{item['sample_id']}",
            "source_task_id": f"task-primary-{item['sample_id']}",
            "attempt_role": "primary",
            "pattern_label": "not_coil",
            "tops": [],
        }
        for item in manifest["items"]
        if item["partition"] == partition
    ]


def _write_task_manifest(root, manifest):
    primary_tasks = [
        {
            "task_id": f"task-primary-{item['sample_id']}",
            "sample_id": item["sample_id"],
            "ticker": item["ticker"],
            "partition": item["partition"],
            "attempt": 1,
        }
        for item in manifest["items"]
    ]
    repeat_tasks = [
        {
            "task_id": f"task-repeat-{item['sample_id']}",
            "sample_id": item["sample_id"],
            "ticker": item["ticker"],
            "partition": "development_repeat",
            "attempt": 2,
        }
        for item in manifest["items"]
        if item["partition"] == "development"
    ][:12]
    _write(
        root / "review-task-manifest.internal.json",
        {"benchmark_id": BENCHMARK_ID, "tasks": primary_tasks + repeat_tasks},
    )


def test_builder_emits_workbench_manifest_kind_accepted_by_loader():
    assert build_v24_benchmark.REVIEW_CORPUS_MANIFEST_KIND == (
        REVIEW_CORPUS_MANIFEST_KIND
    )


def _selection_report(manifest_path, protocol_path):
    configs = registered_validation_configs()
    metrics = {
        "major_top_precision": 0.8,
        "pattern_precision": 0.8,
        "band_agreement": 0.7,
        "major_top_recall": 0.65,
    }
    sweep = []
    for config in configs:
        row_metrics = dict(metrics)
        if config == DEFAULT_CONFIG:
            row_metrics["major_top_precision"] = 0.9
        sweep.append(
            {
                "config": config.__dict__,
                "config_fingerprint": config_fingerprint(config),
                "development_metrics": row_metrics,
            }
        )
    selected_config, _ = select_development_configuration(
        [
            (ValidationConfig(**row["config"]), row["development_metrics"])
            for row in sweep
        ]
    )
    selected = next(
        row
        for row in sweep
        if row["config_fingerprint"] == config_fingerprint(selected_config)
    )
    development_labels = manifest_path.parent / "development-labels.json"
    validation_labels = manifest_path.parent / "validation-labels.json"
    _write(development_labels, {"fixture": "development"})
    _write(validation_labels, {"fixture": "validation"})
    return {
        "schema_version": 1,
        "kind": "coilingview.v24-development-selection",
        "benchmark_id": BENCHMARK_ID,
        "manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
        "protocol_sha256": __import__("hashlib").sha256(protocol_path.read_bytes()).hexdigest(),
        "code_commit": "a" * 40,
        "development_labels_sha256": __import__("hashlib").sha256(
            development_labels.read_bytes()
        ).hexdigest(),
        "validation_labels_sha256": __import__("hashlib").sha256(
            validation_labels.read_bytes()
        ).hexdigest(),
        "label_stability": {"repeat_pair_count": 12, "passed": True},
        "development_sweep": sweep,
        "selected": {
            **selected,
            "single_use_validation_metrics": {"labeled_sample_count": 18},
        },
    }


def test_manifest_enforces_exact_balances_and_disjoint_families():
    manifest = _manifest()
    assert validate_manifest(manifest)["valid"] is True

    leaked = deepcopy(manifest)
    leaked["items"][54]["ticker_family"] = leaked["items"][0]["ticker_family"]
    report = validate_manifest(leaked)

    assert report["valid"] is False
    assert any("ticker-family leakage" in error for error in report["errors"])


def test_manifest_rejects_noncanonical_and_interview_holdout_samples():
    manifest = _manifest()
    manifest["items"][54]["adjustment_mode"] = "unknown"
    manifest["items"][55]["ticker"] = "TXN"

    report = validate_manifest(manifest)

    assert report["valid"] is False
    assert any("not split_adjusted" in error for error in report["errors"])
    assert any("interview example TXN leaked" in error for error in report["errors"])


def test_manifest_requires_completed_calendar_quarter_cutoffs():
    manifest = _manifest()
    manifest["items"][0]["as_of"] = "2026-05-31"
    manifest["items"][1]["as_of"] = "2026-09-30"

    report = validate_manifest(manifest)

    assert report["valid"] is False
    assert any("not a calendar quarter-end" in error for error in report["errors"])
    assert any("not a completed quarter" in error for error in report["errors"])


def test_frozen_evaluation_command_includes_repeat_and_holdout_artifacts():
    protocol = json.loads((CORPUS_ROOT / "protocol.json").read_text(encoding="utf-8"))
    command = shlex.split(protocol["evaluation_command"])

    assert command[0:2] == ["python3", "scripts/evaluate_v24_benchmark.py"]
    assert command[command.index("--repeat-labels") + 1].endswith(
        "/development-labels.json"
    )
    assert command[command.index("--holdout-labels") + 1].endswith(
        "/holdout-labels.json"
    )


def test_maximum_top_match_is_one_to_one_and_not_greedy():
    result = maximum_top_match(
        ["2020-03-31", "2020-09-30"],
        ["2020-06-30", "2020-12-31"],
    )

    assert result["true_positive"] == 2
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["f1"] == 1.0


def test_band_agreement_requires_iou_and_centre_proximity():
    human = {"lower": 95, "upper": 105}
    assert band_agrees(human, {"lower": 96, "upper": 106}) is True
    assert band_agrees(human, {"lower": 103, "upper": 113}) is False
    assert band_agrees(human, {"lower": 200, "upper": 210}) is False


def test_holdout_labels_cannot_load_until_matching_freeze(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    protocol_path = tmp_path / "protocol.json"
    labels_path = tmp_path / "holdout.json"
    freeze_path = tmp_path / "freeze.json"
    selection_path = tmp_path / "selection.json"
    manifest = _manifest()
    command = "python3 scripts/evaluate_v24_benchmark.py"
    _write(manifest_path, manifest)
    _write_task_manifest(tmp_path, manifest)
    _write(protocol_path, {"benchmark_id": BENCHMARK_ID, "evaluation_command": command})
    _write(selection_path, _selection_report(manifest_path, protocol_path))
    monkeypatch.setattr(
        benchmark_module,
        "validate_selection_report",
        lambda *args, **kwargs: DEFAULT_CONFIG,
    )
    _write(
        labels_path,
        {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "partition": "holdout",
            "manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
            "labels": _partition_labels(manifest, "holdout"),
        },
    )

    with pytest.raises(HoldoutSealedError):
        load_labels(
            labels_path,
            partition="holdout",
            manifest_path=manifest_path,
            protocol_path=protocol_path,
        )

    freeze = make_freeze(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        code_commit="a" * 40,
        evaluation_command=command,
        selection_report_sha256=__import__("hashlib").sha256(selection_path.read_bytes()).hexdigest(),
    )
    _write(freeze_path, freeze)
    with pytest.raises(HoldoutSealedError):
        load_labels(
            labels_path,
            partition="holdout",
            manifest_path=manifest_path,
            protocol_path=protocol_path,
            freeze_path=freeze_path,
            selection_report_path=selection_path,
            current_code_commit="a" * 40,
        )

    freeze = register_holdout_labels(freeze, labels_path, revealed_at="2026-08-13T12:00:00Z")
    _write(freeze_path, freeze)
    loaded = load_labels(
        labels_path,
        partition="holdout",
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        freeze_path=freeze_path,
        selection_report_path=selection_path,
        current_code_commit="a" * 40,
    )
    assert len(loaded["labels"]) == 18


def test_freeze_rejects_manifest_or_configuration_drift(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    protocol_path = tmp_path / "protocol.json"
    selection_path = tmp_path / "selection.json"
    command = "python3 scripts/evaluate_v24_benchmark.py"
    _write(manifest_path, _manifest())
    _write(protocol_path, {"benchmark_id": BENCHMARK_ID, "evaluation_command": command})
    _write(selection_path, _selection_report(manifest_path, protocol_path))
    monkeypatch.setattr(
        benchmark_module,
        "validate_selection_report",
        lambda *args, **kwargs: DEFAULT_CONFIG,
    )
    freeze = make_freeze(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        code_commit="a" * 40,
        evaluation_command=command,
        selection_report_sha256=__import__("hashlib").sha256(selection_path.read_bytes()).hexdigest(),
    )
    verify_freeze(
        freeze,
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        selection_report_path=selection_path,
        current_code_commit="a" * 40,
    )
    protocol_path.write_text("{}", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="protocol_sha256 mismatch"):
        verify_freeze(
            freeze,
            manifest_path=manifest_path,
            protocol_path=protocol_path,
            selection_report_path=selection_path,
            current_code_commit="a" * 40,
        )


def test_hash_bound_runtime_selection_receipt_skips_detector_recomputation(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "manifest.json"
    protocol_path = tmp_path / "protocol.json"
    _write(manifest_path, _manifest())
    _write(protocol_path, {"benchmark_id": BENCHMARK_ID})
    report = _selection_report(manifest_path, protocol_path)

    def unexpected_detector_run(*_args, **_kwargs):
        raise AssertionError("runtime receipt validation must not rerun detectors")

    monkeypatch.setattr(benchmark_module, "run_detectors", unexpected_detector_run)
    selected = validate_selection_report(
        report,
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        current_code_commit="a" * 40,
        recompute_metrics=False,
    )

    assert selected == DEFAULT_CONFIG


def test_selection_receipt_recomputes_every_metric_and_the_winner(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "manifest.json"
    protocol_path = tmp_path / "protocol.json"
    manifest = _manifest()
    _write(manifest_path, manifest)
    _write_task_manifest(tmp_path, manifest)
    _write(protocol_path, {"benchmark_id": BENCHMARK_ID})
    report = _selection_report(manifest_path, protocol_path)
    expected_metrics = {
        row["config_fingerprint"]: deepcopy(row["development_metrics"])
        for row in report["development_sweep"]
    }
    expected_validation = deepcopy(
        report["selected"]["single_use_validation_metrics"]
    )
    monkeypatch.setattr(
        benchmark_module,
        "load_labels",
        lambda *args, **kwargs: {"labels": []},
    )
    monkeypatch.setattr(
        benchmark_module,
        "run_detectors",
        lambda manifest, root, *, config: {
            "fingerprint": config_fingerprint(config),
            "partition": (
                manifest["items"][0]["partition"] if manifest["items"] else ""
            ),
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "detector_metrics",
        lambda labels, outputs, **kwargs: (
            expected_validation
            if outputs["partition"] == "validation"
            else expected_metrics[outputs["fingerprint"]]
        ),
    )

    assert (
        validate_selection_report(
            report,
            manifest_path=manifest_path,
            protocol_path=protocol_path,
            current_code_commit="a" * 40,
        )
        == DEFAULT_CONFIG
    )
    tampered = deepcopy(report)
    tampered["selected"]["single_use_validation_metrics"][
        "pattern_precision"
    ] = 1.0
    with pytest.raises(BenchmarkError, match="validation metrics mismatch"):
        validate_selection_report(
            tampered,
            manifest_path=manifest_path,
            protocol_path=protocol_path,
            current_code_commit="a" * 40,
        )


def test_label_loader_rejects_duplicate_or_cross_partition_rows(tmp_path):
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    protocol_path = tmp_path / "protocol.json"
    labels_path = tmp_path / "validation.json"
    _write(manifest_path, manifest)
    _write_task_manifest(tmp_path, manifest)
    _write(protocol_path, {"benchmark_id": BENCHMARK_ID})
    rows = _partition_labels(manifest, "validation")
    rows[-1] = dict(rows[0], attempt_id="different-attempt")
    _write(
        labels_path,
        {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "partition": "validation",
            "manifest_sha256": __import__("hashlib").sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "labels": rows,
        },
    )

    with pytest.raises(BenchmarkError, match="frozen task"):
        load_labels(
            labels_path,
            partition="validation",
            manifest_path=manifest_path,
            protocol_path=protocol_path,
        )


def test_development_labels_require_the_exact_frozen_repeat_tasks(tmp_path):
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    protocol_path = tmp_path / "protocol.json"
    labels_path = tmp_path / "development.json"
    _write(manifest_path, manifest)
    _write_task_manifest(tmp_path, manifest)
    _write(protocol_path, {"benchmark_id": BENCHMARK_ID})
    rows = _partition_labels(manifest, "development")
    wrong_repeats = []
    development_items = [
        item for item in manifest["items"] if item["partition"] == "development"
    ]
    for item in development_items[12:24]:
        wrong_repeats.append(
            {
                "sample_id": item["sample_id"],
                "ticker": item["ticker"],
                "attempt_id": f"task-repeat-{item['sample_id']}",
                "source_task_id": f"task-repeat-{item['sample_id']}",
                "attempt_role": "repeat",
                "pattern_label": "not_coil",
                "tops": [],
            }
        )
    _write(
        labels_path,
        {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "partition": "development",
            "manifest_sha256": __import__("hashlib").sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "labels": rows + wrong_repeats,
        },
    )

    with pytest.raises(BenchmarkError, match="frozen task"):
        load_labels(
            labels_path,
            partition="development",
            manifest_path=manifest_path,
            protocol_path=protocol_path,
        )


def test_label_stability_requires_all_twelve_repeat_pairs():
    labels = []
    for index in range(12):
        for attempt in (1, 2):
            labels.append(
                {
                    "sample_id": f"sample-{index}",
                    "attempt_id": str(attempt),
                    "attempt_role": "primary" if attempt == 1 else "repeat",
                    "pattern_label": "coil",
                    "tops": [{"peak_date": "2020-03-31"}],
                    "band": {"lower": 95, "upper": 105, "confidence": "high"},
                }
            )

    report = label_stability(labels)

    assert report["repeat_pair_count"] == 12
    assert report["pattern_agreement"] == 1.0
    assert report["matched_top_f1"] == 1.0
    assert report["band_agreement"] == 1.0
    assert report["passed"] is True


def test_label_stability_uses_attempt_role_not_random_task_id_order():
    labels = []
    for index in range(12):
        labels.extend(
            [
                {
                    "sample_id": f"sample-{index}",
                    "attempt_id": f"z-primary-{index}",
                    "attempt_role": "primary",
                    "pattern_label": "not_coil",
                    "tops": [],
                    "band": None,
                    "blind_active_seconds": 999,
                    "assisted_active_seconds": 999,
                    "timing_order": "wrong",
                },
                {
                    "sample_id": f"sample-{index}",
                    "attempt_id": f"a-repeat-{index}",
                    "attempt_role": "repeat",
                    "pattern_label": "not_coil",
                    "tops": [],
                    "band": None,
                    "blind_active_seconds": 100,
                    "assisted_active_seconds": 50,
                    "timing_order": "blind_first" if index < 6 else "assisted_first",
                },
            ]
        )

    report = label_stability(labels)

    assert report["counterbalanced"] is True
    assert report["median_review_time_reduction"] == 0.5


def test_gate_is_inconclusive_without_labels_but_no_go_on_pit_failure():
    manifest_validation = {"valid": True}
    pit = {"violation_count": 0}
    decision = gate_decision(
        manifest_validation=manifest_validation,
        stability=None,
        v24_metrics=None,
        point_in_time=pit,
        accepted_hard_invalid=0,
    )
    assert decision["outcome"] == "inconclusive"
    assert decision["promotion_authorized"] is False

    failed = gate_decision(
        manifest_validation=manifest_validation,
        stability=None,
        v24_metrics=None,
        point_in_time={"violation_count": 1},
        accepted_hard_invalid=0,
    )
    assert failed["outcome"] == "no_go"


def test_frozen_72_sample_corpus_is_review_workbench_compatible():
    with pytest.raises(ReviewSnapshotError, match="unexpected frozen review manifest kind"):
        load_review_manifest(CORPUS_SOURCE)
    manifest = load_review_manifest(DEVELOPMENT_SOURCE)

    assert manifest["source_run"]["algorithm_version"] == "2.3.1"
    assert len(manifest["items"]) == 36
    for item in manifest["items"]:
        identity = review_snapshot_identity(DEVELOPMENT_SOURCE, item["ticker"])
        verify_manifest_identity(identity, item)
        blind = load_blind_review_context(DEVELOPMENT_SOURCE, item["ticker"])
        assert blind["model_revealed"] is False
        assert "detector_outputs" not in blind


def test_historical_lifecycle_snapshot_uses_its_frozen_quarter_end():
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    item = next(
        row
        for row in manifest["items"]
        if row["cohort"] == "point_in_time_lifecycle"
        and row["as_of"] != "2026-06-30"
    )

    context = load_review_context(CORPUS_SOURCE, item["ticker"])

    assert context["as_of"] == item["as_of"]
    for result in context["detector_outputs"].values():
        assert result["analysis_metadata"]["evidence_cutoff"] == item["as_of"]


def test_prediction_summary_does_not_claim_labeled_performance():
    manifest = {"items": [{"sample_id": "a", "partition": "holdout"}]}
    outputs = {
        "a": {
            "v2_3_1": {
                "grade": "A",
                "lifecycle": "forming",
                "major_highs": [],
                "analysis_metadata": {"classification_blocked": False},
                "resistance": None,
            },
            "v2_4_validation": {
                "pattern_assessment": {
                    "structure_state": "qualified",
                    "confidence": "high",
                    "abstained": False,
                },
                "top_candidates": [],
                "analysis_metadata": {"classification_blocked": False},
                "resistance_band": None,
            },
        }
    }

    report = prediction_summary(manifest, outputs)

    assert report["paired"]["both_positive"] == 1
    assert "not precision" in report["warning"]


def test_review_export_import_uses_only_blind_assessment_and_keeps_repeats():
    manifest = {
        "items": [
            {"ticker": "AAA", "sample_id": "sample-a", "partition": "development"},
            {"ticker": "BBB", "sample_id": "sample-b", "partition": "development"},
        ]
    }

    tasks = {
        ("dev.csv", "AAA"): {
            "task_id": "task-a1", "source": "dev.csv", "ticker": "AAA",
            "sample_id": "sample-a", "attempt": 1, "partition": "development",
            "workbench_sample_id": "work-a1", "workbench_bars_hash": "bars-a",
            "data_date": "2026-06-01",
        },
        ("dev.csv", "BBB"): {
            "task_id": "task-b1", "source": "dev.csv", "ticker": "BBB",
            "sample_id": "sample-b", "attempt": 1, "partition": "development",
            "workbench_sample_id": "work-b1", "workbench_bars_hash": "bars-b",
            "data_date": "2026-06-01",
        },
        ("repeat.csv", "AAA"): {
            "task_id": "task-a2", "source": "repeat.csv", "ticker": "AAA",
            "sample_id": "sample-a", "attempt": 2, "partition": "development_repeat",
            "workbench_sample_id": "work-a2", "workbench_bars_hash": "bars-a",
            "data_date": "2026-06-01", "timing_order": "blind_first",
        },
    }

    def wrapper(source, ticker, event, created):
        task = tasks[(source, ticker)]
        return {
            "event_id": event,
            "ticker": ticker,
            "created_at": created,
            "record": {
                "asOf": task["data_date"],
                "blindAssessment": {
                    "patternLabel": "coil" if ticker == "AAA" else "not_coil",
                    "lifecycleLabel": "forming" if ticker == "AAA" else "no_pattern",
                    "humanTops": (
                        [{"date": "2020-03-31", "price": 100, "role": "major_top"}]
                        if ticker == "AAA"
                        else []
                    ),
                    "resistanceBand": (
                        {"lower": 95, "upper": 105} if ticker == "AAA" else None
                    ),
                },
                "confidence": "high",
                "detectorOutputs": {"must_not_become_truth": True},
                "detectorReview": {
                    "timing": {
                        "blindActiveSeconds": 100,
                        "assistedActiveSeconds": 60,
                        "reviewOrder": task.get("timing_order"),
                    }
                },
                "provenance": {
                    "frozen": True,
                    "source": source,
                    "sampleId": task["workbench_sample_id"],
                    "barsHash": task["workbench_bars_hash"],
                    "dataDate": task["data_date"],
                    "reviewOverrideApplied": False,
                },
            },
        }

    exports = [
        {
            "schema_version": 5,
            "reviewer": {"name": "Expert One"},
            "session": {"source": "dev.csv"},
            "records": [
                wrapper("dev.csv", "AAA", "event-a1", "2026-08-13T10:00:00Z"),
                wrapper("dev.csv", "BBB", "event-b1", "2026-08-13T10:01:00Z"),
            ],
        },
        {
            "schema_version": 5,
            "reviewer": {"name": "Expert One"},
            "session": {"source": "repeat.csv"},
            "records": [
                wrapper("repeat.csv", "AAA", "event-a2", "2026-08-13T11:00:00Z")
            ],
        },
    ]

    labels = labels_from_review_exports(
        exports,
        manifest,
        partition="development",
        manifest_sha256="abc",
        task_manifest={"tasks": list(tasks.values())},
    )

    assert len(labels["labels"]) == 3
    attempts = [row for row in labels["labels"] if row["sample_id"] == "sample-a"]
    assert [row["attempt_role"] for row in attempts] == ["primary", "repeat"]
    assert attempts[0]["tops"] == [
        {
            "peak_date": "2020-03-31",
            "price": 100.0,
            "role": "major_top",
            "lid_member": None,
        }
    ]
    assert "detectorOutputs" not in attempts[0]

    corrupted = deepcopy(exports)
    corrupted[0]["records"][0]["record"]["provenance"]["sampleId"] = "wrong"
    with pytest.raises(BenchmarkError, match="mismatched sampleId"):
        labels_from_review_exports(
            corrupted,
            manifest,
            partition="development",
            manifest_sha256="abc",
            task_manifest={"tasks": list(tasks.values())},
        )


def test_registered_sweep_is_bounded_and_lexicographic_selection_is_stable():
    configs = registered_validation_configs()
    assert len(configs) == 54
    assert len({config.__repr__() for config in configs}) == 54
    scored = [
        (
            configs[0],
            {
                "major_top_precision": 0.9,
                "pattern_precision": 0.8,
                "band_agreement": 0.8,
                "major_top_recall": 0.7,
            },
        ),
        (
            configs[1],
            {
                "major_top_precision": 0.9,
                "pattern_precision": 0.8,
                "band_agreement": 0.7,
                "major_top_recall": 0.95,
            },
        ),
    ]

    selected, _ = select_development_configuration(scored)

    assert selected == configs[0]
