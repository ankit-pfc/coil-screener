from __future__ import annotations

import copy
import inspect
from typing import Any, Callable

import pytest

import automatic_exemplar_evaluator as evaluator
from gold_labels import CAPTURE_KIND, materialize_gold_label


CUTOFF = "2020-06-01"
DECISION_AS_OF = "2020-06-30"
DETECTOR_BAR_FIELDS = {"date", "open", "high", "low", "close", "volume"}


def _monthly_bars() -> list[dict[str, Any]]:
    bars: list[dict[str, Any]] = []
    for month in range(1, 10):
        price = 100.0 + month
        bars.append(
            {
                "date": f"2020-{month:02d}-01",
                "open": price,
                "high": price + 2.0,
                "low": price - 2.0,
                "close": price + 0.5,
                "volume": 1_000_000.0 + month,
                "gold_label": {"shape": "coil"},
                "algorithm_analysis": {"review": {"effective": "human"}},
                "review_override": {"points": [{"date": "2020-03-01"}]},
            }
        )
    return bars


def _gold_capture(
    episode_id: str,
    *,
    setup_id: str = "synthetic-setup",
    evaluation_role: str = "development",
    outcome_visible: bool = False,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "kind": CAPTURE_KIND,
        "episodeId": episode_id,
        "setupId": setup_id,
        "evaluationRole": evaluation_role,
        "labeler": "Synthetic Reviewer",
        "labeledAt": "2026-08-22T00:00:00Z",
        "cutoffDate": CUTOFF,
        "decisionAsOf": DECISION_AS_OF,
        "interval": "3M",
        "outcomeVisibleDuringLabel": outcome_visible,
        "judgments": {
            "shape": "not_coil",
            "maturity": "immature",
            "lifecycle": "no_structure",
            "readiness": "not_ready",
            "action": "avoid",
            "confidence": "high",
        },
        "activeStructureId": None,
        "topReviewComplete": True,
        "structures": [],
        "bottoms": [],
        "events": [],
        "phases": [],
        "sourceEvidence": [
            {
                "kind": "blind_review_session",
                "reference": "synthetic detector-only evaluator fixture",
                "sha256": "a" * 64,
            }
        ],
        "notes": [],
    }


def _algorithm_analysis(
    bars: list[dict[str, Any]], requested_as_of: str
) -> dict[str, Any]:
    history_end = bars[-1]["date"] if bars else requested_as_of
    return {
        "algorithm_version": evaluator.ALGORITHM_VERSION,
        "as_of": history_end,
        "bar_count": len(bars),
        "analysis_metadata": {
            "history_end": history_end,
            "bar_count_monthly": len(bars),
            "algorithm_version": evaluator.ALGORITHM_VERSION,
        },
        "review": {"reviewed": False, "effective": "algorithm"},
        "lifecycle": "no_structure",
        "points": [],
        "major_highs": [],
        "active_lid": None,
        "breakout": None,
        "pattern_anatomy": {
            "recognized": False,
            "maturity": {"passes": False},
            "boundary": None,
            "base": None,
            "congestion": None,
            "compression": None,
            "actionability": {
                "state": "none",
                "eligible": False,
                "signal_state": "breakout_now",
            },
        },
    }


def _install_algorithm_spy(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    observed = calls if calls is not None else []

    def spy(
        bars: list[dict[str, Any]],
        *,
        as_of: str,
        review_override: dict[str, Any] | None,
    ) -> dict[str, Any]:
        observed.append(
            {"bars": bars, "as_of": as_of, "review_override": review_override}
        )
        return _algorithm_analysis(bars, as_of)

    monkeypatch.setattr(evaluator, "analyze_coil", spy)
    return observed


def _blind_protocol(composition: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": "blind-protocol-test-v1",
        "finalized_at": "2026-08-22T00:00:00Z",
        "evidence_sha256": "d" * 64,
        "frozen_detector_version": evaluator.ALGORITHM_VERSION,
        "attestations": {
            name: True for name in evaluator.BLIND_PROTOCOL_ATTESTATIONS
        },
        "composition": composition
        or {
            "predicted_positive": 1,
            "near_boundary_negative": 0,
            "hard_trap": 0,
            "random_population": 0,
        },
    }


def _aggregate_item(
    *,
    setup_id: str,
    expected_shape: str,
    predicted_shape: str,
    counts: dict[str, int],
    correction_required: bool,
) -> dict[str, Any]:
    expected = {
        "shape": expected_shape,
        "maturity": "mature",
        "lifecycle": "pre_breakout",
        "readiness": "ready",
        "action": "actionable",
    }
    predicted = {**expected, "shape": predicted_shape}
    empty_counts = {"tp": 0, "fp": 0, "fn": 0}
    return {
        "setup_id": setup_id,
        "cutoff_date": CUTOFF,
        "expected": expected,
        "predicted": predicted,
        "tops": {"supported": True, **counts},
        "selected_top_membership_proxy": counts,
        "construction_anchors": counts,
        "supporting_touches": counts,
        "structures": {
            "matched": counts,
            "primary": counts,
            "alternate": empty_counts,
            "parent_child": empty_counts,
            "active_boundary": {
                "direction_correct": True,
                "projected_line_error_pct": 1.0,
            },
        },
        "excluded_highs": {"expected_count": 1, "rejected": 1},
        "bottoms": {
            "major_bottom": {"supported": True, **counts},
            "undercut": {"supported": False, "expected_count": 0},
            "outlier": {"supported": False, "expected_count": 0},
        },
        "phases": {
            kind: {
                "expected_present": True,
                "predicted_present": True,
                "start_error_candles": 0,
                "end_error_candles": 0,
                "timing_correct": True,
            }
            for kind in evaluator.PHASE_KINDS
        },
        "events": {
            "timing": {kind: empty_counts for kind in evaluator.EVENT_KINDS},
            "retest_state": {"expected": None, "predicted": None},
            "relative_volume_confirmation": {
                "expected": None,
                "predicted": None,
            },
        },
        "abstained": None,
        "false_action": False,
        "correction_required": correction_required,
        "critical_correction_required": correction_required,
        "intervention_checks": [],
    }


def _gold_structure(
    structure_id: str,
    *,
    selection: str = "primary",
    role: str = "primary_lid",
    relationship: str = "standalone",
    parent_id: str | None = None,
    value: float = 100.0,
    slope: float = 0.0,
) -> dict[str, Any]:
    return {
        "id": structure_id,
        "selection": selection,
        "role": role,
        "boundary_kind": "line",
        "relationship": relationship,
        "parent_id": parent_id,
        "construction_anchors": [
            {"date": "2019-12-01", "price_field": "high"},
            {"date": "2020-03-01", "price_field": "high"},
        ],
        "supporting_touches": [],
        "excluded_highs": [],
        "line": {
            "direction": "flat",
            "value_at_cutoff": value,
            "slope_pct_per_year": slope,
        },
    }


def _predicted_structure(
    structure_id: str,
    *,
    selection: str = "primary",
    role: str = "primary_lid",
    relationship: str = "standalone",
    parent_id: str | None = None,
    value: float = 100.0,
    slope: float = 0.0,
) -> dict[str, Any]:
    return {
        "id": structure_id,
        "selection": selection,
        "role": role,
        "boundary_kind": "line",
        "relationship": relationship,
        "parent_id": parent_id,
        "construction_anchors": [
            {"date": "2019-12-01"},
            {"date": "2020-03-01"},
        ],
        "line": {
            "direction": "flat",
            "value_at_cutoff": value,
            "slope_pct_per_year": slope,
        },
    }


def test_facade_has_decision_time_but_no_override_parameter():
    signature = inspect.signature(evaluator.run_detector_only)
    assert tuple(signature.parameters) == (
        "monthly_bars",
        "cutoff_date",
        "decision_as_of",
    )
    assert "review_override" not in signature.parameters


def test_detector_call_disables_override_and_sanitizes_cutoff_prefix(monkeypatch):
    calls = _install_algorithm_spy(monkeypatch)
    analysis, detector_bars = evaluator.run_detector_only(
        _monthly_bars(), CUTOFF, DECISION_AS_OF
    )
    assert analysis["review"] == {"reviewed": False, "effective": "algorithm"}
    assert len(calls) == 1
    assert calls[0]["as_of"] == DECISION_AS_OF
    assert calls[0]["review_override"] is None
    assert calls[0]["bars"] == detector_bars
    assert [bar["date"] for bar in detector_bars] == [
        f"2020-{month:02d}-01" for month in range(1, 7)
    ]
    assert all(set(bar) == DETECTOR_BAR_FIELDS for bar in detector_bars)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("algorithm_version",), "wrong-version"),
        (("as_of",), "2020-05-01"),
        (("bar_count",), 5),
        (("analysis_metadata", "history_end"), "2020-05-01"),
        (("analysis_metadata", "bar_count_monthly"), 5),
        (("analysis_metadata", "algorithm_version"), "wrong-version"),
    ],
)
def test_detector_provenance_must_match_exactly(
    monkeypatch, path: tuple[str, ...], bad_value: Any
):
    def corrupt_result(
        bars: list[dict[str, Any]],
        *,
        as_of: str,
        review_override: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result = _algorithm_analysis(bars, as_of)
        target = result
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad_value
        return result

    monkeypatch.setattr(evaluator, "analyze_coil", corrupt_result)
    with pytest.raises(evaluator.DetectorOnlyViolation):
        evaluator.run_detector_only(_monthly_bars(), CUTOFF, DECISION_AS_OF)


def test_human_effective_result_fails_closed(monkeypatch):
    def human_result(
        bars: list[dict[str, Any]],
        *,
        as_of: str,
        review_override: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result = _algorithm_analysis(bars, as_of)
        result["review"] = {"reviewed": True, "effective": "human"}
        return result

    monkeypatch.setattr(evaluator, "analyze_coil", human_result)
    with pytest.raises(evaluator.DetectorOnlyViolation, match="human-effective"):
        evaluator.run_detector_only(_monthly_bars(), CUTOFF, DECISION_AS_OF)


def test_effective_actionability_ignores_diagnostic_signal_state():
    analysis = _algorithm_analysis(_monthly_bars()[:6], DECISION_AS_OF)
    predicted = evaluator._classification_predictions(analysis)
    assert predicted["action"] == "avoid"
    assert predicted["readiness"] == "not_ready"
    assert predicted["abstained"] is None


def test_future_bar_values_cannot_change_historical_episode(monkeypatch):
    _install_algorithm_spy(monkeypatch)
    bars = _monthly_bars()
    gold = materialize_gold_label(_gold_capture("future-invariance"), bars)
    changed_future = copy.deepcopy(bars)
    for bar in changed_future[6:]:
        bar.update(open=9_000.0, high=10_000.0, low=8_000.0, close=9_500.0)
        bar["review_override"] = {"force": "actionable"}
    baseline = evaluator.evaluate_episode(gold, bars)
    future_changed = evaluator.evaluate_episode(gold, changed_future)
    assert future_changed == baseline
    assert future_changed["detector"]["bar_count"] == 6
    assert future_changed["detector"]["default_config_sha256"] == evaluator.DEFAULT_CONFIG_SHA256


def test_date_matching_maximizes_cardinality_before_distance():
    matched = evaluator._match_dates(
        ["2020-03-01", "2020-06-01"],
        ["2020-06-01", "2020-09-01"],
        tolerance=1,
    )
    assert (matched["tp"], matched["fp"], matched["fn"]) == (2, 0, 0)
    assert [(pair["expected"], pair["predicted"]) for pair in matched["pairs"]] == [
        ("2020-03-01", "2020-06-01"),
        ("2020-06-01", "2020-09-01"),
    ]


def test_missed_classes_have_zero_f1_and_null_f1_interval():
    missed = evaluator._prf(
        {"tp": 0, "fp": 0, "fn": 1}, evaluator.EvaluatorConfig().confidence_z
    )
    summary = evaluator._classification_summary(
        [("coil", "not_coil")], evaluator.EvaluatorConfig().confidence_z
    )
    assert missed["f1"]["value"] == 0.0
    assert missed["f1"]["ci_low"] is None
    assert missed["f1"]["ci_high"] is None
    assert summary["per_class"]["coil"]["f1"]["value"] == 0.0
    assert summary["per_class"]["not_coil"]["f1"]["value"] == 0.0
    assert summary["macro_f1"]["value"] == 0.0


def test_aggregate_bounds_and_unsupported_abstention():
    items = [
        _aggregate_item(
            setup_id="a",
            expected_shape="coil",
            predicted_shape="coil",
            counts={"tp": 1, "fp": 0, "fn": 0},
            correction_required=False,
        ),
        _aggregate_item(
            setup_id="b",
            expected_shape="coil",
            predicted_shape="not_coil",
            counts={"tp": 0, "fp": 1, "fn": 1},
            correction_required=True,
        ),
    ]
    aggregate = evaluator.aggregate_results(items, config=evaluator.EvaluatorConfig())
    top = aggregate["top_detection"]
    assert top["supported"] is True
    assert (top["tp"], top["fp"], top["fn"]) == (1, 1, 1)
    assert 0 <= top["precision"]["ci_low"] < 0.5 < top["precision"]["ci_high"] <= 1
    assert top["f1"]["value"] == pytest.approx(0.5)
    assert top["f1"]["ci_low"] == pytest.approx(0.0)
    assert top["f1"]["ci_high"] == pytest.approx(1.0)
    assert top["f1"]["ci_method"] == "setup_cluster_percentile_bootstrap"
    assert aggregate["gate_confidence"]["cluster_count"] == 2
    assert aggregate["gate_metrics"]["top_f1"]["point"] == pytest.approx(0.5)
    assert aggregate["structures"]["primary_selection"]["tp"] == 1
    assert aggregate["supporting_touches"]["tp"] == 1
    assert aggregate["bottoms"]["major_bottom"]["tp"] == 1
    assert aggregate["abstention_rate"]["supported"] is False
    assert aggregate["abstention_rate"]["value"] is None
    assert "human_correction_rate" not in aggregate
    assert aggregate["observed_human_correction_rate"]["supported"] is False
    assert aggregate["observed_human_correction_rate"]["value"] is None
    assert aggregate["estimated_human_intervention_rate"]["value"] == 0.5


def test_general_tops_require_candidates_and_membership_is_selected_only():
    analysis = {
        "major_highs": [{"date": "2020-06-01"}],
        "points": [
            {"date": "2020-06-01", "lid_member": False},
            {"date": "2020-09-01", "lid_member": True},
        ],
        "active_lid": {"anchors": [{"date": "2020-03-01"}], "touches": []},
        "pattern_anatomy": {},
    }
    assert evaluator._candidate_top_dates(analysis) == (False, [])
    selected = evaluator._selected_lid_member_dates(analysis)
    assert selected == ["2020-03-01", "2020-09-01"]
    assert evaluator._match_dates(["2020-06-01"], selected, tolerance=0)["tp"] == 0
    analysis["candidate_tops"] = [{"date": "2020-06-01"}]
    assert evaluator._candidate_top_dates(analysis) == (True, ["2020-06-01"])


def test_supporting_touches_are_scored_separately_from_anchors():
    analysis = {
        "active_lid": {
            "anchors": [{"date": "2020-03-01"}],
            # The analyzer repeats anchors in touches; they must not be
            # double-counted as supporting construction evidence.
            "touches": [
                {"date": "2020-03-01"},
                {"date": "2020-06-01"},
            ],
        },
        "pattern_anatomy": {},
        "points": [],
    }
    gold = {
        "structures": [
            {
                "construction_anchors": [{"date": "2020-03-01"}],
                "supporting_touches": [{"date": "2020-06-01"}],
            }
        ]
    }

    anchors = evaluator._match_dates(
        evaluator._expected_anchor_dates(gold),
        evaluator._predicted_anchor_dates(analysis),
        tolerance=0,
    )
    touches = evaluator._match_dates(
        evaluator._expected_supporting_touch_dates(gold),
        evaluator._predicted_supporting_touch_dates(analysis),
        tolerance=0,
    )

    assert anchors == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "pairs": [
            {
                "expected": "2020-03-01",
                "predicted": "2020-03-01",
                "candle_error": 0,
            }
        ],
        "unmatched_expected": [],
        "unmatched_predicted": [],
    }
    assert touches["tp"] == 1
    assert touches["fp"] == 0
    assert evaluator._predicted_supporting_touch_dates(analysis) == ["2020-06-01"]


def test_bottom_roles_are_grouped_and_only_major_bottom_is_supported():
    gold = {
        "bottoms": [
            {"role": "major_bottom", "point": {"date": "2020-03-01"}},
            {"role": "undercut", "point": {"date": "2020-06-01"}},
            {"role": "outlier", "point": {"date": "2020-09-01"}},
        ]
    }
    analysis = {
        "metrics": {
            "pullback_lows": [
                {"date": "2020-03-01"},
                # Pullback lows must not be reused as undercut predictions.
                {"date": "2020-06-01"},
            ]
        }
    }

    bottoms = evaluator._bottom_metrics(
        gold, analysis, evaluator.EvaluatorConfig(candle_tolerance=0)
    )

    assert bottoms["major_bottom"]["supported"] is True
    assert (bottoms["major_bottom"]["tp"], bottoms["major_bottom"]["fp"]) == (1, 1)
    assert bottoms["undercut"]["supported"] is False
    assert bottoms["undercut"]["expected_count"] == 1
    assert bottoms["undercut"]["tp"] is None
    assert bottoms["outlier"]["supported"] is False
    assert bottoms["outlier"]["expected_count"] == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(role="support"),
        lambda value: value.update(boundary_kind="resistance_band"),
        lambda value: value["line"].update(direction="rising"),
        lambda value: value["line"].update(value_at_cutoff=106.0),
        lambda value: value["line"].update(slope_pct_per_year=1.1),
        lambda value: value.update(
            construction_anchors=[
                {"date": "2018-12-01"},
                {"date": "2019-03-01"},
            ]
        ),
    ],
    ids=["role", "kind", "direction", "projection", "slope", "anchors"],
)
def test_structure_matching_requires_geometry_and_anchor_linkage(
    mutate: Callable[[dict[str, Any]], None],
):
    gold = {
        "active_structure_id": "gold-lid",
        "structures": [_gold_structure("gold-lid")],
    }
    prediction = _predicted_structure("predicted-lid")
    exact = evaluator._structure_metrics(
        gold,
        {"pattern_anatomy": {"structures": [prediction]}},
        evaluator.EvaluatorConfig(),
    )
    assert exact["matched"]["tp"] == 1
    changed = copy.deepcopy(prediction)
    mutate(changed)
    rejected = evaluator._structure_metrics(
        gold,
        {"pattern_anatomy": {"structures": [changed]}},
        evaluator.EvaluatorConfig(),
    )
    assert rejected["matched"]["tp"] == 0


def test_nested_primaries_are_counted_and_active_id_drives_boundary():
    gold = {
        "active_structure_id": "inner",
        "structures": [
            _gold_structure("outer", relationship="parent"),
            _gold_structure(
                "inner",
                relationship="child",
                parent_id="outer",
                role="secondary_lid",
                value=90.0,
            ),
        ],
    }
    predicted = [
        _predicted_structure("pred-outer", relationship="parent"),
        _predicted_structure(
            "pred-inner",
            relationship="child",
            parent_id="pred-outer",
            role="secondary_lid",
            value=90.0,
        ),
    ]
    metrics = evaluator._structure_metrics(
        gold,
        {"pattern_anatomy": {"structures": predicted}},
        evaluator.EvaluatorConfig(),
    )
    assert metrics["primary"] == {"tp": 2, "fp": 0, "fn": 0}
    assert metrics["parent_child"] == {"tp": 1, "fp": 0, "fn": 0}
    assert metrics["active_boundary"]["matched_predicted_id"] == "pred-inner"
    predicted[1]["selection"] = "alternate"
    mismatch = evaluator._structure_metrics(
        gold,
        {"pattern_anatomy": {"structures": predicted}},
        evaluator.EvaluatorConfig(),
    )
    assert mismatch["primary"] == {"tp": 1, "fp": 0, "fn": 1}
    assert mismatch["active_boundary"]["selection_correct"] is False


def test_phase_aggregate_reports_timing_correctness_coverage_and_error():
    first = _aggregate_item(
        setup_id="phase-a",
        expected_shape="coil",
        predicted_shape="coil",
        counts={"tp": 0, "fp": 0, "fn": 0},
        correction_required=False,
    )
    second = _aggregate_item(
        setup_id="phase-b",
        expected_shape="coil",
        predicted_shape="coil",
        counts={"tp": 0, "fp": 0, "fn": 0},
        correction_required=True,
    )
    second["phases"]["base"].update(
        start_error_candles=2,
        end_error_candles=1,
        timing_correct=False,
    )

    aggregate = evaluator.aggregate_results(
        [first, second], config=evaluator.EvaluatorConfig()
    )
    base = aggregate["phases"]["base"]

    assert base["presence"]["accuracy"]["value"] == 1.0
    assert base["timing_correctness"]["value"] == 0.5
    assert base["timing_coverage"]["value"] == 1.0
    assert base["start_error_candles"]["mean"] == 1.0
    assert base["end_error_candles"]["mean"] == 0.5


def test_intervention_policy_includes_whole_pattern_checks_but_not_unsupported_tops():
    empty_counts = {"tp": 0, "fp": 0, "fn": 0}
    structures = {
        "matched": empty_counts,
        "primary": {"tp": 0, "fp": 0, "fn": 1},
        "alternate": empty_counts,
        "parent_child": {"tp": 0, "fp": 0, "fn": 1},
        "active_boundary": {
            "matched": False,
            "selection_correct": False,
            "direction_correct": False,
            "within_tolerance": False,
        },
    }
    phases = {
        "base": {
            "expected_present": True,
            "predicted_present": True,
            "presence_correct": True,
            "predicted_start": "2020-03-01",
            "predicted_end": "2020-06-01",
            "timing_correct": False,
        },
        "congestion": {
            "expected_present": True,
            "predicted_present": False,
            "presence_correct": False,
            "predicted_start": None,
            "predicted_end": None,
            "timing_correct": None,
        },
        "compression": {
            "expected_present": False,
            "predicted_present": False,
            "presence_correct": True,
            "predicted_start": None,
            "predicted_end": None,
            "timing_correct": None,
        },
    }
    events = {
        "timing": {kind: empty_counts for kind in evaluator.EVENT_KINDS},
        "retest_state": {
            "expected": "shallow",
            "predicted": "deep",
            "correct": False,
            "supported": True,
        },
        "relative_volume_confirmation": {
            "expected": "confirmed",
            "predicted": None,
            "correct": False,
            "supported": False,
        },
    }
    checks = evaluator._intervention_policy_checks(
        gold={"top_review_complete": True, "active_structure_id": "lid"},
        classification_correct={
            field: True
            for field in ("shape", "maturity", "lifecycle", "readiness", "action")
        },
        top_metrics=evaluator._unsupported_top_metric(1),
        selected_membership_proxy=empty_counts,
        anchor_metrics=empty_counts,
        supporting_touch_metrics={"tp": 0, "fp": 0, "fn": 1},
        bottom_metrics={
            "major_bottom": {"tp": 0, "fp": 0, "fn": 1},
            "undercut": {
                "expected_count": 0,
                "reason": "unsupported",
            },
            "outlier": {
                "expected_count": 1,
                "reason": "unsupported",
            },
        },
        excluded_highs={"incorrectly_selected": 1},
        structures=structures,
        phases=phases,
        events=events,
    )
    by_name = {check["name"]: check for check in checks}

    assert by_name["general_top_detection"]["supported"] is False
    assert by_name["general_top_detection"]["included_in_estimate"] is False
    assert by_name["general_top_detection"]["requires_intervention"] is False
    for name in (
        "supporting_touches",
        "excluded_highs",
        "structure.primary_selection",
        "structure.parent_child_topology",
        "phase.base.timing",
        "phase.congestion.presence",
        "event.retest_state",
        "event.relative_volume_confirmation",
    ):
        assert by_name[name]["requires_intervention"] is True
    assert by_name["bottom.outlier"]["supported"] is False
    assert by_name["bottom.outlier"]["included_in_estimate"] is False


def test_materialized_gold_must_match_capture_rematerialization(monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "analyze_coil",
        lambda *args, **kwargs: pytest.fail("detector ran before gold check"),
    )
    bars = _monthly_bars()
    tampered = materialize_gold_label(_gold_capture("rematerialize"), bars)
    tampered["derived_dates"]["first_watch_date"] = "2020-03-01"
    unhashed = dict(tampered)
    unhashed.pop("label_sha256")
    tampered["label_sha256"] = evaluator.sha256_json(unhashed)
    with pytest.raises(evaluator.BenchmarkCorpusError, match="rematerialization"):
        evaluator.evaluate_episode(tampered, bars)


def test_episode_split_must_match_gold_role_before_detection(monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "analyze_coil",
        lambda *args, **kwargs: pytest.fail("detector ran before split check"),
    )
    corpus = {
        "schema_version": evaluator.CORPUS_SCHEMA_VERSION,
        "kind": evaluator.CORPUS_KIND,
        "corpus_id": "split-role-mismatch",
        "episodes": [
            {
                "split": "validation",
                "monthly_bars": _monthly_bars(),
                "gold_capture": _gold_capture("split-mismatch"),
            }
        ],
    }
    with pytest.raises(evaluator.BenchmarkCorpusError, match="split must exactly"):
        evaluator.evaluate_corpus(corpus)


def test_same_setup_cannot_cross_splits(monkeypatch):
    _install_algorithm_spy(monkeypatch)
    corpus = {
        "schema_version": evaluator.CORPUS_SCHEMA_VERSION,
        "kind": evaluator.CORPUS_KIND,
        "corpus_id": "same-setup-split-leakage",
        "episodes": [
            {
                "split": "development",
                "monthly_bars": _monthly_bars(),
                "gold_capture": _gold_capture("dev", setup_id="shared"),
            },
            {
                "split": "validation",
                "monthly_bars": _monthly_bars(),
                "gold_capture": _gold_capture(
                    "validation",
                    setup_id="shared",
                    evaluation_role="validation",
                ),
            },
        ],
    }
    with pytest.raises(evaluator.BenchmarkCorpusError, match="multiple splits"):
        evaluator.evaluate_corpus(corpus)


def test_outcome_visible_blind_label_is_rejected_before_detection(monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "analyze_coil",
        lambda *args, **kwargs: pytest.fail("detector ran before leakage check"),
    )
    bars = _monthly_bars()
    gold = materialize_gold_label(
        _gold_capture(
            "outcome-visible",
            evaluation_role="blind_benchmark",
            outcome_visible=True,
        ),
        bars,
    )
    with pytest.raises(evaluator.BenchmarkCorpusError, match="outcome-visible"):
        evaluator.evaluate_episode(gold, bars)


def test_one_item_blind_report_is_ineligible_with_reasons(monkeypatch):
    _install_algorithm_spy(monkeypatch)
    corpus = {
        "schema_version": evaluator.CORPUS_SCHEMA_VERSION,
        "kind": evaluator.CORPUS_KIND,
        "corpus_id": "one-is-not-holdout",
        "blind_protocol": _blind_protocol(),
        "episodes": [
            {
                "split": "blind_benchmark",
                "sampling_stratum": "predicted_positive",
                "monthly_bars": _monthly_bars(),
                "gold_capture": _gold_capture(
                    "one-blind",
                    setup_id="one",
                    evaluation_role="blind_benchmark",
                ),
            }
        ],
    }
    report = evaluator.evaluate_corpus(corpus)
    assert report["holdout_claim_eligible"] is False
    codes = {reason["code"] for reason in report["holdout_claim_failures"]}
    assert "insufficient_unique_setups" in codes
    eligibility = report["aggregate"]["holdout_claim_eligibility"]
    assert eligibility["unique_setup_count"] == 1
    assert report["episodes"][0]["top_review_complete"] is True
    assert report["aggregate"]["top_detection"]["supported"] is False
    assert report["aggregate"]["abstention_rate"]["supported"] is False
    assert report["episodes"][0]["intervention_policy_version"] == (
        evaluator.INTERVENTION_POLICY_VERSION
    )
    top_check = next(
        check
        for check in report["episodes"][0]["intervention_checks"]
        if check["name"] == "general_top_detection"
    )
    assert top_check["requires_intervention"] is False


def test_complete_400_setup_blind_protocol_is_eligible():
    results = [
        {
            "setup_id": f"blind-{index:03d}",
            "evaluation_role": "blind_benchmark",
            "outcome_visible_during_label": False,
            "top_review_complete": True,
            "sampling_stratum": (
                "predicted_positive"
                if index < 120
                else (
                    "near_boundary_negative"
                    if index < 240
                    else ("hard_trap" if index < 320 else "random_population")
                )
            ),
        }
        for index in range(400)
    ]
    corpus = {
        "blind_protocol": _blind_protocol(
            {
                "predicted_positive": 120,
                "near_boundary_negative": 120,
                "hard_trap": 80,
                "random_population": 80,
            }
        )
    }
    eligibility = evaluator._holdout_claim_eligibility(corpus, results)
    assert eligibility["eligible"] is True
    assert eligibility["failure_reasons"] == []

    results[0]["sampling_stratum"] = "random_population"
    changed = evaluator._holdout_claim_eligibility(corpus, results)
    assert changed["eligible"] is False
    assert "blind_composition_not_reproducible" in {
        reason["code"] for reason in changed["failure_reasons"]
    }

    results[0]["sampling_stratum"] = "predicted_positive"
    results[0]["top_review_complete"] = False
    incomplete = evaluator._holdout_claim_eligibility(corpus, results)
    assert incomplete["eligible"] is False
    assert "incomplete_top_review" in {
        reason["code"] for reason in incomplete["failure_reasons"]
    }


def test_blind_protocol_requires_frozen_identity_evidence():
    protocol = _blind_protocol()
    protocol.pop("evidence_sha256")
    eligibility = evaluator._holdout_claim_eligibility(
        {"blind_protocol": protocol},
        [
            {
                "setup_id": "only",
                "evaluation_role": "blind_benchmark",
                "outcome_visible_during_label": False,
                "top_review_complete": True,
                "sampling_stratum": "predicted_positive",
            }
        ],
    )

    assert "blind_protocol_identity_invalid" in {
        reason["code"] for reason in eligibility["failure_reasons"]
    }


def test_real_analyzer_smoke_is_algorithm_only_and_records_config():
    bars = _monthly_bars()
    gold = materialize_gold_label(_gold_capture("real-smoke"), bars)
    result = evaluator.evaluate_episode(gold, bars)
    assert result["detector"]["algorithm_version"] == evaluator.ALGORITHM_VERSION
    assert result["detector"]["default_config_sha256"] == evaluator.DEFAULT_CONFIG_SHA256
    assert result["detector"]["review_override_applied"] is False
    assert all(result["classification_correct"].values())
    assert result["correction_required"] is False
    assert result["abstained"] is None
