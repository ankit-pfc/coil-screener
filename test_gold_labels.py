from __future__ import annotations

from copy import deepcopy

import pytest

from gold_labels import (
    CAPTURE_KIND,
    MATERIALIZED_KIND,
    WholePatternGoldCapture,
    materialize_gold_label,
    sha256_json,
    validate_materialized_gold_label,
    validate_materialized_gold_label_against_bars,
)


def _monthly_bars() -> list[dict[str, object]]:
    """Four completed quarters through the cutoff, plus future evidence."""
    rows = [
        ("2023-01-31", 100, 106, 96, 102, 10),
        ("2023-02-28", 102, 108, 98, 104, 20),
        ("2023-03-31", 104, 110, 99, 107, 70),
        ("2023-04-30", 106, 109, 100, 105, 20),
        ("2023-05-31", 105, 107, 99, 103, 40),
        ("2023-06-30", 103, 108, 97, 102, 60),
        ("2023-07-31", 101, 107, 96, 100, 30),
        ("2023-08-31", 100, 106, 95, 99, 40),
        ("2023-09-30", 99, 105, 94, 98, 70),
        ("2023-10-31", 99, 108, 95, 104, 100),
        ("2023-11-30", 104, 110, 100, 107, 80),
        ("2023-12-31", 107, 112, 103, 111, 120),
        ("2024-01-31", 112, 114, 108, 113, 160),
        ("2024-02-29", 113, 115, 109, 114, 160),
        ("2024-03-31", 114, 116, 110, 115, 160),
    ]
    return [
        {
            "date": bar_date,
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        }
        for bar_date, open_, high, low, close, volume in rows
    ]


def _click(bar_date: str, price_field: str = "high") -> dict[str, str]:
    return {"date": bar_date, "priceField": price_field}


def _single_line_capture() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "kind": CAPTURE_KIND,
        "episodeId": "TEST-2023-Q4",
        "setupId": "TEST-primary",
        "evaluationRole": "validation",
        "labeler": "Test Reviewer",
        "labeledAt": "2026-08-22T00:00:00Z",
        "cutoffDate": "2023-12-31",
        "decisionAsOf": "2023-12-31",
        "interval": "3M",
        "outcomeVisibleDuringLabel": False,
        "judgments": {
            "shape": "coil",
            "maturity": "mature",
            "lifecycle": "breaking_out",
            "readiness": "ready",
            "action": "actionable",
            "confidence": "high",
        },
        "activeStructureId": "lid",
        "topReviewComplete": True,
        "structures": [
            {
                "id": "lid",
                "relationship": "standalone",
                "selection": "primary",
                "role": "primary_lid",
                "boundaryKind": "line",
                "confidence": "high",
                "constructionAnchors": [
                    _click("2023-03-31"),
                    _click("2023-06-30"),
                ],
                "recognitionConfirmation": _click("2023-06-30", "close"),
                "supportingTouches": [_click("2023-09-30")],
                "excludedHighs": [_click("2023-12-31")],
            }
        ],
        "bottoms": [
            {
                "id": "base-low",
                "structureId": "lid",
                "role": "major_bottom",
                "click": _click("2023-06-30", "low"),
                "confidence": "high",
            }
        ],
        "phases": [
            {
                "id": "compression",
                "structureId": "lid",
                "kind": "compression",
                "start": _click("2023-09-30", "low"),
                "end": _click("2023-12-31", "close"),
                "present": True,
                "confidence": "high",
            }
        ],
        "events": [
            {
                "id": "release",
                "structureId": "lid",
                "kind": "breakout",
                "trigger": _click("2023-12-31"),
                "relativeVolume": "confirmed",
                "actionSignal": True,
                "confidence": "high",
            }
        ],
        "sourceEvidence": [
            {
                "kind": "blind_review_session",
                "reference": "session/test/item/TEST",
                "sha256": "a" * 64,
            }
        ],
        "notes": ["Synthetic point-in-time materialization fixture."],
    }


def _nested_capture() -> dict[str, object]:
    capture = _single_line_capture()
    capture["setupId"] = "TEST-nested"
    capture["activeStructureId"] = "inner"
    capture["structures"] = [
        {
            "id": "outer",
            "relationship": "parent",
            "selection": "primary",
            "role": "primary_lid",
            "boundaryKind": "line",
            "confidence": "high",
            "constructionAnchors": [
                _click("2023-03-31"),
                _click("2023-06-30"),
            ],
            "recognitionConfirmation": _click("2023-06-30", "close"),
        },
        {
            "id": "inner",
            "parentId": "outer",
            "relationship": "child",
            "selection": "primary",
            "role": "secondary_lid",
            "boundaryKind": "line",
            "confidence": "medium",
            "constructionAnchors": [
                _click("2023-06-30"),
                _click("2023-09-30"),
            ],
            "recognitionConfirmation": _click("2023-09-30", "close"),
        },
    ]
    capture["bottoms"] = []
    capture["events"] = []
    capture["judgments"]["action"] = "watch"  # type: ignore[index]
    capture["phases"] = [
        {
            "id": "inner-compression",
            "structureId": "inner",
            "kind": "compression",
            "start": _click("2023-09-30", "close"),
            "end": _click("2023-12-31", "close"),
            "present": True,
            "confidence": "medium",
        }
    ]
    return capture


def _month_start_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    quarter_end_to_month_start = {
        "2023-03-31": "2023-03-01",
        "2023-06-30": "2023-06-01",
        "2023-09-30": "2023-09-01",
        "2023-12-31": "2023-12-01",
    }
    bars = _monthly_bars()
    for bar in bars:
        bar["date"] = quarter_end_to_month_start.get(
            str(bar["date"]), str(bar["date"])
        )

    capture = _single_line_capture()
    capture["cutoffDate"] = "2023-12-01"

    def replace_click_dates(value: object) -> None:
        if isinstance(value, dict):
            if "date" in value:
                value["date"] = quarter_end_to_month_start.get(
                    str(value["date"]), str(value["date"])
                )
            for nested in value.values():
                replace_click_dates(nested)
        elif isinstance(value, list):
            for nested in value:
                replace_click_dates(nested)

    replace_click_dates(capture)
    return capture, bars


def test_materializes_single_line_coordinates_milestones_volume_and_hashes() -> None:
    bars = _monthly_bars()
    capture = _single_line_capture()

    label = materialize_gold_label(capture, bars)

    assert label["kind"] == MATERIALIZED_KIND
    assert label["cutoff_date"] == "2023-12-31"
    assert label["decision_as_of"] == "2023-12-31"
    assert label["top_review_complete"] is True
    assert label["structures"][0]["construction_anchors"] == [
        {"date": "2023-03-31", "idx": 0, "price": 110.0, "price_field": "high"},
        {"date": "2023-06-30", "idx": 1, "price": 109.0, "price_field": "high"},
    ]
    assert label["structures"][0]["recognition_confirmation"] == {
        "date": "2023-06-30",
        "idx": 1,
        "price": 102.0,
        "price_field": "close",
    }
    assert label["structures"][0]["supporting_touches"][0]["price"] == 107.0
    assert label["structures"][0]["excluded_highs"][0]["price"] == 112.0
    assert label["bottoms"][0]["point"] == {
        "date": "2023-06-30",
        "idx": 1,
        "price": 97.0,
        "price_field": "low",
    }
    assert label["phases"][0]["start"]["price"] == 94.0
    assert label["phases"][0]["end"]["price"] == 111.0

    line = label["structures"][0]["line"]
    assert line == {
        "slope_per_bar": -1.0,
        "slope_pct_per_year": -3.7383,
        "intercept": 110.0,
        "value_at_cutoff": 107.0,
        "projected_idx": 3,
        "direction": "falling",
        "fit_error_pct": 0.0,
        "recognition_date": "2023-06-30",
    }
    assert label["derived_dates"] == {
        "first_recognizable_date": "2023-06-30",
        "first_watch_date": "2023-09-30",
        "first_actionable_date": "2023-12-31",
    }
    event = label["events"][0]
    assert event["trigger"]["price"] == 112.0
    assert event["relative_volume_label"] == "confirmed"
    assert event["relative_volume_observed"] == {
        "lookback_bars": 8,
        "ratio": 2.5,
        "available": True,
    }

    expected_monthly = bars[:12]
    expected_quarterly = [
        {
            "date": "2023-03-31",
            "open": 100.0,
            "high": 110.0,
            "low": 96.0,
            "close": 107.0,
            "volume": 100.0,
        },
        {
            "date": "2023-06-30",
            "open": 106.0,
            "high": 109.0,
            "low": 97.0,
            "close": 102.0,
            "volume": 120.0,
        },
        {
            "date": "2023-09-30",
            "open": 101.0,
            "high": 107.0,
            "low": 94.0,
            "close": 98.0,
            "volume": 140.0,
        },
        {
            "date": "2023-12-31",
            "open": 99.0,
            "high": 112.0,
            "low": 95.0,
            "close": 111.0,
            "volume": 300.0,
        },
    ]
    provenance = label["provenance"]
    assert provenance["bars_through_cutoff_sha256"] == sha256_json(expected_monthly)
    assert provenance["completed_quarterly_bars_sha256"] == sha256_json(
        expected_quarterly
    )
    assert "future_bars_discarded" not in provenance
    normalized_capture = WholePatternGoldCapture.model_validate(capture).model_dump(
        mode="json", by_alias=True
    )
    assert provenance["capture_sha256"] == sha256_json(normalized_capture)
    assert label["capture"] == normalized_capture
    assert validate_materialized_gold_label(label) is label
    assert materialize_gold_label(label["capture"], bars) == label


def test_future_suffix_does_not_change_canonical_gold_identity() -> None:
    capture = _single_line_capture()
    bars = _monthly_bars()

    prefix_label = materialize_gold_label(capture, bars[:12])
    label_with_future = materialize_gold_label(capture, bars)

    assert label_with_future == prefix_label
    assert label_with_future["label_sha256"] == prefix_label["label_sha256"]


def test_labeled_at_requires_timezone_normalizes_utc_and_bounds_decision() -> None:
    capture = _single_line_capture()
    capture["labeledAt"] = "2026-08-22T05:30:00+05:30"

    label = materialize_gold_label(capture, _monthly_bars())

    assert label["labeled_at"] == "2026-08-22T00:00:00+00:00"
    assert label["capture"]["labeledAt"] == "2026-08-22T00:00:00Z"

    capture["labeledAt"] = "2026-08-22T00:00:00"
    with pytest.raises(ValueError, match="labeledAt must include a timezone offset"):
        materialize_gold_label(capture, _monthly_bars())

    capture["labeledAt"] = "2023-12-30T23:59:59Z"
    with pytest.raises(ValueError, match="decisionAsOf cannot be later than labeledAt"):
        materialize_gold_label(capture, _monthly_bars())


def test_materializes_nested_parent_child_and_uses_active_child_milestones() -> None:
    label = materialize_gold_label(_nested_capture(), _monthly_bars())

    outer, inner = label["structures"]
    assert (outer["id"], outer["relationship"], outer["parent_id"]) == (
        "outer",
        "parent",
        None,
    )
    assert outer["selection"] == "primary"
    assert (inner["id"], inner["relationship"], inner["parent_id"]) == (
        "inner",
        "child",
        "outer",
    )
    assert inner["selection"] == "primary"
    assert inner["construction_anchors"][0]["price"] == 109.0
    assert inner["construction_anchors"][1]["price"] == 107.0
    assert label["derived_dates"] == {
        "first_recognizable_date": "2023-09-30",
        "first_watch_date": "2023-09-30",
        "first_actionable_date": None,
    }


def test_materializes_three_level_parent_child_topology() -> None:
    capture = _nested_capture()
    inner = capture["structures"][1]  # type: ignore[index]
    inner["relationship"] = "parent_child"
    capture["activeStructureId"] = "innermost"
    capture["structures"].append(  # type: ignore[union-attr]
        {
            "id": "innermost",
            "parentId": "inner",
            "relationship": "child",
            "selection": "primary",
            "role": "secondary_lid",
            "boundaryKind": "line",
            "confidence": "medium",
            "constructionAnchors": [
                _click("2023-06-30"),
                _click("2023-12-31"),
            ],
            "recognitionConfirmation": _click("2023-12-31", "close"),
        }
    )

    label = materialize_gold_label(capture, _monthly_bars())

    assert [item["relationship"] for item in label["structures"]] == [
        "parent",
        "parent_child",
        "child",
    ]
    assert label["active_structure_id"] == "innermost"


def test_rejects_relationships_that_disagree_with_topology() -> None:
    capture = _nested_capture()
    capture["structures"][0]["relationship"] = "standalone"  # type: ignore[index]

    with pytest.raises(ValueError, match="relationship must be parent"):
        materialize_gold_label(capture, _monthly_bars())


def test_rejects_active_structure_below_alternate_ancestor() -> None:
    capture = _nested_capture()
    outer, inner = capture["structures"]  # type: ignore[misc]
    outer["relationship"] = "standalone"
    inner["parentId"] = "alternate-parent"
    capture["structures"].append(  # type: ignore[union-attr]
        {
            "id": "alternate-parent",
            "relationship": "parent",
            "selection": "alternate",
            "role": "primary_lid",
            "boundaryKind": "line",
            "confidence": "high",
            "constructionAnchors": [
                _click("2023-03-31"),
                _click("2023-09-30"),
            ],
            "recognitionConfirmation": _click("2023-09-30", "close"),
        }
    )

    with pytest.raises(
        ValueError, match="activeStructureId and every ancestor must be primary"
    ):
        materialize_gold_label(capture, _monthly_bars())


def test_month_start_cutoff_uses_distinct_quarter_end_decision_date() -> None:
    capture, bars = _month_start_fixture()

    label = materialize_gold_label(capture, bars)

    assert label["cutoff_date"] == "2023-12-01"
    assert label["decision_as_of"] == "2023-12-31"
    assert label["structures"][0]["line"]["recognition_date"] == "2023-06-01"
    assert label["derived_dates"] == {
        "first_recognizable_date": "2023-06-01",
        "first_watch_date": "2023-09-01",
        "first_actionable_date": "2023-12-01",
    }


def test_rejects_decision_before_calendar_quarter_end() -> None:
    capture, bars = _month_start_fixture()
    capture["decisionAsOf"] = "2023-12-30"

    with pytest.raises(
        ValueError, match="decisionAsOf must be on or after the calendar quarter end"
    ):
        materialize_gold_label(capture, bars)


def test_quarterly_volume_is_unavailable_unless_all_three_months_exist() -> None:
    capture = _single_line_capture()
    capture["events"][0]["relativeVolume"] = "unavailable"  # type: ignore[index]
    bars = _monthly_bars()
    bars[10]["volume"] = None

    label = materialize_gold_label(capture, bars)

    assert label["events"][0]["relative_volume_observed"] == {
        "lookback_bars": 8,
        "ratio": None,
        "available": False,
    }

    capture["events"][0]["relativeVolume"] = "confirmed"  # type: ignore[index]
    with pytest.raises(
        ValueError, match="labels relative volume without completed volume evidence"
    ):
        materialize_gold_label(capture, bars)


def test_rejects_impossible_ohlc_and_duplicate_calendar_months() -> None:
    capture = _single_line_capture()
    bars = _monthly_bars()
    bars[0]["high"] = 99.0

    with pytest.raises(ValueError, match="impossible OHLC containment"):
        materialize_gold_label(capture, bars)

    bars = _monthly_bars()
    duplicate = dict(bars[0])
    duplicate["date"] = "2023-01-30"
    bars.insert(0, duplicate)
    with pytest.raises(ValueError, match="at most one bar per month"):
        materialize_gold_label(capture, bars)


def test_rejects_sparse_interior_and_cutoff_quarters_but_allows_leading_fragment() -> None:
    capture = _single_line_capture()
    capture["events"] = []
    bars = _monthly_bars()

    sparse_cutoff = [
        bar
        for bar in bars
        if not str(bar["date"]).startswith(("2023-10", "2023-11"))
    ]
    with pytest.raises(ValueError, match="2023-Q4 requires exactly one bar"):
        materialize_gold_label(capture, sparse_cutoff)

    sparse_interior = [
        bar
        for bar in bars
        if not str(bar["date"]).startswith("2023-05")
    ]
    with pytest.raises(ValueError, match="2023-Q2 requires exactly one bar"):
        materialize_gold_label(capture, sparse_interior)

    leading_fragment = [
        bar for bar in bars if not str(bar["date"]).startswith("2023-01")
    ]
    structure = capture["structures"][0]  # type: ignore[index]
    structure["constructionAnchors"] = [
        _click("2023-06-30"),
        _click("2023-09-30"),
    ]
    structure["recognitionConfirmation"] = _click("2023-09-30", "close")
    structure["supportingTouches"] = []
    label = materialize_gold_label(capture, leading_fragment)
    assert label["provenance"]["completed_quarterly_bar_count"] == 3


def test_rejects_missing_complete_quarter_between_geometry_bars() -> None:
    capture = _single_line_capture()
    capture["events"] = []
    bars = [
        bar
        for bar in _monthly_bars()
        if not str(bar["date"]).startswith(("2023-04", "2023-05", "2023-06"))
    ]

    with pytest.raises(ValueError, match="calendar-contiguous"):
        materialize_gold_label(capture, bars)


def test_rejects_client_supplied_click_coordinates() -> None:
    capture = _single_line_capture()
    capture["structures"][0]["constructionAnchors"][0]["price"] = 999.0  # type: ignore[index]

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        materialize_gold_label(capture, _monthly_bars())


@pytest.mark.parametrize(
    ("first_anchor", "second_anchor"),
    [
        pytest.param("2022-12-31", "2023-06-30", id="not-in-frozen-bars"),
        pytest.param("2023-03-31", "2023-05-31", id="incomplete-quarter"),
        pytest.param("2023-03-31", "2024-03-31", id="post-cutoff"),
    ],
)
def test_rejects_arbitrary_incomplete_and_post_cutoff_clicks(
    first_anchor: str, second_anchor: str
) -> None:
    capture = _single_line_capture()
    capture["structures"][0]["constructionAnchors"] = [  # type: ignore[index]
        _click(first_anchor),
        _click(second_anchor),
    ]
    capture["structures"][0]["recognitionConfirmation"] = _click(  # type: ignore[index]
        second_anchor, "close"
    )
    capture["events"] = []

    with pytest.raises(
        ValueError, match="is not a completed frozen 3M candle at the cutoff"
    ):
        materialize_gold_label(capture, _monthly_bars())


def test_rejects_parent_child_hierarchy_cycles() -> None:
    capture = _nested_capture()
    outer, inner = capture["structures"]  # type: ignore[misc]
    outer["relationship"] = "child"
    outer["parentId"] = "inner"
    inner["parentId"] = "outer"

    with pytest.raises(ValueError, match="parent relationships must be acyclic"):
        materialize_gold_label(capture, _monthly_bars())


def test_rejects_multiple_primary_structures_in_one_sibling_scope() -> None:
    capture = _nested_capture()
    second_inner = deepcopy(capture["structures"][1])  # type: ignore[index]
    second_inner["id"] = "inner-2"
    capture["structures"].append(second_inner)  # type: ignore[union-attr]

    with pytest.raises(
        ValueError, match="each parentId group requires exactly one primary structure"
    ):
        materialize_gold_label(capture, _monthly_bars())


def test_rejects_conflicting_point_roles() -> None:
    capture = _single_line_capture()
    capture["structures"][0]["excludedHighs"] = [  # type: ignore[index]
        _click("2023-03-31")
    ]

    with pytest.raises(ValueError, match="excluded highs cannot also be line members"):
        materialize_gold_label(capture, _monthly_bars())


def test_rejects_recognition_confirmation_before_construction_is_complete() -> None:
    capture = _single_line_capture()
    capture["structures"][0]["recognitionConfirmation"] = _click(  # type: ignore[index]
        "2023-03-31", "close"
    )

    with pytest.raises(
        ValueError,
        match="recognitionConfirmation cannot precede construction anchors",
    ):
        materialize_gold_label(capture, _monthly_bars())


def test_rejects_inverted_resistance_band() -> None:
    capture = _single_line_capture()
    structure = capture["structures"][0]  # type: ignore[index]
    structure["boundaryKind"] = "resistance_band"
    structure["constructionAnchors"] = [
        _click("2023-03-31", "low"),
        _click("2023-06-30", "low"),
    ]
    structure["lowerBandTouches"] = [
        _click("2023-03-31", "high"),
        _click("2023-06-30", "high"),
    ]

    with pytest.raises(
        ValueError,
        match="lower line must remain below its upper line throughout the labeled interval",
    ):
        materialize_gold_label(capture, _monthly_bars())


def test_rejects_bad_derived_milestone_ordering() -> None:
    capture = _single_line_capture()
    capture["phases"][0]["start"] = _click("2023-03-31", "low")  # type: ignore[index]

    with pytest.raises(
        ValueError,
        match="derived recognition/watch/action milestones must be chronological",
    ):
        materialize_gold_label(capture, _monthly_bars())


def test_rejects_event_state_or_time_before_same_structure_breakout() -> None:
    capture = _single_line_capture()
    capture["events"] = [
        {
            "id": "early-continuation",
            "structureId": "lid",
            "kind": "continuation",
            "trigger": _click("2023-09-30", "close"),
            "actionSignal": True,
            "confidence": "high",
        }
    ]

    with pytest.raises(ValueError, match="requires an earlier active breakout"):
        materialize_gold_label(capture, _monthly_bars())

    capture = _single_line_capture()
    capture["structures"][0]["recognitionConfirmation"] = _click(  # type: ignore[index]
        "2023-09-30", "close"
    )
    capture["events"][0]["trigger"] = _click("2023-06-30", "high")  # type: ignore[index]
    with pytest.raises(ValueError, match="cannot precede structure recognition"):
        materialize_gold_label(capture, _monthly_bars())


def test_rejects_nonchronological_events_within_one_structure() -> None:
    capture = _single_line_capture()
    capture["events"] = [  # type: ignore[index]
        {
            "id": "release",
            "structureId": "lid",
            "kind": "breakout",
            "trigger": _click("2023-12-31"),
            "actionSignal": True,
            "confidence": "high",
        },
        {
            "id": "earlier-release",
            "structureId": "lid",
            "kind": "breakout",
            "trigger": _click("2023-09-30"),
            "actionSignal": False,
            "confidence": "medium",
        },
    ]

    with pytest.raises(ValueError, match="events for structure lid must be chronological"):
        materialize_gold_label(capture, _monthly_bars())


def test_rejects_non_action_capable_event_signal() -> None:
    capture = _single_line_capture()
    capture["events"] = [
        {
            "id": "invalidated",
            "structureId": "lid",
            "kind": "invalidation",
            "trigger": _click("2023-12-31", "close"),
            "actionSignal": True,
            "confidence": "high",
        }
    ]

    with pytest.raises(ValueError, match="actionSignal requires"):
        materialize_gold_label(capture, _monthly_bars())


def test_first_actionable_ignores_action_event_on_alternate_structure() -> None:
    capture = _single_line_capture()
    capture["structures"].append(  # type: ignore[union-attr]
        {
            "id": "alternate",
            "relationship": "standalone",
            "selection": "alternate",
            "role": "breakout_level",
            "boundaryKind": "line",
            "confidence": "medium",
            "constructionAnchors": [
                _click("2023-03-31"),
                _click("2023-06-30"),
            ],
            "recognitionConfirmation": _click("2023-06-30", "close"),
        }
    )
    capture["events"] = [  # type: ignore[index]
        {
            "id": "alternate-release",
            "structureId": "alternate",
            "kind": "breakout",
            "trigger": _click("2023-09-30"),
            "actionSignal": True,
            "confidence": "medium",
        },
        {
            "id": "active-release",
            "structureId": "lid",
            "kind": "breakout",
            "trigger": _click("2023-12-31"),
            "actionSignal": True,
            "confidence": "high",
        },
    ]

    label = materialize_gold_label(capture, _monthly_bars())

    assert label["derived_dates"]["first_actionable_date"] == "2023-12-31"


def test_rejects_tampered_materialized_label_hash() -> None:
    label = materialize_gold_label(_single_line_capture(), _monthly_bars())
    tampered = deepcopy(label)
    tampered["structures"][0]["line"]["slope_per_bar"] = 42.0

    with pytest.raises(
        ValueError, match="materialized gold label content hash does not match"
    ):
        validate_materialized_gold_label(tampered)


def test_frozen_bar_validation_rejects_rehashed_derived_geometry() -> None:
    bars = _monthly_bars()
    label = materialize_gold_label(_single_line_capture(), bars)
    tampered = deepcopy(label)
    tampered["structures"][0]["line"]["slope_per_bar"] = 42.0
    unhashed = dict(tampered)
    unhashed.pop("label_sha256")
    tampered["label_sha256"] = sha256_json(unhashed)

    assert validate_materialized_gold_label(tampered) is tampered
    with pytest.raises(ValueError, match="frozen-bar rematerialization"):
        validate_materialized_gold_label_against_bars(tampered, bars)


def test_rejects_non_normalized_embedded_capture_even_with_recomputed_label_hash() -> None:
    label = materialize_gold_label(_single_line_capture(), _monthly_bars())
    tampered = deepcopy(label)
    tampered["capture"]["labeler"] = " Test Reviewer "
    unhashed = dict(tampered)
    unhashed.pop("label_sha256")
    tampered["label_sha256"] = sha256_json(unhashed)

    with pytest.raises(
        ValueError, match="materialized gold label capture is not normalized"
    ):
        validate_materialized_gold_label(tampered)
