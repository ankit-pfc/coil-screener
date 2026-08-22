from __future__ import annotations

import pytest

from benchmark_gate_confidence import (
    _lifecycle_macro_f1,
    benchmark_gate_confidence,
)


def _item(
    setup_id: str,
    *,
    expected_lifecycle: str,
    predicted_lifecycle: str,
    expected_action: str,
    predicted_action: str,
    top_supported: bool = True,
    correction: bool = False,
    critical: bool = False,
) -> dict:
    actionable = predicted_action == "actionable"
    return {
        "setup_id": setup_id,
        "expected": {
            "lifecycle": expected_lifecycle,
            "action": expected_action,
        },
        "predicted": {
            "lifecycle": predicted_lifecycle,
            "action": predicted_action,
        },
        "tops": {
            "supported": top_supported,
            "tp": 1,
            "fp": 0,
            "fn": 0,
        },
        "false_action": actionable and expected_action != "actionable",
        "correction_required": correction,
        "critical_correction_required": critical,
    }


def test_gate_bounds_cluster_repeated_checkpoints_by_setup() -> None:
    items = [
        _item(
            "A",
            expected_lifecycle="forming",
            predicted_lifecycle="forming",
            expected_action="watch",
            predicted_action="watch",
        ),
        _item(
            "A",
            expected_lifecycle="pre_breakout",
            predicted_lifecycle="forming",
            expected_action="actionable",
            predicted_action="actionable",
            correction=True,
        ),
        _item(
            "B",
            expected_lifecycle="pre_breakout",
            predicted_lifecycle="pre_breakout",
            expected_action="avoid",
            predicted_action="actionable",
            correction=True,
            critical=True,
        ),
        _item(
            "C",
            expected_lifecycle="post_breakout",
            predicted_lifecycle="post_breakout",
            expected_action="avoid",
            predicted_action="avoid",
        ),
    ]

    result = benchmark_gate_confidence(items, samples=500, seed=11)

    assert result["cluster_count"] == 3
    assert result["samples"] == 500
    assert result["metrics"]["top_f1"]["point"] == pytest.approx(1.0)
    assert result["metrics"]["estimated_human_intervention_rate"][
        "point"
    ] == pytest.approx(0.5)
    assert result["metrics"]["estimated_human_intervention_rate"][
        "ci_low"
    ] is not None


def test_macro_f1_keeps_full_label_universe_when_replicate_omits_class() -> None:
    sample = [
        _item(
            "A",
            expected_lifecycle="forming",
            predicted_lifecycle="forming",
            expected_action="watch",
            predicted_action="watch",
        )
    ]

    assert _lifecycle_macro_f1(
        sample, ["forming", "post_breakout"]
    ) == pytest.approx(0.5)


def test_unsupported_top_and_one_cluster_fail_closed() -> None:
    items = [
        _item(
            "only",
            expected_lifecycle="forming",
            predicted_lifecycle="forming",
            expected_action="watch",
            predicted_action="watch",
            top_supported=False,
        )
    ]

    result = benchmark_gate_confidence(items, samples=50)

    assert result["metrics"]["top_f1"]["point"] is None
    assert result["metrics"]["top_f1"]["ci_low"] is None
    assert result["metrics"]["lifecycle_macro_f1"]["point"] == pytest.approx(1.0)
    assert result["metrics"]["lifecycle_macro_f1"]["ci_low"] is None
