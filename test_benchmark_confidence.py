from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import pytest

from benchmark_confidence import METHOD, setup_cluster_bootstrap_interval


def _mean(checkpoints: Sequence[dict[str, Any]]) -> float | None:
    if not checkpoints:
        return None
    return sum(float(item["value"]) for item in checkpoints) / len(checkpoints)


def test_bootstrap_keeps_every_checkpoint_from_a_sampled_setup_together() -> None:
    checkpoints = [
        {"setup_id": "A", "checkpoint": "A-1", "value": 0.0},
        {"setup_id": "A", "checkpoint": "A-2", "value": 1.0},
        {"setup_id": "B", "checkpoint": "B-1", "value": 1.0},
    ]
    observed: list[list[str]] = []

    def recording_mean(sample: Sequence[dict[str, Any]]) -> float | None:
        observed.append([str(item["checkpoint"]) for item in sample])
        return _mean(sample)

    result = setup_cluster_bootstrap_interval(
        checkpoints,
        recording_mean,
        samples=80,
        seed=13,
    )

    assert result["cluster_count"] == 2
    assert len(observed) == 81  # one point estimate plus 80 bootstrap replicates
    for replicate in observed[1:]:
        counts = Counter(replicate)
        # A may be selected zero, one, or two times, but its two checkpoints
        # always appear with the same multiplicity.
        assert counts["A-1"] == counts["A-2"]


def test_bootstrap_is_deterministic_for_seed_and_semantic_input_order() -> None:
    checkpoints = [
        {"setup_id": "gamma", "value": 0.1},
        {"setup_id": "alpha", "value": 0.5},
        {"setup_id": "beta", "value": 0.9},
        {"setup_id": "gamma", "value": 0.3},
    ]

    first = setup_cluster_bootstrap_interval(
        checkpoints,
        _mean,
        confidence=0.9,
        samples=500,
        seed=2026,
    )
    second = setup_cluster_bootstrap_interval(
        list(reversed(checkpoints)),
        _mean,
        confidence=0.9,
        samples=500,
        seed=2026,
    )

    assert second == first
    assert first["method"] == METHOD


def test_bootstrap_returns_ordered_bounded_percentile_interval() -> None:
    checkpoints = [
        {"setup_id": "low", "value": 0.0},
        {"setup_id": "middle", "value": 0.5},
        {"setup_id": "high", "value": 1.0},
    ]

    result = setup_cluster_bootstrap_interval(
        checkpoints,
        _mean,
        confidence=0.95,
        samples=2_000,
        seed=7,
    )

    assert result["point"] == pytest.approx(0.5)
    assert 0.0 <= result["ci_low"] <= result["point"]
    assert result["point"] <= result["ci_high"] <= 1.0


def test_one_setup_returns_point_with_null_interval() -> None:
    checkpoints = [
        {"setup_id": "only", "value": 0.25},
        {"setup_id": "only", "value": 0.75},
    ]

    result = setup_cluster_bootstrap_interval(
        checkpoints,
        _mean,
        samples=100,
        seed=9,
    )

    assert result == {
        "point": pytest.approx(0.5),
        "ci_low": None,
        "ci_high": None,
        "cluster_count": 1,
        "method": METHOD,
    }


@pytest.mark.parametrize("unavailable", [None, float("nan"), float("inf")])
def test_unavailable_point_statistic_returns_null_interval(unavailable: float | None) -> None:
    checkpoints = [
        {"setup_id": "A", "value": 0.0},
        {"setup_id": "B", "value": 1.0},
    ]

    result = setup_cluster_bootstrap_interval(
        checkpoints,
        lambda _: unavailable,
        samples=20,
        seed=4,
    )

    assert result["point"] is None
    assert result["ci_low"] is None
    assert result["ci_high"] is None
    assert result["cluster_count"] == 2


def test_unavailable_bootstrap_replicate_fails_closed() -> None:
    checkpoints = [
        {"setup_id": "A", "value": 0.0},
        {"setup_id": "B", "value": 1.0},
    ]

    def unavailable_for_single_setup(sample: Sequence[dict[str, Any]]) -> float | None:
        if len({item["setup_id"] for item in sample}) < 2:
            return None
        return _mean(sample)

    result = setup_cluster_bootstrap_interval(
        checkpoints,
        unavailable_for_single_setup,
        samples=50,
        seed=2,
    )

    assert result["point"] == pytest.approx(0.5)
    assert result["ci_low"] is None
    assert result["ci_high"] is None
