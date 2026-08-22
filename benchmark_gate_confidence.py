"""Setup-cluster confidence bounds for detector benchmark release gates."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from benchmark_confidence import setup_cluster_bootstrap_interval


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return 2.0 * tp / denominator if denominator else None


def _top_f1(items: Sequence[Mapping[str, Any]]) -> float | None:
    if not items or any(
        item.get("tops", {}).get("supported") is not True for item in items
    ):
        return None
    return _f1(
        sum(int(item["tops"]["tp"]) for item in items),
        sum(int(item["tops"]["fp"]) for item in items),
        sum(int(item["tops"]["fn"]) for item in items),
    )


def _lifecycle_macro_f1(
    items: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> float | None:
    if not items or not labels:
        return None
    scores: list[float] = []
    for label in labels:
        tp = sum(
            item["expected"]["lifecycle"]
            == item["predicted"]["lifecycle"]
            == label
            for item in items
        )
        fp = sum(
            item["expected"]["lifecycle"] != label
            and item["predicted"]["lifecycle"] == label
            for item in items
        )
        fn = sum(
            item["expected"]["lifecycle"] == label
            and item["predicted"]["lifecycle"] != label
            for item in items
        )
        # The full-corpus label universe is fixed before resampling. A class
        # absent from a replicate contributes zero instead of disappearing and
        # inflating that replicate's macro score.
        scores.append(_f1(tp, fp, fn) or 0.0)
    return sum(scores) / len(scores)


def _action_precision(items: Sequence[Mapping[str, Any]]) -> float | None:
    predicted = sum(
        item["predicted"]["action"] == "actionable" for item in items
    )
    if not predicted:
        return None
    correct = sum(
        item["predicted"]["action"] == "actionable"
        and item["expected"]["action"] == "actionable"
        for item in items
    )
    return correct / predicted


def _false_action_rate(items: Sequence[Mapping[str, Any]]) -> float | None:
    predicted = sum(
        item["predicted"]["action"] == "actionable" for item in items
    )
    if not predicted:
        return None
    return sum(bool(item["false_action"]) for item in items) / predicted


def _boolean_rate(
    items: Sequence[Mapping[str, Any]], field: str
) -> float | None:
    if not items:
        return None
    return sum(bool(item[field]) for item in items) / len(items)


def benchmark_gate_confidence(
    items: Sequence[Mapping[str, Any]],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return statistically valid intervals for progressive automation gates."""
    lifecycle_labels = sorted(
        {
            str(value)
            for item in items
            for value in (
                item["expected"]["lifecycle"],
                item["predicted"]["lifecycle"],
            )
        }
    )
    statistics = {
        "top_f1": _top_f1,
        "lifecycle_macro_f1": lambda sample: _lifecycle_macro_f1(
            sample, lifecycle_labels
        ),
        "action_precision": _action_precision,
        "false_action_rate": _false_action_rate,
        "estimated_human_intervention_rate": lambda sample: _boolean_rate(
            sample, "correction_required"
        ),
        "critical_intervention_rate": lambda sample: _boolean_rate(
            sample, "critical_correction_required"
        ),
    }
    metrics = {
        name: setup_cluster_bootstrap_interval(
            items,
            statistic,
            confidence=confidence,
            samples=samples,
            seed=seed,
        )
        for name, statistic in statistics.items()
    }
    cluster_count = len({str(item["setup_id"]) for item in items})
    return {
        "method": "setup_cluster_percentile_bootstrap",
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
        "cluster_count": cluster_count,
        "lifecycle_label_universe": lifecycle_labels,
        "metrics": metrics,
    }


__all__ = ["benchmark_gate_confidence"]
