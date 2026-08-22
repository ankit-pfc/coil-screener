"""Confidence intervals for benchmark statistics with repeated checkpoints.

Benchmark episodes are not independent when several historical cutoffs belong
to the same setup.  This module therefore resamples whole setup clusters: every
checkpoint from a selected setup is copied into a bootstrap replicate, and a
setup selected more than once contributes all of its checkpoints each time.

The implementation intentionally has no dependency on the detector, gold-label
schema, NumPy, or SciPy so it can be reused by offline gates and report tooling.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar


T = TypeVar("T", bound=Mapping[str, Any])
ScalarStatistic = Callable[[Sequence[T]], float | int | None]

METHOD = "setup_cluster_percentile_bootstrap"


def _statistic_value(value: float | int | None) -> float | None:
    """Normalize a statistic result, treating missing/non-finite values as absent."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("statistic must return a numeric scalar or None") from exc
    return number if math.isfinite(number) else None


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile from an already sorted sample."""
    if not sorted_values:
        raise ValueError("a percentile requires at least one value")
    if probability <= 0.0:
        return float(sorted_values[0])
    if probability >= 1.0:
        return float(sorted_values[-1])
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = float(sorted_values[lower_index])
    upper = float(sorted_values[upper_index])
    if lower_index == upper_index:
        return lower
    return lower + (upper - lower) * (position - lower_index)


def setup_cluster_bootstrap_interval(
    checkpoints: Sequence[T],
    statistic: ScalarStatistic[T],
    *,
    setup_key: str = "setup_id",
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate a percentile interval by resampling complete setup clusters.

    ``statistic`` is evaluated once on the original checkpoints to obtain the
    point estimate.  Each bootstrap replicate then samples the same number of
    unique setup IDs with replacement and evaluates ``statistic`` on all
    checkpoints belonging to those sampled IDs.

    Bounds are deliberately unavailable when fewer than two unique setups are
    present, the point statistic is unavailable, or any bootstrap replicate
    cannot produce the statistic.  Dropping unavailable replicates would make a
    gate's interval depend on a selectively easier subset of resamples.
    """
    if not callable(statistic):
        raise TypeError("statistic must be callable")
    if not isinstance(setup_key, str) or not setup_key:
        raise ValueError("setup_key must be a non-empty string")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    clusters: dict[str, list[T]] = {}
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"checkpoint {index} must be a mapping")
        setup_id = checkpoint.get(setup_key)
        if not isinstance(setup_id, str) or not setup_id.strip():
            raise ValueError(
                f"checkpoint {index} requires a non-empty string {setup_key}"
            )
        clusters.setdefault(setup_id, []).append(checkpoint)

    point = _statistic_value(statistic(checkpoints))
    cluster_ids = sorted(clusters)
    cluster_count = len(cluster_ids)
    result: dict[str, Any] = {
        "point": point,
        "ci_low": None,
        "ci_high": None,
        "cluster_count": cluster_count,
        "method": METHOD,
    }
    if cluster_count < 2 or point is None:
        return result

    rng = random.Random(seed)
    bootstrap_values: list[float] = []
    for _ in range(samples):
        replicate: list[T] = []
        for _ in range(cluster_count):
            sampled_id = cluster_ids[rng.randrange(cluster_count)]
            replicate.extend(clusters[sampled_id])
        value = _statistic_value(statistic(replicate))
        if value is None:
            return result
        bootstrap_values.append(value)

    bootstrap_values.sort()
    tail = (1.0 - confidence) / 2.0
    result["ci_low"] = _percentile(bootstrap_values, tail)
    result["ci_high"] = _percentile(bootstrap_values, 1.0 - tail)
    return result


__all__ = ["METHOD", "setup_cluster_bootstrap_interval"]
