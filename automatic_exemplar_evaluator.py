"""Detector-only, point-in-time evaluation against whole-pattern gold labels.

This is intentionally separate from ``review_corpus.py`` and the review API.
Those paths summarize feedback or may apply a persisted human override.  This
module physically truncates frozen OHLCV bars at each historical cutoff, calls
``analyze_coil`` with ``review_override=None``, and fails the entire run if the
returned analysis is human-effective.
"""
from __future__ import annotations

import argparse
import calendar
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from benchmark_gate_confidence import benchmark_gate_confidence
from coil_analysis import ALGORITHM_VERSION, DEFAULT_CONFIG, analyze_coil
from gold_labels import (
    MATERIALIZED_KIND,
    _completed_quarterly_bars,
    _normalize_monthly_bars,
    canonical_json,
    materialize_gold_label,
    sha256_json,
    validate_materialized_gold_label_against_bars,
)


CORPUS_SCHEMA_VERSION = 1
CORPUS_KIND = "coilingview.detector-benchmark-corpus"
REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "coilingview.detector-only-exemplar-evaluation"
EVENT_KINDS = (
    "breakout",
    "failed_breakout",
    "retest",
    "continuation",
    "invalidation",
)
PHASE_KINDS = ("base", "congestion", "compression")
BOTTOM_ROLES = ("major_bottom", "undercut", "outlier")
INTERVENTION_POLICY_VERSION = "whole-pattern-discrepancy-v1"
DEFAULT_CONFIG_SHA256 = sha256_json(asdict(DEFAULT_CONFIG))
BLIND_PROTOCOL_ATTESTATIONS = (
    "ticker_identity_hidden",
    "detector_output_hidden",
    "reference_answer_hidden",
    "session_finalized_before_reveal",
    "future_data_leakage_audited",
    "independent_double_review_at_least_25pct",
    "repeat_review_after_washout_at_least_10pct",
    "setup_grouped_split",
)
BLIND_COMPOSITION_TARGETS = {
    "predicted_positive": 0.30,
    "near_boundary_negative": 0.30,
    "hard_trap": 0.20,
    "random_population": 0.20,
}
MIN_BLIND_SETUP_COUNT = 400
BLIND_PROTOCOL_IDENTITY_FIELDS = (
    "protocol_id",
    "finalized_at",
    "evidence_sha256",
    "frozen_detector_version",
)
_FULL_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class DetectorOnlyViolation(RuntimeError):
    """The evaluation path observed an override or non-point-in-time output."""


class BenchmarkCorpusError(ValueError):
    """The benchmark corpus is incomplete, leaky, or internally inconsistent."""


@dataclass(frozen=True)
class EvaluatorConfig:
    candle_tolerance: int = 1
    projected_line_tolerance_pct: float = 5.0
    slope_tolerance_pct_per_year: float = 1.0
    phase_tolerance: int = 1
    confidence_z: float = 1.959963984540054
    bootstrap_confidence: float = 0.95
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        if self.candle_tolerance < 0:
            raise ValueError("candle_tolerance must be nonnegative")
        if self.projected_line_tolerance_pct < 0:
            raise ValueError("projected_line_tolerance_pct must be nonnegative")
        if self.slope_tolerance_pct_per_year < 0:
            raise ValueError("slope_tolerance_pct_per_year must be nonnegative")
        if self.phase_tolerance < 0:
            raise ValueError("phase_tolerance must be nonnegative")
        if self.confidence_z <= 0:
            raise ValueError("confidence_z must be positive")
        if isinstance(self.bootstrap_confidence, bool) or not (
            0.0 < self.bootstrap_confidence < 1.0
        ):
            raise ValueError("bootstrap_confidence must be between zero and one")
        if isinstance(self.bootstrap_samples, bool) or self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(
            self.bootstrap_seed, int
        ):
            raise ValueError("bootstrap_seed must be an integer")


def _sanitize_detector_bars(
    raw_bars: list[dict[str, Any]], cutoff_date: str
) -> list[dict[str, Any]]:
    """Return an OHLCV-only prefix, stripping every label/model field."""
    clean = _normalize_monthly_bars(raw_bars, cutoff_date)
    return [
        {
            "date": bar["date"],
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": bar["close"],
            "volume": bar.get("volume"),
        }
        for bar in clean
    ]


def _validated_decision_as_of(cutoff_date: str, decision_as_of: str) -> str:
    try:
        cutoff = date.fromisoformat(str(cutoff_date))
        decision = date.fromisoformat(str(decision_as_of))
    except (TypeError, ValueError) as exc:
        raise DetectorOnlyViolation(
            "cutoff_date and decision_as_of must use ISO YYYY-MM-DD"
        ) from exc
    month_end = date(
        cutoff.year,
        cutoff.month,
        calendar.monthrange(cutoff.year, cutoff.month)[1],
    )
    if decision < month_end:
        raise DetectorOnlyViolation(
            "decision_as_of precedes completion of the cutoff monthly candle"
        )
    return decision.isoformat()


def _assert_detector_only_analysis(
    analysis: Any,
    *,
    cutoff_date: str,
    decision_as_of: str,
    expected_bar_count: int,
) -> dict[str, Any]:
    if not isinstance(analysis, dict):
        raise DetectorOnlyViolation("analyze_coil returned a non-object result")
    review = analysis.get("review")
    if not isinstance(review, dict):
        raise DetectorOnlyViolation("analysis is missing detector provenance")
    if review.get("reviewed") is not False or review.get("effective") != "algorithm":
        raise DetectorOnlyViolation(
            "human-effective analysis is prohibited in the automatic evaluator"
        )
    anatomy = analysis.get("pattern_anatomy") or {}
    boundary = anatomy.get("boundary") if isinstance(anatomy, dict) else None
    if isinstance(boundary, dict) and boundary.get("family") == "human_review":
        raise DetectorOnlyViolation(
            "human_review boundary leaked into detector-only evaluation"
        )
    active_lid = analysis.get("active_lid")
    if isinstance(active_lid, dict) and active_lid.get("boundary_family") == "human_review":
        raise DetectorOnlyViolation(
            "human_review active lid leaked into detector-only evaluation"
        )
    if analysis.get("algorithm_version") != ALGORITHM_VERSION:
        raise DetectorOnlyViolation(
            "analysis algorithm_version does not match the current detector"
        )
    if analysis.get("as_of") != cutoff_date:
        raise DetectorOnlyViolation(
            "analysis as_of does not exactly match the physical bar cutoff"
        )
    if analysis.get("bar_count") != expected_bar_count:
        raise DetectorOnlyViolation(
            "analysis bar_count does not match the sanitized cutoff prefix"
        )
    metadata = analysis.get("analysis_metadata")
    if not isinstance(metadata, dict):
        raise DetectorOnlyViolation("analysis is missing point-in-time metadata")
    if metadata.get("history_end") != cutoff_date:
        raise DetectorOnlyViolation(
            "analysis history_end does not exactly match the physical bar cutoff"
        )
    if metadata.get("bar_count_monthly") != expected_bar_count:
        raise DetectorOnlyViolation(
            "analysis metadata bar count does not match the cutoff prefix"
        )
    if metadata.get("algorithm_version") != ALGORITHM_VERSION:
        raise DetectorOnlyViolation(
            "analysis metadata algorithm_version does not match the detector"
        )
    # The result reports the last physical bar as ``as_of``. The distinct
    # decision timestamp controls calendar completeness inside the analyzer.
    _validated_decision_as_of(cutoff_date, decision_as_of)
    return analysis


def run_detector_only(
    monthly_bars: list[dict[str, Any]],
    cutoff_date: str,
    decision_as_of: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the only authorized analyzer call for benchmark predictions.

    There is deliberately no override parameter on this facade.  The explicit
    ``None`` below makes the invariant visible to spies and future refactors.
    """
    decision = _validated_decision_as_of(cutoff_date, decision_as_of)
    sanitized = _sanitize_detector_bars(monthly_bars, cutoff_date)
    analysis = analyze_coil(
        sanitized,
        as_of=decision,
        review_override=None,
    )
    return _assert_detector_only_analysis(
        analysis,
        cutoff_date=cutoff_date,
        decision_as_of=decision,
        expected_bar_count=len(sanitized),
    ), sanitized


def _quarter_ordinal(value: str) -> int:
    parsed = date.fromisoformat(str(value)[:10])
    return parsed.year * 4 + (parsed.month - 1) // 3


def _date_from_point(value: Any) -> str | None:
    if isinstance(value, dict) and value.get("date"):
        return str(value["date"])[:10]
    return None


def _unique_dates(points: Iterable[Any]) -> list[str]:
    return sorted(
        {
            point_date
            for point in points
            if (point_date := _date_from_point(point)) is not None
        }
    )


def _match_dates(
    expected_dates: Iterable[str],
    predicted_dates: Iterable[str],
    *,
    tolerance: int,
) -> dict[str, Any]:
    expected = sorted(set(expected_dates))
    predicted = sorted(set(predicted_dates))
    expected_ordinals = [_quarter_ordinal(value) for value in expected]
    predicted_ordinals = [_quarter_ordinal(value) for value in predicted]

    # Dynamic programming is important here.  A greedy exact-date match can
    # consume the only prediction available to a neighbouring gold candle and
    # reduce cardinality.  The ordered solution first maximizes the number of
    # one-to-one matches, then minimizes total timing error.
    @lru_cache(maxsize=None)
    def solve(
        expected_idx: int, predicted_idx: int
    ) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
        if expected_idx >= len(expected) or predicted_idx >= len(predicted):
            return (0, 0, ())
        options = [
            solve(expected_idx + 1, predicted_idx),
            solve(expected_idx, predicted_idx + 1),
        ]
        distance = abs(
            expected_ordinals[expected_idx] - predicted_ordinals[predicted_idx]
        )
        if distance <= tolerance:
            matched, total_error, suffix = solve(
                expected_idx + 1, predicted_idx + 1
            )
            options.append(
                (
                    matched + 1,
                    total_error + distance,
                    ((expected_idx, predicted_idx, distance),) + suffix,
                )
            )
        return min(
            options,
            key=lambda result: (-result[0], result[1], result[2]),
        )

    _, _, matched_indices = solve(0, 0)
    used_expected = {expected_idx for expected_idx, _, _ in matched_indices}
    used_predicted = {predicted_idx for _, predicted_idx, _ in matched_indices}
    pairs = [
        {
            "expected": expected[expected_idx],
            "predicted": predicted[predicted_idx],
            "candle_error": distance,
        }
        for expected_idx, predicted_idx, distance in matched_indices
    ]
    return {
        "tp": len(pairs),
        "fp": len(predicted) - len(pairs),
        "fn": len(expected) - len(pairs),
        "pairs": pairs,
        "unmatched_expected": [
            value for idx, value in enumerate(expected) if idx not in used_expected
        ],
        "unmatched_predicted": [
            value for idx, value in enumerate(predicted) if idx not in used_predicted
        ],
    }


def _wilson(successes: int, total: int, z: float) -> dict[str, Any]:
    if total <= 0:
        return {
            "numerator": successes,
            "denominator": total,
            "value": None,
            "ci_low": None,
            "ci_high": None,
        }
    point = successes / total
    denominator = 1.0 + z * z / total
    center = (point + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            point * (1.0 - point) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "numerator": successes,
        "denominator": total,
        "value": round(point, 6),
        "ci_low": round(max(0.0, center - margin), 6),
        "ci_high": round(min(1.0, center + margin), 6),
    }


def _f1_metric(tp: int, fp: int, fn: int) -> dict[str, Any]:
    denominator = 2 * tp + fp + fn
    value = (2.0 * tp / denominator) if denominator else None
    return {
        "value": round(value, 6) if value is not None else None,
        # Precision and recall Wilson intervals are correlated.  Taking their
        # harmonic means is not a confidence interval for F1.  A setup-cluster
        # bootstrap is required before these bounds can be reported.
        "ci_low": None,
        "ci_high": None,
        "ci_method": "pending_setup_cluster_bootstrap",
    }


def _prf(counts: dict[str, int], z: float) -> dict[str, Any]:
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    precision = _wilson(tp, tp + fp, z)
    recall = _wilson(tp, tp + fn, z)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1_metric(tp, fp, fn),
    }


def _expected_top_dates(gold: dict[str, Any]) -> list[str]:
    points: list[dict[str, Any]] = []
    for structure in gold["structures"]:
        if structure.get("role") not in {
            "primary_lid",
            "secondary_lid",
            "breakout_level",
        }:
            continue
        points.extend(structure.get("construction_anchors") or [])
        points.extend(structure.get("supporting_touches") or [])
    return _unique_dates(
        point for point in points if point.get("price_field") == "high"
    )


def _expected_anchor_dates(gold: dict[str, Any]) -> list[str]:
    return _unique_dates(
        point
        for structure in gold["structures"]
        for point in (structure.get("construction_anchors") or [])
    )


def _expected_supporting_touch_dates(gold: dict[str, Any]) -> list[str]:
    return _unique_dates(
        point
        for structure in gold.get("structures") or []
        for point in (structure.get("supporting_touches") or [])
    )


def _candidate_top_dates(
    analysis: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Return general top candidates only when the detector declares them.

    ``points`` and ``major_highs`` are display/diagnostic collections whose
    semantics have changed over time.  Treating either as a complete top
    candidate set would manufacture an unsupported benchmark capability.
    """
    points = analysis.get("candidate_tops")
    if not isinstance(points, list):
        return False, []
    return True, _unique_dates(points)


def _selected_lid_member_dates(analysis: dict[str, Any]) -> list[str]:
    points: list[Any] = []
    active_lid = analysis.get("active_lid")
    if isinstance(active_lid, dict):
        points.extend(active_lid.get("anchors") or [])
        points.extend(active_lid.get("touches") or [])

    anatomy = analysis.get("pattern_anatomy") or {}
    structures = anatomy.get("structures") if isinstance(anatomy, dict) else None
    if not isinstance(structures, list):
        structures = analysis.get("structures")
    if isinstance(structures, list):
        for structure in structures:
            if not isinstance(structure, dict) or structure.get("selection") != "primary":
                continue
            points.extend(
                structure.get("construction_anchors")
                or structure.get("anchors")
                or []
            )
            points.extend(
                structure.get("supporting_touches")
                or structure.get("touches")
                or []
            )

    # Explicit membership on the algorithm's role points is authoritative.
    points.extend(
        point
        for point in (analysis.get("points") or [])
        if isinstance(point, dict) and point.get("lid_member") is True
    )
    return _unique_dates(points)


def _predicted_anchor_dates(analysis: dict[str, Any]) -> list[str]:
    points: list[Any] = []
    active_lid = analysis.get("active_lid")
    if isinstance(active_lid, dict):
        points.extend(active_lid.get("anchors") or [])
    anatomy = analysis.get("pattern_anatomy") or {}
    structures = anatomy.get("structures") if isinstance(anatomy, dict) else None
    if not isinstance(structures, list):
        structures = analysis.get("structures")
    if isinstance(structures, list):
        for structure in structures:
            if isinstance(structure, dict) and structure.get("selection") == "primary":
                points.extend(
                    structure.get("construction_anchors")
                    or structure.get("anchors")
                    or []
                )
    return _unique_dates(points)


def _predicted_supporting_touch_dates(analysis: dict[str, Any]) -> list[str]:
    """Return selected-lid touches without re-counting construction anchors."""
    anchors = _predicted_anchor_dates(analysis)
    anchor_quarters = {_quarter_ordinal(value) for value in anchors}
    points: list[Any] = []
    active_lid = analysis.get("active_lid")
    if isinstance(active_lid, dict):
        points.extend(active_lid.get("touches") or [])

    anatomy = analysis.get("pattern_anatomy") or {}
    structures = anatomy.get("structures") if isinstance(anatomy, dict) else None
    if not isinstance(structures, list):
        structures = analysis.get("structures")
    if isinstance(structures, list):
        for structure in structures:
            if not isinstance(structure, dict) or structure.get("selection") != "primary":
                continue
            points.extend(
                structure.get("supporting_touches")
                or structure.get("touches")
                or []
            )

    points.extend(
        point
        for point in (analysis.get("points") or [])
        if isinstance(point, dict)
        and point.get("role") == "structural_retest"
        and point.get("lid_member") is not True
    )
    return [
        value
        for value in _unique_dates(points)
        if _quarter_ordinal(value) not in anchor_quarters
    ]


def _bottom_metrics(
    gold: dict[str, Any],
    analysis: dict[str, Any],
    config: EvaluatorConfig,
) -> dict[str, Any]:
    expected_by_role = {
        role: _unique_dates(
            bottom.get("point")
            for bottom in (gold.get("bottoms") or [])
            if bottom.get("role") == role
        )
        for role in BOTTOM_ROLES
    }
    metrics = analysis.get("metrics")
    pullback_lows = (
        metrics.get("pullback_lows")
        if isinstance(metrics, dict)
        and isinstance(metrics.get("pullback_lows"), list)
        else []
    )
    major = {
        "supported": True,
        "source": "metrics.pullback_lows",
        "reason": None,
        **_match_dates(
            expected_by_role["major_bottom"],
            _unique_dates(pullback_lows),
            tolerance=config.candle_tolerance,
        ),
    }
    output = {"major_bottom": major}
    for role in ("undercut", "outlier"):
        output[role] = {
            "supported": False,
            "source": None,
            "reason": f"detector_does_not_emit_{role}_bottoms",
            "expected_count": len(expected_by_role[role]),
            "predicted_count": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "pairs": [],
            "unmatched_expected": None,
            "unmatched_predicted": None,
        }
    return output


def _predicted_action_state(
    analysis: dict[str, Any]
) -> tuple[str, str, bool | None]:
    anatomy = analysis.get("pattern_anatomy") or {}
    actionability = anatomy.get("actionability") or {}
    state = str(actionability.get("state") or "none")
    eligible = actionability.get("eligible")
    explicit_abstention = (
        bool(analysis["abstained"])
        if isinstance(analysis.get("abstained"), bool)
        else None
    )
    if explicit_abstention is True:
        return "abstain", "uncertain", True
    if state in {"ready", "breakout_now"} and eligible is not False:
        action = "actionable"
        readiness = "ready"
    elif state == "watch":
        action = "watch"
        readiness = "not_ready"
    elif state in {"historical_or_late", "none"} or eligible is False:
        action = "avoid"
        readiness = "not_ready"
    else:
        # Unknown effective states fail closed as a decisive negative.  The
        # diagnostic ``signal_state`` is deliberately never consulted.
        action = "avoid"
        readiness = "not_ready"
    return action, readiness, explicit_abstention


def _classification_predictions(analysis: dict[str, Any]) -> dict[str, Any]:
    anatomy = analysis.get("pattern_anatomy") or {}
    action, readiness, abstained = _predicted_action_state(analysis)
    if abstained is True:
        shape = "uncertain"
    else:
        shape = "coil" if bool(anatomy.get("recognized")) else "not_coil"
    maturity = anatomy.get("maturity") or {}
    maturity_prediction = (
        "mature"
        if maturity.get("passes") is True
        else ("immature" if maturity.get("passes") is False else "uncertain")
    )
    lifecycle = str(analysis.get("lifecycle") or "uncertain")
    return {
        "shape": shape,
        "maturity": maturity_prediction,
        "lifecycle": lifecycle,
        "readiness": readiness,
        "action": action,
        "abstained": abstained,
    }


def _prediction_structures(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    anatomy = analysis.get("pattern_anatomy") or {}
    raw_structures = (
        anatomy.get("structures")
        if isinstance(anatomy, dict)
        else None
    ) or analysis.get("structures")
    if isinstance(raw_structures, list) and raw_structures:
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_structures):
            if not isinstance(raw, dict):
                continue
            line = raw.get("line") if isinstance(raw.get("line"), dict) else raw
            value_at_cutoff = next(
                (
                    value
                    for value in (
                        line.get("value_at_cutoff"),
                        line.get("value_at_last_bar"),
                        raw.get("value_at_cutoff"),
                        raw.get("value_at_last_bar"),
                    )
                    if value is not None
                ),
                None,
            )
            slope = next(
                (
                    value
                    for value in (
                        line.get("slope_pct_per_year"),
                        raw.get("slope_pct_per_year"),
                    )
                    if value is not None
                ),
                None,
            )
            boundary_kind = (
                raw.get("boundary_kind")
                or raw.get("boundaryKind")
                or line.get("boundary_kind")
                or line.get("boundaryKind")
            )
            normalized.append(
                {
                    "id": str(raw.get("id") or f"predicted-{index}"),
                    "role": raw.get("role") or line.get("role"),
                    "boundary_kind": boundary_kind,
                    "direction": line.get("direction") or raw.get("direction"),
                    "value_at_cutoff": value_at_cutoff,
                    "slope_pct_per_year": slope,
                    "anchors": raw.get("construction_anchors")
                    or raw.get("anchors")
                    or [],
                    "touches": raw.get("supporting_touches")
                    or raw.get("touches")
                    or [],
                    "selection": raw.get("selection", "alternate"),
                    "relationship": raw.get("relationship", "standalone"),
                    "parent_id": raw.get("parent_id"),
                }
            )
        if normalized:
            return normalized

    boundary = anatomy.get("boundary")
    active_lid = analysis.get("active_lid")
    if not isinstance(boundary, dict) and not isinstance(active_lid, dict):
        return []
    boundary = boundary if isinstance(boundary, dict) else {}
    active_lid = active_lid if isinstance(active_lid, dict) else {}
    return [
        {
            "id": "active-boundary",
            "role": boundary.get("role") or "primary_lid",
            "boundary_kind": (
                boundary.get("boundary_kind")
                or boundary.get("boundaryKind")
                or "line"
            ),
            "direction": boundary.get("direction"),
            "value_at_cutoff": active_lid.get("value_at_last_bar"),
            "slope_pct_per_year": (
                boundary.get("slope_pct_per_year")
                if boundary.get("slope_pct_per_year") is not None
                else active_lid.get("slope_pct_per_year")
            ),
            "anchors": active_lid.get("anchors") or boundary.get("anchors") or [],
            "touches": active_lid.get("touches") or boundary.get("touches") or [],
            "selection": "primary",
            "relationship": "standalone",
            "parent_id": None,
        }
    ]


def _projected_error_pct(expected: dict[str, Any], predicted: dict[str, Any]) -> float | None:
    try:
        expected_value = float(expected["line"]["value_at_cutoff"])
        predicted_value = float(predicted["value_at_cutoff"])
    except (KeyError, TypeError, ValueError):
        return None
    if expected_value <= 0 or not math.isfinite(predicted_value):
        return None
    return abs(predicted_value - expected_value) / expected_value * 100.0


def _slope_error_pct_per_year(
    expected: dict[str, Any], predicted: dict[str, Any]
) -> float | None:
    try:
        expected_slope = float(expected["line"]["slope_pct_per_year"])
        predicted_slope = float(predicted["slope_pct_per_year"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(expected_slope) or not math.isfinite(predicted_slope):
        return None
    return abs(predicted_slope - expected_slope)


def _structure_anchor_linkage(
    expected: dict[str, Any],
    predicted: dict[str, Any],
    *,
    tolerance: int,
) -> dict[str, Any]:
    expected_dates = _unique_dates(expected.get("construction_anchors") or [])
    predicted_dates = _unique_dates(predicted.get("anchors") or [])
    matching = _match_dates(
        expected_dates,
        predicted_dates,
        tolerance=tolerance,
    )
    required = min(2, len(expected_dates))
    return {
        **matching,
        "required_matches": required,
        "linked": (
            required > 0
            and matching["tp"] >= required
            and matching["fp"] == 0
        ),
    }


def _structure_metrics(
    gold: dict[str, Any],
    analysis: dict[str, Any],
    config: EvaluatorConfig,
) -> dict[str, Any]:
    expected = list(gold.get("structures") or [])
    predicted = _prediction_structures(analysis)
    candidates: dict[int, list[tuple[int, float, float, dict[str, Any]]]] = (
        defaultdict(list)
    )
    for expected_idx, expected_structure in enumerate(expected):
        for predicted_idx, predicted_structure in enumerate(predicted):
            projected_error = _projected_error_pct(
                expected_structure, predicted_structure
            )
            slope_error = _slope_error_pct_per_year(
                expected_structure, predicted_structure
            )
            anchor_linkage = _structure_anchor_linkage(
                expected_structure,
                predicted_structure,
                tolerance=config.candle_tolerance,
            )
            if (
                predicted_structure.get("role") == expected_structure.get("role")
                and predicted_structure.get("boundary_kind")
                == expected_structure.get("boundary_kind")
                and predicted_structure.get("direction")
                == expected_structure.get("line", {}).get("direction")
                and projected_error is not None
                and projected_error <= config.projected_line_tolerance_pct
                and slope_error is not None
                and slope_error <= config.slope_tolerance_pct_per_year
                and anchor_linkage["linked"]
            ):
                # Anchor timing is only a deterministic third-order cost: the
                # eligibility check above carries the semantic requirement.
                anchor_error = sum(
                    int(pair["candle_error"])
                    for pair in anchor_linkage["pairs"]
                )
                cost = projected_error + slope_error + anchor_error / 1000.0
                candidates[expected_idx].append(
                    (
                        predicted_idx,
                        cost,
                        projected_error,
                        {
                            "slope_error_pct_per_year": round(slope_error, 6),
                            "anchor_linkage": anchor_linkage,
                        },
                    )
                )

    # Structures are not inherently ordered, so use a bounded assignment DP.
    # Gold labels cap each side at 16, keeping the bitmask state tractable.
    @lru_cache(maxsize=None)
    def solve(
        expected_idx: int, used_predicted_mask: int
    ) -> tuple[int, float, tuple[tuple[int, int, float, float], ...]]:
        if expected_idx >= len(expected):
            return (0, 0.0, ())
        options = [solve(expected_idx + 1, used_predicted_mask)]
        for predicted_idx, cost, projected_error, _ in candidates.get(
            expected_idx, []
        ):
            bit = 1 << predicted_idx
            if used_predicted_mask & bit:
                continue
            matched, total_cost, suffix = solve(
                expected_idx + 1, used_predicted_mask | bit
            )
            options.append(
                (
                    matched + 1,
                    total_cost + cost,
                    (
                        (
                            expected_idx,
                            predicted_idx,
                            projected_error,
                            cost,
                        ),
                    )
                    + suffix,
                )
            )
        return min(
            options,
            key=lambda result: (-result[0], round(result[1], 12), result[2]),
        )

    _, _, matched_indices = solve(0, 0)
    pairs: list[dict[str, Any]] = []
    for expected_idx, predicted_idx, projected_error, _ in matched_indices:
        details = next(
            details
            for candidate_idx, _, _, details in candidates[expected_idx]
            if candidate_idx == predicted_idx
        )
        pairs.append(
            {
                "expected_id": expected[expected_idx].get("id"),
                "predicted_id": predicted[predicted_idx].get("id"),
                "expected_selection": expected[expected_idx].get("selection"),
                "expected_relationship": expected[expected_idx].get("relationship"),
                "predicted_selection": predicted[predicted_idx].get("selection"),
                "predicted_relationship": predicted[predicted_idx].get("relationship"),
                "projected_line_error_pct": round(projected_error, 6),
                **details,
            }
        )

    primary_expected = sum(
        item.get("selection") == "primary" for item in expected
    )
    primary_predicted = sum(
        item.get("selection") == "primary" for item in predicted
    )
    primary_tp = sum(
        pair["expected_selection"] == pair["predicted_selection"] == "primary"
        for pair in pairs
    )
    alternate_expected = sum(item.get("selection") == "alternate" for item in expected)
    alternate_predicted = sum(item.get("selection") == "alternate" for item in predicted)
    alternate_tp = sum(
        pair["expected_selection"] == pair["predicted_selection"] == "alternate"
        for pair in pairs
    )
    expected_edges: set[tuple[str, str]] = {
        (str(item.get("parent_id")), str(item.get("id")))
        for item in expected
        if item.get("relationship") == "child"
    }
    predicted_to_expected = {
        str(pair["predicted_id"]): str(pair["expected_id"]) for pair in pairs
    }
    raw_predicted_edges = {
        (str(item.get("parent_id")), str(item.get("id")))
        for item in predicted
        if item.get("relationship") == "child"
    }
    predicted_edges = {
        (
            predicted_to_expected.get(parent_id, f"unmatched:{parent_id}"),
            predicted_to_expected.get(child_id, f"unmatched:{child_id}"),
        )
        for parent_id, child_id in raw_predicted_edges
    }
    edge_tp = len(expected_edges & predicted_edges)
    active = next(
        (
            structure
            for structure in expected
            if structure.get("id") == gold.get("active_structure_id")
        ),
        None,
    )
    active_pair = next(
        (
            pair
            for pair in pairs
            if pair["expected_id"] == gold.get("active_structure_id")
        ),
        None,
    )
    predicted_by_id = {str(item.get("id")): item for item in predicted}
    predicted_boundary = (
        predicted_by_id.get(str(active_pair["predicted_id"]))
        if active_pair is not None
        else None
    )
    active_error = (
        _projected_error_pct(active, predicted_boundary)
        if active is not None and predicted_boundary is not None
        else None
    )
    direction_expected = active.get("line", {}).get("direction") if active else None
    direction_predicted = (
        predicted_boundary.get("direction") if predicted_boundary else None
    )
    return {
        "matched": {"tp": len(pairs), "fp": len(predicted) - len(pairs), "fn": len(expected) - len(pairs)},
        "pairs": pairs,
        "primary": {
            "tp": primary_tp,
            "fp": primary_predicted - primary_tp,
            "fn": primary_expected - primary_tp,
        },
        "alternate": {
            "tp": alternate_tp,
            "fp": alternate_predicted - alternate_tp,
            "fn": alternate_expected - alternate_tp,
        },
        "parent_child": {
            "tp": edge_tp,
            "fp": len(predicted_edges) - edge_tp,
            "fn": len(expected_edges) - edge_tp,
        },
        "active_boundary": {
            "expected_id": gold.get("active_structure_id"),
            "matched_predicted_id": (
                active_pair["predicted_id"] if active_pair is not None else None
            ),
            "matched": active_pair is not None,
            "selection_correct": (
                active_pair["predicted_selection"] == "primary"
                if active_pair is not None
                else False
            ),
            "direction_expected": direction_expected,
            "direction_predicted": direction_predicted,
            "direction_correct": (
                direction_expected == direction_predicted
                if direction_expected is not None
                else None
            ),
            "projected_line_error_pct": (
                round(active_error, 6) if active_error is not None else None
            ),
            "within_tolerance": (
                active_error <= config.projected_line_tolerance_pct
                if active_error is not None
                else False
            ),
        },
    }


def _phase_metrics(
    gold: dict[str, Any], analysis: dict[str, Any], config: EvaluatorConfig
) -> dict[str, Any]:
    anatomy = analysis.get("pattern_anatomy") or {}
    output: dict[str, Any] = {}
    for kind in PHASE_KINDS:
        expected_phases = [
            phase
            for phase in gold.get("phases") or []
            if phase.get("kind") == kind and phase.get("present") is True
        ]
        predicted_phase = anatomy.get(kind)
        if kind == "base":
            predicted_present = isinstance(predicted_phase, dict)
        else:
            predicted_present = (
                isinstance(predicted_phase, dict)
                and predicted_phase.get("present") is True
            )
        expected_present = bool(expected_phases)
        expected_start = (
            min(str(phase["start"]["date"]) for phase in expected_phases)
            if expected_phases
            else None
        )
        expected_end = (
            max(str(phase["end"]["date"]) for phase in expected_phases)
            if expected_phases
            else None
        )
        predicted_start = (
            str(predicted_phase.get("start"))
            if isinstance(predicted_phase, dict) and predicted_phase.get("start")
            else None
        )
        predicted_end = (
            str(predicted_phase.get("end"))
            if isinstance(predicted_phase, dict) and predicted_phase.get("end")
            else None
        )
        start_error = (
            abs(_quarter_ordinal(expected_start) - _quarter_ordinal(predicted_start))
            if expected_start and predicted_start
            else None
        )
        end_error = (
            abs(_quarter_ordinal(expected_end) - _quarter_ordinal(predicted_end))
            if expected_end and predicted_end
            else None
        )
        output[kind] = {
            "expected_present": expected_present,
            "predicted_present": predicted_present,
            "presence_correct": expected_present == predicted_present,
            "expected_start": expected_start,
            "predicted_start": predicted_start,
            "start_error_candles": start_error,
            "expected_end": expected_end,
            "predicted_end": predicted_end,
            "end_error_candles": end_error,
            "timing_correct": (
                start_error <= config.phase_tolerance
                and end_error <= config.phase_tolerance
                if start_error is not None and end_error is not None
                else None
            ),
        }
    return output


def _gold_event_dates(gold: dict[str, Any], kind: str) -> list[str]:
    dates: list[str] = []
    for event in gold.get("events") or []:
        if event.get("kind") != kind:
            continue
        point = (
            event.get("resolution")
            if kind == "failed_breakout" and event.get("resolution")
            else event.get("trigger")
        )
        point_date = _date_from_point(point)
        if point_date:
            dates.append(point_date)
    return sorted(set(dates))


def _predicted_event_dates(analysis: dict[str, Any], kind: str) -> list[str]:
    breakout = analysis.get("breakout") or {}
    if not isinstance(breakout, dict):
        return []
    if kind == "breakout":
        return _unique_dates([breakout.get("first_escape")])
    if kind == "failed_breakout":
        return _unique_dates(
            event.get("failed")
            for event in breakout.get("failed_breakouts") or []
            if isinstance(event, dict)
        )
    if kind == "retest":
        return _unique_dates([breakout.get("retest")])
    if kind == "continuation":
        confirmed = breakout.get("confirmed")
        if isinstance(confirmed, dict) and confirmed.get("via") == "retest_continuation":
            return _unique_dates([confirmed])
        continuation = breakout.get("continuation")
        return _unique_dates(
            continuation if isinstance(continuation, list) else [continuation]
        )
    if kind == "invalidation":
        invalidation = breakout.get("invalidation")
        return _unique_dates(
            invalidation if isinstance(invalidation, list) else [invalidation]
        )
    return []


def _event_metrics(
    gold: dict[str, Any], analysis: dict[str, Any], config: EvaluatorConfig
) -> dict[str, Any]:
    timing = {
        kind: _match_dates(
            _gold_event_dates(gold, kind),
            _predicted_event_dates(analysis, kind),
            tolerance=config.candle_tolerance,
        )
        for kind in EVENT_KINDS
    }
    breakout = analysis.get("breakout") or {}
    predicted_retest = breakout.get("retest") if isinstance(breakout, dict) else None
    predicted_retest_state: str | None = None
    if isinstance(predicted_retest, dict):
        predicted_retest_state = predicted_retest.get("retest_state")
        if predicted_retest_state is None and predicted_retest.get("holds") is False:
            predicted_retest_state = "failed"
        elif predicted_retest_state is None and predicted_retest.get("holds") is True:
            predicted_retest_state = "holding_unspecified"
    expected_retests = sorted(
        (
            event
            for event in gold.get("events") or []
            if event.get("kind") == "retest"
        ),
        key=lambda event: str((event.get("trigger") or {}).get("date") or ""),
    )
    expected_retest_state = (
        expected_retests[-1].get("retest_state") if expected_retests else None
    )

    expected_volume_events = sorted(
        (
            event
            for event in gold.get("events") or []
            if event.get("relative_volume_label") not in {None, "unavailable"}
        ),
        key=lambda event: str((event.get("trigger") or {}).get("date") or ""),
    )
    expected_volume = (
        expected_volume_events[-1].get("relative_volume_label")
        if expected_volume_events
        else None
    )
    predicted_volume: str | None = None
    if isinstance(breakout, dict):
        candidate = breakout.get("relative_volume_confirmation")
        if isinstance(candidate, bool):
            predicted_volume = "confirmed" if candidate else "not_confirmed"
        elif candidate in {"confirmed", "not_confirmed", "unavailable"}:
            predicted_volume = str(candidate)
    return {
        "timing": timing,
        "retest_state": {
            "expected": expected_retest_state,
            "predicted": predicted_retest_state,
            "correct": (
                expected_retest_state == predicted_retest_state
                if expected_retest_state is not None
                else None
            ),
            "supported": predicted_retest_state
            not in {None, "holding_unspecified"},
        },
        "relative_volume_confirmation": {
            "expected": expected_volume,
            "predicted": predicted_volume,
            "correct": (
                expected_volume == predicted_volume
                if expected_volume is not None
                else None
            ),
            "supported": predicted_volume is not None,
        },
    }


def _counts_are_perfect(metric: dict[str, Any]) -> bool:
    return (
        int(metric.get("fp") or 0) == 0
        and int(metric.get("fn") or 0) == 0
    )


def _intervention_policy_checks(
    *,
    gold: dict[str, Any],
    classification_correct: dict[str, bool],
    top_metrics: dict[str, Any],
    selected_membership_proxy: dict[str, Any],
    anchor_metrics: dict[str, Any],
    supporting_touch_metrics: dict[str, Any],
    bottom_metrics: dict[str, Any],
    excluded_highs: dict[str, Any],
    structures: dict[str, Any],
    phases: dict[str, Any],
    events: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the explicit discrepancy-to-intervention policy audit trail."""
    checks: list[dict[str, Any]] = []

    def add(
        name: str,
        *,
        applicable: bool,
        supported: bool,
        passed: bool | None,
        included_in_estimate: bool = True,
        reason: str | None = None,
    ) -> None:
        normalized_passed = bool(passed) if passed is not None else None
        checks.append(
            {
                "name": name,
                "applicable": applicable,
                "supported": supported,
                "passed": normalized_passed,
                "included_in_estimate": included_in_estimate,
                "requires_intervention": bool(
                    applicable
                    and included_in_estimate
                    and normalized_passed is not True
                ),
                "reason": reason,
            }
        )

    for field in ("shape", "maturity", "lifecycle", "readiness", "action"):
        add(
            f"classification.{field}",
            applicable=True,
            supported=True,
            passed=classification_correct[field],
        )

    top_review_complete = bool(gold.get("top_review_complete"))
    top_supported = top_metrics.get("supported") is True
    add(
        "general_top_detection",
        applicable=top_review_complete,
        supported=top_supported,
        passed=(
            _counts_are_perfect(top_metrics) if top_supported else None
        ),
        included_in_estimate=top_supported,
        reason=(
            None
            if top_supported
            else "unsupported general tops are excluded; selected membership is scored"
        ),
    )
    add(
        "selected_top_membership_proxy",
        applicable=top_review_complete,
        supported=True,
        passed=_counts_are_perfect(selected_membership_proxy),
    )
    add(
        "construction_anchors",
        applicable=True,
        supported=True,
        passed=_counts_are_perfect(anchor_metrics),
    )
    add(
        "supporting_touches",
        applicable=True,
        supported=True,
        passed=_counts_are_perfect(supporting_touch_metrics),
    )
    add(
        "excluded_highs",
        applicable=True,
        supported=True,
        passed=excluded_highs["incorrectly_selected"] == 0,
    )

    for name, key in (
        ("structure.geometry", "matched"),
        ("structure.primary_selection", "primary"),
        ("structure.alternate_selection", "alternate"),
        ("structure.parent_child_topology", "parent_child"),
    ):
        add(
            name,
            applicable=True,
            supported=True,
            passed=_counts_are_perfect(structures[key]),
        )
    active_applicable = gold.get("active_structure_id") is not None
    active = structures["active_boundary"]
    add(
        "structure.active_boundary",
        applicable=active_applicable,
        supported=True,
        passed=(
            bool(
                active["matched"]
                and active["selection_correct"]
                and active["direction_correct"]
                and active["within_tolerance"]
            )
            if active_applicable
            else None
        ),
    )

    major_bottom = bottom_metrics["major_bottom"]
    add(
        "bottom.major_bottom",
        applicable=True,
        supported=True,
        passed=_counts_are_perfect(major_bottom),
    )
    for role in ("undercut", "outlier"):
        metric = bottom_metrics[role]
        add(
            f"bottom.{role}",
            applicable=int(metric["expected_count"]) > 0,
            supported=False,
            passed=None,
            included_in_estimate=False,
            reason=metric["reason"],
        )

    for kind in PHASE_KINDS:
        metric = phases[kind]
        add(
            f"phase.{kind}.presence",
            applicable=True,
            supported=True,
            passed=metric["presence_correct"],
        )
        expected_present = metric["expected_present"] is True
        timing_supported = (
            metric["predicted_start"] is not None
            and metric["predicted_end"] is not None
        )
        add(
            f"phase.{kind}.timing",
            applicable=expected_present,
            supported=timing_supported,
            passed=(metric["timing_correct"] if expected_present else None),
        )

    for kind in EVENT_KINDS:
        add(
            f"event.{kind}.timing",
            applicable=True,
            supported=True,
            passed=_counts_are_perfect(events["timing"][kind]),
        )
    for name in ("retest_state", "relative_volume_confirmation"):
        metric = events[name]
        applicable = metric["expected"] is not None
        add(
            f"event.{name}",
            applicable=applicable,
            supported=metric["supported"] is True,
            passed=(metric["correct"] if applicable else None),
        )
    return checks


def _classification_summary(
    pairs: list[tuple[str, str]], z: float
) -> dict[str, Any]:
    labels = sorted({value for pair in pairs for value in pair})
    confusion: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in labels} for expected in labels
    }
    for expected, predicted in pairs:
        confusion[expected][predicted] += 1
    correct = sum(expected == predicted for expected, predicted in pairs)
    per_class: dict[str, Any] = {}
    for label in labels:
        tp = sum(expected == predicted == label for expected, predicted in pairs)
        fp = sum(expected != label and predicted == label for expected, predicted in pairs)
        fn = sum(expected == label and predicted != label for expected, predicted in pairs)
        per_class[label] = _prf({"tp": tp, "fp": fp, "fn": fn}, z)
    f1_values = [
        metric["f1"]["value"]
        for metric in per_class.values()
        if metric["f1"]["value"] is not None
    ]
    return {
        "count": len(pairs),
        "accuracy": _wilson(correct, len(pairs), z),
        "macro_f1": {
            "value": round(sum(f1_values) / len(f1_values), 6) if f1_values else None,
            "ci_low": None,
            "ci_high": None,
            "ci_method": "pending_setup_cluster_bootstrap",
        },
        "per_class": per_class,
        "confusion": confusion,
    }


def _rematerialize_gold_label(
    raw_label: dict[str, Any], monthly_bars: list[dict[str, Any]]
) -> dict[str, Any]:
    """Rebuild every derived gold field and compare canonical bytes."""
    try:
        rematerialized = validate_materialized_gold_label_against_bars(
            raw_label, monthly_bars
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkCorpusError(
            "materialized gold-label rematerialization from its embedded capture failed"
        ) from exc
    return rematerialized


def _unsupported_top_metric(
    expected_count: int,
    *,
    reason: str = "detector_does_not_emit_explicit_candidate_tops",
) -> dict[str, Any]:
    return {
        "supported": False,
        "reason": reason,
        "expected_count": expected_count,
        "predicted_count": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "pairs": [],
        "unmatched_expected": None,
        "unmatched_predicted": None,
    }


def evaluate_episode(
    gold_label: dict[str, Any],
    monthly_bars: list[dict[str, Any]],
    *,
    config: EvaluatorConfig = EvaluatorConfig(),
) -> dict[str, Any]:
    """Evaluate one historical checkpoint without applying human truth."""
    gold = _rematerialize_gold_label(gold_label, monthly_bars)
    cutoff = str(gold["cutoff_date"])
    decision_as_of = str(gold["decision_as_of"])
    normalized = _normalize_monthly_bars(monthly_bars, cutoff)
    expected_bars_hash = (gold.get("provenance") or {}).get(
        "bars_through_cutoff_sha256"
    )
    if expected_bars_hash != sha256_json(normalized):
        raise BenchmarkCorpusError(
            f"{gold.get('episode_id')} frozen bars do not match the gold label"
        )
    expected_quarter_hash = (gold.get("provenance") or {}).get(
        "completed_quarterly_bars_sha256"
    )
    if expected_quarter_hash != sha256_json(_completed_quarterly_bars(normalized)):
        raise BenchmarkCorpusError(
            f"{gold.get('episode_id')} quarterly evidence does not match the label"
        )
    if (
        gold.get("evaluation_role") == "blind_benchmark"
        and gold.get("outcome_visible_during_label") is True
    ):
        raise BenchmarkCorpusError(
            "outcome-visible labels cannot enter the blind benchmark"
        )

    analysis, detector_bars = run_detector_only(
        monthly_bars,
        cutoff,
        decision_as_of,
    )
    predictions = _classification_predictions(analysis)
    expected = dict(gold["judgments"])
    expected_top_dates = _expected_top_dates(gold)
    candidate_tops_supported, predicted_top_dates = _candidate_top_dates(analysis)
    if gold.get("top_review_complete") is not True:
        top_metrics = _unsupported_top_metric(
            len(expected_top_dates),
            reason="gold_top_review_incomplete",
        )
    elif candidate_tops_supported:
        top_metrics = {
            "supported": True,
            "reason": None,
            **_match_dates(
                expected_top_dates,
                predicted_top_dates,
                tolerance=config.candle_tolerance,
            ),
        }
    else:
        top_metrics = _unsupported_top_metric(len(expected_top_dates))
    selected_lid_member_dates = _selected_lid_member_dates(analysis)
    selected_membership_proxy = _match_dates(
        expected_top_dates,
        selected_lid_member_dates,
        tolerance=config.candle_tolerance,
    )
    anchor_metrics = _match_dates(
        _expected_anchor_dates(gold),
        _predicted_anchor_dates(analysis),
        tolerance=config.candle_tolerance,
    )
    supporting_touch_metrics = _match_dates(
        _expected_supporting_touch_dates(gold),
        _predicted_supporting_touch_dates(analysis),
        tolerance=config.candle_tolerance,
    )
    excluded_dates = _unique_dates(
        point
        for structure in gold.get("structures") or []
        for point in (structure.get("excluded_highs") or [])
    )
    excluded_matches = _match_dates(
        excluded_dates,
        selected_lid_member_dates,
        tolerance=0,
    )
    excluded_high_metrics = {
        "expected_count": len(excluded_dates),
        "selected_lid_member_dates": selected_lid_member_dates,
        "matching_tolerance_candles": 0,
        "incorrectly_selected": excluded_matches["tp"],
        "rejected": len(excluded_dates) - excluded_matches["tp"],
    }
    structures = _structure_metrics(gold, analysis, config)
    bottoms = _bottom_metrics(gold, analysis, config)
    phases = _phase_metrics(gold, analysis, config)
    events = _event_metrics(gold, analysis, config)

    classification_correct = {
        field: expected[field] == predictions[field]
        for field in ("shape", "maturity", "lifecycle", "readiness", "action")
    }
    boundary_correct = bool(
        structures["active_boundary"]["matched"]
        and structures["active_boundary"]["selection_correct"]
        and structures["active_boundary"]["direction_correct"]
        and structures["active_boundary"]["within_tolerance"]
    ) if gold.get("active_structure_id") else True
    intervention_checks = _intervention_policy_checks(
        gold=gold,
        classification_correct=classification_correct,
        top_metrics=top_metrics,
        selected_membership_proxy=selected_membership_proxy,
        anchor_metrics=anchor_metrics,
        supporting_touch_metrics=supporting_touch_metrics,
        bottom_metrics=bottoms,
        excluded_highs=excluded_high_metrics,
        structures=structures,
        phases=phases,
        events=events,
    )
    correction_required = any(
        check["requires_intervention"] for check in intervention_checks
    )
    critical_correction_required = not all(
        classification_correct[field]
        for field in ("shape", "lifecycle", "action")
    ) or not boundary_correct

    return {
        "episode_id": gold["episode_id"],
        "setup_id": gold["setup_id"],
        "evaluation_role": gold["evaluation_role"],
        "outcome_visible_during_label": bool(
            gold.get("outcome_visible_during_label")
        ),
        "top_review_complete": gold.get("top_review_complete") is True,
        "cutoff_date": cutoff,
        "decision_as_of": decision_as_of,
        "gold_label_sha256": gold["label_sha256"],
        "detector": {
            "algorithm_version": analysis.get("algorithm_version"),
            "default_config_sha256": DEFAULT_CONFIG_SHA256,
            "prediction_sha256": sha256_json(analysis),
            "bars_sha256": sha256_json(detector_bars),
            "bar_count": len(detector_bars),
            "review_override_applied": False,
        },
        "expected": expected,
        "predicted": predictions,
        "classification_correct": classification_correct,
        "tops": top_metrics,
        "selected_top_membership_proxy": selected_membership_proxy,
        "construction_anchors": anchor_metrics,
        "supporting_touches": supporting_touch_metrics,
        "excluded_highs": excluded_high_metrics,
        "structures": structures,
        "bottoms": bottoms,
        "phases": phases,
        "events": events,
        "abstained": predictions["abstained"],
        "false_action": (
            predictions["action"] == "actionable"
            and expected["action"] != "actionable"
        ),
        "intervention_policy_version": INTERVENTION_POLICY_VERSION,
        "intervention_checks": intervention_checks,
        "estimated_human_intervention_required": correction_required,
        "correction_required": correction_required,
        "critical_correction_required": critical_correction_required,
    }


def _sum_counts(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    total = {"tp": 0, "fp": 0, "fn": 0}
    for item in items:
        for key in total:
            total[key] += int(item.get(key, 0))
    return total


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "p95": round(ordered[p95_idx], 6),
        "max": round(max(values), 6),
    }


def _transition_timing_summary(
    items: list[dict[str, Any]],
    *,
    field: str,
    config: EvaluatorConfig,
) -> dict[str, Any]:
    """Compare state-entry dates across point-in-time checkpoints per setup."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["setup_id"])].append(item)
    expected_by_state: dict[str, list[str]] = defaultdict(list)
    predicted_by_state: dict[str, list[str]] = defaultdict(list)
    eligible_setups = 0
    for setup_items in grouped.values():
        ordered = sorted(setup_items, key=lambda value: value["cutoff_date"])
        if len(ordered) < 2:
            continue
        eligible_setups += 1
        prior_expected = ordered[0]["expected"][field]
        prior_predicted = ordered[0]["predicted"][field]
        for item in ordered[1:]:
            expected = item["expected"][field]
            predicted = item["predicted"][field]
            if expected != prior_expected:
                expected_by_state[str(expected)].append(item["cutoff_date"])
            if predicted != prior_predicted:
                predicted_by_state[str(predicted)].append(item["cutoff_date"])
            prior_expected = expected
            prior_predicted = predicted
    states = sorted(set(expected_by_state) | set(predicted_by_state))
    per_state = {
        state: _match_dates(
            expected_by_state[state],
            predicted_by_state[state],
            tolerance=config.candle_tolerance,
        )
        for state in states
    }
    counts = _sum_counts(per_state.values())
    errors = [
        float(pair["candle_error"])
        for metric in per_state.values()
        for pair in metric["pairs"]
    ]
    return {
        "eligible_setup_count": eligible_setups,
        "matching": _prf(counts, config.confidence_z),
        "timing_error_candles": _numeric_summary(errors),
        "per_entered_state": per_state,
    }


def aggregate_results(
    items: list[dict[str, Any]], *, config: EvaluatorConfig
) -> dict[str, Any]:
    z = config.confidence_z
    classifications = {
        field: _classification_summary(
            [(item["expected"][field], item["predicted"][field]) for item in items],
            z,
        )
        for field in ("shape", "maturity", "lifecycle", "readiness", "action")
    }
    supported_top_items = [
        item["tops"] for item in items if item["tops"].get("supported") is True
    ]
    if items and len(supported_top_items) == len(items):
        top = {
            "supported": True,
            "reason": None,
            "supported_episode_count": len(supported_top_items),
            "episode_count": len(items),
            **_prf(_sum_counts(supported_top_items), z),
        }
    else:
        null_ratio = {
            "numerator": None,
            "denominator": None,
            "value": None,
            "ci_low": None,
            "ci_high": None,
        }
        top = {
            "supported": False,
            "reason": "not_all_episodes_emit_explicit_candidate_tops",
            "supported_episode_count": len(supported_top_items),
            "episode_count": len(items),
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": dict(null_ratio),
            "recall": dict(null_ratio),
            "f1": {
                "value": None,
                "ci_low": None,
                "ci_high": None,
                "ci_method": "pending_setup_cluster_bootstrap",
            },
        }
    selected_top_membership = _prf(
        _sum_counts(item["selected_top_membership_proxy"] for item in items), z
    )
    anchors = _prf(_sum_counts(item["construction_anchors"] for item in items), z)
    supporting_touches = _prf(
        _sum_counts(item["supporting_touches"] for item in items), z
    )
    structures = _prf(
        _sum_counts(item["structures"]["matched"] for item in items), z
    )
    alternates = _prf(
        _sum_counts(item["structures"]["alternate"] for item in items), z
    )
    hierarchy = _prf(
        _sum_counts(item["structures"]["parent_child"] for item in items), z
    )
    primary = _prf(
        _sum_counts(item["structures"]["primary"] for item in items), z
    )
    events = {
        kind: _prf(
            _sum_counts(item["events"]["timing"][kind] for item in items), z
        )
        for kind in EVENT_KINDS
    }
    event_state_metrics: dict[str, Any] = {}
    for state_name in ("retest_state", "relative_volume_confirmation"):
        pairs = [
            (
                str(item["events"][state_name]["expected"]),
                str(item["events"][state_name]["predicted"]),
            )
            for item in items
            if item["events"][state_name]["expected"] is not None
        ]
        event_state_metrics[state_name] = _classification_summary(pairs, z)
    phase_metrics: dict[str, Any] = {}
    for kind in PHASE_KINDS:
        pairs = [
            (
                str(item["phases"][kind]["expected_present"]),
                str(item["phases"][kind]["predicted_present"]),
            )
            for item in items
        ]
        expected_timing = [
            item["phases"][kind]
            for item in items
            if item["phases"][kind]["expected_present"] is True
        ]
        comparable_timing = [
            metric
            for metric in expected_timing
            if metric.get("start_error_candles") is not None
            and metric.get("end_error_candles") is not None
        ]
        phase_metrics[kind] = {
            "presence": _classification_summary(pairs, z),
            # Missing predicted boundaries count as timing failures, while the
            # separate coverage metric makes that incompleteness explicit.
            "timing_correctness": _wilson(
                sum(metric.get("timing_correct") is True for metric in expected_timing),
                len(expected_timing),
                z,
            ),
            "timing_coverage": _wilson(
                len(comparable_timing), len(expected_timing), z
            ),
            "start_error_candles": _numeric_summary(
                [
                    float(metric["start_error_candles"])
                    for metric in comparable_timing
                ]
            ),
            "end_error_candles": _numeric_summary(
                [
                    float(metric["end_error_candles"])
                    for metric in comparable_timing
                ]
            ),
        }

    bottom_metrics: dict[str, Any] = {
        "major_bottom": {
            "supported": True,
            "source": "metrics.pullback_lows",
            **_prf(
                _sum_counts(item["bottoms"]["major_bottom"] for item in items), z
            ),
        }
    }
    for role in ("undercut", "outlier"):
        bottom_metrics[role] = {
            "supported": False,
            "source": None,
            "reason": f"detector_does_not_emit_{role}_bottoms",
            "expected_count": sum(
                int(item["bottoms"][role]["expected_count"]) for item in items
            ),
            "predicted_count": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    actionable_tp = sum(
        item["expected"]["action"] == item["predicted"]["action"] == "actionable"
        for item in items
    )
    actionable_fp = sum(
        item["expected"]["action"] != "actionable"
        and item["predicted"]["action"] == "actionable"
        for item in items
    )
    actionable_fn = sum(
        item["expected"]["action"] == "actionable"
        and item["predicted"]["action"] != "actionable"
        for item in items
    )
    action_binary = _prf(
        {"tp": actionable_tp, "fp": actionable_fp, "fn": actionable_fn}, z
    )
    count = len(items)
    projection_errors = [
        float(error)
        for item in items
        if (
            error := item["structures"]["active_boundary"].get(
                "projected_line_error_pct"
            )
        )
        is not None
    ]
    excluded_total = sum(item["excluded_highs"]["expected_count"] for item in items)
    excluded_rejected = sum(item["excluded_highs"]["rejected"] for item in items)
    abstention_supported = bool(items) and all(
        isinstance(item.get("abstained"), bool) for item in items
    )
    if abstention_supported:
        abstention_rate = {
            "supported": True,
            "reason": None,
            **_wilson(
                sum(item["abstained"] is True for item in items),
                len(items),
                z,
            ),
        }
    else:
        abstention_rate = {
            "supported": False,
            "reason": "detector_does_not_emit_explicit_abstention",
            "numerator": None,
            "denominator": None,
            "value": None,
            "ci_low": None,
            "ci_high": None,
        }
    false_actions = sum(item["false_action"] for item in items)
    predicted_actions = sum(
        item["predicted"]["action"] == "actionable" for item in items
    )
    corrections = sum(item["correction_required"] for item in items)
    critical_corrections = sum(
        item["critical_correction_required"] for item in items
    )
    direction_evaluable = [
        item
        for item in items
        if item["structures"]["active_boundary"]["direction_correct"] is not None
    ]
    direction_correct = sum(
        item["structures"]["active_boundary"]["direction_correct"]
        for item in direction_evaluable
    )
    policy_check_names = list(
        dict.fromkeys(
            check["name"]
            for item in items
            for check in (item.get("intervention_checks") or [])
        )
    )
    aggregate_policy_checks: list[dict[str, Any]] = []
    for name in policy_check_names:
        checks = [
            check
            for item in items
            for check in (item.get("intervention_checks") or [])
            if check["name"] == name
        ]
        applicable = [check for check in checks if check["applicable"]]
        included = [
            check
            for check in applicable
            if check["included_in_estimate"]
        ]
        aggregate_policy_checks.append(
            {
                "name": name,
                "episode_count": len(checks),
                "applicable_episode_count": len(applicable),
                "supported_episode_count": sum(
                    check["supported"] for check in applicable
                ),
                "included_episode_count": len(included),
                "intervention_required_count": sum(
                    check["requires_intervention"] for check in included
                ),
                "estimated_intervention_rate": _wilson(
                    sum(check["requires_intervention"] for check in included),
                    len(included),
                    z,
                ),
            }
        )
    estimated_intervention_rate = _wilson(corrections, count, z)
    critical_intervention_rate = _wilson(critical_corrections, count, z)
    observed_correction_rate = {
        "supported": False,
        "reason": "benchmark_records_gold_discrepancies_not_observed_human_actions",
        "numerator": None,
        "denominator": None,
        "value": None,
        "ci_low": None,
        "ci_high": None,
    }
    gate_confidence = benchmark_gate_confidence(
        items,
        confidence=config.bootstrap_confidence,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )
    gate_intervals = gate_confidence["metrics"]

    def apply_cluster_interval(
        metric: dict[str, Any], interval: dict[str, Any]
    ) -> None:
        metric["ci_low"] = (
            round(float(interval["ci_low"]), 6)
            if interval["ci_low"] is not None
            else None
        )
        metric["ci_high"] = (
            round(float(interval["ci_high"]), 6)
            if interval["ci_high"] is not None
            else None
        )
        metric["ci_method"] = interval["method"]
        metric["cluster_count"] = interval["cluster_count"]
        metric["bootstrap_samples"] = config.bootstrap_samples
        metric["bootstrap_seed"] = config.bootstrap_seed
        metric["confidence"] = config.bootstrap_confidence

    apply_cluster_interval(top["f1"], gate_intervals["top_f1"])
    apply_cluster_interval(
        classifications["lifecycle"]["macro_f1"],
        gate_intervals["lifecycle_macro_f1"],
    )
    apply_cluster_interval(
        action_binary["precision"], gate_intervals["action_precision"]
    )
    apply_cluster_interval(
        estimated_intervention_rate,
        gate_intervals["estimated_human_intervention_rate"],
    )
    apply_cluster_interval(
        critical_intervention_rate,
        gate_intervals["critical_intervention_rate"],
    )
    release_gate_metrics = {
        "top_f1": gate_intervals["top_f1"],
        "lifecycle_macro_f1": gate_intervals["lifecycle_macro_f1"],
        "action_precision": gate_intervals["action_precision"],
        "false_action_rate": gate_intervals["false_action_rate"],
        "estimated_human_intervention_rate": gate_intervals[
            "estimated_human_intervention_rate"
        ],
        "critical_intervention_rate": gate_intervals[
            "critical_intervention_rate"
        ],
    }
    return {
        "episode_count": count,
        "classifications": classifications,
        "lifecycle_transition_timing": _transition_timing_summary(
            items, field="lifecycle", config=config
        ),
        "top_detection": top,
        "selected_top_membership_proxy": selected_top_membership,
        "construction_anchors": anchors,
        "supporting_touches": supporting_touches,
        "structures": {
            "geometry": structures,
            "primary_selection": primary,
            "alternate_selection": alternates,
            "parent_child_topology": hierarchy,
            "line_direction_accuracy": _wilson(
                direction_correct, len(direction_evaluable), z
            ),
            "projected_line_error_pct": _numeric_summary(projection_errors),
        },
        "excluded_high_rejection_rate": _wilson(
            excluded_rejected, excluded_total, z
        ),
        "bottoms": bottom_metrics,
        "phases": phase_metrics,
        "events": {
            "timing": events,
            **event_state_metrics,
        },
        "actionable_binary": action_binary,
        "abstention_rate": abstention_rate,
        "false_action_rate": _wilson(false_actions, predicted_actions, z),
        "observed_human_correction_rate": observed_correction_rate,
        "estimated_human_intervention_rate": estimated_intervention_rate,
        "critical_intervention_rate": critical_intervention_rate,
        "intervention_policy": {
            "version": INTERVENTION_POLICY_VERSION,
            "check_names": policy_check_names,
            "checks": aggregate_policy_checks,
        },
        "intervention_policy_version": INTERVENTION_POLICY_VERSION,
        "intervention_policy_checks": aggregate_policy_checks,
        "gate_confidence": gate_confidence,
        "gate_metrics": release_gate_metrics,
    }


def _load_episode_input(
    episode: dict[str, Any],
    *,
    source_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eodhd_snapshot = episode.get("eodhd_snapshot")
    if isinstance(eodhd_snapshot, (dict, str)):
        from eodhd_ingestion import (
            EodhdIngestionError,
            load_frozen_snapshot,
            validate_frozen_snapshot,
        )

        try:
            if isinstance(eodhd_snapshot, dict):
                snapshot = validate_frozen_snapshot(eodhd_snapshot)
            else:
                snapshot_path = Path(eodhd_snapshot)
                if not snapshot_path.is_absolute():
                    snapshot_path = (source_root or Path.cwd()) / snapshot_path
                snapshot = load_frozen_snapshot(snapshot_path)
        except EodhdIngestionError as exc:
            raise BenchmarkCorpusError(f"invalid EODHD snapshot: {exc}") from exc
        expected_ticker = str(episode.get("ticker") or "").strip().upper()
        if expected_ticker and expected_ticker != snapshot["symbol"]:
            raise BenchmarkCorpusError(
                "episode ticker does not match its EODHD snapshot symbol"
            )
        params = snapshot["request"]["params"]
        return snapshot["monthly_bars"], {
            "kind": snapshot["kind"],
            "provider": snapshot["provider"],
            "symbol": snapshot["symbol"],
            "requested_from": params["from"],
            "requested_to": params["to"],
            "fetched_at": snapshot["fetched_at"],
            "code_sha": snapshot["code_sha"],
            "provider_rows_sha256": snapshot["provider_rows_sha256"],
            "monthly_bars_sha256": snapshot["monthly_bars_sha256"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
        }

    bars = episode.get("monthly_bars")
    if isinstance(bars, list):
        return bars, {
            "kind": "coilingview.embedded-monthly-bars",
            "monthly_bars_sha256": sha256_json(bars),
        }
    source = episode.get("source")
    ticker = episode.get("ticker")
    if source and ticker:
        # Import the candle-only snapshot loader lazily.  Never call
        # load_review_context(): it can return stored model/human analysis.
        from review_snapshots import load_blind_review_context

        context = load_blind_review_context(str(source), str(ticker))
        return context["monthly_bars"], {
            "kind": "coilingview.saved-run-review-snapshot",
            "source": str(source),
            "ticker": str(ticker),
            "sample_id": context.get("sample_id"),
            "monthly_bars_sha256": context.get("bars_hash"),
        }
    raise BenchmarkCorpusError(
        "each episode requires monthly_bars, eodhd_snapshot, or a frozen "
        "source/ticker snapshot"
    )


def _load_episode_bars(
    episode: dict[str, Any],
    *,
    source_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Compatibility facade for callers that only need the validated bars."""
    return _load_episode_input(episode, source_root=source_root)[0]


def _load_episode_gold(
    episode: dict[str, Any], monthly_bars: list[dict[str, Any]]
) -> dict[str, Any]:
    label = episode.get("gold_label")
    if isinstance(label, dict) and label.get("kind") == MATERIALIZED_KIND:
        rematerialized = _rematerialize_gold_label(label, monthly_bars)
        capture = episode.get("gold_capture")
        if isinstance(capture, dict):
            separately_materialized = materialize_gold_label(capture, monthly_bars)
            if canonical_json(separately_materialized).encode(
                "utf-8"
            ) != canonical_json(rematerialized).encode("utf-8"):
                raise BenchmarkCorpusError(
                    "episode gold_capture differs from materialized gold_label"
                )
        return rematerialized
    capture = episode.get("gold_capture")
    if isinstance(capture, dict):
        materialized = materialize_gold_label(capture, monthly_bars)
        return _rematerialize_gold_label(materialized, monthly_bars)
    raise BenchmarkCorpusError(
        "each episode requires gold_capture or a materialized gold_label"
    )


def _holdout_claim_eligibility(
    raw_corpus: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    setup_count = len({str(item["setup_id"]) for item in results})
    blind_count = sum(
        item["evaluation_role"] == "blind_benchmark" for item in results
    )
    outcome_visible_count = sum(
        bool(item["outcome_visible_during_label"]) for item in results
    )
    incomplete_top_review_count = sum(
        item.get("top_review_complete") is not True for item in results
    )
    setup_strata: dict[str, str] = {}
    stratum_conflicts: set[str] = set()
    invalid_stratum_setups: set[str] = set()
    for item in results:
        setup_id = str(item["setup_id"])
        stratum = item.get("sampling_stratum")
        if stratum not in BLIND_COMPOSITION_TARGETS:
            invalid_stratum_setups.add(setup_id)
            continue
        prior = setup_strata.setdefault(setup_id, str(stratum))
        if prior != stratum:
            stratum_conflicts.add(setup_id)
    observed_composition = Counter(setup_strata.values())
    failures: list[dict[str, str]] = []

    def fail(code: str, message: str) -> None:
        failures.append({"code": code, "message": message})

    if blind_count != len(results):
        fail(
            "non_blind_episode_present",
            "every episode and split must have the blind_benchmark role",
        )
    if outcome_visible_count:
        fail(
            "outcome_visible_label_present",
            "blind holdout labels must be finalized without outcome visibility",
        )
    if incomplete_top_review_count:
        fail(
            "incomplete_top_review",
            "every blind setup must attest top_review_complete=true",
        )
    if setup_count < MIN_BLIND_SETUP_COUNT:
        fail(
            "insufficient_unique_setups",
            f"at least {MIN_BLIND_SETUP_COUNT} unique blind setups are required",
        )

    protocol = raw_corpus.get("blind_protocol")
    if not isinstance(protocol, dict):
        fail(
            "missing_blind_protocol",
            "corpus blind_protocol attestations and composition are required",
        )
        attestations: dict[str, Any] = {}
        composition: dict[str, Any] = {}
    else:
        if protocol.get("schema_version") != 1:
            fail(
                "unsupported_blind_protocol_schema",
                "blind_protocol.schema_version must equal 1",
            )
        attestations = (
            protocol.get("attestations")
            if isinstance(protocol.get("attestations"), dict)
            else {}
        )
        composition = (
            protocol.get("composition")
            if isinstance(protocol.get("composition"), dict)
            else {}
        )
        missing_identity = [
            field
            for field in BLIND_PROTOCOL_IDENTITY_FIELDS
            if not isinstance(protocol.get(field), str)
            or not str(protocol.get(field)).strip()
        ]
        evidence_sha256 = str(protocol.get("evidence_sha256") or "")
        if evidence_sha256 and (
            len(evidence_sha256) != 64
            or any(character not in "0123456789abcdef" for character in evidence_sha256)
        ):
            missing_identity.append("evidence_sha256")
        finalized_at = str(protocol.get("finalized_at") or "")
        if finalized_at:
            try:
                parsed_finalized_at = datetime.fromisoformat(
                    finalized_at.replace("Z", "+00:00")
                )
                if parsed_finalized_at.utcoffset() is None:
                    raise ValueError
            except ValueError:
                missing_identity.append("finalized_at")
        if missing_identity:
            fail(
                "blind_protocol_identity_invalid",
                "blind protocol requires server-verifiable identity fields: "
                + ", ".join(sorted(set(missing_identity))),
            )
        if protocol.get("frozen_detector_version") != ALGORITHM_VERSION:
            fail(
                "blind_protocol_detector_version_mismatch",
                "blind protocol frozen_detector_version must match the evaluated detector",
            )

    missing_attestations = [
        name for name in BLIND_PROTOCOL_ATTESTATIONS if attestations.get(name) is not True
    ]
    if missing_attestations:
        fail(
            "blind_protocol_attestations_incomplete",
            "required true attestations are missing: "
            + ", ".join(missing_attestations),
        )

    if invalid_stratum_setups:
        fail(
            "blind_setup_stratum_missing_or_invalid",
            "every setup requires one recorded benchmark sampling stratum",
        )
    if stratum_conflicts:
        fail(
            "blind_setup_stratum_conflict",
            "all checkpoints from one setup must retain the same sampling stratum",
        )

    composition_counts: dict[str, int] = {}
    invalid_composition = False
    for name in BLIND_COMPOSITION_TARGETS:
        value = composition.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            invalid_composition = True
        else:
            composition_counts[name] = value
    if invalid_composition:
        fail(
            "blind_composition_invalid",
            "blind composition requires nonnegative integer counts for every stratum",
        )
    elif sum(composition_counts.values()) != setup_count:
        fail(
            "blind_composition_count_mismatch",
            "blind composition counts must sum to the unique setup count",
        )
    elif any(
        composition_counts[name] != observed_composition.get(name, 0)
        for name in BLIND_COMPOSITION_TARGETS
    ):
        fail(
            "blind_composition_not_reproducible",
            "blind composition metadata must equal setup-level episode strata",
        )
    else:
        off_target = [
            name
            for name, target in BLIND_COMPOSITION_TARGETS.items()
            if abs(composition_counts[name] - target * setup_count) > 1.0
        ]
        if off_target:
            fail(
                "blind_composition_off_target",
                "composition must be 30% predicted-positive, 30% near-boundary "
                "negative, 20% hard-trap, and 20% random-population; off target: "
                + ", ".join(off_target),
            )

    return {
        "eligible": not failures,
        "failure_reasons": failures,
        "episode_count": len(results),
        "blind_episode_count": blind_count,
        "unique_setup_count": setup_count,
        "outcome_visible_count": outcome_visible_count,
        "incomplete_top_review_count": incomplete_top_review_count,
        "observed_setup_composition": {
            name: observed_composition.get(name, 0)
            for name in BLIND_COMPOSITION_TARGETS
        },
        "requirements": {
            "all_episodes_blind": True,
            "all_top_reviews_complete": True,
            "minimum_unique_setups": MIN_BLIND_SETUP_COUNT,
            "required_attestations": list(BLIND_PROTOCOL_ATTESTATIONS),
            "required_protocol_identity_fields": list(
                BLIND_PROTOCOL_IDENTITY_FIELDS
            ),
            "composition_targets": BLIND_COMPOSITION_TARGETS,
        },
    }


def evaluate_corpus(
    raw_corpus: dict[str, Any],
    *,
    config: EvaluatorConfig = EvaluatorConfig(),
    source_root: Path | None = None,
    expected_code_sha: str | None = None,
) -> dict[str, Any]:
    """Evaluate a complete corpus, failing the run on any leakage/integrity error."""
    if not isinstance(raw_corpus, dict):
        raise BenchmarkCorpusError("benchmark corpus must be a JSON object")
    if raw_corpus.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise BenchmarkCorpusError("unsupported detector benchmark corpus schema")
    if raw_corpus.get("kind") != CORPUS_KIND:
        raise BenchmarkCorpusError("unexpected detector benchmark corpus kind")
    corpus_id = str(raw_corpus.get("corpus_id") or "").strip()
    if not corpus_id:
        raise BenchmarkCorpusError("benchmark corpus_id is required")
    episodes = raw_corpus.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise BenchmarkCorpusError("benchmark corpus requires episodes")
    has_eodhd_snapshot = any(
        isinstance(episode, dict) and "eodhd_snapshot" in episode
        for episode in episodes
    )
    raw_code_sha = raw_corpus.get("code_sha")
    corpus_code_sha = (
        str(raw_code_sha).strip().lower() if raw_code_sha is not None else None
    )
    if has_eodhd_snapshot and not (
        isinstance(corpus_code_sha, str)
        and _FULL_GIT_SHA_RE.fullmatch(corpus_code_sha)
    ):
        raise BenchmarkCorpusError(
            "an EODHD benchmark corpus requires the exact full pushed code_sha"
        )
    if expected_code_sha is not None:
        runtime_sha = str(expected_code_sha).strip().lower()
        if not _FULL_GIT_SHA_RE.fullmatch(runtime_sha):
            raise BenchmarkCorpusError("expected_code_sha must be a full Git SHA")
        if corpus_code_sha != runtime_sha:
            raise BenchmarkCorpusError(
                "benchmark corpus code_sha does not match the running checkout"
            )

    setup_splits: dict[str, str] = {}
    seen_episode_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    roles: Counter[str] = Counter()
    outcome_visible_count = 0
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise BenchmarkCorpusError(f"episode {index} must be an object")
        bars, input_source = _load_episode_input(
            episode,
            source_root=source_root,
        )
        if (
            input_source.get("kind")
            == "coilingview.eodhd-monthly-ohlcv-snapshot"
            and input_source.get("code_sha") != corpus_code_sha
        ):
            raise BenchmarkCorpusError(
                "EODHD snapshot code_sha does not match the benchmark corpus"
            )
        gold = _load_episode_gold(episode, bars)
        episode_id = str(gold["episode_id"])
        if episode_id in seen_episode_ids:
            raise BenchmarkCorpusError(f"duplicate episode_id: {episode_id}")
        seen_episode_ids.add(episode_id)
        if "split" not in episode:
            raise BenchmarkCorpusError(
                f"episode {episode_id} requires an explicit split"
            )
        split = str(episode["split"])
        evaluation_role = str(gold["evaluation_role"])
        if split != evaluation_role:
            raise BenchmarkCorpusError(
                f"episode {episode_id} split must exactly match gold evaluation_role"
            )
        setup_id = str(gold["setup_id"])
        prior_split = setup_splits.setdefault(setup_id, split)
        if prior_split != split:
            raise BenchmarkCorpusError(
                f"setup {setup_id} appears in multiple splits"
            )
        roles[evaluation_role] += 1
        outcome_visible_count += bool(gold.get("outcome_visible_during_label"))
        result = evaluate_episode(gold, bars, config=config)
        result["input_source"] = input_source
        result["sampling_stratum"] = episode.get(
            "sampling_stratum", episode.get("stratum")
        )
        results.append(result)

    aggregate = aggregate_results(results, config=config)
    holdout_claim = _holdout_claim_eligibility(raw_corpus, results)
    aggregate["holdout_claim_eligibility"] = holdout_claim
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "corpus_id": corpus_id,
        "corpus_sha256": sha256_json(raw_corpus),
        "code_sha": corpus_code_sha,
        "algorithm_version": ALGORITHM_VERSION,
        "default_config_sha256": DEFAULT_CONFIG_SHA256,
        "intervention_policy_version": INTERVENTION_POLICY_VERSION,
        "detector_only": True,
        "analyzer_call_count": len(results),
        "review_override_attempts": 0,
        "config": {
            "candle_tolerance": config.candle_tolerance,
            "projected_line_tolerance_pct": config.projected_line_tolerance_pct,
            "slope_tolerance_pct_per_year": config.slope_tolerance_pct_per_year,
            "phase_tolerance": config.phase_tolerance,
            "confidence_z": config.confidence_z,
            "bootstrap_confidence": config.bootstrap_confidence,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
        },
        "evaluation_roles": dict(sorted(roles.items())),
        "holdout_claim_eligible": holdout_claim["eligible"],
        "holdout_claim_failures": holdout_claim["failure_reasons"],
        "aggregate": aggregate,
        "episodes": results,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detector-only point-in-time exemplar evaluation."
    )
    parser.add_argument("corpus", type=Path, help="benchmark corpus JSON")
    parser.add_argument("--output", type=Path, help="write report JSON here")
    parser.add_argument("--candle-tolerance", type=int, default=1)
    parser.add_argument("--line-tolerance-pct", type=float, default=5.0)
    parser.add_argument("--slope-tolerance-pct-per-year", type=float, default=1.0)
    parser.add_argument("--phase-tolerance", type=int, default=1)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    raw = json.loads(args.corpus.read_text(encoding="utf-8"))
    has_eodhd_snapshot = any(
        isinstance(episode, dict) and "eodhd_snapshot" in episode
        for episode in raw.get("episodes", [])
    )
    expected_code_sha = None
    if has_eodhd_snapshot:
        from eodhd_ingestion import repository_identity

        expected_code_sha = repository_identity(Path(__file__).resolve().parent)
    report = evaluate_corpus(
        raw,
        config=EvaluatorConfig(
            candle_tolerance=args.candle_tolerance,
            projected_line_tolerance_pct=args.line_tolerance_pct,
            slope_tolerance_pct_per_year=args.slope_tolerance_pct_per_year,
            phase_tolerance=args.phase_tolerance,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
        source_root=args.corpus.resolve().parent,
        expected_code_sha=expected_code_sha,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
