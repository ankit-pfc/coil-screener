"""Lean v2.4 validation-only major-top and resistance-band detector.

This module is deliberately separate from the production v2.3.1 detector.  It
implements only the bounded structural hypotheses needed by the blind pilot:
point-in-time top evidence, alternative lid hypotheses, a consensus band,
structure/readiness separation, and explicit abstention.  It contains no ML,
trading, sizing, execution, or universe-selection logic.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable

from bar_integrity import (
    ADJUSTMENT_UNKNOWN,
    DATA_QUALITY_BLOCKED,
    inspect_monthly_bars,
)

ALGORITHM_VERSION = "2.4.0-validation"
VARIANT = "v2_4_validation"
MODE = "algorithm_only"
SCHEMA_VERSION = 2
SOURCE = "timeseries"


@dataclass(frozen=True)
class ValidationConfig:
    """Frozen lean-pilot parameters; only four fields may be swept."""

    zone_candidate_prominence_pct: float = 18.5
    zone_similarity_pct: float = 5.0
    touch_tolerance_pct: float = 3.5
    max_qualifying_lid_slope_pct_per_year: float = 12.0
    strict_wick_prominence_pct: float = 28.0
    strict_body_prominence_pct: float = 21.0
    strict_min_range_position_pct: float = 35.0
    secondary_wick_prominence_pct: float = 14.0
    secondary_body_prominence_pct: float = 10.0
    required_strict_major_count: int = 1
    min_separation_quarters: int = 4
    min_structure_years: float = 10.0
    min_slope_pct_per_year: float = -3.0
    max_violations: int = 2
    steady_trend_r2_veto: float = 0.85
    pressing_min_pct: float = 90.0
    equivalent_fit_error_delta_pct: float = 1.0
    equivalent_projection_delta_pct: float = 5.0

    def __post_init__(self) -> None:
        allowed = {
            "zone_candidate_prominence_pct": {15.0, 18.5, 22.5},
            "zone_similarity_pct": {3.5, 5.0, 7.5},
            "touch_tolerance_pct": {2.5, 3.5, 5.0},
            "max_qualifying_lid_slope_pct_per_year": {6.5, 12.0},
        }
        for name, values in allowed.items():
            if float(getattr(self, name)) not in values:
                raise ValueError(
                    f"{name} is outside the registered lean sweep: "
                    f"{sorted(values)}"
                )
        frozen = {
            "strict_wick_prominence_pct": 28.0,
            "strict_body_prominence_pct": 21.0,
            "strict_min_range_position_pct": 35.0,
            "secondary_wick_prominence_pct": 14.0,
            "secondary_body_prominence_pct": 10.0,
            "required_strict_major_count": 1,
            "min_separation_quarters": 4,
            "min_structure_years": 10.0,
            "min_slope_pct_per_year": -3.0,
            "max_violations": 2,
            "steady_trend_r2_veto": 0.85,
            "pressing_min_pct": 90.0,
            "equivalent_fit_error_delta_pct": 1.0,
            "equivalent_projection_delta_pct": 5.0,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is frozen for the lean validation pilot")


DEFAULT_CONFIG = ValidationConfig()


def config_fingerprint(config: ValidationConfig = DEFAULT_CONFIG) -> str:
    payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _stable_id(prefix: str, payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _scope_output_ids(
    result: dict[str, Any],
    *,
    ticker: str,
    as_of: str | None,
    config: ValidationConfig,
) -> dict[str, Any]:
    """Bind every reviewable id to sample identity and frozen configuration."""
    scope = {
        "ticker": ticker.strip().upper(),
        "as_of": as_of or result.get("as_of"),
        "config_fingerprint": config_fingerprint(config),
    }
    top_id_map: dict[str, str] = {}
    for top in result.get("top_candidates", []):
        old = str(top["id"])
        new = _stable_id("top", {"scope": scope, "event": top})
        top["id"] = new
        top_id_map[old] = new
    hypothesis_id_map: dict[str, str] = {}
    for hypothesis in result.get("lid_hypotheses", []):
        old = str(hypothesis["id"])
        contacts = [top_id_map.get(str(value), str(value)) for value in hypothesis.get("contact_ids", [])]
        hypothesis["contact_ids"] = contacts
        if isinstance(hypothesis.get("contacts"), list):
            for contact in hypothesis["contacts"]:
                if isinstance(contact, dict) and "id" in contact:
                    contact["id"] = top_id_map.get(str(contact["id"]), str(contact["id"]))
        new = _stable_id(
            "lid",
            {"scope": scope, "contacts": contacts, "rank": hypothesis.get("rank")},
        )
        hypothesis["id"] = new
        hypothesis_id_map[old] = new
    band = result.get("resistance_band")
    if isinstance(band, dict):
        band["hypothesis_ids"] = [
            hypothesis_id_map.get(str(value), str(value))
            for value in band.get("hypothesis_ids", [])
        ]
    for field in ("points", "major_highs"):
        for point in result.get(field, []):
            evidence = point.get("evidence") if isinstance(point, dict) else None
            if isinstance(evidence, dict) and evidence.get("top_candidate_id"):
                evidence["top_candidate_id"] = top_id_map.get(
                    str(evidence["top_candidate_id"]),
                    str(evidence["top_candidate_id"]),
                )
    return result


def _month_end(text: str) -> str:
    parsed = date.fromisoformat(text[:10])
    return date(
        parsed.year, parsed.month, calendar.monthrange(parsed.year, parsed.month)[1]
    ).isoformat()


def _quarter_key(text: str) -> tuple[int, int]:
    parsed = date.fromisoformat(text[:10])
    return parsed.year, (parsed.month - 1) // 3 + 1


def _quarter_ordinal(key: tuple[int, int]) -> int:
    return key[0] * 4 + key[1] - 1


def _point_coordinate(point: dict[str, Any]) -> int:
    return int(point.get("calendar_quarter_index", point["quarter_index"]))


def _aggregate_quarters(bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quarters: list[dict[str, Any]] = []
    for monthly_idx, bar in enumerate(bars):
        key = _quarter_key(str(bar["date"]))
        parsed = date.fromisoformat(str(bar["date"])[:10])
        volume = bar.get("volume")
        if quarters and quarters[-1]["quarter_key"] == key:
            quarter = quarters[-1]
            if float(bar["high"]) > float(quarter["high"]):
                quarter["high"] = float(bar["high"])
                quarter["high_month_idx"] = monthly_idx
                quarter["peak_date"] = str(bar["date"])
            quarter["low"] = min(float(quarter["low"]), float(bar["low"]))
            quarter["close"] = float(bar["close"])
            quarter["date"] = str(bar["date"])
            quarter["last_month"] = parsed.month
            quarter["close_month_idx"] = monthly_idx
            if quarter["volume"] is None or volume is None:
                quarter["volume"] = None
            else:
                quarter["volume"] = float(quarter["volume"]) + float(volume)
        else:
            quarters.append(
                {
                    "quarter_key": key,
                    "calendar_quarter_index": _quarter_ordinal(key),
                    "date": str(bar["date"]),
                    "last_month": parsed.month,
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": float(volume) if volume is not None else None,
                    "high_month_idx": monthly_idx,
                    "close_month_idx": monthly_idx,
                    "peak_date": str(bar["date"]),
                }
            )
    return quarters


def _quarter_complete(quarter: dict[str, Any], *, as_of: str | None) -> bool:
    month = int(quarter["last_month"])
    if month % 3 != 0:
        return False
    if as_of is None:
        parsed = date.fromisoformat(str(quarter["date"])[:10])
        today = date.today()
        return (parsed.year, parsed.month) < (today.year, today.month)
    return _month_end(str(quarter["date"])) <= as_of


def _trailing_partial_quarter(
    quarters: list[dict[str, Any]], *, as_of: str | None
) -> dict[str, Any] | None:
    if not quarters or _quarter_complete(quarters[-1], as_of=as_of):
        return None
    return quarters[-1]


def _pivot_indexes(highs: list[float]) -> list[int]:
    return [
        idx
        for idx in range(1, len(highs) - 1)
        if highs[idx] >= highs[idx - 1]
        and highs[idx] >= highs[idx + 1]
        and (highs[idx] > highs[idx - 1] or highs[idx] > highs[idx + 1])
    ]


def _side_floor(
    values: list[float], idx: int, peak: float, step: int
) -> float | None:
    floor: float | None = None
    cursor = idx + step
    while 0 <= cursor < len(values):
        if values[cursor] > peak:
            break
        floor = values[cursor] if floor is None else min(floor, values[cursor])
        cursor += step
    return floor


def _two_sided_prominence(values: list[float], idx: int, peak: float) -> float:
    floors = [
        value
        for value in (
            _side_floor(values, idx, peak, -1),
            _side_floor(values, idx, peak, 1),
        )
        if value is not None
    ]
    if len(floors) < 2 or peak <= 0:
        return 0.0
    return max(0.0, (peak - max(floors)) / peak * 100.0)


def _plateau_clusters(
    indexes: list[int], highs: list[float], config: ValidationConfig
) -> list[list[int]]:
    clusters: list[list[int]] = []
    for idx in indexes:
        if (
            clusters
            # Every bar between sparse pivot edges must remain in the same
            # price shoulder. A later same-zone contact after a real valley is
            # a new retest, never a retroactive replacement.
            and all(
                abs(highs[cursor] - highs[clusters[-1][0]])
                / max(highs[cursor], highs[clusters[-1][0]])
                <= config.zone_similarity_pct / 100.0
                for cursor in range(clusters[-1][-1], idx + 1)
            )
            and abs(highs[idx] - highs[clusters[-1][0]])
            / max(highs[idx], highs[clusters[-1][0]])
            <= config.zone_similarity_pct / 100.0
        ):
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return clusters


def _cluster_plateaus(
    indexes: list[int], highs: list[float], config: ValidationConfig
) -> list[int]:
    return [
        sorted(cluster, key=lambda idx: (-highs[idx], idx))[0]
        for cluster in _plateau_clusters(indexes, highs, config)
    ]


def _top_candidates(
    quarters: list[dict[str, Any]], config: ValidationConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    highs = [float(item["high"]) for item in quarters]
    indexes = _cluster_plateaus(_pivot_indexes(highs), highs, config)
    candidates: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []

    for idx in indexes:
        evidence: dict[str, Any] | None = None
        latest: dict[str, Any] | None = None
        rejection_evidence: dict[str, Any] | None = None
        # Replay prefixes and freeze the first prefix on which the required
        # decline evidence passes. Later quarters cannot improve or re-role an
        # already-confirmed event.
        for end in range(idx + 2, len(quarters) + 1):
            prefix = quarters[:end]
            prefix_highs = [float(item["high"]) for item in prefix]
            prefix_bodies = [
                max(float(item["open"]), float(item["close"])) for item in prefix
            ]
            prefix_lows = [float(item["low"]) for item in prefix]
            prefix_clusters = _plateau_clusters(
                _pivot_indexes(prefix_highs), prefix_highs, config
            )
            prefix_indexes = [
                sorted(cluster, key=lambda position: (-prefix_highs[position], position))[0]
                for cluster in prefix_clusters
            ]
            if idx not in prefix_indexes:
                continue
            plateau = next(
                cluster
                for cluster, representative in zip(prefix_clusters, prefix_indexes)
                if representative == idx
            )
            active_start = max(0, end - 40)
            active_low = min(prefix_lows[active_start:])
            active_high = max(prefix_highs[active_start:])
            active_range = max(1e-9, active_high - active_low)
            latest = {
                "wick_prominence_pct": _two_sided_prominence(
                    prefix_highs, idx, prefix_highs[idx]
                ),
                "body_prominence_pct": _two_sided_prominence(
                    prefix_bodies, idx, prefix_bodies[idx]
                ),
                "range_position_pct": (
                    prefix_highs[idx] - active_low
                )
                / active_range
                * 100.0,
                "confirmed_quarter_index": end - 1,
                "plateau_start_quarter_index": min(plateau),
                "plateau_end_quarter_index": max(plateau),
                "plateau_quarters": max(plateau) - min(plateau) + 1,
            }
            if (
                latest["wick_prominence_pct"]
                >= config.zone_candidate_prominence_pct
            ):
                evidence = latest
                break
            # A strictly higher later high closes this peak's prominence basin.
            # If the structural gate still has not passed, rejection is now
            # final and gets one stable observation timestamp.
            if any(value > prefix_highs[idx] for value in prefix_highs[idx + 1 :]):
                rejection_evidence = latest
                break

        observed = evidence or rejection_evidence or latest or {
            "wick_prominence_pct": 0.0,
            "body_prominence_pct": 0.0,
            "range_position_pct": 0.0,
            "confirmed_quarter_index": None,
            "plateau_start_quarter_index": idx,
            "plateau_end_quarter_index": idx,
            "plateau_quarters": 1,
        }
        wick_prominence = float(observed["wick_prominence_pct"])
        body_prominence = float(observed["body_prominence_pct"])
        range_position = float(observed["range_position_pct"])
        structural = evidence is not None
        rejected = rejection_evidence is not None
        rejection_reasons: list[str] = []
        if not structural:
            rejection_reasons.append("zone_candidate_prominence_below_threshold")
        strict_failures: list[str] = []
        if wick_prominence < config.strict_wick_prominence_pct:
            strict_failures.append("strict_wick_prominence_below_threshold")
        if body_prominence < config.strict_body_prominence_pct:
            strict_failures.append("strict_body_prominence_below_threshold")
        if range_position < config.strict_min_range_position_pct:
            strict_failures.append("strict_range_position_below_threshold")
        strict = structural and not strict_failures
        secondary_eligible = (
            wick_prominence >= config.secondary_wick_prominence_pct
            and body_prominence >= config.secondary_body_prominence_pct
            and range_position >= config.strict_min_range_position_pct
        )
        role = (
            "confirmed_major_top"
            if strict and structural
            else "confirmed_structural_retest"
            if structural
            else "rejected_high"
            if rejected
            else "pending_top"
        )
        confirmed_quarter_idx = (
            int(observed["confirmed_quarter_index"])
            if (evidence is not None or rejection_evidence is not None)
            and observed["confirmed_quarter_index"] is not None
            else None
        )
        confirmed_month_idx = (
            int(quarters[confirmed_quarter_idx]["close_month_idx"])
            if confirmed_quarter_idx is not None
            else None
        )
        confirmed_at = (
            _month_end(str(quarters[confirmed_quarter_idx]["date"]))
            if confirmed_quarter_idx is not None
            else None
        )
        item = {
            "id": _stable_id(
                "top",
                [quarters[idx]["peak_date"], highs[idx], confirmed_quarter_idx],
            ),
            "peak_index": int(quarters[idx]["high_month_idx"]),
            "quarter_index": idx,
            "calendar_quarter_index": int(
                quarters[idx].get("calendar_quarter_index", idx)
            ),
            "peak_date": quarters[idx]["peak_date"],
            "confirmed_at_index": confirmed_month_idx,
            "confirmed_at": confirmed_at,
            "confirmed_quarter_index": confirmed_quarter_idx,
            "confirmed_calendar_quarter_index": (
                int(
                    quarters[confirmed_quarter_idx].get(
                        "calendar_quarter_index", confirmed_quarter_idx
                    )
                )
                if confirmed_quarter_idx is not None
                else None
            ),
            "price": round(highs[idx], 4),
            "role": role,
            "strict_major": strict,
            "structural_eligible": structural,
            "secondary_compatibility_eligible": secondary_eligible,
            "wick_prominence_pct": round(wick_prominence, 3),
            "body_prominence_pct": round(body_prominence, 3),
            "range_position_pct": round(range_position, 3),
            "confirmation_lag_quarters": (
                confirmed_quarter_idx - idx
                if structural and confirmed_quarter_idx is not None
                else None
            ),
            "rejection_quarters": (
                confirmed_quarter_idx - idx
                if rejected and confirmed_quarter_idx is not None
                else None
            ),
            "plateau_start_quarter_index": int(
                observed["plateau_start_quarter_index"]
            ),
            "plateau_end_quarter_index": int(
                observed["plateau_end_quarter_index"]
            ),
            "plateau_quarters": int(observed["plateau_quarters"]),
            "rejection_reasons": rejection_reasons + strict_failures,
        }
        candidates.append(item)
        if structural:
            eligible.append(item)

    if len(quarters) >= 2:
        idx = len(quarters) - 1
        if highs[idx] >= highs[idx - 1]:
            pending_start = max(0, len(quarters) - 40)
            pending_low = min(
                float(item["low"]) for item in quarters[pending_start:]
            )
            pending_high = max(highs[pending_start:])
            pending_range = max(1e-9, pending_high - pending_low)
            candidates.append(
                {
                    "id": _stable_id(
                        "top_pending", [quarters[idx]["peak_date"], highs[idx]]
                    ),
                    "peak_index": int(quarters[idx]["high_month_idx"]),
                    "quarter_index": idx,
                    "calendar_quarter_index": int(
                        quarters[idx].get("calendar_quarter_index", idx)
                    ),
                    "peak_date": quarters[idx]["peak_date"],
                    "confirmed_at_index": None,
                    "confirmed_at": None,
                    "confirmed_quarter_index": None,
                    "confirmed_calendar_quarter_index": None,
                    "price": round(highs[idx], 4),
                    "role": "pending_top",
                    "strict_major": False,
                    "structural_eligible": False,
                    "secondary_compatibility_eligible": False,
                    "wick_prominence_pct": 0.0,
                    "body_prominence_pct": 0.0,
                    "range_position_pct": round(
                        (highs[idx] - pending_low) / pending_range * 100.0, 3
                    ),
                    "confirmation_lag_quarters": None,
                    "rejection_quarters": None,
                    "plateau_start_quarter_index": idx,
                    "plateau_end_quarter_index": idx,
                    "plateau_quarters": 1,
                    "rejection_reasons": ["right_pivot_not_yet_observable"],
                }
            )
    candidates.sort(key=lambda item: (item["quarter_index"], item["id"]))
    eligible.sort(key=lambda item: item["quarter_index"])
    return candidates, eligible


def _least_squares(points: list[dict[str, Any]]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    xs = [float(_point_coordinate(point)) for point in points]
    ys = [float(point["price"]) for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    return slope, mean_y - slope * mean_x


def _fit_error(
    points: list[dict[str, Any]], slope: float, intercept: float
) -> float:
    errors = []
    for point in points:
        expected = intercept + slope * _point_coordinate(point)
        if expected <= 0:
            return math.inf
        errors.append(((float(point["price"]) - expected) / expected) ** 2)
    return math.sqrt(sum(errors) / len(errors)) * 100.0


def _independent_contacts(
    contacts: list[dict[str, Any]],
    *,
    min_separation_quarters: int,
    slope: float,
    intercept: float,
) -> list[dict[str, Any]]:
    """Return the strongest deterministic quarter-separated contact set."""
    ordered = sorted(
        contacts,
        key=lambda item: (_point_coordinate(item), str(item["id"])),
    )

    def score(items: list[dict[str, Any]]) -> tuple[int, int, float, int, int]:
        if not items:
            return (0, 0, 0.0, 0, -1)
        errors = []
        for item in items:
            expected = intercept + slope * _point_coordinate(item)
            errors.append(
                abs(float(item["price"]) - expected) / max(expected, 1e-9) * 100.0
            )
        indexes = [_point_coordinate(item) for item in items]
        return (
            len(items),
            sum(bool(item["strict_major"]) for item in items),
            -round(sum(errors), 12),
            max(indexes) - min(indexes),
            max(indexes),
        )

    def better(
        left: list[dict[str, Any]], right: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        left_score = score(left)
        right_score = score(right)
        if left_score != right_score:
            return left if left_score > right_score else right
        left_ids = tuple(str(item["id"]) for item in left)
        right_ids = tuple(str(item["id"]) for item in right)
        return left if left_ids <= right_ids else right

    best: list[list[dict[str, Any]]] = [[]]
    for pos, item in enumerate(ordered, start=1):
        item_idx = _point_coordinate(item)
        compatible_count = pos - 1
        while compatible_count > 0:
            prior_idx = _point_coordinate(ordered[compatible_count - 1])
            if item_idx - prior_idx >= min_separation_quarters:
                break
            compatible_count -= 1
        take = [*best[compatible_count], item]
        skip = best[pos - 1]
        best.append(better(take, skip))
    return best[-1]


def _slope_grade(value: float) -> str | None:
    if -1.0 <= value < 5.0:
        return "A"
    if -3.0 <= value < 6.5:
        return "B"
    if -3.0 <= value < 12.0:
        return "C"
    return None


def _hypotheses(
    quarters: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    config: ValidationConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_by_membership: dict[tuple[str, ...], dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    last_idx = int(
        quarters[-1].get("calendar_quarter_index", len(quarters) - 1)
    )
    for left_pos, left in enumerate(eligible):
        for right in eligible[left_pos + 1 :]:
            pair_id = _stable_id("hypothesis_pair", [left["id"], right["id"]])
            reasons: list[str] = []
            if (
                _point_coordinate(right) - _point_coordinate(left)
                < config.min_separation_quarters
            ):
                reasons.append("insufficient_contact_separation")
            initial = _least_squares([left, right])
            if initial is None:
                reasons.append("degenerate_anchor_pair")
            if reasons:
                rejected.append({"id": pair_id, "rejection_reasons": reasons})
                continue
            slope, intercept = initial
            contacts = []
            for point in eligible:
                expected = intercept + slope * _point_coordinate(point)
                if expected <= 0:
                    continue
                error = abs(float(point["price"]) - expected) / expected * 100.0
                if error <= config.touch_tolerance_pct:
                    contacts.append(point)
            contacts = _independent_contacts(
                contacts,
                min_separation_quarters=config.min_separation_quarters,
                slope=slope,
                intercept=intercept,
            )
            refit = _least_squares(contacts)
            if refit is None or len(contacts) < 2:
                rejected.append(
                    {"id": pair_id, "rejection_reasons": ["fewer_than_two_contacts"]}
                )
                continue
            slope, intercept = refit
            projection = intercept + slope * last_idx
            if projection <= 0:
                reasons.append("nonpositive_projection")
                slope_pct = math.inf
            else:
                slope_pct = slope * 4.0 / projection * 100.0
            if not (
                config.min_slope_pct_per_year
                <= slope_pct
                < config.max_qualifying_lid_slope_pct_per_year
            ):
                reasons.append("slope_outside_validation_range")
            first_position = min(int(point["quarter_index"]) for point in contacts)
            violations = [
                position
                for position in range(first_position, len(quarters))
                if float(quarters[position]["close"])
                > (
                    intercept
                    + slope
                    * int(
                        quarters[position].get(
                            "calendar_quarter_index", position
                        )
                    )
                )
                * (1.0 + config.touch_tolerance_pct / 100.0)
            ]
            if len(violations) > config.max_violations:
                reasons.append("excessive_close_violations")
            membership = tuple(sorted(str(point["id"]) for point in contacts))
            hypothesis_id = _stable_id("lid", membership)
            if reasons:
                rejected.append(
                    {
                        "id": hypothesis_id,
                        "contact_ids": list(membership),
                        "rejection_reasons": reasons,
                    }
                )
                continue
            fit_error = _fit_error(contacts, slope, intercept)
            strict_count = sum(bool(point["strict_major"]) for point in contacts)
            item = {
                "id": hypothesis_id,
                "rank": None,
                "contact_ids": list(membership),
                "contacts": contacts,
                "contact_count": len(contacts),
                "strict_major_count": strict_count,
                "slope_per_quarter": round(slope, 8),
                "intercept": round(intercept, 8),
                "slope_pct_per_year": round(slope_pct, 4),
                "slope_grade": _slope_grade(slope_pct),
                "fit_error_pct": round(fit_error, 4),
                "projected_lid": round(projection, 4),
                "span_quarters": max(
                    _point_coordinate(point) for point in contacts
                )
                - min(_point_coordinate(point) for point in contacts),
                "latest_contact_index": max(
                    _point_coordinate(point) for point in contacts
                ),
                "violation_indexes": violations,
                "rejection_reasons": [],
            }
            prior = accepted_by_membership.get(membership)
            if prior is None or item["fit_error_pct"] < prior["fit_error_pct"]:
                accepted_by_membership[membership] = item

    grade_order = {"A": 0, "B": 1, "C": 2, None: 3}
    accepted = sorted(
        accepted_by_membership.values(),
        key=lambda item: (
            -int(item["contact_count"]),
            -int(item["strict_major_count"]),
            float(item["fit_error_pct"]),
            grade_order[item["slope_grade"]],
            -int(item["span_quarters"]),
            -int(item["latest_contact_index"]),
            str(item["id"]),
        ),
    )
    for rank, item in enumerate(accepted, start=1):
        item["rank"] = rank
    rejected.sort(key=lambda item: item["id"])
    return accepted, rejected


def _log_r2(values: list[float]) -> float | None:
    if len(values) < 3 or any(value <= 0 for value in values):
        return None
    ys = [math.log(value) for value in values]
    xs = list(range(len(ys)))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - mean_y) ** 2 for y in ys)
    return None if total <= 0 else 1.0 - residual / total


def _equivalent_leaders(
    hypotheses: list[dict[str, Any]], config: ValidationConfig
) -> tuple[list[dict[str, Any]], bool]:
    if not hypotheses:
        return [], False
    best = hypotheses[0]
    same_support = [
        item
        for item in hypotheses
        if item["contact_count"] == best["contact_count"]
        and item["strict_major_count"] == best["strict_major_count"]
        and abs(item["fit_error_pct"] - best["fit_error_pct"])
        <= config.equivalent_fit_error_delta_pct
    ]
    equivalent = [
        item
        for item in same_support
        if abs(item["projected_lid"] - best["projected_lid"])
        / max(item["projected_lid"], best["projected_lid"])
        * 100.0
        <= config.equivalent_projection_delta_pct
    ]
    conflict = len(equivalent) != len(same_support)
    return equivalent, conflict


def _readiness_signals(
    quarters: list[dict[str, Any]], primary: dict[str, Any], centre: float
) -> dict[str, Any]:
    contacts = primary["contacts"]
    lows: list[float] = []
    for left, right in zip(contacts, contacts[1:]):
        start = int(left["quarter_index"])
        end = int(right["quarter_index"])
        if end - start > 1:
            lows.append(min(float(item["low"]) for item in quarters[start + 1 : end]))
    depths = [
        max(0.0, (centre - value) / centre * 100.0) for value in lows if centre > 0
    ]
    contraction = len(depths) >= 2 and depths[-1] < depths[0]
    rising_lows = len(lows) >= 2 and lows[-1] > lows[0]
    last_close = float(quarters[-1]["close"])
    proximity = last_close / centre * 100.0 if centre > 0 else 0.0
    repeated = int(primary["contact_count"]) >= 3
    near_count = 0
    for idx in range(max(0, len(quarters) - 8), len(quarters)):
        coordinate = int(quarters[idx].get("calendar_quarter_index", idx))
        projected = float(primary["intercept"]) + float(
            primary["slope_per_quarter"]
        ) * coordinate
        if projected > 0 and float(quarters[idx]["close"]) / projected >= 0.9:
            near_count += 1
    time_near = near_count >= 2
    volumes = [item.get("volume") for item in quarters]
    known_volume = [float(value) for value in volumes if value is not None]
    relative_volume = None
    if len(known_volume) >= 8:
        recent = sum(known_volume[-4:]) / 4.0
        prior = sum(known_volume[-8:-4]) / 4.0
        relative_volume = recent / prior if prior > 0 else None
    volume_contraction = relative_volume is not None and relative_volume < 1.0
    return {
        "pullback_depth_contraction": contraction,
        "rising_structural_lows": rising_lows,
        "price_pressing_lid": proximity >= 90.0,
        "repeated_approaches": repeated,
        "time_near_lid": time_near,
        "completed_period_relative_volume": (
            round(relative_volume, 4) if relative_volume is not None else None
        ),
        "completed_period_volume_contraction": volume_contraction,
        "pullback_depths_pct": [round(value, 3) for value in depths],
        "proximity_pct": round(proximity, 3),
        "independent_signal_count": sum(
            [
                contraction,
                rising_lows,
                repeated,
                time_near,
                volume_contraction,
            ]
        ),
    }


def _validation_state(
    quarters: list[dict[str, Any]],
    partial_quarter: dict[str, Any] | None,
    primary: dict[str, Any],
    band: dict[str, Any],
    signals: dict[str, Any],
    structure_qualified: bool,
    leaders: list[dict[str, Any]] | None = None,
) -> str:
    active_leaders = leaders or [primary]
    tolerance = float(band["touch_tolerance_pct"]) / 100.0

    def dynamic_band(idx: int) -> tuple[float, float, float]:
        projections = [
            float(item["intercept"])
            + float(item["slope_per_quarter"]) * idx
            for item in active_leaders
        ]
        return (
            min(projections) * (1.0 - tolerance),
            sum(projections) / len(projections),
            max(projections) * (1.0 + tolerance),
        )

    evidence_ready_idx = max(
        int(item["confirmed_quarter_index"])
        for leader in active_leaders
        for item in leader["contacts"]
        if item.get("confirmed_quarter_index") is not None
    )
    start_idx = evidence_ready_idx + 1
    consecutive_escape = 0
    terminal_after_escape: str | None = None
    confirmed_breakout = False
    for idx in range(start_idx, len(quarters)):
        coordinate = int(quarters[idx].get("calendar_quarter_index", idx))
        lower, centre, upper = dynamic_band(coordinate)
        close = float(quarters[idx]["close"])
        if close > upper:
            consecutive_escape += 1
            terminal_after_escape = "breaking_out"
            if consecutive_escape >= 2:
                confirmed_breakout = True
        elif consecutive_escape:
            terminal_after_escape = (
                "retest" if lower <= close <= upper else "failed_breakout"
            )
            consecutive_escape = 0
    if confirmed_breakout:
        return "post_breakout"
    if consecutive_escape == 1:
        return "breaking_out"
    if terminal_after_escape in {"retest", "failed_breakout"}:
        return terminal_after_escape
    # A live partial quarter can be provisional only when no completed-price
    # sequence already established a stronger lifecycle state.
    if partial_quarter is not None:
        partial_idx = int(
            partial_quarter.get("calendar_quarter_index", len(quarters))
        )
        _, _, partial_upper = dynamic_band(partial_idx)
        last_completed_idx = int(
            quarters[-1].get("calendar_quarter_index", len(quarters) - 1)
        )
        if (
            partial_idx > last_completed_idx
            and float(partial_quarter["close"]) > partial_upper
        ):
            return "breakout_provisional"
    proximity = float(signals["proximity_pct"])
    if (
        structure_qualified
        and 90.0 <= proximity <= 100.0
        and int(signals["independent_signal_count"]) >= 1
    ):
        return "pre_breakout"
    return "forming"


def _empty_result(
    bars: list[dict[str, Any]],
    data_quality: dict[str, Any],
    config: ValidationConfig,
    *,
    status: str,
    structure_state: str,
    failed_rules: list[str],
    abstained: bool,
) -> dict[str, Any]:
    last_date = bars[-1]["date"] if bars else None
    lifecycle = "no_structure" if structure_state != "watch_immature" else "forming"
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "source": SOURCE,
        "as_of": last_date,
        "bar_count": len(bars),
        "analysis_metadata": {
            "history_start": bars[0]["date"] if bars else None,
            "history_end": last_date,
            "bar_count_monthly": len(bars),
            "analysis_interval": "3M",
            "algorithm_version": ALGORITHM_VERSION,
            "variant": VARIANT,
            "mode": MODE,
            "config_fingerprint": config_fingerprint(config),
            "evidence_cutoff": data_quality.get("evidence_cutoff"),
            "adjustment_mode": data_quality.get("adjustment_mode"),
            "data_quality": data_quality,
            "classification_blocked": data_quality.get("blocked", False),
            "data_freshness": {"last_bar_date": last_date},
        },
        "lifecycle": lifecycle,
        "status": status,
        "grade": None,
        "coil_score": 0.0,
        "points": [],
        "major_highs": [],
        "active_lid": None,
        "breakout": None,
        "review": {"reviewed": False, "effective": "algorithm", "analysis_mode": MODE},
        "resistance": None,
        "support": None,
        "metrics": None,
        "diagnostics": {"rejected_hypotheses": []},
        "notes": list(failed_rules),
        "top_candidates": [],
        "lid_hypotheses": [],
        "resistance_band": None,
        "structure_validity": structure_state,
        "readiness": "no_pattern" if structure_state == "no_structure" else structure_state,
        "confidence": "low" if abstained else "medium",
        "abstained": abstained,
        "pattern_assessment": {
            "structure_state": structure_state,
            "readiness": "no_pattern" if structure_state == "no_structure" else structure_state,
            "confidence": "low" if abstained else "medium",
            "abstained": abstained,
            "passed_rules": [],
            "failed_rules": list(failed_rules),
            "warnings": [item["code"] for item in data_quality.get("issues", []) if item["severity"] == "warning"],
        },
    }


def _enforce_adjustment_gate(result: dict[str, Any]) -> dict[str, Any]:
    """Keep diagnostic evidence visible but never classify an unverified basis."""
    metadata = result.get("analysis_metadata") or {}
    if metadata.get("adjustment_mode") == "split_adjusted":
        return result
    metadata["classification_blocked"] = True
    assessment = result.get("pattern_assessment") or {}
    failed = list(assessment.get("failed_rules") or [])
    if "verified_split_adjustment" not in failed:
        failed.append("verified_split_adjustment")
    assessment.update(
        {
            "structure_state": "invalid_data",
            "readiness": "invalid_data",
            "confidence": "low",
            "abstained": True,
            "failed_rules": failed,
        }
    )
    result.update(
        {
            "structure_validity": "invalid_data",
            "readiness": "invalid_data",
            "confidence": "low",
            "abstained": True,
            "lifecycle": "no_structure",
            "status": "invalid_data",
            "grade": None,
            "points": [],
            "major_highs": [],
            "active_lid": None,
            "resistance": None,
            "support": None,
            "breakout": None,
        }
    )
    result["notes"] = list(dict.fromkeys([*result.get("notes", []), *failed]))
    return result


def analyze_coil_v24(
    bars: Iterable[dict[str, Any]],
    *,
    ticker: str = "UNKNOWN",
    as_of: str | None = None,
    adjustment_mode: str = ADJUSTMENT_UNKNOWN,
    config: ValidationConfig = DEFAULT_CONFIG,
    mode: str = MODE,
) -> dict[str, Any]:
    """Run the deterministic validation detector over immutable monthly bars."""
    if mode != MODE:
        raise ValueError("v2_4_validation is algorithm-only")
    inspected = inspect_monthly_bars(
        list(bars), as_of=as_of, adjustment_mode=adjustment_mode
    )
    clean = inspected.bars
    quality = inspected.report
    if quality["status"] == DATA_QUALITY_BLOCKED:
        return _scope_output_ids(_empty_result(
            clean,
            quality,
            config,
            status="invalid_data",
            structure_state="invalid_data",
            failed_rules=["valid_data"],
            abstained=True,
        ), ticker=ticker, as_of=as_of, config=config)
    all_quarters = _aggregate_quarters(clean)
    completed = [
        item for item in all_quarters if _quarter_complete(item, as_of=as_of)
    ]
    partial_quarter = _trailing_partial_quarter(all_quarters, as_of=as_of)
    if len(completed) < 3:
        return _scope_output_ids(
            _enforce_adjustment_gate(
                _empty_result(
                    clean,
                    quality,
                    config,
                    status="no_structure",
                    structure_state="no_structure",
                    failed_rules=["sufficient_completed_quarters"],
                    abstained=False,
                )
            ),
            ticker=ticker,
            as_of=as_of,
            config=config,
        )

    top_candidates, eligible = _top_candidates(completed, config)
    hypotheses, rejected = _hypotheses(completed, eligible, config)
    if not hypotheses:
        result = _empty_result(
            clean,
            quality,
            config,
            status="no_structure",
            structure_state="no_structure",
            failed_rules=["eligible_resistance_hypothesis"],
            abstained=False,
        )
        result["top_candidates"] = top_candidates
        result["diagnostics"]["rejected_hypotheses"] = rejected
        return _scope_output_ids(
            _enforce_adjustment_gate(result),
            ticker=ticker,
            as_of=as_of,
            config=config,
        )

    equivalent, conflict = _equivalent_leaders(hypotheses, config)
    if conflict:
        result = _empty_result(
            clean,
            quality,
            config,
            status="no_structure",
            structure_state="uncertain_structure",
            failed_rules=["no_unresolved_competing_lid"],
            abstained=True,
        )
        result["top_candidates"] = top_candidates
        result["lid_hypotheses"] = hypotheses
        result["diagnostics"]["rejected_hypotheses"] = rejected
        result["readiness"] = "uncertain_structure"
        result["pattern_assessment"]["readiness"] = "uncertain_structure"
        return _scope_output_ids(
            _enforce_adjustment_gate(result),
            ticker=ticker,
            as_of=as_of,
            config=config,
        )

    primary = hypotheses[0]
    projected = [float(item["projected_lid"]) for item in equivalent]
    tolerance = config.touch_tolerance_pct / 100.0
    band = {
        "hypothesis_ids": [item["id"] for item in equivalent],
        "lower": round(min(projected) * (1.0 - tolerance), 4),
        "upper": round(max(projected) * (1.0 + tolerance), 4),
        "centre": round(sum(projected) / len(projected), 4),
        "touch_tolerance_pct": config.touch_tolerance_pct,
    }
    earliest = min(int(item["quarter_index"]) for item in primary["contacts"])
    structure_start = date.fromisoformat(
        str(completed[earliest]["peak_date"])[:10]
    )
    evidence_end = date.fromisoformat(_month_end(str(completed[-1]["date"])))
    structure_years = (evidence_end - structure_start).days / 365.2425
    base_closes = [float(item["close"]) for item in completed[earliest:]]
    trend_r2 = _log_r2(base_closes)
    passed = ["valid_data", "confirmed_structural_tops", "eligible_resistance_hypothesis"]
    failed: list[str] = []
    if structure_years >= config.min_structure_years:
        passed.append("minimum_ten_year_structure")
    else:
        failed.append("minimum_ten_year_structure")
    if primary["slope_grade"] is not None:
        passed.append("gradeable_lid_slope")
    else:
        failed.append("gradeable_lid_slope")
    if trend_r2 is None or trend_r2 < config.steady_trend_r2_veto:
        passed.append("steady_trend_veto_clear")
    else:
        failed.append("steady_trend_veto_clear")
    if primary["strict_major_count"] >= config.required_strict_major_count:
        passed.append("required_strict_major_evidence")
    else:
        failed.append("required_strict_major_evidence")
    passed.append("no_unresolved_competing_lid")
    structure_qualified = not failed
    if structure_qualified:
        structure_state = "qualified"
    elif "minimum_ten_year_structure" in failed:
        structure_state = "watch_immature"
    elif "required_strict_major_evidence" in failed:
        structure_state = "uncertain_structure"
    else:
        structure_state = "no_pattern"
    signals = _readiness_signals(completed, primary, float(band["centre"]))
    readiness = _validation_state(
        completed,
        partial_quarter,
        primary,
        band,
        signals,
        structure_qualified,
        equivalent,
    )
    if structure_state == "watch_immature":
        readiness = "watch_immature"
    elif structure_state == "uncertain_structure":
        readiness = "uncertain_structure"
    elif structure_state == "no_pattern":
        readiness = "no_pattern"
    confidence = (
        "medium"
        if structure_state in {"watch_immature", "uncertain_structure"}
        else "high"
    )
    abstained = structure_state in {"watch_immature", "uncertain_structure"}

    contacts = primary["contacts"]
    contact_points = [
        {
            "idx": item["peak_index"],
            "date": item["peak_date"],
            "peak_date": item["peak_date"],
            "confirmed_at_idx": item["confirmed_at_index"],
            "confirmed_at": item["confirmed_at"],
            "price": item["price"],
            "prominence_pct": item["wick_prominence_pct"],
            "role": "major_top" if item["strict_major"] else "structural_retest",
            "confirmed": True,
            "lid_member": True,
            "source": SOURCE,
            "evidence": {"top_candidate_id": item["id"]},
        }
        for item in contacts
    ]
    first = contact_points[0]
    last = contact_points[-1]
    lifecycle_map = {
        "no_pattern": "no_structure",
        "watch_immature": "forming",
        "uncertain_structure": "forming",
        "forming": "forming",
        "pre_breakout": "pre_breakout",
        "breakout_provisional": "breaking_out",
        "breaking_out": "breaking_out",
        "failed_breakout": "forming",
        "retest": "forming",
        "post_breakout": "post_breakout",
    }
    lifecycle = lifecycle_map[readiness]
    status_map = {
        "no_structure": "no_structure",
        "forming": "basing",
        "pre_breakout": "coiling",
        "breaking_out": "breaking_out",
        "post_breakout": "broken_out",
    }
    compatibility_allowed = structure_qualified
    warnings = [
        item["code"]
        for item in quality.get("issues", [])
        if item["severity"] == "warning"
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "source": SOURCE,
        "as_of": clean[-1]["date"],
        "bar_count": len(clean),
        "analysis_metadata": {
            "history_start": clean[0]["date"],
            "history_end": clean[-1]["date"],
            "bar_count_monthly": len(clean),
            "bar_count_quarterly": len(all_quarters),
            "completed_quarter_count": len(completed),
            "completed_evidence_cutoff": _month_end(str(completed[-1]["date"])),
            "analysis_interval": "3M",
            "algorithm_version": ALGORITHM_VERSION,
            "variant": VARIANT,
            "mode": MODE,
            "config_fingerprint": config_fingerprint(config),
            "evidence_cutoff": quality.get("evidence_cutoff"),
            "adjustment_mode": quality.get("adjustment_mode"),
            "data_quality": quality,
            "classification_blocked": False,
            "data_freshness": {
                "last_bar_date": clean[-1]["date"],
                "incomplete_last_quarter": partial_quarter is not None,
            },
        },
        "top_candidates": top_candidates,
        "lid_hypotheses": hypotheses,
        "resistance_band": band,
        "structure_validity": structure_state,
        "readiness": readiness,
        "confidence": confidence,
        "abstained": abstained,
        "pattern_assessment": {
            "structure_state": structure_state,
            "readiness": readiness,
            "confidence": confidence,
            "abstained": abstained,
            "passed_rules": passed,
            "failed_rules": failed,
            "warnings": warnings,
        },
        "lifecycle": lifecycle,
        "status": status_map[lifecycle],
        "grade": primary["slope_grade"] if compatibility_allowed else None,
        "coil_score": 0.0,
        "points": contact_points if compatibility_allowed else [],
        "major_highs": contact_points if compatibility_allowed else [],
        "active_lid": None,
        "breakout": {"state": readiness},
        "review": {"reviewed": False, "effective": "algorithm", "analysis_mode": MODE},
        "resistance": None,
        "support": None,
        "metrics": {
            **signals,
            "structure_years": round(structure_years, 3),
            "base_trend_r2": round(trend_r2, 4) if trend_r2 is not None else None,
            "current_price_position": (
                "below_lid_band"
                if signals["proximity_pct"] < 100.0 - config.touch_tolerance_pct
                else "above_lid_band"
                if signals["proximity_pct"] > 100.0 + config.touch_tolerance_pct
                else "within_lid_band"
            ),
        },
        "diagnostics": {"rejected_hypotheses": rejected},
        "notes": failed + warnings,
    }
    if compatibility_allowed:
        endpoint = lambda item: {
            "idx": item["idx"],
            "date": item["date"],
            "price": item["price"],
        }
        result["resistance"] = {
            "from": endpoint(first),
            "to": endpoint(last),
            "slope_per_bar": round(float(primary["slope_per_quarter"]) / 3.0, 8),
            "slope_pct_per_year": primary["slope_pct_per_year"],
            "value_at_last_bar": primary["projected_lid"],
            "touch_count": primary["contact_count"],
            "touches": [endpoint(item) for item in reversed(contact_points)],
            "span_years": round(primary["span_quarters"] / 4.0, 3),
            "wick_overshoots": len(primary["violation_indexes"]),
            "fit_score": 0.0,
            "lid_grade": primary["slope_grade"],
            "source": SOURCE,
        }
        result["active_lid"] = {
            "from": endpoint(first),
            "to": endpoint(last),
            "anchors": [endpoint(item) for item in contact_points],
            "slope_per_bar": round(float(primary["slope_per_quarter"]) / 3.0, 8),
            "slope_pct_per_year": primary["slope_pct_per_year"],
            "grade": primary["slope_grade"],
            "tolerance_band_pct": config.touch_tolerance_pct,
            "fit_error_pct": primary["fit_error_pct"],
            "value_at_last_bar": primary["projected_lid"],
            "touches": [endpoint(item) for item in contact_points],
            "touch_count": primary["contact_count"],
            "span_years": round(primary["span_quarters"] / 4.0, 3),
            "source": SOURCE,
        }
    return _scope_output_ids(
        _enforce_adjustment_gate(result),
        ticker=ticker,
        as_of=as_of,
        config=config,
    )
