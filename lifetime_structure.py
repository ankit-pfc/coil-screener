"""Experimental observed-lifetime top and resistance-family detector.

This module is intentionally separate from :func:`coil_analysis.analyze_coil`.
It turns price history into explicit quarterly top episodes, starts from the
highest confirmed episode, demotes unsupported singleton extremes, and keeps
an outer line plus credible lower price families.

Human review geometry is never accepted as input here.  It belongs only in a
comparison layer so detector output cannot be seeded by the expected answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date as calendar_date
import math
import statistics
from typing import Any, Iterable, Optional

from coil_analysis import (
    _aggregate_quarterly_display_bars,
    _clean_bars,
    _completed_quarters,
)


ALGORITHM_VERSION = "lifetime-reference-exp-0.1"
SCHEMA_VERSION = 1
SOURCE = "timeseries"


@dataclass(frozen=True)
class LifetimeReferenceConfig:
    """Exploratory values, exposed so review can change them explicitly."""

    local_high_window_quarters: int = 4
    reaction_window_quarters: int = 8
    minimum_reaction_quarters: int = 2
    min_rejection_pct: float = 15.0
    plateau_tolerance_pct: float = 1.0
    min_episode_separation_quarters: int = 4
    min_anchor_span_quarters: int = 12
    descending_continuity_floor: float = 0.70
    rising_continuity_floor: float = 0.50
    min_slope_pct_per_year: float = -6.0
    max_slope_pct_per_year: float = 8.0
    travelling_band_lower_pct: float = 14.0
    travelling_band_upper_pct: float = 6.0
    containment_buffer_pct: float = 8.0
    min_containment_ratio: float = 0.88
    confirmed_touch_count: int = 3
    two_anchor_min_span_years: float = 10.0
    new_reference_maturity_quarters: int = 4
    child_zone_lower_pct: float = 20.0
    child_zone_upper_pct: float = 5.0
    child_zone_min_below_parent_pct: float = 8.0
    child_zone_two_touch_span_years: float = 6.0
    child_zone_three_touch_span_years: float = 3.0
    max_structures: int = 3


DEFAULT_LIFETIME_CONFIG = LifetimeReferenceConfig()


@dataclass(frozen=True)
class _Episode:
    id: str
    quarter_idx: int
    time_ordinal: int
    source_month_idx: int
    date: str
    quarter_date: str
    price: float
    status: str
    confirmed_at: Optional[str]
    trough_date: Optional[str]
    trough_price: Optional[float]
    wick_drawdown_pct: float
    close_drawdown_pct: float


@dataclass(frozen=True)
class _LineFit:
    seed_id: str
    anchor_a: _Episode
    anchor_b: _Episode
    slope_per_quarter: float
    slope_pct_per_year: float
    touch_ids: tuple[str, ...]
    later_touch_count: int
    later_overshoot_count: int
    rms_error_pct: float
    containment_ratio: float
    span_years: float
    score: float

    def value_at(self, time_ordinal: int) -> float:
        return self.anchor_a.price + self.slope_per_quarter * (
            time_ordinal - self.anchor_a.time_ordinal
        )


def _quarter_ordinal(value: str) -> int:
    year = int(str(value)[:4])
    month = int(str(value)[5:7])
    return year * 4 + (month - 1) // 3


def _round(value: Optional[float], places: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), places)


def _source_date(
    quarter: dict[str, Any], monthly: list[dict[str, Any]]
) -> tuple[int, str]:
    source_idx = int(quarter.get("_high_source_idx", quarter.get("_close_source_idx", 0)))
    if 0 <= source_idx < len(monthly):
        return source_idx, str(monthly[source_idx]["date"])
    return source_idx, str(quarter["date"])


def _reaction(
    quarterly: list[dict[str, Any]],
    idx: int,
    config: LifetimeReferenceConfig,
) -> Optional[dict[str, Any]]:
    peak = float(quarterly[idx]["high"])
    future = quarterly[
        idx + 1 : min(len(quarterly), idx + config.reaction_window_quarters + 1)
    ]
    if len(future) < config.minimum_reaction_quarters:
        return None

    trough = min(future, key=lambda bar: float(bar["low"]))
    wick_drawdown_pct = (peak - float(trough["low"])) / peak * 100.0
    lowest_close = min(future, key=lambda bar: float(bar["close"]))
    close_drawdown_pct = (peak - float(lowest_close["close"])) / peak * 100.0
    # A deep wick by itself is not enough.  The market must also carry a
    # completed quarterly body/close materially away from the high.
    if min(wick_drawdown_pct, close_drawdown_pct) < config.min_rejection_pct:
        return None

    threshold = peak * (1.0 - config.min_rejection_pct / 100.0)
    confirmed = next(
        (
            bar
            for bar in future
            if float(bar["close"]) <= threshold
        ),
        trough,
    )
    return {
        "confirmed_at": str(confirmed["date"]),
        "trough_date": str(trough["date"]),
        "trough_price": float(trough["low"]),
        "wick_drawdown_pct": wick_drawdown_pct,
        "close_drawdown_pct": close_drawdown_pct,
    }


def _confirmed_peak_candidates(
    quarterly: list[dict[str, Any]],
    monthly: list[dict[str, Any]],
    config: LifetimeReferenceConfig,
) -> list[_Episode]:
    highs = [float(bar["high"]) for bar in quarterly]
    radius = max(1, config.local_high_window_quarters)
    plateau = config.plateau_tolerance_pct / 100.0
    candidates: list[_Episode] = []

    for idx, high in enumerate(highs):
        left = highs[max(0, idx - radius) : idx]
        right = highs[idx + 1 : min(len(highs), idx + radius + 1)]
        # A near-equal earlier quarter owns a plateau.  A materially higher
        # later quarter means this candle was only a shoulder.
        if left and max(left) >= high * (1.0 - plateau):
            continue
        if right and max(right) > high * (1.0 + plateau):
            continue
        reaction = _reaction(quarterly, idx, config)
        if reaction is None:
            continue
        source_idx, source_date = _source_date(quarterly[idx], monthly)
        quarter_date = str(quarterly[idx]["date"])
        candidates.append(
            _Episode(
                id=f"top-{quarter_date[:7]}",
                quarter_idx=idx,
                time_ordinal=_quarter_ordinal(quarter_date),
                source_month_idx=source_idx,
                date=source_date,
                quarter_date=quarter_date,
                price=high,
                status="confirmed_rejection",
                confirmed_at=str(reaction["confirmed_at"]),
                trough_date=str(reaction["trough_date"]),
                trough_price=float(reaction["trough_price"]),
                wick_drawdown_pct=float(reaction["wick_drawdown_pct"]),
                close_drawdown_pct=float(reaction["close_drawdown_pct"]),
            )
        )

    # Independent episodes are chosen price-first inside the separation
    # window, then returned chronologically.  Adjacent quarterly shoulders do
    # not become multiple tops.
    kept: list[_Episode] = []
    for candidate in sorted(candidates, key=lambda item: (-item.price, item.time_ordinal)):
        if all(
            abs(candidate.time_ordinal - other.time_ordinal)
            >= config.min_episode_separation_quarters
            for other in kept
        ):
            kept.append(candidate)
    return sorted(kept, key=lambda item: item.time_ordinal)


def _tracking_lifetime_high(
    full_quarters: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    monthly: list[dict[str, Any]],
    confirmed: list[_Episode],
) -> tuple[dict[str, Any], Optional[_Episode]]:
    lifetime_idx, lifetime_quarter = max(
        enumerate(full_quarters), key=lambda item: float(item[1]["high"])
    )
    quarter_date = str(lifetime_quarter["date"])
    source_idx, source_date = _source_date(lifetime_quarter, monthly)
    complete = lifetime_idx < len(completed)
    matching = next(
        (
            episode
            for episode in confirmed
            if episode.time_ordinal == _quarter_ordinal(quarter_date)
        ),
        None,
    )
    observation = {
        "date": source_date,
        "quarter_date": quarter_date,
        "price": _round(float(lifetime_quarter["high"])),
        "quarter_complete": complete,
        "top_status": matching.status if matching else "tracking_only",
    }
    if matching is not None:
        return observation, None
    return observation, _Episode(
        id=f"top-{quarter_date[:7]}-tracking",
        quarter_idx=lifetime_idx,
        time_ordinal=_quarter_ordinal(quarter_date),
        source_month_idx=source_idx,
        date=source_date,
        quarter_date=quarter_date,
        price=float(lifetime_quarter["high"]),
        status="tracking_only",
        confirmed_at=None,
        trough_date=None,
        trough_price=None,
        wick_drawdown_pct=0.0,
        close_drawdown_pct=0.0,
    )


def _line_value(a: _Episode, b: _Episode, time_ordinal: int) -> float:
    slope = (b.price - a.price) / (b.time_ordinal - a.time_ordinal)
    return a.price + slope * (time_ordinal - a.time_ordinal)


def _fit_line_candidates(
    seed: _Episode,
    episodes: list[_Episode],
    quarterly: list[dict[str, Any]],
    config: LifetimeReferenceConfig,
) -> list[_LineFit]:
    candidates: list[_LineFit] = []
    lower_band = config.travelling_band_lower_pct / 100.0
    upper_band = config.travelling_band_upper_pct / 100.0

    ordered_episodes = sorted(episodes, key=lambda item: item.time_ordinal)
    last_completed_ordinal = _quarter_ordinal(str(quarterly[-1]["date"]))
    for anchor_a_idx, anchor_a in enumerate(ordered_episodes):
        for anchor_b in ordered_episodes[anchor_a_idx + 1 :]:
            span_quarters = anchor_b.time_ordinal - anchor_a.time_ordinal
            if span_quarters < config.min_anchor_span_quarters:
                continue
            price_ratio = anchor_b.price / anchor_a.price
            if price_ratio < 1.0:
                if price_ratio < config.descending_continuity_floor:
                    continue
            elif 1.0 / price_ratio < config.rising_continuity_floor:
                continue

            annualized = (
                math.expm1(math.log(price_ratio) * 4.0 / span_quarters) * 100.0
            )
            if not (
                config.min_slope_pct_per_year
                <= annualized
                <= config.max_slope_pct_per_year
            ):
                continue

            slope = (anchor_b.price - anchor_a.price) / span_quarters
            touches: list[_Episode] = []
            errors: list[float] = []
            for episode in episodes:
                if episode.time_ordinal < anchor_a.time_ordinal:
                    continue
                projected = anchor_a.price + slope * (
                    episode.time_ordinal - anchor_a.time_ordinal
                )
                if projected <= 0:
                    continue
                error = (episode.price - projected) / projected
                if -lower_band <= error <= upper_band:
                    touches.append(episode)
                    errors.append(error * 100.0)
            if seed.id not in {episode.id for episode in touches}:
                continue

            later_touches = [
                episode
                for episode in touches
                if episode.time_ordinal > anchor_b.time_ordinal
            ]
            later_overshoots = []
            for episode in episodes:
                if episode.time_ordinal <= anchor_b.time_ordinal:
                    continue
                if (
                    last_completed_ordinal - episode.time_ordinal
                    < config.new_reference_maturity_quarters
                ):
                    # A newly confirmed high starts the next reference regime;
                    # it does not retrospectively erase the older family.
                    continue
                projected = anchor_a.price + slope * (
                    episode.time_ordinal - anchor_a.time_ordinal
                )
                if projected <= 0 or episode.price > projected * (1.0 + upper_band):
                    later_overshoots.append(episode)

            interval_bars = [
                bar
                for bar in quarterly
                if anchor_a.time_ordinal
                <= _quarter_ordinal(str(bar["date"]))
                <= anchor_b.time_ordinal
            ]
            violations = 0
            for bar in interval_bars:
                ordinal = _quarter_ordinal(str(bar["date"]))
                projected = anchor_a.price + slope * (
                    ordinal - anchor_a.time_ordinal
                )
                if projected > 0 and float(bar["close"]) > projected * (
                    1.0 + config.containment_buffer_pct / 100.0
                ):
                    violations += 1
            containment = 1.0 - violations / max(1, len(interval_bars))
            span_years = span_quarters / 4.0
            credible = (
                len(touches) >= config.confirmed_touch_count
                or span_years >= config.two_anchor_min_span_years
            ) and containment >= config.min_containment_ratio
            # A short early pair that is materially exceeded by later confirmed
            # tops is a historical/broken inner line, not the active lifetime
            # reference.  A decade-plus pair may remain as a genuine released
            # boundary (for example, a later successful breakout).
            if later_overshoots and span_years < config.two_anchor_min_span_years:
                credible = False
            if not credible:
                continue

            rms_error = math.sqrt(
                sum(error * error for error in errors) / max(1, len(errors))
            )
            mean_rejection = statistics.mean(
                max(item.wick_drawdown_pct, item.close_drawdown_pct)
                for item in (anchor_a, anchor_b)
            )
            score = (
                35.0 * min(len(touches), 5) / 5.0
                + 20.0 * min(span_years, 20.0) / 20.0
                + 20.0 * containment
                + 15.0 * max(0.0, 1.0 - rms_error / 14.0)
                + 10.0 * min(mean_rejection, 50.0) / 50.0
            )
            candidates.append(
                _LineFit(
                    seed_id=seed.id,
                    anchor_a=anchor_a,
                    anchor_b=anchor_b,
                    slope_per_quarter=slope,
                    slope_pct_per_year=annualized,
                    touch_ids=tuple(item.id for item in touches),
                    later_touch_count=len(later_touches),
                    later_overshoot_count=len(later_overshoots),
                    rms_error_pct=rms_error,
                    containment_ratio=containment,
                    span_years=span_years,
                    score=score,
                )
            )

    # Earliest credible establishment wins.  Later points can confirm an
    # already drawn line but cannot rotate it simply because they are recent.
    return sorted(
        candidates,
        key=lambda item: (
            -int(item.later_touch_count > 0),
            item.anchor_b.time_ordinal,
            -len(item.touch_ids),
            -item.score,
            item.anchor_a.time_ordinal,
        ),
    )


def _line_structure(
    fit: _LineFit,
    last_ordinal: int,
    config: LifetimeReferenceConfig,
) -> dict[str, Any]:
    structure_id = f"line-{fit.anchor_a.id}-{fit.anchor_b.id}"
    value_at_last = fit.value_at(last_ordinal)
    supporting = [
        item
        for item in fit.touch_ids
        if item not in {fit.anchor_a.id, fit.anchor_b.id}
    ]
    return {
        "id": structure_id,
        "parent_id": None,
        "relationship": "outer_reference",
        "selection": "active",
        "kind": "line",
        "status": (
            "confirmed"
            if len(fit.touch_ids) >= config.confirmed_touch_count
            else "established_low_confidence"
        ),
        "confidence": _round(min(1.0, fit.score / 100.0), 3),
        "construction_anchor_ids": [fit.anchor_a.id, fit.anchor_b.id],
        "supporting_touch_ids": supporting,
        "excluded_top_ids": [],
        "line": {
            "from": {
                "date": fit.anchor_a.date,
                "time_ordinal": fit.anchor_a.time_ordinal,
                "price": _round(fit.anchor_a.price),
            },
            "to": {
                "date": fit.anchor_b.date,
                "time_ordinal": fit.anchor_b.time_ordinal,
                "price": _round(fit.anchor_b.price),
            },
            "slope_per_quarter": _round(fit.slope_per_quarter, 6),
            "slope_pct_per_year": _round(fit.slope_pct_per_year, 3),
            "projected": {
                "time_ordinal": last_ordinal,
                "price": _round(value_at_last),
            },
        },
        "band": {
            "model": "asymmetric_travelling_band",
            "lower_pct": config.travelling_band_lower_pct,
            "upper_pct": config.travelling_band_upper_pct,
            "values_at_last": {
                "lower": _round(
                    value_at_last
                    * (1.0 - config.travelling_band_lower_pct / 100.0)
                ),
                "upper": _round(
                    value_at_last
                    * (1.0 + config.travelling_band_upper_pct / 100.0)
                ),
            },
        },
        "fit": {
            "rms_error_pct": _round(fit.rms_error_pct, 3),
            "touch_count": len(fit.touch_ids),
            "later_touch_count": fit.later_touch_count,
            "later_overshoot_count": fit.later_overshoot_count,
            "span_years": _round(fit.span_years, 2),
            "containment_ratio": _round(fit.containment_ratio, 3),
            "score": _round(fit.score, 2),
        },
        "reason_codes": [
            "highest_supported_reference_family",
            "calendar_quarter_geometry",
            "later_tops_do_not_rotate_anchors",
        ],
        "source": SOURCE,
    }


def _child_zone_structures(
    episodes: list[_Episode],
    fit: _LineFit,
    parent_id: str,
    last_ordinal: int,
    config: LifetimeReferenceConfig,
) -> list[dict[str, Any]]:
    lower = config.child_zone_lower_pct / 100.0
    upper = config.child_zone_upper_pct / 100.0
    anchor_ids = {fit.anchor_a.id, fit.anchor_b.id}
    eligible: list[_Episode] = []
    for episode in episodes:
        if episode.id in anchor_ids or episode.time_ordinal < fit.anchor_a.time_ordinal:
            continue
        parent_value = fit.value_at(episode.time_ordinal)
        if parent_value <= 0 or episode.price > parent_value * (1.0 + upper):
            continue
        eligible.append(episode)

    zone_candidates: list[tuple[float, list[_Episode], float, float]] = []
    for seed in sorted(eligible, key=lambda item: (-item.price, item.time_ordinal)):
        members = [
            episode
            for episode in eligible
            if seed.price * (1.0 - lower)
            <= episode.price
            <= seed.price * (1.0 + upper)
        ]
        members = sorted(members, key=lambda item: item.time_ordinal)
        if len(members) < 2:
            continue
        span_years = (members[-1].time_ordinal - members[0].time_ordinal) / 4.0
        minimum_span = (
            config.child_zone_three_touch_span_years
            if len(members) >= 3
            else config.child_zone_two_touch_span_years
        )
        if span_years < minimum_span:
            continue
        level = statistics.median(item.price for item in members)
        mean_parent_gap = statistics.mean(
            max(
                0.0,
                (fit.value_at(item.time_ordinal) - item.price)
                / fit.value_at(item.time_ordinal)
                * 100.0,
            )
            for item in members
        )
        if mean_parent_gap < config.child_zone_min_below_parent_pct:
            continue
        rms = math.sqrt(
            statistics.mean(((item.price - level) / level * 100.0) ** 2 for item in members)
        )
        score = (
            45.0 * min(len(members), 4) / 4.0
            + 25.0 * min(span_years, 15.0) / 15.0
            + 20.0 * max(0.0, 1.0 - rms / config.child_zone_lower_pct)
            + 10.0 * min(mean_parent_gap, 30.0) / 30.0
        )
        zone_candidates.append((score, members, level, rms))

    structures: list[dict[str, Any]] = []
    used_signatures: list[frozenset[str]] = []
    for score, members, level, rms in sorted(
        zone_candidates,
        key=lambda item: (-item[0], -len(item[1]), item[1][0].time_ordinal),
    ):
        signature = frozenset(item.id for item in members)
        if any(
            signature <= existing
            or existing <= signature
            or len(signature & existing) / max(1, len(signature | existing)) >= 0.60
            for existing in used_signatures
        ):
            continue
        used_signatures.append(signature)
        structure_id = f"zone-{members[0].id}-{members[-1].id}"
        structures.append(
            {
                "id": structure_id,
                "parent_id": parent_id,
                "relationship": "nested_below_outer",
                "selection": "retained",
                "kind": "resistance_band",
                "status": "confirmed" if len(members) >= 3 else "established",
                "confidence": _round(min(1.0, score / 100.0), 3),
                "construction_anchor_ids": [members[0].id, members[-1].id],
                "supporting_touch_ids": [item.id for item in members[1:-1]],
                "excluded_top_ids": [],
                "line": {
                    "from": {
                        "date": members[0].date,
                        "time_ordinal": members[0].time_ordinal,
                        "price": _round(level),
                    },
                    "to": {
                        "date": members[-1].date,
                        "time_ordinal": last_ordinal,
                        "price": _round(level),
                    },
                    "slope_per_quarter": 0.0,
                    "slope_pct_per_year": 0.0,
                    "projected": {
                        "time_ordinal": last_ordinal,
                        "price": _round(level),
                    },
                },
                "band": {
                    "model": "secondary_price_zone",
                    "lower_pct": config.child_zone_lower_pct,
                    "upper_pct": config.child_zone_upper_pct,
                    "values_at_last": {
                        "lower": _round(level * (1.0 - lower)),
                        "upper": _round(level * (1.0 + upper)),
                    },
                },
                "fit": {
                    "rms_error_pct": _round(rms, 3),
                    "touch_count": len(members),
                    "span_years": _round(
                        (members[-1].time_ordinal - members[0].time_ordinal)
                        / 4.0,
                        2,
                    ),
                    "score": _round(score, 2),
                },
                "reason_codes": [
                    "repeated_lower_price_family",
                    "retained_beside_outer_reference",
                ],
                "source": SOURCE,
            }
        )
        if len(structures) >= max(0, config.max_structures - 1):
            break
    return structures


def _episode_dict(
    episode: _Episode,
    *,
    price_rank: int,
    roles: Iterable[str],
    family_ids: Iterable[str],
) -> dict[str, Any]:
    return {
        "id": episode.id,
        "quarter_idx": episode.quarter_idx,
        "time_ordinal": episode.time_ordinal,
        "source_month_idx": episode.source_month_idx,
        "date": episode.date,
        "quarter_date": episode.quarter_date,
        "price": _round(episode.price),
        "price_rank": price_rank,
        "status": episode.status,
        "roles": sorted(set(roles)),
        "reaction": {
            "confirmed_at": episode.confirmed_at,
            "trough": (
                None
                if episode.trough_date is None
                else {
                    "date": episode.trough_date,
                    "price": _round(episode.trough_price),
                }
            ),
            "wick_drawdown_pct": _round(episode.wick_drawdown_pct, 2),
            "close_drawdown_pct": _round(episode.close_drawdown_pct, 2),
        },
        "family_ids": sorted(set(family_ids)),
        "source": SOURCE,
    }


def analyze_lifetime_references(
    bars: Iterable[dict[str, Any]],
    *,
    as_of: Optional[str] = None,
    config: LifetimeReferenceConfig = DEFAULT_LIFETIME_CONFIG,
) -> dict[str, Any]:
    """Analyze price/time top families over the full observed input history."""

    monthly = _clean_bars(bars, as_of)
    if not monthly:
        raise ValueError("at least one valid price bar is required")
    full_quarters = _aggregate_quarterly_display_bars(monthly)
    completed = _completed_quarters(full_quarters)
    if len(completed) < config.local_high_window_quarters + 2:
        raise ValueError("insufficient completed quarterly history")

    confirmed = _confirmed_peak_candidates(completed, monthly, config)
    lifetime_high, tracking_episode = _tracking_lifetime_high(
        full_quarters, completed, monthly, confirmed
    )
    episodes = list(confirmed)
    if tracking_episode is not None:
        episodes.append(tracking_episode)
    episodes.sort(key=lambda item: item.time_ordinal)

    confirmed_by_price = sorted(
        confirmed, key=lambda item: (-item.price, item.time_ordinal)
    )
    last_completed_ordinal = _quarter_ordinal(str(completed[-1]["date"]))
    last_display_ordinal = _quarter_ordinal(str(full_quarters[-1]["date"]))
    demoted_ids: set[str] = set()
    tracking_new_ids: set[str] = set()
    selected_fit: Optional[_LineFit] = None

    for seed in confirmed_by_price:
        fits = _fit_line_candidates(seed, confirmed, completed, config)
        if fits:
            selected_fit = fits[0]
            break
        age = last_completed_ordinal - seed.time_ordinal
        if age < config.new_reference_maturity_quarters:
            tracking_new_ids.add(seed.id)
        else:
            demoted_ids.add(seed.id)

    structures: list[dict[str, Any]] = []
    if selected_fit is not None:
        primary = _line_structure(selected_fit, last_display_ordinal, config)
        structures.append(primary)
        structures.extend(
            _child_zone_structures(
                confirmed,
                selected_fit,
                primary["id"],
                last_display_ordinal,
                config,
            )
        )

    family_by_episode: dict[str, list[str]] = {item.id: [] for item in episodes}
    role_by_episode: dict[str, set[str]] = {item.id: set() for item in episodes}
    for structure in structures:
        for episode_id in (
            structure["construction_anchor_ids"] + structure["supporting_touch_ids"]
        ):
            family_by_episode.setdefault(episode_id, []).append(structure["id"])
            role_by_episode.setdefault(episode_id, set()).add(
                "reference_top"
                if episode_id in structure["construction_anchor_ids"]
                else "supporting_touch"
            )
    lifetime_ordinal = _quarter_ordinal(str(lifetime_high["quarter_date"]))
    for episode in episodes:
        if episode.time_ordinal == lifetime_ordinal:
            role_by_episode[episode.id].add("lifetime_high")
        if episode.id in demoted_ids:
            role_by_episode[episode.id].add("demoted_singleton")
        if episode.id in tracking_new_ids or episode.status == "tracking_only":
            role_by_episode[episode.id].add("tracking_new_high")

    ranked = sorted(episodes, key=lambda item: (-item.price, item.time_ordinal))
    rank_by_id = {item.id: rank for rank, item in enumerate(ranked, start=1)}
    episode_payloads = [
        _episode_dict(
            episode,
            price_rank=rank_by_id[episode.id],
            roles=role_by_episode.get(episode.id, set()),
            family_ids=family_by_episode.get(episode.id, []),
        )
        for episode in episodes
    ]

    active_seed_id = selected_fit.seed_id if selected_fit is not None else None
    reference_ladder = []
    for episode in ranked:
        if episode.status == "tracking_only" or episode.id in tracking_new_ids:
            status = "tracking_new_high"
            reasons = ["not_mature_enough_for_pairing"]
        elif episode.id == active_seed_id:
            status = "active_supported"
            reasons = ["credible_price_time_family"]
        elif episode.id in demoted_ids:
            status = "demoted_singleton"
            reasons = ["no_credible_family", "lower_family_has_better_coverage"]
        elif family_by_episode.get(episode.id):
            status = "supporting_family_member"
            reasons = ["belongs_to_retained_structure"]
        else:
            status = "unassigned_confirmed_top"
            reasons = ["confirmed_reaction_but_no_retained_family"]
        reference_ladder.append(
            {
                "top_id": episode.id,
                "price_rank": rank_by_id[episode.id],
                "reference_price": _round(episode.price),
                "status": status,
                "reason_codes": reasons,
            }
        )

    ordinals = [_quarter_ordinal(str(bar["date"])) for bar in completed]
    missing_ordinals = [
        ordinal
        for ordinal in range(ordinals[0], ordinals[-1] + 1)
        if ordinal not in set(ordinals)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "experimental": True,
        "source": SOURCE,
        "as_of": as_of,
        "interval": "3M",
        "history": {
            "scope": "observed_lifetime",
            "start_date": str(monthly[0]["date"]),
            "end_date": str(monthly[-1]["date"]),
            "completed_through": str(completed[-1]["date"]),
            "quarter_count": len(full_quarters),
            "completed_quarter_count": len(completed),
            "lifetime_high": lifetime_high,
            "provider": "caller_supplied",
            "adjustment_policy": "input_prices_as_provided",
            "completeness": "provider_max_unverified",
            "comparability": (
                "calendar_gaps_present" if missing_ordinals else "calendar_contiguous"
            ),
            "missing_quarter_ordinals": missing_ordinals,
        },
        "config": asdict(config),
        "top_episodes": episode_payloads,
        "reference_ladder": reference_ladder,
        "structures": structures[: config.max_structures],
        "diagnostics": {
            "confirmed_episode_count": len(confirmed),
            "tracking_episode_count": int(tracking_episode is not None),
            "demoted_singleton_ids": sorted(demoted_ids),
            "tracking_new_reference_ids": sorted(tracking_new_ids),
            "primary_structure_id": structures[0]["id"] if structures else None,
            "notes": [
                "Observed lifetime is limited to caller-provided history.",
                "Corporate-action comparability is not inferred by this detector.",
                "Market semantics still require human review in this experiment.",
            ],
        },
    }
