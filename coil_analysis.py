"""Deterministic major-top, resistance-slope, and coiling-structure analysis.

What "coiling" means here
-------------------------
A coiling setup is a long consolidation pressed under a flat-to-gently-rising
lid, before the breakout has played out:

1. Structure  - at least two major tops (high-prominence swing highs) line up
   along one straight resistance line within a small tolerance band.
2. Lid slope  - the line is flat or gently rising. Slope is normalized to
   percent-per-year of the line value at the last bar, and graded:
   A (flattest, best), B (fine), C (steep but acceptable). Steeper than C
   means the move is already trending, not coiling.
3. Sealed     - no monthly close escaped above the line (small tolerance)
   before the most recent bars; the breakout has not happened yet.
4. Winding    - successive pullbacks below the lid get shallower (higher lows
   / ascending compression), so price is being squeezed into the lid.
5. Loaded     - the last close sits near the lid, not mid-base.

The module walks the monthly series, finds swing highs by pivot windows,
ranks them by prominence, searches anchor pairs of major tops for the best
resistance line (most touches, longest span, flattest, still relevant from
the present looking back), then verifies the coil conditions and grades the
slope. Everything is pure Python over the cached bar dicts
(`{"date","open","high","low","close","volume"}`, monthly) so it runs
identically from the API, the CLI, tests, and backtest truncations (`as_of`).

Prominence note: classic (scipy-style) prominence walks until a strictly
higher bar, which collapses on flat coils where the tops are near-equal by
construction. Here the walk also stops at another pivot top within
``prominence_equal_tol_pct`` of the peak, so each top's prominence is its
drop to the valley separating it from its nearest same-height neighbor.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 2
ALGORITHM_VERSION = "2.0.0"
ANALYSIS_INTERVAL = "3M"  # classification is quarterly-native regardless of view
SOURCE = "timeseries"
BARS_PER_YEAR = 12.0  # module operates on monthly bars

# Point roles (schema v2). A point is exactly one of these:
ROLE_MAJOR_TOP = "major_top"  # confirmed two-sided mountain
ROLE_STRUCTURAL_RETEST = "structural_retest"  # weaker anchor needed to define the lid
ROLE_PROVISIONAL_TOP = "provisional_top"  # right-edge peak without future confirmation
ROLE_BREAKOUT_PEAK = "breakout_peak"  # post-breakout high excluded from the lid fit

# Lifecycle buckets (schema v2). ``no_structure`` is the pre-structural bucket.
LIFECYCLE_NO_STRUCTURE = "no_structure"
LIFECYCLE_FORMING = "forming"
LIFECYCLE_PRE_BREAKOUT = "pre_breakout"
LIFECYCLE_BREAKING_OUT = "breaking_out"
LIFECYCLE_POST_BREAKOUT = "post_breakout"


@dataclass(frozen=True)
class CoilConfig:
    """Tunables for monthly bars; window/span/separation values are in bars."""

    min_bars: int = 60
    # Swing-high pivots
    pivot_left: int = 3
    pivot_right: int = 3
    minor_prominence_pct: float = 5.0
    major_prominence_pct: float = 15.0
    major_min_separation: int = 9
    prominence_equal_tol_pct: float = 1.0
    # Chart-overlay majors use a stricter, human-review definition than the
    # broader resistance-search candidate pool above. A displayed top must
    # read like a mountain: a substantial rise into it and fall after it.
    # Nearby same-level pivots collapse into the first bar of one plateau, the
    # primary sequence never steps down, and only the latest three are shown.
    display_lookback_bars: int = 120
    display_pivot_equal_tol_pct: float = 1.0
    display_plateau_gap: int = 12
    display_plateau_tolerance_pct: float = 8.0
    # A later bar can replace the first plateau representative only when it is
    # truly adjacent. A more distant same-zone retest stays collapsed to the
    # original ceiling anchor even if its wick is modestly higher.
    display_plateau_upgrade_gap: int = 6
    display_major_prominence_pct: float = 28.0
    # A deep wick alone is not a meaningful fall. After the latest-three
    # selection, confirmed peaks must also show this much two-sided prominence
    # using closes (candle bodies), or they are shoulders on a continuing move.
    display_body_prominence_pct: float = 21.0
    display_fast_rejection_bars: int = 6
    display_comparable_retest_tolerance_pct: float = 5.0
    # Locally dramatic swings can still be visually minor when they occur near
    # the bottom of the ten-year chart. A displayed major must reach at least
    # this position within the active window's low-to-high price range.
    display_min_range_position_pct: float = 35.0
    display_major_min_separation: int = 9
    display_same_or_higher_tolerance_pct: float = 1.0
    # A plotted ceiling needs at least two anchors to define a slope. If the
    # strict mountain pass leaves only one, admit the strongest comparable
    # structural retest as a secondary line anchor (not a primary mountain).
    display_min_line_anchors: int = 2
    display_secondary_anchor_min_level_pct: float = 85.0
    display_secondary_anchor_min_prominence_pct: float = 14.0
    display_secondary_anchor_body_prominence_pct: float = 10.0
    # With two well-separated endpoints in hand, keep one strong intermediate
    # pivot that retests their line. This captures a three-touch flat lid
    # without admitting short, noisy shelves near either endpoint.
    display_intermediate_anchor_tolerance_pct: float = 7.2
    display_intermediate_anchor_min_separation: int = 18
    display_max_highs: int = 3
    # The current right edge has no future bars with which to confirm a fall.
    # Surface it provisionally when it is the high of the unconfirmed tail,
    # has risen sharply from the valley after the last accepted mountain, and
    # is at least as high as that mountain.
    display_right_edge_bars: int = 3
    display_right_edge_min_rise_pct: float = 30.0
    # Regime selection (v2). Candidate lids are generated from windows anchored
    # at full-history mountains; the active regime is chosen by rule, not by a
    # fixed lookback. ``regime_max_windows`` bounds the hypothesis count.
    regime_max_windows: int = 12
    regime_default_window_bars: int = 120  # legacy 10y window kept as one hypothesis
    regime_relevance_min_pct: float = 50.0  # last close must reach this % of lid value
    regime_relevance_max_pct: float = 400.0  # price far above an old lid = obsolete line
    # A line spanning a confirmed interior mountain that sits this far below it
    # is floating above a newer, lower ceiling regime rather than lidding it.
    regime_interior_major_max_below_pct: float = 15.0
    # Quarterly breakout state machine (v2). The escape band adapts to recent
    # quarterly true range so volatile charts are not flagged by noise.
    breakout_band_min_pct: float = 2.5
    breakout_band_tr_quarters: int = 8
    breakout_band_tr_mult: float = 0.5
    breakout_band_max_pct: float = 5.0
    breakout_confirm_quarters: int = 2
    breakout_expansion_min_pct: float = 10.0
    breakout_post_slope_min_pct: float = 15.0
    breakout_retest_support_tol_pct: float = 3.0
    # A very large extension is a breakout peak even when the preceding lid
    # rises mildly. This separates KN's near-doubling from normal rising-lid
    # continuations such as FCX.
    breakout_peak_extension_min_pct: float = 50.0
    # Resistance line search
    min_anchor_span: int = 24
    touch_tolerance_pct: float = 3.5
    touch_cluster_bars: int = 6
    max_wick_overshoots: int = 2
    break_tolerance_pct: float = 2.5
    breakout_confirm_bars: int = 3
    broken_out_max_age: int = 12
    stale_line_months: int = 48
    min_touches: int = 2
    max_slope_pct_per_year: float = 30.0
    # Falling lids are a different pattern; the search floor matches the
    # grading floor so a steeply descending line never wins over a flat one.
    min_slope_pct_per_year: float = -3.0
    # Coil gates
    min_proximity_pct: float = 70.0
    max_last_depth_ratio: float = 0.65
    # Price this close to the lid counts as "wound" even when the last
    # completed pullback was deep (KN/EAT style: the squeeze arrives late).
    pressing_proximity_pct: float = 90.0
    # A smooth exponential trend can present a plausible upper envelope over
    # decades (mega-cap uptrends); a high log-close R^2 over the base window
    # means the price is trending, not coiling.
    max_base_trend_r2: float = 0.85
    single_pullback_max_depth_pct: float = 30.0
    min_pullback_bars: int = 3
    # Slope grading (percent per year of line value at the last bar).
    # Recalibrated against the reviewed 18-ticker set: A < 5%/yr, B < 6.5%/yr,
    # C < 12%/yr, retaining -1%/yr as the A-grade falling floor and -3%/yr as
    # the overall validity floor (REG/SPG/LH/CF grade A, FCX stays C).
    grade_a_min: float = -1.0
    grade_a_max: float = 5.0
    grade_b_max: float = 6.5
    grade_c_max: float = 12.0
    grade_min: float = -3.0


DEFAULT_CONFIG = CoilConfig()


@dataclass(frozen=True)
class SwingPoint:
    idx: int
    date: str
    price: float
    prominence_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "date": self.date,
            "price": round(self.price, 4),
            "prominence_pct": round(self.prominence_pct, 2),
            "source": SOURCE,
        }


@dataclass
class ResistanceFit:
    anchor_a: SwingPoint  # earlier
    anchor_b: SwingPoint  # later
    slope_per_bar: float
    value_at_last_bar: float
    slope_pct_per_year: float
    touches: list[SwingPoint]  # cluster representatives, oldest -> newest
    wick_overshoots: int
    violation_idxs: list[int]  # closes above line*(1+tol), all within tolerated tail
    score: float

    @property
    def touch_count(self) -> int:
        return len(self.touches)

    @property
    def first_touch_idx(self) -> int:
        return self.touches[0].idx

    @property
    def last_touch_idx(self) -> int:
        return self.touches[-1].idx

    @property
    def span_bars(self) -> int:
        return self.last_touch_idx - self.first_touch_idx

    def value_at(self, idx: int) -> float:
        return self.anchor_a.price + self.slope_per_bar * (idx - self.anchor_a.idx)


@dataclass(frozen=True)
class ActiveLidFit:
    """Least-squares lid through the reviewed display-major sequence."""

    points: list[SwingPoint]
    slope_per_bar: float
    intercept: float
    value_at_last_bar: float
    slope_pct_per_year: float
    violation_idxs: list[int]

    def value_at(self, idx: int) -> float:
        return self.intercept + self.slope_per_bar * idx


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _log_trend_r2(closes: list[float]) -> Optional[float]:
    """R^2 of a linear fit to log-closes: ~1.0 means a steady compounding trend."""
    if len(closes) < 3:
        return None
    ys = [math.log(c) for c in closes]
    n = float(len(ys))
    mean_x = (n - 1) / 2.0
    mean_y = sum(ys) / n
    var_x = sum((i - mean_x) ** 2 for i in range(len(ys)))
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_y == 0:
        return 0.0
    cov = sum((i - mean_x) * (ys[i] - mean_y) for i in range(len(ys)))
    return (cov * cov) / (var_x * var_y)


def _clean_bars(bars: Iterable[dict[str, Any]], as_of: Optional[str]) -> list[dict[str, Any]]:
    """Drop unusable bars, sort by date, and truncate at ``as_of`` (inclusive)."""
    out = []
    for bar in bars:
        date = bar.get("date")
        high, low, close = bar.get("high"), bar.get("low"), bar.get("close")
        if not date or high is None or low is None or close is None:
            continue
        if high <= 0 or low <= 0 or close <= 0:
            continue
        if as_of and str(date) > as_of:
            continue
        out.append(bar)
    out.sort(key=lambda b: str(b["date"]))
    return out


def _pivot_high_indexes(highs: list[float], left: int, right: int) -> list[int]:
    """Bars strictly above their ``left`` window and >= their ``right`` window.

    Plateaus resolve to the first bar of the plateau.
    """
    out = []
    for i in range(left, len(highs) - right):
        v = highs[i]
        if any(highs[j] >= v for j in range(i - left, i)):
            continue
        if any(highs[j] > v for j in range(i + 1, i + right + 1)):
            continue
        out.append(i)
    return out


def _high_prominence(
    highs: list[float],
    lows: list[float],
    i: int,
    pivot_set: frozenset[int],
    equal_tol: float,
) -> float:
    """Drop from the peak to the higher of the two valley floors beside it.

    The walk on each side stops at a strictly higher bar, or at another pivot
    top within ``equal_tol`` (fraction) of this peak — see module docstring.
    """
    p = highs[i]
    cutoff = p * (1.0 - equal_tol)

    def side_min(step: int) -> Optional[float]:
        best: Optional[float] = None
        j = i + step
        while 0 <= j < len(highs):
            if highs[j] > p or (j in pivot_set and highs[j] >= cutoff):
                break
            if best is None or lows[j] < best:
                best = lows[j]
            j += step
        return best

    sides = [b for b in (side_min(-1), side_min(+1)) if b is not None]
    if not sides:
        return 0.0
    base = max(sides) if len(sides) == 2 else sides[0]
    return max(0.0, p - base)


def detect_swing_highs(bars: list[dict[str, Any]], config: CoilConfig = DEFAULT_CONFIG) -> list[SwingPoint]:
    """All confirmed pivot highs with prominence, oldest -> newest.

    The final ``pivot_right`` bars cannot confirm a pivot; an in-progress top
    is not a top yet.
    """
    highs = [float(b["high"]) for b in bars]
    lows = [float(b["low"]) for b in bars]
    idxs = _pivot_high_indexes(highs, config.pivot_left, config.pivot_right)
    pivot_set = frozenset(idxs)
    equal_tol = config.prominence_equal_tol_pct / 100.0
    swings = []
    for i in idxs:
        prom = _high_prominence(highs, lows, i, pivot_set, equal_tol)
        swings.append(
            SwingPoint(
                idx=i,
                date=str(bars[i]["date"]),
                price=highs[i],
                prominence_pct=prom / highs[i] * 100.0,
            )
        )
    return swings


def select_major_highs(swings: list[SwingPoint], config: CoilConfig = DEFAULT_CONFIG) -> list[SwingPoint]:
    """Major tops: prominence-ranked, suppressing neighbors closer than the
    separation window. Returned oldest -> newest."""
    candidates = [s for s in swings if s.prominence_pct >= config.major_prominence_pct]
    candidates.sort(key=lambda s: (-s.prominence_pct, s.idx))
    kept: list[SwingPoint] = []
    for cand in candidates:
        if all(abs(cand.idx - k.idx) >= config.major_min_separation for k in kept):
            kept.append(cand)
    return sorted(kept, key=lambda s: s.idx)


def _mountain_pivot_indexes(
    highs: list[float],
    left: int,
    right: int,
    equal_tol: float,
) -> list[int]:
    """Pivot highs for the displayed overlay, resolving near-equal plateaus
    to their first bar.

    A following bar must exceed the candidate by more than ``equal_tol`` to
    replace it; a preceding bar within that tolerance means the plateau has
    already started and suppresses the later bar. This moves a monthly top to
    the visually earlier quarterly candle when adjacent highs are effectively
    tied (the reviewed AAPL 2021-12 / 2022-01 case).
    """
    out = []
    for i in range(left, len(highs) - right):
        value = highs[i]
        if any(highs[j] >= value * (1.0 - equal_tol) for j in range(i - left, i)):
            continue
        if any(highs[j] > value * (1.0 + equal_tol) for j in range(i + 1, i + right + 1)):
            continue
        out.append(i)
    return out


def _cluster_mountain_plateaus(
    pivot_idxs: list[int],
    highs: list[float],
    max_gap: int,
    tolerance: float,
    upgrade_tolerance: float,
    upgrade_max_gap: int,
) -> list[int]:
    """Collapse nearby same-level pivots to one plateau representative.

    Near-equal highs retain the first bar, but a later pivot that clears the
    first by more than ``upgrade_tolerance`` becomes the representative. This
    preserves visually early flat shelves while preventing a materially higher
    neighboring quarter from invalidating the whole mountain.
    """
    reps: list[int] = []
    for idx in pivot_idxs:
        if reps:
            previous = reps[-1]
            scale = max(highs[idx], highs[previous])
            comparable = scale > 0 and abs(highs[idx] - highs[previous]) / scale <= tolerance
            if idx - previous <= max_gap and comparable:
                upgrade = (
                    idx - previous <= upgrade_max_gap
                    and highs[idx] > highs[previous] * (1.0 + upgrade_tolerance)
                )
                # If the first representative is a lower shoulder beneath an
                # already-established ceiling, let a later retest that restores
                # that ceiling replace it even outside the adjacent upgrade gap.
                if not upgrade and len(reps) >= 2:
                    established = max(highs[older] for older in reps[:-1])
                    upgrade = (
                        highs[previous] < established * (1.0 - upgrade_tolerance)
                        and highs[idx] >= established * (1.0 - upgrade_tolerance)
                    )
                if upgrade:
                    reps[-1] = idx
                continue
        reps.append(idx)
    return reps


def _mountain_prominence(
    highs: list[float],
    lows: list[float],
    idx: int,
    pivot_set: frozenset[int],
    equal_tol: float,
) -> float:
    """Two-sided mountain height, ignoring tiny adjacent overshoots.

    The base is the higher of the valleys on either side, so the score is
    limited by the weaker of the rise into the top and the fall after it. A
    materially higher bar ends a side walk. A separate comparable pivot also
    ends it, while an unselected adjacent plateau bar does not.
    """
    peak = highs[idx]
    comparable_floor = peak * (1.0 - equal_tol)
    materially_higher = peak * (1.0 + equal_tol)

    def side_min(step: int) -> Optional[float]:
        best: Optional[float] = None
        cursor = idx + step
        while 0 <= cursor < len(highs):
            if highs[cursor] > materially_higher:
                break
            if cursor in pivot_set and highs[cursor] >= comparable_floor:
                break
            if best is None or lows[cursor] < best:
                best = lows[cursor]
            cursor += step
        return best

    sides = [value for value in (side_min(-1), side_min(+1)) if value is not None]
    # Confirmed mountains are strictly two-sided. If price exceeds this bar
    # before making a meaningful fall, it was only a shoulder on the climb to
    # the later, higher peak. The separate right-edge path below is the sole
    # exception because future bars do not exist yet.
    if len(sides) < 2:
        return 0.0
    base = max(sides)
    return max(0.0, peak - base)


def _suppress_slower_comparable_retests(
    points: list[SwingPoint],
    closes: list[float],
    edge_start: int,
    config: CoilConfig,
) -> list[SwingPoint]:
    """Drop a later same-level retest when its rejection is materially slower.

    This is intentionally one-directional: an earlier sharp mountain remains
    the representative when a later comparable peak cannot reject within the
    fast confirmation window. A later sharper retest does not erase an already
    accepted older mountain.
    """
    ordered = sorted(points, key=lambda point: point.idx)
    if len(ordered) < 2:
        return ordered
    tolerance = max(0.0, config.display_comparable_retest_tolerance_pct / 100.0)
    fast_bars = max(1, config.display_fast_rejection_bars)
    remove_idxs: set[int] = set()

    def fast_rejection_pct(point: SwingPoint) -> float:
        end = min(len(closes), point.idx + fast_bars + 1)
        following = closes[point.idx + 1 : end]
        if not following:
            return 0.0
        return (point.price - min(following)) / point.price * 100.0

    for earlier, later in zip(ordered, ordered[1:]):
        if later.idx >= edge_start:
            continue
        scale = max(earlier.price, later.price)
        if scale <= 0 or abs(later.price - earlier.price) / scale > tolerance:
            continue
        earlier_fast = fast_rejection_pct(earlier)
        later_fast = fast_rejection_pct(later)
        if (
            earlier_fast >= config.display_body_prominence_pct
            and later_fast < config.display_body_prominence_pct
        ):
            remove_idxs.add(later.idx)

    return [point for point in ordered if point.idx not in remove_idxs]


@dataclass(frozen=True)
class RolePoint:
    """A structure point plus its schema-v2 role and selection evidence."""

    point: SwingPoint
    role: str
    evidence: dict[str, Any]


def _detect_display_major_highs_on_bars(
    bars: list[dict[str, Any]],
    config: CoilConfig = DEFAULT_CONFIG,
) -> list[SwingPoint]:
    """Legacy windowed wrapper: latest chart-worthy mountain peaks, capped."""
    window_start = max(0, len(bars) - max(1, config.display_lookback_bars))
    role_points = _select_window_role_points(
        bars,
        config,
        window_start,
        max_points=config.display_max_highs,
    )
    return [rp.point for rp in role_points]


def _select_window_role_points(
    bars: list[dict[str, Any]],
    config: CoilConfig,
    window_start: int,
    max_points: Optional[int] = None,
) -> list[RolePoint]:
    """Chart-worthy structure points within ``bars[window_start:]``, with roles.

    This is intentionally stricter than ``select_major_highs``, whose wider
    candidate pool feeds resistance-line search. The selection contract follows
    human review: large two-sided mountains only (``major_top``), first bar of
    a plateau, non-decreasing peak levels, plus a strong provisional high in
    the unconfirmed right-edge tail (``provisional_top``). Secondary and
    intermediate line anchors that are not strict mountains are tagged
    ``structural_retest``. ``max_points`` of None keeps every qualifying point
    (v2 uncapped fitting); an integer keeps only the latest N.
    """
    if len(bars) < 2:
        return []

    highs = [float(bar["high"]) for bar in bars]
    lows = [float(bar["low"]) for bar in bars]
    closes = [float(bar["close"]) for bar in bars]
    last_idx = len(bars) - 1
    window_start = max(0, min(window_start, len(bars) - 2))
    roles: dict[int, str] = {}
    evidence: dict[int, dict[str, Any]] = {}
    equal_tol = max(0.0, config.display_pivot_equal_tol_pct / 100.0)
    plateau_tol = max(0.0, config.display_plateau_tolerance_pct / 100.0)

    raw_pivots = [
        idx
        for idx in _mountain_pivot_indexes(
            highs,
            config.pivot_left,
            config.pivot_right,
            equal_tol,
        )
        if idx >= window_start
    ]
    pivot_idxs = _cluster_mountain_plateaus(
        raw_pivots,
        highs,
        max(0, config.display_plateau_gap),
        plateau_tol,
        equal_tol,
        max(0, config.display_plateau_upgrade_gap),
    )
    pivot_set = frozenset(pivot_idxs)

    candidates: list[SwingPoint] = []
    for idx in pivot_idxs:
        prominence = _mountain_prominence(highs, lows, idx, pivot_set, equal_tol)
        prominence_pct = prominence / highs[idx] * 100.0
        if prominence_pct < config.display_major_prominence_pct:
            continue
        candidates.append(
            SwingPoint(
                idx=idx,
                date=str(bars[idx]["date"]),
                price=highs[idx],
                prominence_pct=prominence_pct,
            )
        )

    # When two qualifying mountains are too close, retain the one with the
    # stronger two-sided move; ties resolve to the earlier peak.
    ranked = sorted(candidates, key=lambda point: (-point.prominence_pct, point.idx))
    separated: list[SwingPoint] = []
    for candidate in ranked:
        if all(
            abs(candidate.idx - kept.idx) >= config.display_major_min_separation
            for kept in separated
        ):
            separated.append(candidate)

    # Once a major top establishes the level, later accepted peaks must be at
    # the same level or higher. Lower shoulders and rebound plateaus drop out.
    level_tol = max(0.0, config.display_same_or_higher_tolerance_pct / 100.0)
    progressive: list[SwingPoint] = []
    for candidate in sorted(separated, key=lambda point: point.idx):
        if not progressive or candidate.price >= progressive[-1].price * (1.0 - level_tol):
            progressive.append(candidate)

    # The final bars cannot have a confirmed fall yet. Include their maximum
    # as a provisional mountain only when the rise from the intervening valley
    # is already large and the peak extends (or matches) the established level.
    edge_count = max(1, config.display_right_edge_bars)
    edge_start = max(window_start, len(bars) - edge_count)
    # Use the highest bar in the whole unconfirmed tail. On a quarterly chart
    # this may be the candle immediately left of a new partial quarter, which
    # is still the visually correct extreme-right mountain (and avoids moving
    # the marker onto a lower partial candle).
    edge_idx = max(range(edge_start, len(bars)), key=lambda idx: highs[idx])
    far_enough = all(
        abs(edge_idx - point.idx) >= config.display_major_min_separation
        for point in progressive
    )
    level_ok = not progressive or highs[edge_idx] >= progressive[-1].price * (1.0 - level_tol)
    valley_start = progressive[-1].idx + 1 if progressive else window_start
    if valley_start <= edge_idx:
        valley = min(lows[valley_start : edge_idx + 1])
        rise_pct = (highs[edge_idx] - valley) / highs[edge_idx] * 100.0
    else:
        rise_pct = 0.0
    if far_enough and level_ok and rise_pct >= config.display_right_edge_min_rise_pct:
        progressive.append(
            SwingPoint(
                idx=edge_idx,
                date=str(bars[edge_idx]["date"]),
                price=highs[edge_idx],
                prominence_pct=rise_pct,
            )
        )
        roles[edge_idx] = ROLE_PROVISIONAL_TOP
        evidence[edge_idx] = {"rise_pct": round(rise_pct, 2)}

    if max_points is not None and max_points <= 0:
        return []
    selected = progressive if max_points is None else progressive[-max_points:]

    # Validate candle-body separation only after choosing the latest peaks.
    # This prevents a rejected shoulder from backfilling the overlay with an
    # obsolete older shelf. The provisional tail peak is exempt because it has
    # no future candle bodies yet; its rise gate above is the evidence we have.
    confirmed: list[SwingPoint] = []
    window_low = min(lows[window_start:])
    window_high = max(highs[window_start:])
    window_span = window_high - window_low
    for point in selected:
        range_position_pct = (
            (point.price - window_low) / window_span * 100.0 if window_span > 0 else 100.0
        )
        if range_position_pct < config.display_min_range_position_pct:
            continue
        if point.idx >= edge_start:
            confirmed.append(point)
            evidence.setdefault(point.idx, {})["range_position_pct"] = round(range_position_pct, 2)
            continue
        body_prominence = _mountain_prominence(
            highs,
            closes,
            point.idx,
            pivot_set,
            equal_tol,
        )
        body_prominence_pct = body_prominence / point.price * 100.0
        if body_prominence_pct >= config.display_body_prominence_pct:
            confirmed.append(point)
            roles[point.idx] = ROLE_MAJOR_TOP
            evidence[point.idx] = {
                "body_prominence_pct": round(body_prominence_pct, 2),
                "range_position_pct": round(range_position_pct, 2),
            }
    # A single plotted point cannot define a resistance slope. Preserve the
    # strict mountain list as the primary sequence, but when it leaves only one
    # point, add the strongest comparable retest from the same chart. This is a
    # line-anchor fallback, so its mountain/body thresholds are intentionally
    # lower while its price must remain close to the established ceiling.
    target_min = max(0, config.display_min_line_anchors)
    if max_points is not None:
        target_min = min(max_points, target_min)
    if len(confirmed) < target_min:
        existing_idxs = {point.idx for point in confirmed}
        reference_price = max(
            (point.price for point in confirmed),
            default=window_high,
        )
        min_level = reference_price * (
            max(0.0, config.display_secondary_anchor_min_level_pct) / 100.0
        )
        secondary: list[SwingPoint] = []
        for idx in sorted(set(pivot_idxs + [edge_idx])):
            if idx in existing_idxs or highs[idx] < min_level:
                continue
            range_position_pct = (
                (highs[idx] - window_low) / window_span * 100.0
                if window_span > 0
                else 100.0
            )
            if range_position_pct < config.display_min_range_position_pct:
                continue

            if idx == edge_idx and idx >= edge_start:
                left = min(lows[window_start : idx + 1])
                wick_prominence_pct = (highs[idx] - left) / highs[idx] * 100.0
                body_prominence_pct = wick_prominence_pct
            else:
                wick_prominence_pct = (
                    _mountain_prominence(highs, lows, idx, pivot_set, equal_tol)
                    / highs[idx]
                    * 100.0
                )
                body_prominence_pct = (
                    _mountain_prominence(highs, closes, idx, pivot_set, equal_tol)
                    / highs[idx]
                    * 100.0
                )
            if wick_prominence_pct < config.display_secondary_anchor_min_prominence_pct:
                continue
            if body_prominence_pct < config.display_secondary_anchor_body_prominence_pct:
                continue
            secondary.append(
                SwingPoint(
                    idx=idx,
                    date=str(bars[idx]["date"]),
                    price=highs[idx],
                    prominence_pct=wick_prominence_pct,
                )
            )

        # The closest price to the established ceiling is the best line
        # anchor; if levels tie, prefer the more recent retest.
        secondary.sort(key=lambda point: (-point.price, -point.idx))
        for candidate in secondary:
            if len(confirmed) >= target_min:
                break
            if all(
                abs(candidate.idx - kept.idx) >= config.display_major_min_separation
                for kept in confirmed
            ):
                confirmed.append(candidate)
                roles.setdefault(candidate.idx, ROLE_STRUCTURAL_RETEST)

    # Two endpoint anchors can reveal a meaningful middle retest that is not a
    # strict standalone mountain. Admit at most one such pivot, and only when it
    # is well separated from both endpoints and lies close to their fitted lid.
    if len(confirmed) >= 2 and (max_points is None or len(confirmed) < max_points):
        ordered_confirmed = sorted(confirmed, key=lambda point: point.idx)
        first = ordered_confirmed[0]
        last = ordered_confirmed[-1]
        span = last.idx - first.idx
        line_tolerance = max(
            0.0,
            config.display_intermediate_anchor_tolerance_pct / 100.0,
        )
        intermediate: list[tuple[SwingPoint, float]] = []
        if span > 0:
            for idx in pivot_idxs:
                if not first.idx < idx < last.idx:
                    continue
                if any(
                    abs(idx - point.idx) < config.display_intermediate_anchor_min_separation
                    for point in ordered_confirmed
                ):
                    continue
                line_value = first.price + (last.price - first.price) * (
                    (idx - first.idx) / span
                )
                if line_value <= 0:
                    continue
                line_error = abs(highs[idx] - line_value) / line_value
                if line_error > line_tolerance:
                    continue
                range_position_pct = (
                    (highs[idx] - window_low) / window_span * 100.0
                    if window_span > 0
                    else 100.0
                )
                if range_position_pct < config.display_min_range_position_pct:
                    continue
                wick_prominence_pct = (
                    _mountain_prominence(highs, lows, idx, pivot_set, equal_tol)
                    / highs[idx]
                    * 100.0
                )
                body_prominence_pct = (
                    _mountain_prominence(highs, closes, idx, pivot_set, equal_tol)
                    / highs[idx]
                    * 100.0
                )
                if wick_prominence_pct < config.display_secondary_anchor_min_prominence_pct:
                    continue
                if body_prominence_pct < config.display_secondary_anchor_body_prominence_pct:
                    continue
                intermediate.append(
                    (
                        SwingPoint(
                            idx=idx,
                            date=str(bars[idx]["date"]),
                            price=highs[idx],
                            prominence_pct=wick_prominence_pct,
                        ),
                        line_error,
                    )
                )

        if intermediate:
            intermediate.sort(
                key=lambda item: (-item[0].prominence_pct, item[1], item[0].idx)
            )
            best_intermediate, line_error = intermediate[0]
            confirmed.append(best_intermediate)
            roles.setdefault(best_intermediate.idx, ROLE_STRUCTURAL_RETEST)
            evidence.setdefault(best_intermediate.idx, {})["line_error_pct"] = round(
                line_error * 100.0, 2
            )

    confirmed = _suppress_slower_comparable_retests(
        confirmed,
        closes,
        edge_start,
        config,
    )

    ordered = sorted(confirmed, key=lambda point: point.idx)
    if max_points is not None:
        ordered = ordered[-max_points:]
    return [
        RolePoint(
            point=point,
            role=roles.get(point.idx, ROLE_STRUCTURAL_RETEST),
            evidence=evidence.get(point.idx, {}),
        )
        for point in ordered
    ]


def _aggregate_quarterly_display_bars(
    monthly: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate monthly bars for the 3M overlay while retaining high origin."""
    quarters: list[dict[str, Any]] = []
    for source_idx, bar in enumerate(monthly):
        text = str(bar["date"])
        year = int(text[:4])
        month = int(text[5:7])
        key = (year, (month - 1) // 3 + 1)
        if quarters and quarters[-1]["_quarter_key"] == key:
            quarter = quarters[-1]
            if float(bar["high"]) > float(quarter["high"]):
                quarter["high"] = float(bar["high"])
                quarter["_high_source_idx"] = source_idx
            quarter["low"] = min(float(quarter["low"]), float(bar["low"]))
            quarter["close"] = float(bar["close"])
            quarter["date"] = text
            quarter["_close_source_idx"] = source_idx
            quarter["_last_month"] = month
        else:
            quarters.append(
                {
                    "_quarter_key": key,
                    "_high_source_idx": source_idx,
                    "_close_source_idx": source_idx,
                    "_last_month": month,
                    "date": text,
                    "open": float(bar.get("open", bar["close"])),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": bar.get("volume"),
                }
            )
    return quarters


def _quarter_is_complete(quarter: dict[str, Any]) -> bool:
    """A quarter is complete when it contains its calendar-final month."""
    return int(quarter["_last_month"]) % 3 == 0


def _quarterly_scaled_config(config: CoilConfig) -> CoilConfig:
    """Scale monthly-bar tunables into quarterly-bar coordinates."""
    return replace(
        config,
        pivot_left=1,
        pivot_right=1,
        display_lookback_bars=max(1, math.ceil(config.display_lookback_bars / 3)),
        display_plateau_gap=max(1, math.ceil(config.display_plateau_gap / 3)),
        display_plateau_upgrade_gap=max(
            1,
            math.ceil(config.display_plateau_upgrade_gap / 3),
        ),
        display_major_min_separation=max(
            1,
            math.ceil(config.display_major_min_separation / 3),
        ),
        display_intermediate_anchor_min_separation=max(
            1,
            math.ceil(config.display_intermediate_anchor_min_separation / 3),
        ),
        display_fast_rejection_bars=max(
            1,
            math.ceil(config.display_fast_rejection_bars / 3),
        ),
        min_anchor_span=max(1, math.ceil(config.min_anchor_span / 3)),
        stale_line_months=max(1, math.ceil(config.stale_line_months / 3)),
        broken_out_max_age=max(1, math.ceil(config.broken_out_max_age / 3)),
        touch_cluster_bars=max(1, math.ceil(config.touch_cluster_bars / 3)),
        # Inspect the current and immediately preceding quarterly candle. This
        # keeps the strongest unconfirmed right-edge mountain when the latest
        # quarter is only partially formed.
        display_right_edge_bars=2,
    )


def _remap_quarterly_role_point(
    rp: RolePoint,
    quarterly: list[dict[str, Any]],
    bars: list[dict[str, Any]],
) -> RolePoint:
    """Remap a quarterly-coordinate point to the source month of its high.

    A reviewed point pins its own month (``source_month_idx``); algorithmic
    points land on the month that supplied the quarter's high.
    """
    source_idx = int(
        rp.evidence.get("source_month_idx", quarterly[rp.point.idx]["_high_source_idx"])
    )
    return RolePoint(
        point=SwingPoint(
            idx=source_idx,
            date=str(bars[source_idx]["date"]),
            price=rp.point.price,
            prominence_pct=rp.point.prominence_pct,
        ),
        role=rp.role,
        evidence=rp.evidence,
    )


def detect_display_major_highs(
    bars: list[dict[str, Any]],
    config: CoilConfig = DEFAULT_CONFIG,
) -> list[SwingPoint]:
    """Detect overlay mountains on the same quarterly candles the user sees.

    Monthly highs and lows inside one quarter cannot confirm a post-peak fall:
    confirmation must occur on a later quarterly candle. The returned point is
    remapped to the source month that supplied the quarter's high so API dates
    remain precise while the frontend still lands on the correct 3M candle.

    v2: the points come from the full-history active-regime selection, capped
    to the latest ``display_max_highs`` for the legacy overlay contract.
    """
    structure = _analyze_quarterly_structure(bars, config)
    if structure is None:
        return []
    max_highs = max(0, config.display_max_highs)
    return [rp.point for rp in structure.role_points_monthly][-max_highs:]


def _cluster_touches(touches: list[SwingPoint], max_gap: int) -> list[SwingPoint]:
    """Merge touches within ``max_gap`` bars; representative is the highest."""
    reps: list[SwingPoint] = []
    for touch in sorted(touches, key=lambda s: s.idx):
        if reps and touch.idx - reps[-1].idx <= max_gap:
            if touch.price > reps[-1].price:
                reps[-1] = touch
        else:
            reps.append(touch)
    return reps


def _evaluate_pair(
    a: SwingPoint,
    b: SwingPoint,
    closes: list[float],
    touch_pool: list[SwingPoint],
    last_idx: int,
    config: CoilConfig,
) -> Optional[ResistanceFit]:
    slope = (b.price - a.price) / (b.idx - a.idx)
    value_last = a.price + slope * (last_idx - a.idx)
    if value_last <= 0:
        return None
    slope_pct_per_year = slope * BARS_PER_YEAR / value_last * 100.0
    if not (config.min_slope_pct_per_year <= slope_pct_per_year <= config.max_slope_pct_per_year):
        return None

    touch_tol = config.touch_tolerance_pct / 100.0
    raw_touches: list[SwingPoint] = []
    wick_overshoots = 0
    for swing in touch_pool:
        line_value = a.price + slope * (swing.idx - a.idx)
        if line_value <= 0:
            continue
        err = (swing.price - line_value) / line_value
        if abs(err) <= touch_tol:
            raw_touches.append(swing)
        elif err > touch_tol:
            wick_overshoots += 1
    if wick_overshoots > config.max_wick_overshoots:
        return None

    touches = _cluster_touches(raw_touches, config.touch_cluster_bars)
    if len(touches) < config.min_touches:
        return None
    span = touches[-1].idx - touches[0].idx
    if span < config.min_anchor_span:
        return None

    months_since_touch = last_idx - touches[-1].idx
    if months_since_touch > config.stale_line_months and closes[last_idx] < 0.8 * value_last:
        return None

    # Closes escaping above the lid kill the line unless they sit inside the
    # tolerated tail (a breakout in progress / just happened).
    break_tol = 1.0 + config.break_tolerance_pct / 100.0
    oldest_tolerated = last_idx - config.broken_out_max_age + 1
    violations: list[int] = []
    for x in range(touches[0].idx, last_idx + 1):
        line_value = a.price + slope * (x - a.idx)
        if line_value <= 0 or closes[x] > line_value * break_tol:
            if x < oldest_tolerated:
                return None
            violations.append(x)

    score = (
        0.40 * min(len(touches), 5) / 5.0
        + 0.25 * min(span / 120.0, 1.0)
        + 0.20 * _clamp01(1.0 - abs(slope_pct_per_year) / 15.0)
        + 0.15 * _clamp01(1.0 - months_since_touch / 60.0)
        - 0.10 * wick_overshoots
        - 0.05 * len(violations)
    )
    return ResistanceFit(
        anchor_a=a,
        anchor_b=b,
        slope_per_bar=slope,
        value_at_last_bar=value_last,
        slope_pct_per_year=slope_pct_per_year,
        touches=touches,
        wick_overshoots=wick_overshoots,
        violation_idxs=violations,
        score=score,
    )


def fit_resistance_line(
    bars: list[dict[str, Any]],
    majors: list[SwingPoint],
    swings: list[SwingPoint],
    config: CoilConfig = DEFAULT_CONFIG,
) -> Optional[ResistanceFit]:
    """Best resistance line over anchor pairs of major tops.

    Touches may include minor swing highs; anchors must be majors. Returns
    None when no pair survives the tolerance, staleness, and violation gates.
    """
    closes = [float(b["close"]) for b in bars]
    last_idx = len(bars) - 1
    touch_pool = [s for s in swings if s.prominence_pct >= config.minor_prominence_pct]
    best: Optional[ResistanceFit] = None
    for i, a in enumerate(majors):
        for b in majors[i + 1 :]:
            if b.idx - a.idx < config.min_anchor_span:
                continue
            fit = _evaluate_pair(a, b, closes, touch_pool, last_idx, config)
            if fit is None:
                continue
            if best is None or (fit.score, fit.span_bars, fit.touch_count) > (
                best.score,
                best.span_bars,
                best.touch_count,
            ):
                best = fit
    return best


def _ls_fit(points: list[SwingPoint]) -> Optional[tuple[float, float]]:
    """Least-squares (slope, intercept) through point (idx, price) pairs."""
    if len(points) < 2:
        return None
    mean_idx = sum(point.idx for point in points) / len(points)
    mean_price = sum(point.price for point in points) / len(points)
    variance = sum((point.idx - mean_idx) ** 2 for point in points)
    if variance <= 0:
        return None
    slope = sum(
        (point.idx - mean_idx) * (point.price - mean_price) for point in points
    ) / variance
    return slope, mean_price - slope * mean_idx


def _fit_error_pct(points: list[SwingPoint], slope: float, intercept: float) -> float:
    """RMS distance of the anchors from the line, as percent of line value."""
    errors = []
    for point in points:
        line_value = intercept + slope * point.idx
        if line_value <= 0:
            return math.inf
        errors.append(((point.price - line_value) / line_value) ** 2)
    return math.sqrt(sum(errors) / len(errors)) * 100.0


# ---------------------------------------------------------------------------
# Quarterly breakout state machine (v2)
# ---------------------------------------------------------------------------


@dataclass
class BreakoutAssessment:
    """Lifecycle of price against one lid, walked over completed quarters.

    ``state``: sealed | breaking_out | broken_out. A partial current quarter
    never advances the machine; its overshoot is recorded as
    ``provisional_escape`` only.
    """

    state: str = "sealed"
    band_pct: float = 0.0
    first_escape: Optional[dict[str, Any]] = None
    confirmed: Optional[dict[str, Any]] = None
    provisional_escape: Optional[dict[str, Any]] = None
    failed_breakouts: list[dict[str, Any]] = None  # type: ignore[assignment]
    expansion_pct: Optional[float] = None
    post_break_slope_pct_per_year: Optional[float] = None
    retest: Optional[dict[str, Any]] = None
    violation_idxs: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failed_breakouts is None:
            self.failed_breakouts = []
        if self.violation_idxs is None:
            self.violation_idxs = []


def _adaptive_band_pct(
    quarters: list[dict[str, Any]],
    q_idx: int,
    lid_value: float,
    config: CoilConfig,
) -> float:
    """Escape band: max of the percentage floor and a true-range component."""
    if lid_value <= 0:
        return config.breakout_band_min_pct
    lo = max(0, q_idx - max(1, config.breakout_band_tr_quarters) + 1)
    true_ranges: list[float] = []
    prev_close: Optional[float] = None
    for k in range(max(0, lo - 1), q_idx + 1):
        bar = quarters[k]
        high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if k >= lo:
            true_ranges.append(tr)
        prev_close = close
    if not true_ranges:
        return config.breakout_band_min_pct
    avg_tr_pct = sum(true_ranges) / len(true_ranges) / lid_value * 100.0
    band = max(config.breakout_band_min_pct, config.breakout_band_tr_mult * avg_tr_pct)
    return min(band, config.breakout_band_max_pct)


def _quarter_ref(quarter: dict[str, Any], q_idx: int) -> dict[str, Any]:
    return {
        "q_idx": q_idx,
        "date": str(quarter["date"]),
        "close": round(float(quarter["close"]), 4),
    }


def _run_breakout_state_machine(
    quarters: list[dict[str, Any]],
    value_at: Any,
    start_idx: int,
    config: CoilConfig,
) -> BreakoutAssessment:
    """Walk completed quarterly closes against the lid.

    Rules (see module plan):
    - a small or temporary overshoot remains a lid retest;
    - ``breaking_out`` begins when a completed quarterly close materially
      clears the adaptive band; a partial quarter is provisional only;
    - ``broken_out`` requires two consecutive completed quarters above the
      band with material expansion and a post-break slope substantially
      steeper than the lid, or a breakout-retest-continuation sequence where
      the old lid holds as support and price makes a new high;
    - a close back below the lid before confirmation is a failed breakout and
      the machine returns to pre-breakout evaluation.
    """
    out = BreakoutAssessment()
    last_idx = len(quarters) - 1
    consecutive = 0
    escape_high: Optional[float] = None
    pre_escape_close: Optional[float] = None
    best_close: Optional[float] = None
    retest_holding: Optional[dict[str, Any]] = None
    lid_slope_pct = None

    for q in range(max(0, start_idx), last_idx + 1):
        bar = quarters[q]
        lid_value = float(value_at(q))
        if lid_value <= 0:
            continue
        band = _adaptive_band_pct(quarters, q, lid_value, config)
        out.band_pct = band
        threshold = lid_value * (1.0 + band / 100.0)
        close = float(bar["close"])
        low = float(bar["low"])
        high = float(bar["high"])
        complete = q < last_idx or _quarter_is_complete(bar)
        above_band = close > threshold

        if not complete:
            if above_band and out.state == "sealed":
                out.provisional_escape = _quarter_ref(bar, q)
            break

        if above_band:
            out.violation_idxs.append(q)

        if out.state == "sealed":
            if above_band:
                out.state = "breaking_out"
                out.first_escape = {**_quarter_ref(bar, q), "band_pct": round(band, 2)}
                pre_escape_close = (
                    float(quarters[q - 1]["close"]) if q > 0 else lid_value
                )
                escape_high = high
                best_close = close
                consecutive = 1
                retest_holding = None
        elif out.state == "breaking_out":
            escape_high = max(escape_high or high, high)
            best_close = max(best_close or close, close)
            if above_band:
                consecutive += 1
            elif close < lid_value:
                # Closed back below the lid before confirmation: failed breakout.
                out.failed_breakouts.append(
                    {
                        "escape": out.first_escape,
                        "failed": _quarter_ref(bar, q),
                    }
                )
                out.state = "sealed"
                out.first_escape = None
                out.retest = None
                consecutive = 0
                escape_high = None
                best_close = None
                retest_holding = None
                continue
            else:
                # Close between the lid and the band: the overshoot cooled into
                # a retest of the old lid from above.
                consecutive = 0
                holds = low >= lid_value * (
                    1.0 - config.breakout_retest_support_tol_pct / 100.0
                )
                out.retest = {**_quarter_ref(bar, q), "holds": holds}
                if holds:
                    retest_holding = {**out.retest, "escape_high": escape_high}

            if out.first_escape is not None and pre_escape_close:
                elapsed = q - int(out.first_escape["q_idx"]) + 1
                post_slope = (
                    (close - pre_escape_close)
                    / pre_escape_close
                    / max(1, elapsed)
                    * 4.0
                    * 100.0
                )
                out.post_break_slope_pct_per_year = round(post_slope, 2)
                escape_lid = float(value_at(int(out.first_escape["q_idx"])))
                if escape_lid > 0 and best_close is not None:
                    out.expansion_pct = round(
                        (best_close - escape_lid) / escape_lid * 100.0, 2
                    )
                if lid_slope_pct is None:
                    span = max(1, q - max(0, start_idx))
                    lid_slope_pct = (
                        (float(value_at(q)) - float(value_at(q - span)))
                        / max(lid_value, 1e-9)
                        / span
                        * 4.0
                        * 100.0
                    )
                steep_enough = post_slope >= max(
                    config.breakout_post_slope_min_pct,
                    2.0 * abs(lid_slope_pct),
                )
                expanded = (
                    out.expansion_pct is not None
                    and out.expansion_pct >= config.breakout_expansion_min_pct
                )
                confirmed_by_persistence = (
                    consecutive >= max(1, config.breakout_confirm_quarters)
                    and expanded
                    and steep_enough
                )
                confirmed_by_retest = (
                    retest_holding is not None
                    and above_band
                    and retest_holding.get("escape_high") is not None
                    and high > float(retest_holding["escape_high"])
                )
                if confirmed_by_persistence or confirmed_by_retest:
                    out.state = "broken_out"
                    out.confirmed = {
                        **_quarter_ref(bar, q),
                        "via": "retest_continuation"
                        if confirmed_by_retest and not confirmed_by_persistence
                        else "persistence",
                    }
        else:  # broken_out: keep collecting violations; corrections do not undo it.
            escape_high = max(escape_high or high, high)
            best_close = max(best_close or close, close)

    return out


# ---------------------------------------------------------------------------
# Active-regime selection (v2)
# ---------------------------------------------------------------------------


@dataclass
class LidHypothesis:
    """One candidate lid: window, role points, fitted line, and assessment."""

    window_start: int  # quarterly idx
    role_points: list[RolePoint]  # quarterly coordinates
    anchors: list[SwingPoint]  # lid-fit members, quarterly coordinates
    ejected: list[RolePoint]  # breakout peaks removed from the fit
    slope_per_bar: float
    intercept: float
    value_at_last_bar: float
    slope_pct_per_year: float
    fit_error_pct: float
    breakout: BreakoutAssessment
    score: float = 0.0
    reject_reason: Optional[str] = None

    def value_at(self, idx: int) -> float:
        return self.intercept + self.slope_per_bar * idx


def _full_history_mountains(
    quarterly: list[dict[str, Any]],
    qconfig: CoilConfig,
) -> tuple[list[int], list[int]]:
    """(clustered pivot idxs, strict-mountain idxs) over the whole history."""
    highs = [float(bar["high"]) for bar in quarterly]
    lows = [float(bar["low"]) for bar in quarterly]
    equal_tol = max(0.0, qconfig.display_pivot_equal_tol_pct / 100.0)
    plateau_tol = max(0.0, qconfig.display_plateau_tolerance_pct / 100.0)
    raw = _mountain_pivot_indexes(highs, qconfig.pivot_left, qconfig.pivot_right, equal_tol)
    clustered = _cluster_mountain_plateaus(
        raw,
        highs,
        max(0, qconfig.display_plateau_gap),
        plateau_tol,
        equal_tol,
        max(0, qconfig.display_plateau_upgrade_gap),
    )
    pivot_set = frozenset(clustered)
    mountains = [
        idx
        for idx in clustered
        if _mountain_prominence(highs, lows, idx, pivot_set, equal_tol) / highs[idx] * 100.0
        >= qconfig.display_major_prominence_pct
    ]
    return clustered, mountains


def _regime_window_starts(
    quarterly: list[dict[str, Any]],
    qconfig: CoilConfig,
    clustered_pivots: list[int],
) -> list[int]:
    """Candidate regime windows: full history, the legacy ten-year window, and
    a window opening at each full-history mountain pivot (latest first)."""
    n = len(quarterly)
    starts = {0, max(0, n - max(1, qconfig.display_lookback_bars))}
    for idx in clustered_pivots:
        starts.add(max(0, idx - qconfig.pivot_left))
    ordered = sorted(starts, reverse=True)
    bounded = ordered[: max(2, qconfig.regime_max_windows)]
    if 0 not in bounded:
        bounded.append(0)
    return sorted(bounded)


def _eject_breakout_anchors(
    role_points: list[RolePoint],
    config: CoilConfig,
) -> tuple[list[RolePoint], list[RolePoint]]:
    """Remove trailing anchors that are themselves the breakout, not the lid.

    A materially higher trailing point above a flat-or-falling prior lid is
    the escape from that lid. Rotating it into the fit would erase the
    breakout, so it is ejected and re-tagged ``breakout_peak``. A rising prior
    lid may legitimately extend to a higher new anchor and is left alone.
    """
    kept = sorted(role_points, key=lambda rp: rp.point.idx)
    ejected: list[RolePoint] = []
    tolerance = 1.0 + config.break_tolerance_pct / 100.0
    while len(kept) >= 3:
        prefix = [rp.point for rp in kept[:-1]]
        fit = _ls_fit(prefix)
        if fit is None:
            break
        prior_slope, _ = fit
        prior_ceiling = max(point.price for point in prefix)
        last = kept[-1]
        extension_tolerance = 1.0 + max(
            config.break_tolerance_pct,
            config.breakout_peak_extension_min_pct,
        ) / 100.0
        escaped_flat_lid = prior_slope <= 0 and last.point.price > prior_ceiling * tolerance
        escaped_mildly_rising_lid = last.point.price > prior_ceiling * extension_tolerance
        if escaped_flat_lid or escaped_mildly_rising_lid:
            ejected.append(
                RolePoint(point=last.point, role=ROLE_BREAKOUT_PEAK, evidence=last.evidence)
            )
            kept = kept[:-1]
            continue
        break
    return kept, list(reversed(ejected))


def _build_lid_hypothesis(
    quarterly: list[dict[str, Any]],
    qconfig: CoilConfig,
    config: CoilConfig,
    window_start: int,
) -> Optional[LidHypothesis]:
    role_points = _select_window_role_points(quarterly, qconfig, window_start, max_points=None)
    return _hypothesis_from_role_points(
        quarterly, role_points, qconfig, config, window_start
    )


def _hypothesis_from_role_points(
    quarterly: list[dict[str, Any]],
    role_points: list[RolePoint],
    qconfig: CoilConfig,
    config: CoilConfig,
    window_start: int,
    require_mountain: bool = True,
) -> Optional[LidHypothesis]:
    if len(role_points) < 2:
        return None
    kept, ejected = _eject_breakout_anchors(role_points, config)
    anchors = [rp.point for rp in kept]
    if len(anchors) < 2:
        return None
    # Secondary/intermediate retests only ever supplement a real mountain. A
    # window whose anchors are all fallback retests is not a ceiling regime.
    # (Pair-search-seeded hypotheses carry their own touch/span quality gates.)
    if require_mountain and not any(
        rp.role in (ROLE_MAJOR_TOP, ROLE_PROVISIONAL_TOP) for rp in kept
    ) and not ejected:
        return None
    if anchors[-1].idx - anchors[0].idx < qconfig.min_anchor_span:
        return None
    fit = _ls_fit(anchors)
    if fit is None:
        return None
    slope, intercept = fit
    last_idx = len(quarterly) - 1
    value_at_last = intercept + slope * last_idx
    if value_at_last <= 0:
        return None
    slope_pct_per_year = slope * 4.0 / value_at_last * 100.0

    def value_at(idx: int) -> float:
        return intercept + slope * idx

    breakout = _run_breakout_state_machine(quarterly, value_at, anchors[0].idx, config)
    ordered_points = sorted(kept + ejected, key=lambda rp: rp.point.idx)
    return LidHypothesis(
        window_start=window_start,
        role_points=ordered_points,
        anchors=anchors,
        ejected=ejected,
        slope_per_bar=slope,
        intercept=intercept,
        value_at_last_bar=value_at_last,
        slope_pct_per_year=slope_pct_per_year,
        fit_error_pct=_fit_error_pct(anchors, slope, intercept),
        breakout=breakout,
    )


def _score_lid_hypothesis(
    hypothesis: LidHypothesis,
    quarterly: list[dict[str, Any]],
    qconfig: CoilConfig,
    config: CoilConfig,
    mountains: list[int],
) -> LidHypothesis:
    """Score by the regime-selection rules; a reject_reason marks obsolete fits."""
    last_idx = len(quarterly) - 1
    last_close = float(quarterly[last_idx]["close"])
    slope_pct = hypothesis.slope_pct_per_year

    if not (config.min_slope_pct_per_year <= slope_pct <= config.max_slope_pct_per_year):
        hypothesis.reject_reason = f"slope {slope_pct:.1f}%/yr outside search band"
        return hypothesis
    # Two or more confirmed interior mountains at one level far below the line
    # are their own ceiling regime: the "lid" floats above newer structure. A
    # single deep interior top under a long lid is a normal base (SPG-style).
    structural_idxs = {rp.point.idx for rp in hypothesis.role_points}
    interior_floor = 1.0 - config.regime_interior_major_max_below_pct / 100.0
    plateau_tol = max(0.0, config.display_plateau_tolerance_pct / 100.0)
    interior_below: list[int] = []
    # A provisional right-edge extension is itself the newest structural
    # evidence. Lower interim mountains can be the base underneath that
    # emerging lid (KN); they must not make the hypothesis look like a
    # floating obsolete line before the extension has had a future quarter in
    # which to confirm or fail. Confirmed endpoint fits still receive the
    # interior-ceiling rejection below.
    ends_provisionally = bool(hypothesis.role_points) and (
        hypothesis.role_points[-1].role == ROLE_PROVISIONAL_TOP
    )
    for m_idx in mountains:
        if ends_provisionally:
            break
        if not hypothesis.anchors[0].idx < m_idx < hypothesis.anchors[-1].idx:
            continue
        if m_idx in structural_idxs:
            continue
        line_value = hypothesis.value_at(m_idx)
        if line_value > 0 and float(quarterly[m_idx]["high"]) < line_value * interior_floor:
            interior_below.append(m_idx)
    for i, first in enumerate(interior_below):
        first_price = float(quarterly[first]["high"])
        for second in interior_below[i + 1 :]:
            second_price = float(quarterly[second]["high"])
            scale = max(first_price, second_price)
            if scale > 0 and abs(first_price - second_price) / scale <= plateau_tol:
                hypothesis.reject_reason = (
                    f"interior same-level ceiling at {quarterly[first]['date']} / "
                    f"{quarterly[second]['date']} sits far below the line"
                )
                return hypothesis
    # Mid-history escapes that failed long ago mean price did not respect this
    # lid; a *recent* failed breakout keeps the lid in pre-breakout evaluation.
    stale_cutoff = last_idx - qconfig.broken_out_max_age
    for failure in hypothesis.breakout.failed_breakouts:
        if int(failure["failed"]["q_idx"]) < stale_cutoff:
            hypothesis.reject_reason = (
                f"lid pierced mid-history: failed breakout at {failure['failed']['date']}"
            )
            return hypothesis
    relevance_pct = last_close / hypothesis.value_at_last_bar * 100.0
    # Structural analysis remains available after a sharp pullback from a
    # newly confirmed top. UEC sits just below the normal 50% live-lid floor;
    # 40% is still close enough to preserve the current regime as ``forming``
    # while its steep slope remains ungradeable.
    structural_relevance_floor = min(config.regime_relevance_min_pct, 40.0)
    if relevance_pct < structural_relevance_floor:
        hypothesis.reject_reason = (
            f"price {relevance_pct:.0f}% of lid — line no longer relevant"
        )
        return hypothesis
    if relevance_pct > config.regime_relevance_max_pct:
        hypothesis.reject_reason = (
            f"price {relevance_pct:.0f}% of lid — line long since escaped"
        )
        return hypothesis
    # Touches and recency come from actual quarterly highs meeting the line,
    # not just the fitted anchors — a three-touch flat lid whose latest touch
    # is a minor pivot still reads as freshly relevant.
    touch_tol = config.touch_tolerance_pct / 100.0
    touch_qs: list[int] = []
    for q in range(hypothesis.anchors[0].idx, last_idx + 1):
        line_value = hypothesis.value_at(q)
        if line_value <= 0:
            continue
        if abs(float(quarterly[q]["high"]) - line_value) / line_value <= touch_tol:
            if touch_qs and q - touch_qs[-1] <= qconfig.touch_cluster_bars:
                continue
            touch_qs.append(q)
    quarters_since_touch = last_idx - (touch_qs[-1] if touch_qs else hypothesis.anchors[-1].idx)
    if (
        quarters_since_touch > qconfig.stale_line_months
        and last_close < 0.8 * hypothesis.value_at_last_bar
    ):
        hypothesis.reject_reason = "stale line: no recent touch and price far below"
        return hypothesis
    if hypothesis.fit_error_pct > config.touch_tolerance_pct * 2.0:
        hypothesis.reject_reason = (
            f"fit error {hypothesis.fit_error_pct:.1f}% too large for one straight lid"
        )
        return hypothesis

    span_years = (hypothesis.anchors[-1].idx - hypothesis.anchors[0].idx) / 4.0
    touch_score = min(max(len(touch_qs), len(hypothesis.anchors)), 5) / 5.0
    span_score = _clamp01(span_years / 10.0)
    flatness = _clamp01(1.0 - abs(slope_pct) / config.grade_c_max)
    fit_score = _clamp01(1.0 - hypothesis.fit_error_pct / config.touch_tolerance_pct)
    recency = _clamp01(1.0 - quarters_since_touch / 20.0)
    relevance = _clamp01(1.0 - abs(1.0 - relevance_pct / 100.0))
    score = (
        0.25 * touch_score
        + 0.20 * span_score
        + 0.20 * flatness
        + 0.15 * fit_score
        + 0.10 * recency
        + 0.10 * relevance
    )
    # A lid confirmed-broken far in the past is diagnostics material unless no
    # newer structure exists; rank it below live regimes rather than reject.
    if hypothesis.breakout.state == "broken_out" and hypothesis.breakout.confirmed:
        age = last_idx - int(hypothesis.breakout.confirmed["q_idx"])
        if age > qconfig.broken_out_max_age:
            score *= 0.5
    hypothesis.score = score
    return hypothesis


@dataclass
class QuarterlyStructure:
    """The selected active regime, in both quarterly and monthly coordinates."""

    quarterly: list[dict[str, Any]]
    hypothesis: LidHypothesis
    role_points_monthly: list[RolePoint]
    lid_monthly: ActiveLidFit
    rejected: list[dict[str, Any]]


def _pair_seeded_role_points(
    bars: list[dict[str, Any]],
    quarterly: list[dict[str, Any]],
    pair_fit: ResistanceFit,
    mountains: list[int],
) -> list[RolePoint]:
    """Quarterly role points seeded from the pair-search touches.

    Level-based mountain gates assume a flat ceiling, so valid rising lids can
    lose their early anchors. The pair search is slope-aware; its touch set
    enters regime selection as one more hypothesis under the same gates.
    """
    month_to_q = _month_to_quarter_map(bars, quarterly)
    mountain_set = set(mountains)
    role_points: list[RolePoint] = []
    seen: set[int] = set()
    for touch in pair_fit.touches:
        q_idx = month_to_q.get(touch.idx)
        if q_idx is None or q_idx in seen:
            continue
        seen.add(q_idx)
        role = ROLE_MAJOR_TOP if q_idx in mountain_set else ROLE_STRUCTURAL_RETEST
        role_points.append(
            RolePoint(
                point=SwingPoint(
                    idx=q_idx,
                    date=str(quarterly[q_idx]["date"]),
                    price=touch.price,
                    prominence_pct=touch.prominence_pct,
                ),
                role=role,
                evidence={"seed": "pair_search"},
            )
        )
    return sorted(role_points, key=lambda rp: rp.point.idx)


def _analyze_quarterly_structure(
    bars: list[dict[str, Any]],
    config: CoilConfig = DEFAULT_CONFIG,
    pair_fit: Optional[ResistanceFit] = None,
) -> Optional[QuarterlyStructure]:
    """Full-history quarterly detection + active-regime lid selection."""
    if len(bars) < 2:
        return None
    quarterly = _aggregate_quarterly_display_bars(bars)
    if len(quarterly) < 3:
        return None
    qconfig = _quarterly_scaled_config(config)

    clustered_pivots, mountains = _full_history_mountains(quarterly, qconfig)
    hypotheses: list[LidHypothesis] = []
    seen_anchor_sets: set[tuple[int, ...]] = set()
    for window_start in _regime_window_starts(quarterly, qconfig, clustered_pivots):
        hypothesis = _build_lid_hypothesis(quarterly, qconfig, config, window_start)
        if hypothesis is None:
            continue
        key = tuple(point.idx for point in hypothesis.anchors)
        if key in seen_anchor_sets:
            continue
        seen_anchor_sets.add(key)
        hypotheses.append(
            _score_lid_hypothesis(hypothesis, quarterly, qconfig, config, mountains)
        )
    if pair_fit is None:
        swings = detect_swing_highs(bars, config)
        majors = select_major_highs(swings, config)
        if len(majors) >= 2:
            pair_fit = fit_resistance_line(bars, majors, swings, config)
    if pair_fit is not None:
        seeded = _pair_seeded_role_points(bars, quarterly, pair_fit, mountains)
        hypothesis = _hypothesis_from_role_points(
            quarterly,
            seeded,
            qconfig,
            config,
            window_start=seeded[0].point.idx if seeded else 0,
            require_mountain=False,
        )
        if hypothesis is not None:
            key = tuple(point.idx for point in hypothesis.anchors)
            if key not in seen_anchor_sets:
                seen_anchor_sets.add(key)
                hypotheses.append(
                    _score_lid_hypothesis(hypothesis, quarterly, qconfig, config, mountains)
                )

    # The active regime must account for the latest accepted structure. A lid
    # whose structure ends before newer accepted points exist is a prior
    # regime: keep it in diagnostics, never as the classification line.
    alive = [h for h in hypotheses if h.reject_reason is None]
    if alive:
        latest_structure_idx = max(h.role_points[-1].point.idx for h in alive)
        for h in alive:
            if h.role_points[-1].point.idx < latest_structure_idx - 1:
                h.reject_reason = "superseded: newer accepted structure exists"
        alive = [h for h in alive if h.reject_reason is None]
    # A valid wider regime that explains a strict superset of another
    # hypothesis's structure subsumes it (same points, more history).
    if len(alive) > 1:
        point_sets = {id(h): {rp.point.idx for rp in h.role_points} for h in alive}
        for h in alive:
            for other in alive:
                if other is h or other.reject_reason is not None:
                    continue
                if point_sets[id(other)] > point_sets[id(h)]:
                    added = [
                        rp.point
                        for rp in other.role_points
                        if rp.point.idx not in point_sets[id(h)]
                    ]
                    shared_ceiling = max(
                        rp.point.price
                        for rp in h.role_points
                    )
                    min_comparable_level = shared_ceiling * (
                        config.display_secondary_anchor_min_level_pct / 100.0
                    )
                    # A wider regime only subsumes the newer one when its
                    # additional history belongs to the same ceiling zone.
                    # This keeps SPG/REG-style long lids, but prevents CF's
                    # much lower 2015 shelf from erasing the active 2022+ lid.
                    if any(point.price < min_comparable_level for point in added):
                        continue
                    h.reject_reason = "subsumed by a wider regime with the same structure"
                    break
        alive = [h for h in alive if h.reject_reason is None]
    if not alive:
        return None
    # The reviewed calibration set was graded on the recent display era, so
    # hypotheses whose structure lies inside that horizon form the base pool;
    # accepted structure count beats raw score within it. Full-history regimes
    # win only by explaining strictly more of the same structure (handled by
    # the superset pass above) or when the recent era has no valid structure.
    legacy_start = max(0, len(quarterly) - max(1, qconfig.display_lookback_bars))
    base_pool = [
        h for h in alive if h.role_points[0].point.idx >= legacy_start - 2
    ]
    pool = base_pool if base_pool else alive
    pool.sort(
        key=lambda h: (len(h.role_points), h.score, h.window_start), reverse=True
    )
    active = pool[0]

    rejected = [
        {
            "window_start": h.window_start,
            "anchors": [_point_dict(point) for point in h.anchors],
            "slope_pct_per_year": round(h.slope_pct_per_year, 2),
            "score": round(h.score, 3),
            "reject_reason": h.reject_reason,
        }
        for h in hypotheses
        if h is not active
    ]
    return _finalize_structure(bars, quarterly, active, rejected, config)


def _finalize_structure(
    bars: list[dict[str, Any]],
    quarterly: list[dict[str, Any]],
    active: LidHypothesis,
    rejected: list[dict[str, Any]],
    config: CoilConfig,
) -> Optional[QuarterlyStructure]:
    """Remap the active hypothesis to monthly coordinates and fit its lid."""
    role_points_monthly = [
        _remap_quarterly_role_point(rp, quarterly, bars) for rp in active.role_points
    ]
    anchor_idxs = {point.idx for point in active.anchors}
    monthly_anchors = [
        rp.point
        for rp, qrp in zip(role_points_monthly, active.role_points)
        if qrp.point.idx in anchor_idxs
    ]
    fit = _ls_fit(monthly_anchors)
    if fit is None:
        return None
    slope_m, intercept_m = fit
    last_idx = len(bars) - 1
    value_at_last = intercept_m + slope_m * last_idx
    if value_at_last <= 0:
        return None
    break_tolerance = 1.0 + config.break_tolerance_pct / 100.0
    violations_m = [
        idx
        for idx in range(monthly_anchors[0].idx, last_idx + 1)
        if intercept_m + slope_m * idx > 0
        and float(bars[idx]["close"]) > (intercept_m + slope_m * idx) * break_tolerance
    ]
    lid_monthly = ActiveLidFit(
        points=monthly_anchors,
        slope_per_bar=slope_m,
        intercept=intercept_m,
        value_at_last_bar=value_at_last,
        slope_pct_per_year=slope_m * BARS_PER_YEAR / value_at_last * 100.0,
        violation_idxs=violations_m,
    )
    return QuarterlyStructure(
        quarterly=quarterly,
        hypothesis=active,
        role_points_monthly=role_points_monthly,
        lid_monthly=lid_monthly,
        rejected=rejected,
    )


def _month_to_quarter_map(
    bars: list[dict[str, Any]],
    quarterly: list[dict[str, Any]],
) -> dict[int, int]:
    """Monthly bar index -> index of its containing quarter."""
    mapping: dict[int, int] = {}
    q_iter = 0
    for m_idx, bar in enumerate(bars):
        text = str(bar["date"])
        key = (int(text[:4]), (int(text[5:7]) - 1) // 3 + 1)
        while q_iter < len(quarterly) - 1 and quarterly[q_iter]["_quarter_key"] != key:
            q_iter += 1
        mapping[m_idx] = q_iter
    return mapping


def _structure_from_review_override(
    bars: list[dict[str, Any]],
    override_points: list[dict[str, Any]],
    config: CoilConfig = DEFAULT_CONFIG,
) -> Optional[QuarterlyStructure]:
    """Effective structure from approved human-review points.

    Reviewed points carry authority: no mountain gates, no regime scoring, no
    anchor ejection. Points dated after the analysis window (``as_of`` replay)
    are skipped, so derived slope and status recalculate as new bars arrive.
    Points with role ``breakout_peak`` are plotted but excluded from the fit.
    When any point carries an explicit ``lid_member`` flag (manual line
    anchors from the review UI), only flagged points fit the lid; every
    reviewed top is still plotted. Legacy corrections without membership keep
    fitting all eligible points.
    """
    if len(bars) < 2:
        return None
    quarterly = _aggregate_quarterly_display_bars(bars)
    if len(quarterly) < 3:
        return None
    month_by_prefix = {str(bar["date"])[:7]: idx for idx, bar in enumerate(bars)}
    month_to_q = _month_to_quarter_map(bars, quarterly)

    explicit_membership = any(bool(point.get("lid_member")) for point in override_points)
    role_points: list[RolePoint] = []
    for point in override_points:
        date = str(point.get("date", ""))[:7]
        m_idx = month_by_prefix.get(date)
        if m_idx is None:
            continue
        q_idx = month_to_q[m_idx]
        price = point.get("price")
        role = point.get("role") or ROLE_MAJOR_TOP
        lid_member = bool(point.get("lid_member")) if explicit_membership else None
        role_points.append(
            RolePoint(
                point=SwingPoint(
                    idx=q_idx,
                    date=str(quarterly[q_idx]["date"]),
                    price=float(price) if price is not None else float(bars[m_idx]["high"]),
                    prominence_pct=0.0,
                ),
                role=role,
                evidence={
                    "seed": "human_review",
                    "source_month_idx": m_idx,
                    **({"lid_member": lid_member} if lid_member is not None else {}),
                },
            )
        )
    role_points.sort(key=lambda rp: rp.point.idx)
    anchors = [
        rp.point
        for rp in role_points
        if rp.role != ROLE_BREAKOUT_PEAK
        and (not explicit_membership or rp.evidence.get("lid_member"))
    ]
    if len(anchors) < 2:
        return None
    fit = _ls_fit(anchors)
    if fit is None:
        return None
    slope, intercept = fit
    last_idx = len(quarterly) - 1
    value_at_last = intercept + slope * last_idx
    if value_at_last <= 0:
        return None

    def value_at(idx: int) -> float:
        return intercept + slope * idx

    breakout = _run_breakout_state_machine(quarterly, value_at, anchors[0].idx, config)
    hypothesis = LidHypothesis(
        window_start=anchors[0].idx,
        role_points=role_points,
        anchors=anchors,
        ejected=[rp for rp in role_points if rp.role == ROLE_BREAKOUT_PEAK],
        slope_per_bar=slope,
        intercept=intercept,
        value_at_last_bar=value_at_last,
        slope_pct_per_year=slope * 4.0 / value_at_last * 100.0,
        fit_error_pct=_fit_error_pct(anchors, slope, intercept),
        breakout=breakout,
    )
    return _finalize_structure(bars, quarterly, hypothesis, [], config)


def _resistance_fit_from_active_lid(active_lid: ActiveLidFit) -> ResistanceFit:
    """Adapt a reviewed active lid when the legacy pair search finds no fit."""
    first = active_lid.points[0]
    last = active_lid.points[-1]
    anchor_a = SwingPoint(
        idx=first.idx,
        date=first.date,
        price=active_lid.value_at(first.idx),
        prominence_pct=first.prominence_pct,
    )
    anchor_b = SwingPoint(
        idx=last.idx,
        date=last.date,
        price=active_lid.value_at(last.idx),
        prominence_pct=last.prominence_pct,
    )
    return ResistanceFit(
        anchor_a=anchor_a,
        anchor_b=anchor_b,
        slope_per_bar=active_lid.slope_per_bar,
        value_at_last_bar=active_lid.value_at_last_bar,
        slope_pct_per_year=active_lid.slope_pct_per_year,
        touches=active_lid.points,
        wick_overshoots=0,
        violation_idxs=active_lid.violation_idxs,
        score=0.0,
    )


def _pullback_lows(
    bars: list[dict[str, Any]], fit: ResistanceFit, config: CoilConfig
) -> list[dict[str, Any]]:
    """Deepest low between consecutive lid touches (and after the last touch),
    with its depth below the line, oldest -> newest."""
    lows = [float(b["low"]) for b in bars]
    boundaries = [t.idx for t in fit.touches] + [len(bars) - 1]
    out = []
    for start, end in zip(boundaries, boundaries[1:]):
        lo = start + 1
        hi = end if end == len(bars) - 1 else end - 1
        if hi - lo + 1 < config.min_pullback_bars:
            continue
        x = min(range(lo, hi + 1), key=lambda k: lows[k])
        line_value = fit.value_at(x)
        if line_value <= 0:
            continue
        depth_pct = (line_value - lows[x]) / line_value * 100.0
        out.append(
            {
                "idx": x,
                "date": str(bars[x]["date"]),
                "price": round(lows[x], 4),
                "depth_pct": round(depth_pct, 2),
            }
        )
    return out


def grade_for_slope(slope_pct_per_year: float, config: CoilConfig = DEFAULT_CONFIG) -> Optional[str]:
    if slope_pct_per_year < config.grade_min or slope_pct_per_year >= config.grade_c_max:
        return None
    if config.grade_a_min <= slope_pct_per_year < config.grade_a_max:
        return "A"
    if slope_pct_per_year < config.grade_b_max:
        return "B"
    return "C"


def _point_dict(point: SwingPoint) -> dict[str, Any]:
    return {"idx": point.idx, "date": point.date, "price": round(point.price, 4)}


def _lifecycle_to_status(lifecycle: str) -> str:
    """Map v2 lifecycle buckets onto the legacy status vocabulary."""
    return {
        LIFECYCLE_NO_STRUCTURE: "no_structure",
        LIFECYCLE_FORMING: "basing",
        LIFECYCLE_PRE_BREAKOUT: "coiling",
        LIFECYCLE_BREAKING_OUT: "breaking_out",
        LIFECYCLE_POST_BREAKOUT: "broken_out",
    }[lifecycle]


def _role_point_dict(rp: RolePoint, lid_member: bool) -> dict[str, Any]:
    return {
        "idx": rp.point.idx,
        "date": rp.point.date,
        "price": round(rp.point.price, 4),
        "prominence_pct": round(rp.point.prominence_pct, 2),
        "role": rp.role,
        "confirmed": rp.role != ROLE_PROVISIONAL_TOP,
        "lid_member": lid_member,
        "source": SOURCE,
        "evidence": rp.evidence,
    }


def analyze_coil(
    bars: Iterable[dict[str, Any]],
    config: CoilConfig = DEFAULT_CONFIG,
    as_of: Optional[str] = None,
    review_override: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Full analysis of one monthly bar series. JSON-ready dict, schema v2.

    Detection runs over the complete history in quarterly coordinates; the
    active lid is regime-selected and everything downstream — grading,
    compression, proximity, breakout lifecycle, score — derives from that one
    line. ``lifecycle``: no_structure | forming | pre_breakout | breaking_out
    | post_breakout. ``status`` keeps the legacy vocabulary. ``grade`` (A/B/C)
    is set only when the lid is valid and every coil gate passes; ``notes``
    explains gates that failed or the grade rationale. The obsolete pair
    search is reported under ``diagnostics`` only.
    """
    clean = _clean_bars(bars, as_of)
    last_bar_date = clean[-1]["date"] if clean else as_of
    quarterly = _aggregate_quarterly_display_bars(clean) if clean else []
    incomplete_last_quarter = bool(quarterly) and not _quarter_is_complete(quarterly[-1])
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "source": SOURCE,
        "as_of": last_bar_date,
        "bar_count": len(clean),
        "analysis_metadata": {
            "history_start": clean[0]["date"] if clean else None,
            "history_end": last_bar_date,
            "bar_count_monthly": len(clean),
            "bar_count_quarterly": len(quarterly),
            "analysis_interval": ANALYSIS_INTERVAL,
            "algorithm_version": ALGORITHM_VERSION,
            "data_freshness": {
                "last_bar_date": last_bar_date,
                "incomplete_last_quarter": incomplete_last_quarter,
            },
        },
        "lifecycle": LIFECYCLE_NO_STRUCTURE,
        "status": "no_structure",
        "grade": None,
        "coil_score": 0.0,
        "points": [],
        "major_highs": [],
        "active_lid": None,
        "breakout": None,
        "review": {"reviewed": False, "effective": "algorithm"},
        "resistance": None,
        "support": None,
        "metrics": None,
        "diagnostics": {"pair_search": None, "rejected_hypotheses": []},
        "notes": [],
    }
    if len(clean) < config.min_bars:
        result["notes"].append(f"insufficient history: {len(clean)} bars < {config.min_bars}")
        return result

    # Legacy pair search: kept as a diagnostic so nothing can plot a different
    # line from the one used for classification.
    swings = detect_swing_highs(clean, config)
    resistance_candidates = select_major_highs(swings, config)
    pair_fit = (
        fit_resistance_line(clean, resistance_candidates, swings, config)
        if len(resistance_candidates) >= 2
        else None
    )
    if pair_fit is not None:
        result["diagnostics"]["pair_search"] = {
            "from": _point_dict(pair_fit.anchor_a),
            "to": _point_dict(pair_fit.anchor_b),
            "slope_pct_per_year": round(pair_fit.slope_pct_per_year, 2),
            "value_at_last_bar": round(pair_fit.value_at_last_bar, 4),
            "touch_count": pair_fit.touch_count,
            "fit_score": round(pair_fit.score, 3),
            "lid_grade": grade_for_slope(pair_fit.slope_pct_per_year, config),
        }

    structure = _analyze_quarterly_structure(clean, config, pair_fit=pair_fit)

    # Approved human reviews override the algorithm's structure. The raw
    # algorithm result is retained for comparison and future calibration;
    # slope, grade, and lifecycle are recomputed from the reviewed anchors.
    if review_override and review_override.get("points"):
        overridden = _structure_from_review_override(
            clean, review_override["points"], config
        )
        if overridden is not None:
            algorithm_summary = None
            if structure is not None:
                algorithm_summary = {
                    "major_highs": [
                        rp.point.to_dict() for rp in structure.role_points_monthly
                    ],
                    "slope_pct_per_year": round(
                        structure.lid_monthly.slope_pct_per_year, 2
                    ),
                    "breakout_state": structure.hypothesis.breakout.state,
                }
            result["review"] = {
                "reviewed": True,
                "effective": "human",
                "review_id": review_override.get("review_id"),
                "updated_at": review_override.get("updated_at"),
                "algorithm": algorithm_summary,
            }
            structure = overridden

    if structure is None:
        result["notes"].append(
            "no active lid: no well-separated anchor set survived regime selection"
        )
        return result

    active = structure.hypothesis
    lid = structure.lid_monthly
    sm = active.breakout
    result["diagnostics"]["rejected_hypotheses"] = structure.rejected

    last_idx = len(clean) - 1
    last_close = float(clean[last_idx]["close"])
    lid_member_idxs = {point.idx for point in lid.points}

    # Every meaningful touch of the active lid: the fitted anchors plus minor
    # swing highs inside the regime that land within the touch tolerance.
    touch_tol = config.touch_tolerance_pct / 100.0
    extra_touches: list[SwingPoint] = []
    for swing in swings:
        if swing.idx in lid_member_idxs or swing.idx < lid.points[0].idx:
            continue
        if swing.prominence_pct < config.minor_prominence_pct:
            continue
        line_value = lid.value_at(swing.idx)
        if line_value > 0 and abs(swing.price - line_value) / line_value <= touch_tol:
            extra_touches.append(swing)
    touches = _cluster_touches(
        sorted(list(lid.points) + extra_touches, key=lambda s: s.idx),
        config.touch_cluster_bars,
    )

    compat_fit = _resistance_fit_from_active_lid(lid)
    compat_fit.touches = touches
    compat_fit.score = active.score

    # points: the active structure with roles, then non-anchor lid touches.
    role_points = list(structure.role_points_monthly)
    role_point_idxs = {rp.point.idx for rp in role_points}
    points_out = [
        _role_point_dict(rp, rp.point.idx in lid_member_idxs) for rp in role_points
    ]
    for touch in touches:
        if touch.idx in role_point_idxs:
            continue
        points_out.append(
            _role_point_dict(
                RolePoint(point=touch, role=ROLE_STRUCTURAL_RETEST, evidence={}),
                lid_member=False,
            )
        )
    result["points"] = sorted(points_out, key=lambda p: p["idx"])
    max_highs = max(0, config.display_max_highs)
    result["major_highs"] = [rp.point.to_dict() for rp in role_points][-max_highs:]

    span_years = compat_fit.span_bars / BARS_PER_YEAR
    lid_grade = grade_for_slope(lid.slope_pct_per_year, config)
    result["resistance"] = {
        "from": _point_dict(compat_fit.anchor_a),
        "to": _point_dict(compat_fit.anchor_b),
        "slope_per_bar": round(lid.slope_per_bar, 6),
        "slope_pct_per_year": round(lid.slope_pct_per_year, 2),
        "value_at_last_bar": round(lid.value_at_last_bar, 4),
        "touch_count": len(touches),
        "touches": [_point_dict(t) for t in reversed(touches)],  # newest first
        "span_years": round(span_years, 2),
        "wick_overshoots": 0,
        "fit_score": round(active.score, 3),
        # Slope band of the lid alone (A/B/C/None) — how the team grades a
        # chart. The top-level ``grade`` additionally requires the coil gates
        # (sealed, wound, loaded), so a B lid mid-base shows lid_grade "B"
        # with grade None until price presses the line.
        "lid_grade": lid_grade,
        "source": SOURCE,
        "classification_lid": {
            "points": [_point_dict(point) for point in lid.points],
            "slope_per_bar": round(lid.slope_per_bar, 6),
            "slope_pct_per_year": round(lid.slope_pct_per_year, 2),
            "value_at_last_bar": round(lid.value_at_last_bar, 4),
            "violation_count": len(lid.violation_idxs),
            "source": "active_lid",
        },
    }
    result["active_lid"] = {
        "from": _point_dict(compat_fit.anchor_a),
        "to": _point_dict(compat_fit.anchor_b),
        "anchors": [_point_dict(point) for point in lid.points],
        "slope_per_bar": round(lid.slope_per_bar, 6),
        "slope_pct_per_year": round(lid.slope_pct_per_year, 2),
        "grade": lid_grade,
        "tolerance_band_pct": round(sm.band_pct, 2),
        "fit_error_pct": round(active.fit_error_pct, 3),
        "value_at_last_bar": round(lid.value_at_last_bar, 4),
        "projected": {
            "idx": last_idx,
            "date": str(clean[last_idx]["date"]),
            "price": round(lid.value_at_last_bar, 4),
        },
        "touches": [_point_dict(t) for t in touches],  # oldest -> newest
        "touch_count": len(touches),
        "span_years": round(span_years, 2),
        "source": SOURCE,
    }

    proximity_pct = last_close / lid.value_at_last_bar * 100.0
    pressed = proximity_pct >= config.pressing_proximity_pct

    pullbacks = _pullback_lows(clean, compat_fit, config)
    depths = [p["depth_pct"] for p in pullbacks]
    if len(depths) >= 2:
        last_depth_ratio = depths[-1] / max(depths) if max(depths) > 0 else None
        depth_rule = last_depth_ratio is not None and last_depth_ratio <= config.max_last_depth_ratio
    elif len(depths) == 1:
        last_depth_ratio = None
        depth_rule = depths[0] <= config.single_pullback_max_depth_pct
    else:
        last_depth_ratio = None
        depth_rule = True  # touches so tight there is no measurable pullback
    compression_ok = depth_rule or pressed

    if len(pullbacks) >= 2:
        first_low, last_low = pullbacks[0], pullbacks[-1]
        support_slope = (last_low["price"] - first_low["price"]) / (last_low["idx"] - first_low["idx"])
        support_value_last = last_low["price"] + support_slope * (last_idx - last_low["idx"])
        support_slope_pct = (
            support_slope * BARS_PER_YEAR / support_value_last * 100.0 if support_value_last > 0 else None
        )
        result["support"] = {
            "from": {k: first_low[k] for k in ("idx", "date", "price")},
            "to": {k: last_low[k] for k in ("idx", "date", "price")},
            "slope_pct_per_year": round(support_slope_pct, 2) if support_slope_pct is not None else None,
            "converging": support_slope_pct is not None and support_slope_pct > lid.slope_pct_per_year,
            "source": SOURCE,
        }

    first_anchor_idx = lid.points[0].idx
    base_years = (last_idx - first_anchor_idx) / BARS_PER_YEAR
    base_trend_r2 = _log_trend_r2([float(b["close"]) for b in clean[first_anchor_idx:]])

    volumes = [b.get("volume") for b in clean[first_anchor_idx:]]
    known = [v for v in volumes if v]
    volume_ratio = None
    if len(known) >= max(12, int(0.6 * len(volumes))):
        third = max(1, len(known) // 3)
        early = sum(known[:third]) / third
        late = sum(known[-12:]) / min(12, len(known))
        if early > 0:
            volume_ratio = late / early

    result["metrics"] = {
        "proximity_pct": round(proximity_pct, 2),
        "base_years": round(base_years, 2),
        "pullback_depths_pct": depths,
        "pullback_lows": pullbacks,
        "compression_ok": compression_ok,
        "pressed_at_lid": pressed,
        "last_depth_ratio": round(last_depth_ratio, 3) if last_depth_ratio is not None else None,
        "volume_contraction_ratio": round(volume_ratio, 3) if volume_ratio is not None else None,
        "violation_count": len(lid.violation_idxs),
        "base_trend_r2": round(base_trend_r2, 3) if base_trend_r2 is not None else None,
    }

    result["breakout"] = {
        "state": sm.state,
        "band_pct": round(sm.band_pct, 2),
        "first_escape": sm.first_escape,
        "confirmed": sm.confirmed,
        "provisional_escape": sm.provisional_escape,
        "failed_breakouts": sm.failed_breakouts,
        "expansion_pct": sm.expansion_pct,
        "post_break_slope_pct_per_year": sm.post_break_slope_pct_per_year,
        "retest": sm.retest,
    }

    sealed = sm.state == "sealed"
    grade = lid_grade
    failed = []
    if grade is None:
        failed.append(f"lid slope {lid.slope_pct_per_year:.1f}%/yr outside gradeable range")
    if not compression_ok:
        failed.append("pullbacks not compressing (no higher-lows squeeze)")
    if proximity_pct < config.min_proximity_pct and sealed:
        failed.append(f"last close {proximity_pct:.0f}% of lid, below {config.min_proximity_pct:.0f}% proximity gate")
    if base_trend_r2 is not None and base_trend_r2 >= config.max_base_trend_r2:
        failed.append(f"steady trend in base (log R2 {base_trend_r2:.2f}) — trending, not coiling")

    if sm.state == "broken_out":
        lifecycle = LIFECYCLE_POST_BREAKOUT
        result["grade"] = None
        result["notes"].append("lid broken and confirmed by the quarterly state machine")
    elif sm.state == "breaking_out":
        lifecycle = LIFECYCLE_BREAKING_OUT
        if failed:
            result["grade"] = None
            result["notes"].extend(failed)
        else:
            result["grade"] = grade
    elif failed:
        lifecycle = LIFECYCLE_FORMING
        result["grade"] = None
        result["notes"].extend(failed)
    else:
        lifecycle = LIFECYCLE_PRE_BREAKOUT
        result["grade"] = grade
    result["lifecycle"] = lifecycle
    result["status"] = _lifecycle_to_status(lifecycle)
    if sm.provisional_escape is not None:
        result["notes"].append(
            "partial quarter closed above the band — provisional only, not a breakout"
        )
    if result["grade"] is not None:
        result["notes"].append(
            f"grade {result['grade']}: lid {lid.slope_pct_per_year:+.1f}%/yr, "
            f"{len(touches)} touches over {span_years:.1f}y"
        )
        if pressed and not depth_rule:
            result["notes"].append(f"pressed at lid ({proximity_pct:.0f}% of line)")
        if depths:
            result["notes"].append(
                "pullback depths " + " -> ".join(f"{d:.0f}%" for d in depths)
            )

    flat = _clamp01(1.0 - abs(lid.slope_pct_per_year) / config.grade_c_max)
    touch_score = min(len(touches), 5) / 5.0
    compress = (
        _clamp01(1.0 - last_depth_ratio) if last_depth_ratio is not None else (0.5 if compression_ok else 0.0)
    )
    prox = _clamp01((proximity_pct - 60.0) / 40.0)
    span_score = _clamp01(span_years / 10.0)
    result["coil_score"] = round(
        100.0 * (0.30 * flat + 0.20 * touch_score + 0.20 * compress + 0.15 * prox + 0.15 * span_score), 1
    )
    return result


# ---------------------------------------------------------------------------
# CLI: run the analyzer over tickers or a saved screener run, like the vision
# pipeline does. Uses the cache-first history path so repeated runs are free.
# ---------------------------------------------------------------------------


def _tickers_from_csv(path: Path, limit: Optional[int]) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    tickers = [row["ticker"].strip().upper() for row in rows if row.get("ticker", "").strip()]
    return tickers[:limit] if limit else tickers


def _run_cli(args: argparse.Namespace) -> None:
    import history_cache
    import screen_monthly

    if args.saved_run:
        tickers = _tickers_from_csv(Path(args.saved_run), args.limit)
    else:
        tickers = [t.strip().upper() for t in args.tickers if t.strip()]
    if not tickers:
        print("No tickers given.")
        return

    json_dir = Path(args.json_dir) if args.json_dir else None
    if json_dir:
        json_dir.mkdir(parents=True, exist_ok=True)

    header = f"{'ticker':<8}{'status':<14}{'grade':<7}{'lid':<5}{'slope%/yr':>10}{'touches':>9}{'base_y':>8}{'prox%':>8}{'score':>8}"
    print(header)
    print("-" * len(header))
    for ticker in tickers:
        payload = history_cache.get_history_payload(
            ticker, screen_monthly.fetch_monthly_history, screen_monthly.compute_features
        )
        if payload is None:
            print(f"{ticker:<8}{'no data':<14}")
            continue
        analysis = analyze_coil(payload["bars"], as_of=args.as_of)
        res = analysis.get("resistance") or {}
        metrics = analysis.get("metrics") or {}
        print(
            f"{ticker:<8}{analysis['status']:<14}{analysis['grade'] or '-':<7}"
            f"{res.get('lid_grade') or '-':<5}"
            f"{res.get('slope_pct_per_year', float('nan')):>10.2f}"
            f"{res.get('touch_count', 0):>9}"
            f"{metrics.get('base_years', float('nan')):>8.1f}"
            f"{metrics.get('proximity_pct', float('nan')):>8.1f}"
            f"{analysis['coil_score']:>8.1f}"
        )
        if json_dir:
            with (json_dir / f"{ticker}.json").open("w", encoding="utf-8") as fh:
                json.dump({"ticker": ticker, **analysis}, fh, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic coil-structure analysis over monthly bars")
    parser.add_argument("tickers", nargs="*", help="Tickers to analyze")
    parser.add_argument("--saved-run", help="Screener results CSV to pull tickers from")
    parser.add_argument("--limit", type=int, help="Cap tickers taken from --saved-run")
    parser.add_argument("--as-of", help="Truncate history at this date (YYYY-MM-DD) for backtest checks")
    parser.add_argument("--json-dir", help="Write per-ticker analysis JSON into this directory")
    return parser


if __name__ == "__main__":
    _run_cli(build_parser().parse_args())
