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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 1
SOURCE = "timeseries"
BARS_PER_YEAR = 12.0  # module operates on monthly bars


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
    # Slope grading (percent per year of line value at the last bar)
    grade_a_min: float = -1.0
    grade_a_max: float = 2.0
    grade_b_max: float = 6.0
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


def analyze_coil(
    bars: Iterable[dict[str, Any]],
    config: CoilConfig = DEFAULT_CONFIG,
    as_of: Optional[str] = None,
) -> dict[str, Any]:
    """Full analysis of one monthly bar series. JSON-ready dict.

    ``status``: no_structure | basing | coiling | breaking_out | broken_out.
    ``grade`` (A/B/C) is set only when the lid is valid and every coil gate
    passes; ``notes`` explains gates that failed or the grade rationale.
    """
    clean = _clean_bars(bars, as_of)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "as_of": clean[-1]["date"] if clean else as_of,
        "bar_count": len(clean),
        "status": "no_structure",
        "grade": None,
        "coil_score": 0.0,
        "major_highs": [],
        "resistance": None,
        "support": None,
        "metrics": None,
        "notes": [],
    }
    if len(clean) < config.min_bars:
        result["notes"].append(f"insufficient history: {len(clean)} bars < {config.min_bars}")
        return result

    swings = detect_swing_highs(clean, config)
    majors = select_major_highs(swings, config)
    result["major_highs"] = [s.to_dict() for s in majors]
    if len(majors) < 2:
        result["notes"].append(f"need >=2 major tops, found {len(majors)}")
        return result

    fit = fit_resistance_line(clean, majors, swings, config)
    if fit is None:
        result["notes"].append("no resistance line: no major-top pair with clean touches survived")
        return result

    last_idx = len(clean) - 1
    last_close = float(clean[last_idx]["close"])
    span_years = fit.span_bars / BARS_PER_YEAR
    result["resistance"] = {
        "from": _point_dict(fit.anchor_a),
        "to": _point_dict(fit.anchor_b),
        "slope_per_bar": round(fit.slope_per_bar, 6),
        "slope_pct_per_year": round(fit.slope_pct_per_year, 2),
        "value_at_last_bar": round(fit.value_at_last_bar, 4),
        "touch_count": fit.touch_count,
        "touches": [_point_dict(t) for t in reversed(fit.touches)],  # newest first
        "span_years": round(span_years, 2),
        "wick_overshoots": fit.wick_overshoots,
        "fit_score": round(fit.score, 3),
        # Slope band of the lid alone (A/B/C/None) — how the team grades a
        # chart. The top-level ``grade`` additionally requires the coil gates
        # (sealed, wound, loaded), so a B lid mid-base shows lid_grade "B"
        # with grade None until price presses the line.
        "lid_grade": grade_for_slope(fit.slope_pct_per_year, config),
        "source": SOURCE,
    }

    proximity_pct = last_close / fit.value_at_last_bar * 100.0
    pressed = proximity_pct >= config.pressing_proximity_pct

    pullbacks = _pullback_lows(clean, fit, config)
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
            "converging": support_slope_pct is not None and support_slope_pct > fit.slope_pct_per_year,
            "source": SOURCE,
        }

    base_years = (last_idx - fit.first_touch_idx) / BARS_PER_YEAR
    base_trend_r2 = _log_trend_r2([float(b["close"]) for b in clean[fit.first_touch_idx :]])

    volumes = [b.get("volume") for b in clean[fit.first_touch_idx :]]
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
        "violation_count": len(fit.violation_idxs),
        "base_trend_r2": round(base_trend_r2, 3) if base_trend_r2 is not None else None,
    }

    # Status: violations only exist inside the tolerated tail by construction.
    if fit.violation_idxs:
        oldest_age = last_idx - min(fit.violation_idxs)
        sealed = False
        status = "breaking_out" if oldest_age < config.breakout_confirm_bars else "broken_out"
    else:
        sealed = True
        status = "coiling"

    grade = grade_for_slope(fit.slope_pct_per_year, config)
    failed = []
    if grade is None:
        failed.append(f"lid slope {fit.slope_pct_per_year:.1f}%/yr outside gradeable range")
    if not compression_ok:
        failed.append("pullbacks not compressing (no higher-lows squeeze)")
    if proximity_pct < config.min_proximity_pct and sealed:
        failed.append(f"last close {proximity_pct:.0f}% of lid, below {config.min_proximity_pct:.0f}% proximity gate")
    if base_trend_r2 is not None and base_trend_r2 >= config.max_base_trend_r2:
        failed.append(f"steady trend in base (log R2 {base_trend_r2:.2f}) — trending, not coiling")

    if status == "broken_out":
        result["status"] = status
        result["grade"] = None
        result["notes"].append("lid already broken: closes above the line beyond the confirm window")
    elif failed:
        result["status"] = "basing" if sealed else status
        result["grade"] = None
        result["notes"].extend(failed)
    else:
        result["status"] = status
        result["grade"] = grade
        result["notes"].append(
            f"grade {grade}: lid {fit.slope_pct_per_year:+.1f}%/yr, "
            f"{fit.touch_count} touches over {span_years:.1f}y"
        )
        if pressed and not depth_rule:
            result["notes"].append(f"pressed at lid ({proximity_pct:.0f}% of line)")
        if depths:
            result["notes"].append(
                "pullback depths " + " -> ".join(f"{d:.0f}%" for d in depths)
            )

    flat = _clamp01(1.0 - abs(fit.slope_pct_per_year) / config.grade_c_max)
    touch_score = min(fit.touch_count, 5) / 5.0
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
