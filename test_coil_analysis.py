"""Behavior tests for the deterministic coil-structure analyzer.

Synthetic monthly series are built from piecewise-linear close paths with
exact lid touches (bar high == line value) and exact pullback lows, so the
expected slope, grade, touch count, and depth sequence are known by
construction.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from coil_analysis import (
    ALGORITHM_VERSION,
    DEFAULT_CONFIG,
    PRICE_POSITION_ABOVE,
    PRICE_POSITION_BELOW,
    PRICE_POSITION_WITHIN,
    ROLE_PROVISIONAL_TOP,
    LidZone,
    SwingPoint,
    _aggregate_quarterly_display_bars,
    _cluster_mountain_plateaus,
    _cluster_price_zones,
    _cluster_touches,
    _completed_quarters,
    _detect_display_major_highs_on_bars,
    _mountain_prominence,
    _pivot_high_indexes,
    _price_position,
    _quarterly_scaled_config,
    _rank_lid_zones,
    _select_lid_zone,
    _suppress_slower_comparable_retests,
    _tickers_from_csv,
    _zone_reject_reason,
    analyze_coil,
    detect_display_major_highs,
    detect_swing_highs,
    grade_for_slope,
    select_major_highs,
)


VALIDATION_FEEDBACK = json.loads(
    (Path(__file__).parent / "validation" / "major_high_feedback.json").read_text(
        encoding="utf-8"
    )
)["reviews"]


def month_dates(n: int, start_year: int = 2010, start_month: int = 1) -> list[str]:
    out = []
    for k in range(n):
        month0 = start_month - 1 + k
        out.append(f"{start_year + month0 // 12:04d}-{month0 % 12 + 1:02d}-01")
    return out


def piecewise_closes(keys: list[tuple[int, float]], n: int) -> list[float]:
    keys = sorted(keys)
    assert keys[0][0] == 0 and keys[-1][0] == n - 1
    out = [0.0] * n
    for (x0, y0), (x1, y1) in zip(keys, keys[1:]):
        for x in range(x0, x1 + 1):
            out[x] = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return out


def bars_from_closes(
    closes: list[float],
    high_overrides: dict[int, float] | None = None,
    low_overrides: dict[int, float] | None = None,
) -> list[dict]:
    dates = month_dates(len(closes))
    bars = []
    for x, c in enumerate(closes):
        bars.append(
            {
                "date": dates[x],
                "open": c,
                "high": c * 1.005,
                "low": c * 0.995,
                "close": c,
                "volume": 1_000_000.0,
            }
        )
    for idx, value in (high_overrides or {}).items():
        bars[idx]["high"] = value
    for idx, value in (low_overrides or {}).items():
        bars[idx]["low"] = value
    return bars


def make_coil_bars(
    n: int = 120,
    touches: tuple[int, ...] = (20, 60, 100),
    lid_start: float = 100.0,
    slope_per_bar: float = 0.0,
    depths: tuple[float, ...] = (0.50, 0.40, 0.30),
    start_frac: float = 0.65,
    end_frac: float = 0.96,
    end_touch: bool = True,
) -> list[dict]:
    """Coil with the lid passing exactly through each touch bar's high.

    The default pullbacks are deep enough (>= the 28% display prominence gate)
    that every interior touch is a confirmed quarterly mountain, so the lid
    zone is known by construction. ``end_touch`` wicks the final bar up to the
    lid without moving its close, which is what a coil pressing its ceiling
    looks like and what lets the latest completed quarter join the zone.
    """
    t0 = touches[0]

    def lid(x: int) -> float:
        return lid_start + slope_per_bar * (x - t0)

    keys = [(0, lid(0) * start_frac)]
    high_overrides: dict[int, float] = {}
    low_overrides: dict[int, float] = {}
    for i, t in enumerate(touches):
        keys.append((t, lid(t) * 0.99))
        high_overrides[t] = lid(t)
        nxt = touches[i + 1] if i + 1 < len(touches) else n - 1
        mid = (t + nxt) // 2
        if t < mid < n - 1 and i < len(depths):
            dip = lid(mid) * (1.0 - depths[i])
            keys.append((mid, dip))
            low_overrides[mid] = dip
    keys.append((n - 1, lid(n - 1) * end_frac))
    if end_touch:
        high_overrides[n - 1] = lid(n - 1)
    return bars_from_closes(piecewise_closes(keys, n), high_overrides, low_overrides)


def lid_override(bars: list[dict], idxs: tuple[int, ...]) -> dict:
    """An approved human review pinning ``idxs`` as the lid anchors."""
    return {
        "review_id": "rev-test",
        "updated_at": "2026-07-26T00:00:00Z",
        "points": [
            {
                "date": bars[idx]["date"],
                "price": bars[idx]["high"],
                "role": "major_top",
                "lid_member": True,
            }
            for idx in idxs
        ],
    }


def zone_from(prices: dict[int, float]) -> LidZone:
    """A LidZone from {quarterly_idx: price}, level = mean of member prices."""
    members = tuple(
        SwingPoint(idx=idx, date=f"q{idx}", price=price, prominence_pct=30.0)
        for idx, price in sorted(prices.items())
    )
    return LidZone(
        level=sum(m.price for m in members) / len(members),
        members=members,
    )


def quarterly_bars(highs: list[float], last_month: int = 12) -> list[dict]:
    """Minimal quarterly bar dicts for the zone helpers under test."""
    out = []
    for idx, high in enumerate(highs):
        out.append(
            {
                "_quarter_key": (2000 + idx // 4, idx % 4 + 1),
                "_high_source_idx": idx,
                "_close_source_idx": idx,
                "_last_month": 12 if idx < len(highs) - 1 else last_month,
                "date": f"{2000 + idx // 4:04d}-{(idx % 4) * 3 + 1:02d}-01",
                "open": high * 0.9,
                "high": high,
                "low": high * 0.7,
                "close": high * 0.9,
                "volume": 1_000_000.0,
            }
        )
    return out


def spb_for_end_slope(pct: float, lid_start: float = 100.0, t0: int = 20, n: int = 120) -> float:
    """Slope per bar so the end-normalized slope equals ``pct`` %/yr exactly."""
    return pct * lid_start / (1200.0 - pct * (n - 1 - t0))


# ---------------------------------------------------------------------------
# Unit pieces
# ---------------------------------------------------------------------------


def test_pivot_plateau_resolves_to_first_bar():
    highs = [1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0]
    assert _pivot_high_indexes(highs, left=2, right=2) == [2]


def test_monotonic_rise_has_no_pivots():
    highs = [float(i) for i in range(20)]
    assert _pivot_high_indexes(highs, left=3, right=3) == []


# Recalibrated against the reviewed 18-ticker set: A < 5%/yr, B < 6.5%/yr,
# C < 12%/yr, -1%/yr A-grade falling floor, -3%/yr overall validity floor.
@pytest.mark.parametrize(
    ("slope", "expected"),
    [
        (-5.0, None),
        (-2.0, "B"),
        (-0.5, "A"),
        (0.0, "A"),
        (1.9, "A"),
        (4.9, "A"),
        (5.0, "B"),
        (6.4, "B"),
        (6.5, "C"),
        (11.9, "C"),
        (12.0, None),
        (25.0, None),
    ],
)
def test_grade_bands(slope, expected):
    assert grade_for_slope(slope) == expected


def test_cluster_touches_merges_adjacent_keeps_highest():
    touches = [
        SwingPoint(10, "2010-11-01", 99.0, 10.0),
        SwingPoint(12, "2011-01-01", 100.5, 10.0),
        SwingPoint(30, "2012-07-01", 100.0, 10.0),
    ]
    reps = _cluster_touches(touches, max_gap=6)
    assert [r.idx for r in reps] == [12, 30]
    assert reps[0].price == 100.5


def test_major_separation_keeps_higher_prominence():
    swings = [
        SwingPoint(50, "2014-03-01", 100.0, 20.0),
        SwingPoint(54, "2014-07-01", 99.0, 30.0),
        SwingPoint(80, "2016-09-01", 98.0, 25.0),
    ]
    majors = select_major_highs(swings, DEFAULT_CONFIG)
    assert [m.idx for m in majors] == [54, 80]


def test_plateau_upgrade_requires_an_adjacent_higher_candle():
    # NSC: the higher representative is two quarterly candles away, so it
    # replaces the earlier plateau bar. UEC: the retest is three candles away,
    # so the original same-zone ceiling remains the representative.
    highs = [100.0, 80.0, 102.0, 107.0]

    adjacent = _cluster_mountain_plateaus(
        [0, 2],
        highs,
        max_gap=4,
        tolerance=0.08,
        upgrade_tolerance=0.01,
        upgrade_max_gap=2,
    )
    distant = _cluster_mountain_plateaus(
        [0, 3],
        highs,
        max_gap=4,
        tolerance=0.08,
        upgrade_tolerance=0.01,
        upgrade_max_gap=2,
    )

    assert adjacent == [2]
    assert distant == [0]

    restored_ceiling = _cluster_mountain_plateaus(
        [0, 3, 6],
        [100.0, 80.0, 80.0, 95.0, 80.0, 80.0, 103.0],
        max_gap=4,
        tolerance=0.08,
        upgrade_tolerance=0.01,
        upgrade_max_gap=2,
    )
    assert restored_ceiling == [0, 6]


def test_slower_later_comparable_retest_is_suppressed():
    points = [
        SwingPoint(2, "2020-03-01", 100.0, 35.0),
        SwingPoint(6, "2021-03-01", 103.0, 35.0),
        SwingPoint(10, "2022-03-01", 120.0, 35.0),
    ]
    closes = [80.0, 90.0, 100.0, 70.0, 72.0, 90.0, 102.0, 95.0, 94.0, 100.0, 118.0]

    kept = _suppress_slower_comparable_retests(
        points,
        closes,
        edge_start=10,
        config=DEFAULT_CONFIG,
    )

    assert [point.idx for point in kept] == [2, 10]


def test_display_majors_never_include_the_unconfirmed_right_edge():
    # v2.2: the same chart that used to surface bar 139 as a provisional
    # right-edge top. Bar 139 is the highest bar on the chart and has risen
    # ~97% from the valley at 126, so every old right-edge gate would admit
    # it — but it has no future candles, so it is not structure.
    n = 140
    closes = piecewise_closes(
        [
            (0, 40.0),
            (20, 80.0),
            (32, 48.0),
            (48, 105.0),
            (62, 65.0),
            (80, 180.0),
            (81, 181.0),
            (96, 115.0),
            (110, 260.0),
            (126, 160.0),
            (139, 315.0),
        ],
        n,
    )
    bars = bars_from_closes(
        closes,
        high_overrides={20: 82.0, 48: 108.0, 80: 182.0, 81: 183.0, 110: 260.0, 139: 315.0},
    )

    majors = _detect_display_major_highs_on_bars(bars)

    assert [point.idx for point in majors] == [80, 110]
    assert [point.price for point in majors] == [182.0, 260.0]
    assert all(point.idx < 139 for point in majors)


def test_display_majors_reject_lower_shoulders_after_level_is_established():
    n = 120
    bars = make_coil_bars(
        n=n,
        touches=(10, 50, 90),
        lid_start=100.0,
        depths=(0.40, 0.35, 0.30),
        end_frac=0.99,
    )
    # A deep but lower rebound after the 100 ceiling is still a shoulder, not
    # a new major top once the established level is higher.
    bars[75]["high"] = 85.0
    bars[75]["open"] = 82.0
    bars[75]["close"] = 82.0
    bars[75]["low"] = 80.0

    majors = _detect_display_major_highs_on_bars(bars)

    prices = [point.price for point in majors]
    assert prices == sorted(prices)
    assert 85.0 not in prices


def test_display_mountain_requires_a_valley_on_both_sides():
    # The 100 bar has a deep rise on its left, but price proceeds directly to
    # 115 before any fall. It is a shoulder on the later mountain, not a peak.
    highs = [50.0, 60.0, 100.0, 115.0, 90.0, 70.0]
    lows = [45.0, 55.0, 95.0, 105.0, 80.0, 65.0]

    prominence = _mountain_prominence(
        highs,
        lows,
        idx=2,
        pivot_set=frozenset({2, 3}),
        equal_tol=0.01,
    )

    assert prominence == 0.0


def test_display_major_rejects_deep_wick_without_body_fall():
    # The idx-60 candidate has a dramatic low wick after it, but quarterly-style
    # bodies stay near the top and continue directly to the higher idx-80 peak.
    # It must not survive the candle-body confirmation pass.
    n = 120
    closes = piecewise_closes(
        [
            (0, 50.0),
            (20, 80.0),
            (35, 50.0),
            (60, 100.0),
            (70, 95.0),
            (80, 130.0),
            (95, 80.0),
            (119, 120.0),
        ],
        n,
    )
    bars = bars_from_closes(closes, low_overrides={70: 50.0})

    majors = _detect_display_major_highs_on_bars(bars)

    assert 60 not in [point.idx for point in majors]


def test_display_major_ignores_low_peak_in_chart_range():
    # The first swing is locally deep, but it sits in the bottom third of the
    # ten-year price range and is visually minor beside the later mountains.
    n = 120
    closes = piecewise_closes(
        [
            (0, 12.0),
            (20, 60.0),
            (35, 10.0),
            (60, 110.0),
            (75, 55.0),
            (95, 160.0),
            (108, 90.0),
            (119, 200.0),
        ],
        n,
    )
    bars = bars_from_closes(
        closes,
        high_overrides={20: 60.0, 60: 110.0, 95: 160.0, 119: 200.0},
        low_overrides={35: 10.0},
    )

    majors = _detect_display_major_highs_on_bars(bars)

    assert 20 not in [point.idx for point in majors]
    assert all(point.price >= 110.0 for point in majors)


CACHE_DIR = Path(__file__).parent / "cache"
CACHED_TICKERS = sorted(path.stem for path in CACHE_DIR.glob("*.json"))


def _review_fixture(ticker: str) -> Path:
    root = Path(__file__).parent
    seeded = root / "seed_cache" / f"{ticker}.json"
    return seeded if seeded.exists() else root / "cache" / f"{ticker}.json"


def _cached_bars(ticker: str) -> list[dict]:
    return json.loads((CACHE_DIR / f"{ticker}.json").read_text(encoding="utf-8"))["bars"]


# ---------------------------------------------------------------------------
# Corpus invariants (v2.2). The per-ticker expectations in
# validation/major_high_feedback.json were labelled at algorithm_version
# 2.1.0, when 24 of these 79 charts anchored their lid to a ``provisional_top``
# drawn through the live price. Those labels are stale pending human
# re-review, so the corpus is asserted on the invariants the new algorithm
# guarantees rather than on 2.1.0 point-for-point output.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", CACHED_TICKERS)
def test_corpus_structure_never_comes_from_the_live_right_edge(ticker):
    """No provisional top, and nothing from the incomplete final quarter.

    This is the whole point of v2.2: the lid must be a repeated historical
    ceiling. If any of the three deleted live-price leaks comes back, a
    ``provisional_top`` role or a point inside the partial quarter shows up
    here on the tickers that used to produce them.
    """
    bars = _cached_bars(ticker)
    result = analyze_coil(bars)
    quarterly = _aggregate_quarterly_display_bars(bars)
    completed = _completed_quarters(quarterly)
    assert completed, f"{ticker} has no completed quarter"
    last_structural_month = str(completed[-1]["date"])

    roles = [point["role"] for point in result["points"]]
    assert ROLE_PROVISIONAL_TOP not in roles
    assert all(point["confirmed"] for point in result["points"])
    assert all(point["prominence_pct"] > 0 for point in result["points"])
    assert all(
        "immediate_qualification" not in point["evidence"]
        for point in result["points"]
    )
    for point in result["points"]:
        assert point["date"] <= last_structural_month, (
            f"{ticker}: {point['date']} is inside the incomplete final quarter"
        )
    for high in result["major_highs"]:
        assert high["date"] <= last_structural_month


@pytest.mark.parametrize("ticker", CACHED_TICKERS)
def test_corpus_lid_anchors_are_one_repeated_ceiling(ticker):
    """Every selected lid is two members of one price zone, far apart in time.

    Anchors inside one zone sit within ``zone_similarity_pct`` of the seed, so
    the pair can never spread further than twice that. A lid drawn between two
    unrelated price eras (the growth-trend failure mode) breaks both bounds.
    """
    result = analyze_coil(_cached_bars(ticker))
    if result["resistance"] is None:
        assert result["status"] in {"no_structure", "invalid_data"}
        if result["status"] == "invalid_data":
            assert result["analysis_metadata"]["data_quality"]["blocked"] is True
        return
    anchors = result["resistance"]["classification_lid"]["points"]
    assert len(anchors) == 2
    earlier, later = anchors
    spread = abs(later["price"] - earlier["price"]) / max(earlier["price"], later["price"])
    assert spread <= 2 * DEFAULT_CONFIG.zone_similarity_pct / 100.0
    # Separation is enforced in quarterly coordinates, but anchors report the
    # source month of each quarterly high. Two quarters N apart are at least
    # 3N-2 months apart (last month of the first, first month of the last), so
    # a flat 3N floor would reject spec-compliant zones such as CBA.AX's
    # 2025-06 -> 2026-04 pair (4 quarters, 10 months).
    months_apart = later["idx"] - earlier["idx"]
    assert months_apart >= 3 * DEFAULT_CONFIG.zone_min_separation_quarters - 2


@pytest.mark.parametrize("ticker", CACHED_TICKERS)
def test_corpus_lid_anchors_are_always_marked_on_the_overlay(ticker):
    """The overlay may not draw a lid from a candle it does not mark.

    ``display_max_highs`` caps ``major_highs`` at 3, but a zone can hold more
    members than that. Truncating to the latest N drops the earliest member,
    which is the lid's own first anchor — the line then starts at an unmarked
    candle. Interior touches may be dropped by the cap; anchors may not.
    """
    result = analyze_coil(_cached_bars(ticker))
    if result["active_lid"] is None:
        assert result["major_highs"] == []
        return
    marked = {high["idx"] for high in result["major_highs"]}
    anchors = {anchor["idx"] for anchor in result["active_lid"]["anchors"]}
    assert anchors <= marked
    assert len(result["major_highs"]) >= len(anchors)


@pytest.mark.parametrize("ticker", CACHED_TICKERS)
def test_corpus_price_position_matches_the_reported_proximity(ticker):
    result = analyze_coil(_cached_bars(ticker))
    metrics = result["metrics"]
    if metrics is None:
        assert result["resistance"] is None
        return
    assert metrics["current_price_position"] == _price_position(
        metrics["proximity_pct"], DEFAULT_CONFIG
    )
    if metrics["current_price_position"] != PRICE_POSITION_WITHIN:
        assert result["grade"] is None
        # The lid is retained for diagnosis even when it cannot grade.
        assert result["resistance"]["value_at_last_bar"] > 0


def test_analyzer_never_emits_a_provisional_top_anywhere():
    """The analyzer-wide guarantee, on the corpus that used to break it."""
    offenders = []
    for ticker in CACHED_TICKERS:
        result = analyze_coil(_cached_bars(ticker))
        offenders.extend(
            (ticker, point["date"])
            for point in result["points"]
            if point["role"] == ROLE_PROVISIONAL_TOP
        )
    assert offenders == []


def test_enb_lid_stays_on_the_sixty_dollar_ceiling():
    """ENB's repeated ~59-60 ceiling, not the 62.5 breakout pass-through.

    2024-12 printed 61.99 on the way up through the lid. Admitting any
    completed quarter into a zone (rather than only the latest one) pulls that
    bar in and drags the line from 60.0 to 62.5 — live-price anchoring one step
    removed. The immediate-qualification rule is scoped to prevent exactly this.
    """
    result = analyze_coil(_cached_bars("ENB.TO"))

    if result["status"] == "invalid_data":
        assert result["resistance"] is None
        assert result["analysis_metadata"]["data_quality"]["blocked"] is True
        return

    lid = result["resistance"]["classification_lid"]
    assert lid["value_at_last_bar"] == pytest.approx(60.0, abs=0.5)
    assert [(point["date"], point["price"]) for point in lid["points"]] == [
        ("2016-09-01", pytest.approx(59.19, abs=0.01)),
        ("2022-06-01", pytest.approx(59.69, abs=0.01)),
    ]
    assert all(point["date"] != "2024-12-01" for point in result["points"])


@pytest.mark.parametrize("ticker", ["AAPL", "ASML.AS"])
def test_trending_megacap_without_a_repeated_ceiling_has_no_lid(ticker):
    """No zone qualifies -> no structure. That is the correct answer.

    These charts never built a repeated ceiling; the 2.1.0 algorithm still
    drew one by joining unrelated price eras and then reported 99%+ proximity
    because the line ran through the live price.
    """
    result = analyze_coil(_cached_bars(ticker))

    assert result["status"] == "no_structure"
    assert result["resistance"] is None
    assert result["active_lid"] is None
    assert result["grade"] is None
    assert result["points"] == []


def test_bg_lifetime_ceiling_is_the_repeated_135_level():
    # Ankit's explicit read: BG's ceiling is the 2008 high at 135 — the same
    # level as the confirmed 2022 mountain. The current quarter may be near the
    # same level, but it cannot replace the confirmed anchor before rejecting.
    payload = json.loads(_review_fixture("BG").read_text(encoding="utf-8"))
    result = analyze_coil(payload["bars"])

    lid = result["resistance"]["classification_lid"]
    assert [(point["date"], point["price"]) for point in lid["points"]] == [
        ("2008-01-01", pytest.approx(135.0, abs=0.02)),
        ("2022-04-01", pytest.approx(128.4, abs=0.02)),
    ]
    assert result["status"] == "coiling"
    assert result["grade"] == "A"
    assert result["metrics"]["current_price_position"] == PRICE_POSITION_WITHIN


def test_algorithm_version_is_reported():
    result = analyze_coil(make_coil_bars())

    assert ALGORITHM_VERSION == "2.3.1"
    assert result["algorithm_version"] == ALGORITHM_VERSION
    assert result["analysis_metadata"]["algorithm_version"] == ALGORITHM_VERSION


def test_tickers_from_csv(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("ticker,score_total\nkn,0.9\nUEC,0.8\n ,0.1\n", encoding="utf-8")
    assert _tickers_from_csv(path, limit=None) == ["KN", "UEC"]
    assert _tickers_from_csv(path, limit=1) == ["KN"]


# ---------------------------------------------------------------------------
# Whole-chart behavior
# ---------------------------------------------------------------------------


def test_flat_coil_grades_A():
    bars = make_coil_bars(slope_per_bar=0.0)
    result = analyze_coil(bars)

    assert result["status"] == "coiling"
    assert result["grade"] == "A"
    res = result["resistance"]
    assert res["slope_pct_per_year"] == pytest.approx(0.0, abs=0.1)
    # Only the three confirmed constructed tops count. The latest quarter's
    # tag of the same level still lacks a later-quarter rejection.
    assert res["touch_count"] == 3
    assert res["value_at_last_bar"] == pytest.approx(100.0, rel=1e-6)
    assert [point["price"] for point in res["classification_lid"]["points"]] == [
        pytest.approx(100.0, abs=1e-6),
        pytest.approx(100.0, abs=1e-6),
    ]
    metrics = result["metrics"]
    assert metrics["pullback_depths_pct"] == pytest.approx([50.0, 40.0, 30.0], abs=0.2)
    assert metrics["compression_ok"] is True
    assert metrics["proximity_pct"] == pytest.approx(96.0, abs=0.5)
    assert metrics["current_price_position"] == PRICE_POSITION_WITHIN
    assert result["support"]["converging"] is True
    assert result["coil_score"] > 60


def test_near_equal_tops_still_detected_as_majors():
    # Classic scipy prominence collapses when coil tops are near-equal; the
    # equal-tolerance stop must keep at least the first two tops major.
    bars = make_coil_bars(slope_per_bar=0.0)
    bars[20]["high"] = 100.0
    bars[60]["high"] = 100.2
    bars[100]["high"] = 99.9

    majors = select_major_highs(detect_swing_highs(bars))
    assert len(majors) >= 2

    result = analyze_coil(bars)
    assert result["grade"] == "A"
    assert result["resistance"]["touch_count"] == 3


def test_rising_lid_spanning_price_eras_is_not_one_zone():
    """A lid rising 4%/yr is not a repeated ceiling.

    Both of its confirmed tops are found, but they sit ~17% apart: two price
    levels, not one level retested. v2.2 declines to draw a lid through them;
    the human review workspace exists for exactly this override (see the
    reviewed-lid tests below).
    """
    spb = spb_for_end_slope(4.0)
    bars = make_coil_bars(slope_per_bar=spb)
    highs = [bars[20]["high"], bars[60]["high"]]
    assert abs(highs[1] - highs[0]) / highs[1] > DEFAULT_CONFIG.zone_similarity_pct / 100.0

    result = analyze_coil(bars)

    assert result["status"] == "no_structure"
    assert result["resistance"] is None


def test_wider_zone_tolerance_admits_the_same_rising_tops():
    """The 5% tolerance is the only reason, and it is a config field."""
    spb = spb_for_end_slope(4.0)
    bars = make_coil_bars(slope_per_bar=spb)
    wide = replace(DEFAULT_CONFIG, zone_similarity_pct=25.0)

    result = analyze_coil(bars, config=wide)

    assert result["status"] == "coiling"
    assert result["resistance"]["slope_pct_per_year"] == pytest.approx(4.0, abs=0.05)
    assert [
        (point["date"], point["price"])
        for point in result["resistance"]["classification_lid"]["points"]
    ] == [
        ("2015-01-01", pytest.approx(119.9, abs=0.01)),
        ("2018-05-01", pytest.approx(139.801, abs=0.01)),
    ]


@pytest.mark.parametrize(
    ("slope_pct", "depths", "expected_grade"),
    [
        (4.0, (0.40, 0.28, 0.20), "A"),
        (6.0, (0.45, 0.30, 0.22), "B"),
        (8.0, (0.45, 0.55, 0.25), "C"),
    ],
)
def test_reviewed_lid_gets_the_same_slope_and_grade_math(slope_pct, depths, expected_grade):
    """Reviewed anchors bypass zone eligibility but not the lid math.

    None of these rising lids is a price zone, so the algorithm declines them.
    A reviewer pinning the same three tops must still get the constructed
    slope, its grade band, and a band placement.
    """
    spb = spb_for_end_slope(slope_pct)
    bars = make_coil_bars(slope_per_bar=spb, depths=depths)
    assert analyze_coil(bars)["status"] == "no_structure"

    result = analyze_coil(bars, review_override=lid_override(bars, (20, 60, 100)))

    assert result["review"]["effective"] == "human"
    assert result["resistance"]["slope_pct_per_year"] == pytest.approx(slope_pct, abs=0.05)
    assert result["resistance"]["lid_grade"] == expected_grade
    assert result["grade"] == expected_grade
    assert result["status"] == "coiling"
    assert result["metrics"]["current_price_position"] == PRICE_POSITION_WITHIN


def test_reviewed_anchors_bypass_zone_eligibility():
    """Two anchors that no zone rule would ever accept still make a lid.

    They are 25% apart in price (five times ``zone_similarity_pct``) and one
    quarter apart in time (below ``zone_min_separation_quarters``). Reviewer
    authority is the point of the override path.
    """
    bars = make_coil_bars()
    bars[100]["high"] = 100.0
    bars[103]["high"] = 125.0

    zone_view = _select_lid_zone(
        _completed_quarters(_aggregate_quarterly_display_bars(bars)),
        DEFAULT_CONFIG,
        DEFAULT_CONFIG,
        float(bars[-1]["close"]),
    )[0]
    assert 34 not in [member.idx for member in zone_view.members]

    result = analyze_coil(bars, review_override=lid_override(bars, (100, 103)))

    anchors = result["resistance"]["classification_lid"]["points"]
    assert [point["date"] for point in anchors] == ["2018-05-01", "2018-08-01"]
    assert [point["price"] for point in anchors] == [100.0, 125.0]
    assert result["metrics"]["current_price_position"] is not None
    assert result["resistance"]["lid_grade"] == grade_for_slope(
        result["resistance"]["slope_pct_per_year"]
    )


def test_reviewed_too_steep_lid_is_basing_without_grade():
    n, t0 = 120, 70
    spb = spb_for_end_slope(14.0, t0=t0, n=n)

    def lid(x: int) -> float:
        return 100.0 + spb * (x - t0)

    keys = [(0, 60.0), (70, lid(70) * 0.99), (85, 80.0), (100, lid(100) * 0.99), (110, 120.0), (119, lid(119) * 0.96)]
    bars = bars_from_closes(
        piecewise_closes(keys, n),
        high_overrides={70: lid(70), 100: lid(100)},
        low_overrides={85: 80.0, 110: 120.0},
    )

    result = analyze_coil(bars, review_override=lid_override(bars, (70, 100)))

    assert result["resistance"] is not None
    assert result["resistance"]["slope_pct_per_year"] == pytest.approx(14.0, abs=0.5)
    assert result["resistance"]["lid_grade"] is None
    assert result["status"] == "basing"
    assert result["grade"] is None
    assert any("slope" in note for note in result["notes"])


def test_expanding_pullbacks_fail_compression_gate():
    # Widening pullbacks AND price not pressed at the lid -> not wound yet.
    bars = make_coil_bars(depths=(0.25, 0.35, 0.45), end_frac=0.82)
    result = analyze_coil(bars)

    assert result["status"] == "basing"
    assert result["grade"] is None
    assert any("compress" in note for note in result["notes"])


def test_pressed_at_lid_overrides_deep_late_pullback():
    # KN/EAT style: a deep pullback late in the base, but price has come all
    # the way back to the lid — the coil is wound, grade stands.
    bars = make_coil_bars(depths=(0.45, 0.30, 0.44), end_frac=0.97)
    result = analyze_coil(bars)

    assert result["status"] == "coiling"
    assert result["grade"] == "A"
    assert result["metrics"]["pressed_at_lid"] is True
    assert any("pressed at lid" in note for note in result["notes"])


def test_broken_out_lid():
    bars = make_coil_bars()
    dates = month_dates(126)
    for k in range(6):
        bars.append(
            {"date": dates[120 + k], "open": 110.0, "high": 110.5, "low": 108.0, "close": 110.0, "volume": 1e6}
        )
    result = analyze_coil(bars)

    assert result["status"] == "broken_out"
    assert result["grade"] is None
    # The lid it escaped from is still reported, with its slope band intact.
    assert result["resistance"]["lid_grade"] == "A"


def test_breaking_out_now_keeps_grade():
    bars = make_coil_bars()
    for idx in (118, 119):
        bars[idx].update(open=109.0, close=110.0, high=110.5, low=108.0)
    result = analyze_coil(bars)

    assert result["status"] == "breaking_out"
    assert result["grade"] == "A"


def test_as_of_truncation_restores_pre_breakout_coil():
    bars = make_coil_bars()
    dates = month_dates(126)
    for k in range(6):
        bars.append(
            {"date": dates[120 + k], "open": 110.0, "high": 110.5, "low": 108.0, "close": 110.0, "volume": 1e6}
        )

    full = analyze_coil(bars)
    truncated = analyze_coil(bars, as_of="2019-12-31")

    assert full["status"] == "broken_out"
    assert truncated["status"] == "coiling"
    assert truncated["grade"] == "A"
    assert truncated["bar_count"] == 120


def test_pure_trend_has_no_structure():
    import math

    closes = [100.0 * (1.015**x) * (1.0 + 0.02 * math.sin(2 * math.pi * x / 10)) for x in range(120)]
    result = analyze_coil(bars_from_closes(closes))

    assert result["status"] == "no_structure"
    assert result["grade"] is None


def test_decades_long_exponential_trend_is_never_graded():
    # A mega-cap style compounding uptrend with deep cyclical swings offers a
    # plausible 2-touch upper envelope, but log-close R^2 over the base stays
    # high — it must never earn a coil grade (the AAPL false-positive case).
    import math

    closes = [
        2.0 * (1.12 ** (x / 12.0)) * (1.0 + 0.22 * math.sin(2 * math.pi * x / 60))
        for x in range(360)
    ]
    result = analyze_coil(bars_from_closes(closes))

    assert result["grade"] is None
    assert result["status"] != "coiling"
    if result["resistance"] is not None:
        assert result["metrics"]["base_trend_r2"] > 0.85
        assert any("trend" in note for note in result["notes"])


def test_insufficient_history_notes():
    bars = make_coil_bars()[:40]
    result = analyze_coil(bars)

    assert result["status"] == "no_structure"
    assert any("insufficient history" in note for note in result["notes"])


def test_unsorted_input_is_sorted_but_invalid_rows_block_analysis():
    bars = make_coil_bars()
    clean_expected = analyze_coil(bars)

    shuffled = list(reversed(bars))
    shuffled.insert(5, {"date": "2012-06-15", "open": 1.0, "high": None, "low": 1.0, "close": 1.0})
    result = analyze_coil(shuffled)

    assert result["bar_count"] == clean_expected["bar_count"]
    assert result["status"] == "invalid_data"
    quality = result["analysis_metadata"]["data_quality"]
    assert quality["blocked"] is True
    assert {issue["code"] for issue in quality["issues"]} >= {
        "out_of_order",
        "missing_or_nonfinite_ohlc",
    }


def test_volume_contraction_ratio():
    bars = make_coil_bars()
    for idx, bar in enumerate(bars):
        bar["volume"] = 2_000_000.0 if idx < 60 else 1_000_000.0
    result = analyze_coil(bars)

    assert result["metrics"]["volume_contraction_ratio"] == pytest.approx(0.5, abs=0.01)


def test_seed_cache_fixture_end_to_end():
    seed = Path(__file__).parent / "seed_cache" / "BG.json"
    import json

    payload = json.loads(seed.read_text(encoding="utf-8"))
    result = analyze_coil(payload["bars"])

    assert result["schema_version"] == 2
    assert result["status"] in {"no_structure", "basing", "coiling", "breaking_out", "broken_out"}
    assert result["lifecycle"] in {
        "no_structure",
        "forming",
        "pre_breakout",
        "breaking_out",
        "post_breakout",
    }
    assert result["bar_count"] > 100
    assert isinstance(result["major_highs"], list)
    for high in result["major_highs"]:
        assert set(high) == {
            "idx",
            "date",
            "peak_date",
            "confirmed_at_idx",
            "confirmed_at",
            "price",
            "prominence_pct",
            "source",
        }


def test_lh_above_the_band_keeps_the_lid_and_drops_the_grade():
    # LH's repeated 163/168 ceiling from 2018-2020 is 38% below today's price.
    # The line stays on the result for diagnosis; the grade does not.
    payload = json.loads(_review_fixture("LH").read_text(encoding="utf-8"))
    result = analyze_coil(payload["bars"])

    metrics = result["metrics"]
    assert metrics["proximity_pct"] > DEFAULT_CONFIG.lid_band_upper_pct
    assert metrics["current_price_position"] == PRICE_POSITION_ABOVE
    assert result["lifecycle"] == "post_breakout"
    assert result["grade"] is None
    # Retained for diagnosis, with its slope band intact.
    assert result["resistance"]["value_at_last_bar"] > 0
    assert result["resistance"]["lid_grade"] == "A"
    assert any("above the" in note for note in result["notes"])


def test_msci_unconfirmed_retest_cannot_manufacture_a_zone():
    """MSCI's 679.85 and 626.28 tops are 5.5% apart — a real boundary case.

    At the 5% default they are two levels, and the apparent recent retest has no
    confirmed two-sided fall, so there is no repeated ceiling. Widening only
    ``zone_similarity_pct`` to 8% merges the two confirmed mountains and
    produces a lid, pinning the tolerance as the sole reason for the default
    no-structure result.
    """
    payload = json.loads(_review_fixture("MSCI").read_text(encoding="utf-8"))
    bars = payload["bars"]

    assert abs(679.85 - 626.28) / 679.85 > DEFAULT_CONFIG.zone_similarity_pct / 100.0

    default = analyze_coil(bars)
    assert default["status"] == "no_structure"
    assert default["resistance"] is None

    widened = analyze_coil(bars, config=replace(DEFAULT_CONFIG, zone_similarity_pct=8.0))
    assert [
        (point["date"], point["price"])
        for point in widened["resistance"]["classification_lid"]["points"]
    ] == [
        ("2021-11-01", pytest.approx(679.85, abs=0.02)),
        ("2024-12-01", pytest.approx(642.45, abs=0.02)),
    ]


# ---------------------------------------------------------------------------
# Lid zone selection (v2.2)
# ---------------------------------------------------------------------------


def test_completed_quarters_excludes_every_incomplete_quarter():
    full = quarterly_bars([10.0, 11.0, 12.0], last_month=12)
    partial = quarterly_bars([10.0, 11.0, 12.0], last_month=10)

    assert len(_completed_quarters(full)) == 3
    assert [bar["high"] for bar in _completed_quarters(partial)] == [10.0, 11.0]
    # Incomplete historical quarters remain display warnings, never structure.
    partial[1]["_last_month"] = 11
    assert len(_completed_quarters(partial)) == 1
    assert _completed_quarters([]) == []


def test_quarter_end_month_is_not_complete_while_its_candle_is_live():
    quarters = quarterly_bars([10.0, 11.0], last_month=9)
    quarters[-1]["date"] = "2026-09-01"

    assert _completed_quarters(quarters, today=date(2026, 9, 15)) == quarters[:-1]
    assert _completed_quarters(quarters, today=date(2026, 10, 1)) == quarters
    assert _completed_quarters(quarters, as_of="2026-09-15") == quarters[:-1]
    assert _completed_quarters(quarters, as_of="2026-09-30") == quarters


def test_cluster_price_zones_handles_empty_and_single_candidates():
    assert _cluster_price_zones([], DEFAULT_CONFIG) == []

    zones = _cluster_price_zones([SwingPoint(4, "2011-01-01", 50.0, 30.0)], DEFAULT_CONFIG)

    assert len(zones) == 1
    assert zones[0].member_count == 1
    assert zones[0].level == 50.0
    assert zones[0].span == 0


def test_cluster_price_zones_absorbs_exactly_at_tolerance_and_splits_beyond():
    # Seed 100. 95.0 is exactly 5.0% below (inclusive), 94.99 is beyond.
    inside = _cluster_price_zones(
        [
            SwingPoint(0, "2010-01-01", 100.0, 30.0),
            SwingPoint(8, "2012-01-01", 95.0, 30.0),
        ],
        DEFAULT_CONFIG,
    )
    outside = _cluster_price_zones(
        [
            SwingPoint(0, "2010-01-01", 100.0, 30.0),
            SwingPoint(8, "2012-01-01", 94.99, 30.0),
        ],
        DEFAULT_CONFIG,
    )

    assert [zone.member_count for zone in inside] == [2]
    assert [zone.member_count for zone in outside] == [1, 1]


def test_cluster_price_zones_level_is_the_member_mean_and_seeds_from_the_top():
    zones = _cluster_price_zones(
        [
            SwingPoint(0, "2010-01-01", 98.0, 30.0),
            SwingPoint(8, "2012-01-01", 100.0, 30.0),
            SwingPoint(16, "2014-01-01", 96.0, 30.0),
            SwingPoint(24, "2016-01-01", 50.0, 30.0),
        ],
        DEFAULT_CONFIG,
    )

    assert len(zones) == 2
    # Highest candidate seeds first; 96/98 are both within 5% of 100.
    assert [member.idx for member in zones[0].members] == [0, 8, 16]
    assert zones[0].level == pytest.approx(98.0)
    assert [member.idx for member in zones[1].members] == [24]


def test_zone_qualification_boundaries():
    at_boundary = zone_from({0: 100.0, 4: 100.0})
    below_boundary = zone_from({0: 100.0, 3: 100.0})
    single = zone_from({0: 100.0})

    assert at_boundary.span == DEFAULT_CONFIG.zone_min_separation_quarters
    assert _zone_reject_reason(at_boundary, last_close=100.0) is None
    assert "quarters apart" in _zone_reject_reason(below_boundary, last_close=100.0)
    assert "single touch" in _zone_reject_reason(single, last_close=100.0)


def test_era_relevance_boundaries_are_inclusive():
    zone = zone_from({0: 100.0, 8: 100.0})

    assert _zone_reject_reason(zone, last_close=50.0) is None  # exactly 50%
    assert _zone_reject_reason(zone, last_close=400.0) is None  # exactly 400%
    assert "different price era" in _zone_reject_reason(zone, last_close=49.9)
    assert "long since escaped" in _zone_reject_reason(zone, last_close=400.1)


def test_ranking_prefers_recency_over_a_more_repeated_older_zone():
    """Recency is the primary key, and it must beat repetition.

    Repetition-first was measured and is wrong: a long history holds far more
    repeated pivots at its ancient low prices than at today's level, so
    repetition-first always returns the oldest zone.
    """
    old_and_busy = zone_from({0: 10.0, 6: 10.0, 12: 10.0, 20: 10.0})
    recent_and_sparse = zone_from({30: 100.0, 38: 100.0})

    ranked = _rank_lid_zones([old_and_busy, recent_and_sparse])

    assert ranked[0] is recent_and_sparse
    assert old_and_busy.member_count > recent_and_sparse.member_count


def test_ranking_tiebreaks_on_member_count_then_span_then_first_index():
    # All three end at quarter 40, so recency ties and the next keys decide.
    busiest = zone_from({20: 100.0, 30: 100.0, 40: 100.0})
    longest = zone_from({4: 100.0, 40: 100.0})
    shortest = zone_from({30: 100.0, 40: 100.0})

    assert _rank_lid_zones([shortest, longest, busiest]) == [busiest, longest, shortest]

    # Equal recency, count, and span: the earlier start wins, deterministically.
    early = zone_from({10: 100.0, 40: 100.0})
    same = zone_from({10: 100.0, 40: 100.0})
    assert _rank_lid_zones([same, early])[0].first_idx == 10


def test_latest_completed_quarter_cannot_join_a_zone_without_confirmation():
    """Being near an old ceiling is not enough to make the latest bar a top."""
    quarters = quarterly_bars(
        [80.0, 100.0, 70.0, 70.0, 70.0, 70.0, 100.0, 70.0, 70.0, 70.0, 70.0, 98.0]
    )

    zone, _ = _select_lid_zone(
        quarters,
        _quarterly_scaled_config(DEFAULT_CONFIG),
        DEFAULT_CONFIG,
        last_close=98.0,
    )

    assert zone is not None
    assert [member.idx for member in zone.members] == [1, 6]
    assert zone.last_idx != len(quarters) - 1


def test_select_lid_zone_returns_nothing_when_no_zone_qualifies():
    # A monotone staircase: every pivot is its own level, none repeats.
    quarters = quarterly_bars([10.0 * (1.3**k) for k in range(20)])

    zone, rejected = _select_lid_zone(
        quarters,
        _quarterly_scaled_config(DEFAULT_CONFIG),
        DEFAULT_CONFIG,
        last_close=float(quarters[-1]["close"]),
    )

    assert zone is None
    assert all(item.reject_reason is not None for item in rejected)


# ---------------------------------------------------------------------------
# The +/-20% lid band (v2.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("proximity_pct", "expected"),
    [
        (0.0, PRICE_POSITION_BELOW),
        (79.99, PRICE_POSITION_BELOW),
        (80.0, PRICE_POSITION_WITHIN),
        (100.0, PRICE_POSITION_WITHIN),
        (120.0, PRICE_POSITION_WITHIN),
        (120.01, PRICE_POSITION_ABOVE),
        (400.0, PRICE_POSITION_ABOVE),
    ],
)
def test_price_position_band_edges_are_inclusive(proximity_pct, expected):
    assert _price_position(proximity_pct, DEFAULT_CONFIG) == expected


def test_price_position_follows_the_configured_band():
    tight = replace(DEFAULT_CONFIG, lid_band_lower_pct=95.0, lid_band_upper_pct=105.0)

    assert _price_position(90.0, tight) == PRICE_POSITION_BELOW
    assert _price_position(90.0, DEFAULT_CONFIG) == PRICE_POSITION_WITHIN


def test_below_the_band_is_forming_and_keeps_the_lid():
    bars = make_coil_bars(end_frac=0.70)

    result = analyze_coil(bars)

    assert result["metrics"]["proximity_pct"] == pytest.approx(70.0, abs=0.5)
    assert result["metrics"]["current_price_position"] == PRICE_POSITION_BELOW
    assert result["lifecycle"] == "forming"
    assert result["grade"] is None
    # Retained for diagnosis: value, anchors, and slope band all still present.
    assert result["resistance"]["value_at_last_bar"] == pytest.approx(100.0, rel=1e-6)
    assert result["resistance"]["lid_grade"] == "A"
    assert any("below the 80% lid band" in note for note in result["notes"])


def test_above_the_band_is_post_breakout_and_keeps_the_lid():
    bars = make_coil_bars()
    dates = month_dates(126)
    for k in range(6):
        bars.append(
            {"date": dates[120 + k], "open": 130.0, "high": 130.5, "low": 128.0, "close": 130.0, "volume": 1e6}
        )

    result = analyze_coil(bars)

    assert result["metrics"]["proximity_pct"] == pytest.approx(130.0, abs=0.5)
    assert result["metrics"]["current_price_position"] == PRICE_POSITION_ABOVE
    assert result["lifecycle"] == "post_breakout"
    assert result["grade"] is None
    assert result["resistance"]["value_at_last_bar"] == pytest.approx(100.0, rel=1e-6)
    assert result["resistance"]["lid_grade"] == "A"
    assert any("above the 120% lid band" in note for note in result["notes"])


# ---------------------------------------------------------------------------
# The incomplete-final-quarter invariant (v2.2)
# ---------------------------------------------------------------------------


def _with_partial_quarter(bars: list[dict]) -> list[dict]:
    """Append a January bar that spikes to 140 and closes at 128."""
    return [dict(bar) for bar in bars] + [
        {"date": "2020-01-01", "open": 100.0, "high": 140.0, "low": 99.0, "close": 128.0, "volume": 1e6}
    ]


def _with_ongoing_quarter_end_month(bars: list[dict]) -> list[dict]:
    """Append a still-open July/August/September quarter with a live spike."""
    return [dict(bar) for bar in bars] + [
        {
            "date": "2026-07-01",
            "open": 100.0,
            "high": 120.0,
            "low": 98.0,
            "close": 115.0,
            "volume": 1e6,
        },
        {
            "date": "2026-08-01",
            "open": 115.0,
            "high": 130.0,
            "low": 110.0,
            "close": 125.0,
            "volume": 1e6,
        },
        {
            "date": "2026-09-01",
            "open": 125.0,
            "high": 140.0,
            "low": 120.0,
            "close": 138.0,
            "volume": 1e6,
        },
    ]


def test_ongoing_quarter_end_month_never_becomes_a_top():
    result = analyze_coil(
        _with_ongoing_quarter_end_month(make_coil_bars()),
        as_of="2026-09-15",
    )

    assert (
        result["analysis_metadata"]["data_freshness"]["incomplete_last_quarter"]
        is True
    )
    assert all(point["date"] < "2026-07-01" for point in result["points"])
    assert all(high["date"] < "2026-07-01" for high in result["major_highs"])
    assert all(
        touch["date"] < "2026-07-01"
        for touch in result["resistance"]["touches"]
    )


def test_incomplete_final_quarter_never_becomes_structure():
    """A 40% spike in the partial quarter must not move a single point.

    The last close still counts — it is what proximity and band placement are
    measured from — but nothing inside that quarter may be a top, a marker, a
    touch, or an anchor.
    """
    completed_only = make_coil_bars()
    with_partial = _with_partial_quarter(completed_only)

    baseline = analyze_coil(completed_only)
    result = analyze_coil(with_partial)

    assert result["analysis_metadata"]["data_freshness"]["incomplete_last_quarter"] is True
    lid, baseline_lid = (
        result["resistance"]["classification_lid"],
        baseline["resistance"]["classification_lid"],
    )
    assert lid["points"] == baseline_lid["points"]
    assert lid["slope_pct_per_year"] == baseline_lid["slope_pct_per_year"]
    assert lid["value_at_last_bar"] == baseline_lid["value_at_last_bar"]
    assert [point["date"] for point in result["points"]] == [
        point["date"] for point in baseline["points"]
    ]
    assert all(point["date"] < "2020-01-01" for point in result["points"])
    assert all(touch["date"] < "2020-01-01" for touch in result["resistance"]["touches"])
    assert all(high["date"] < "2020-01-01" for high in result["major_highs"])
    # The live close is still read against the (unchanged) lid.
    assert result["metrics"]["proximity_pct"] == pytest.approx(128.0, abs=0.5)
    assert result["metrics"]["current_price_position"] == PRICE_POSITION_ABOVE


def test_reviewed_anchor_inside_the_incomplete_quarter_is_dropped():
    """Reviewers keep zone authority, but not the partial quarter."""
    bars = _with_partial_quarter(make_coil_bars())
    legal = lid_override(bars, (20, 100))
    with_partial_anchor = {
        **legal,
        "points": legal["points"] + [
            {"date": "2020-01-01", "price": 140.0, "role": "major_top", "lid_member": True}
        ],
    }

    accepted = analyze_coil(bars, review_override=legal)
    filtered = analyze_coil(bars, review_override=with_partial_anchor)

    assert filtered["review"]["effective"] == "human"
    assert all(point["date"] != "2020-01-01" for point in filtered["points"])
    assert (
        filtered["resistance"]["classification_lid"]
        == accepted["resistance"]["classification_lid"]
    )


REVIEWED_TICKERS = sorted(VALIDATION_FEEDBACK)


@pytest.mark.parametrize("ticker", REVIEWED_TICKERS)
def test_reviewed_charts_never_regress_to_a_live_price_anchor(ticker):
    """The 23 human-reviewed charts, on their own review-time fixtures.

    ``validation/major_high_feedback.json`` records ``algorithm_version_at_review``
    of 2.1.0, and several of its ``intended_highs`` carry the
    ``provisional_top`` role that v2.2 abolished. Those point-for-point labels
    are stale pending human re-review, so this asserts the invariants instead:
    the reviewed charts still analyze, and none of them anchors to the live
    right edge.
    """
    fixture = _review_fixture(ticker)
    assert fixture.exists(), f"missing reviewed history fixture for {ticker}"
    bars = json.loads(fixture.read_text(encoding="utf-8"))["bars"]

    result = analyze_coil(bars)

    completed = _completed_quarters(_aggregate_quarterly_display_bars(bars))
    last_structural_month = str(completed[-1]["date"])
    assert all(point["role"] != ROLE_PROVISIONAL_TOP for point in result["points"])
    assert all(point["date"] <= last_structural_month for point in result["points"])
    if result["resistance"] is None:
        assert result["status"] in {"no_structure", "invalid_data"}
        if result["status"] == "invalid_data":
            assert result["analysis_metadata"]["data_quality"]["blocked"] is True
        return
    anchors = result["resistance"]["classification_lid"]["points"]
    assert len(anchors) == 2
    assert anchors[1]["idx"] - anchors[0]["idx"] >= (
        3 * DEFAULT_CONFIG.zone_min_separation_quarters
    )
    assert result["metrics"]["current_price_position"] in {
        PRICE_POSITION_BELOW,
        PRICE_POSITION_WITHIN,
        PRICE_POSITION_ABOVE,
    }


def test_detect_display_major_highs_returns_the_capped_active_structure():
    bars = make_coil_bars()

    highs = detect_display_major_highs(bars)

    assert [point.date for point in highs] == [
        point["date"] for point in analyze_coil(bars)["major_highs"]
    ]
    assert len(highs) <= DEFAULT_CONFIG.display_max_highs
    assert [point.price for point in highs] == [pytest.approx(100.0)] * len(highs)


def test_detect_display_major_highs_is_empty_without_structure():
    spb = spb_for_end_slope(4.0)

    assert detect_display_major_highs(make_coil_bars(slope_per_bar=spb)) == []
    assert detect_display_major_highs([]) == []


def test_every_algorithmic_lid_member_is_a_confirmed_mountain():
    result = analyze_coil(make_coil_bars())

    assert result["points"]
    assert all(point["role"] == "major_top" for point in result["points"])
    assert all(point["confirmed"] for point in result["points"])
    assert all(point["prominence_pct"] > 0 for point in result["points"])
    assert all(
        "immediate_qualification" not in point["evidence"]
        for point in result["points"]
    )
