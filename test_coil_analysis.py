"""Behavior tests for the deterministic coil-structure analyzer.

Synthetic monthly series are built from piecewise-linear close paths with
exact lid touches (bar high == line value) and exact pullback lows, so the
expected slope, grade, touch count, and depth sequence are known by
construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from coil_analysis import (
    DEFAULT_CONFIG,
    SwingPoint,
    _cluster_mountain_plateaus,
    _cluster_touches,
    _detect_display_major_highs_on_bars,
    _mountain_prominence,
    _pivot_high_indexes,
    _suppress_slower_comparable_retests,
    _tickers_from_csv,
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
    depths: tuple[float, ...] = (0.30, 0.20, 0.10),
    start_frac: float = 0.65,
    end_frac: float = 0.96,
) -> list[dict]:
    """Coil with the lid passing exactly through each touch bar's high."""
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
    return bars_from_closes(piecewise_closes(keys, n), high_overrides, low_overrides)


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


def test_display_majors_are_latest_non_decreasing_mountains_with_right_edge():
    # Five large mountains rise across the window. The fourth starts as a
    # near-equal two-bar plateau; it must resolve to the first bar. Only the
    # latest three chart-worthy tops are returned, including the unconfirmed
    # extreme-right extension.
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

    assert [point.idx for point in majors] == [80, 110, 139]
    assert [point.price for point in majors] == [182.0, 260.0, 315.0]


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


def test_nsc_reviewed_quarter_replaces_early_plateau_and_pandemic_shoulder():
    payload = json.loads(
        (Path(__file__).parent / "seed_cache" / "NSC.json").read_text(encoding="utf-8")
    )

    majors = detect_display_major_highs(payload["bars"])

    assert [(point.date, point.price) for point in majors] == [
        ("2021-12-01", pytest.approx(299.2, abs=0.01)),
        ("2026-05-01", pytest.approx(326.0, abs=0.01)),
    ]


def test_reg_reviewed_highs_include_well_separated_intermediate_lid_retest():
    payload = json.loads(
        (Path(__file__).parent / "seed_cache" / "REG.json").read_text(encoding="utf-8")
    )

    majors = detect_display_major_highs(payload["bars"])

    assert [(point.date, point.price) for point in majors] == [
        ("2016-07-01", pytest.approx(85.35, abs=0.01)),
        ("2021-11-01", pytest.approx(78.07, abs=0.01)),
        ("2026-04-01", pytest.approx(81.66, abs=0.01)),
    ]


def test_spg_active_display_lid_is_grade_a_and_not_broken_out():
    payload = json.loads(
        (Path(__file__).parent / "seed_cache" / "SPG.json").read_text(encoding="utf-8")
    )

    result = analyze_coil(payload["bars"])

    assert result["status"] == "coiling"
    assert result["grade"] == "A"
    active = result["resistance"]["classification_lid"]
    assert active["slope_pct_per_year"] == pytest.approx(-0.99, abs=0.02)
    assert active["value_at_last_bar"] == pytest.approx(208.79, abs=0.02)
    assert active["violation_count"] == 0


def test_fcx_active_display_lid_prevents_obsolete_line_false_breakout():
    payload = json.loads(
        (Path(__file__).parent / "seed_cache" / "FCX.json").read_text(encoding="utf-8")
    )

    result = analyze_coil(payload["bars"])

    assert result["status"] != "broken_out"
    active = result["resistance"]["classification_lid"]
    assert active["slope_pct_per_year"] == pytest.approx(6.64, abs=0.02)
    assert active["value_at_last_bar"] == pytest.approx(68.94, abs=0.02)
    assert active["violation_count"] == 0


def test_vtr_reviewed_intermediate_retest_supplies_active_structure_fallback():
    payload = json.loads(
        (Path(__file__).parent / "seed_cache" / "VTR.json").read_text(encoding="utf-8")
    )

    result = analyze_coil(payload["bars"])

    assert [(high["date"], high["price"]) for high in result["major_highs"]] == [
        ("2016-07-01", pytest.approx(76.8, abs=0.01)),
        ("2019-09-01", pytest.approx(75.4, abs=0.01)),
        ("2026-05-01", pytest.approx(91.06, abs=0.01)),
    ]
    assert result["status"] == "broken_out"
    assert result["resistance"] is not None
    assert result["resistance"]["classification_lid"]["points"] == [
        {"idx": 230, "date": "2016-07-01", "price": pytest.approx(76.8, abs=0.01)},
        {"idx": 268, "date": "2019-09-01", "price": pytest.approx(75.4, abs=0.01)},
    ]
    assert result["resistance"]["classification_lid"]["violation_count"] == 7


def test_csx_reviewed_highs_drop_the_slower_comparable_rejection():
    payload = json.loads(
        (Path(__file__).parent / "seed_cache" / "CSX.json").read_text(encoding="utf-8")
    )

    majors = detect_display_major_highs(payload["bars"])

    assert [(point.date, point.price) for point in majors] == [
        ("2022-03-01", pytest.approx(38.63, abs=0.01)),
        ("2026-05-01", pytest.approx(47.18, abs=0.01)),
    ]


def _review_fixture(ticker: str) -> Path:
    root = Path(__file__).parent
    seeded = root / "seed_cache" / f"{ticker}.json"
    return seeded if seeded.exists() else root / "cache" / f"{ticker}.json"


def _quarter_key(value: str) -> tuple[int, int]:
    if "-Q" in value:
        year, quarter = value.split("-Q", 1)
        return int(year), int(quarter)
    year, month = value[:7].split("-")
    return int(year), (int(month) - 1) // 3 + 1


@pytest.mark.parametrize("ticker", sorted(VALIDATION_FEEDBACK))
def test_golden_reviewed_major_highs_match_intended_quarters_and_prices(ticker):
    """Every human-reviewed 3M chart is a detector golden fixture.

    Algorithm points retain the source month of each quarterly high, while a
    few review labels name the containing quarter (REG 2021-12 and LH
    2026-Q1). Compare interval-native quarter identity plus exact high price.
    """
    fixture = _review_fixture(ticker)
    assert fixture.exists(), f"missing reviewed history fixture for {ticker}"
    bars = json.loads(fixture.read_text(encoding="utf-8"))["bars"]
    actual = detect_display_major_highs(bars)
    intended = VALIDATION_FEEDBACK[ticker]["intended_highs"]

    assert len(actual) == len(intended)
    for point, expected in zip(actual, intended):
        assert _quarter_key(point.date) == _quarter_key(expected["date"])
        assert point.price == pytest.approx(expected["price"], abs=0.02)


@pytest.mark.parametrize("ticker", ["REG", "SPG", "LH", "CF"])
def test_reviewed_grade_a_regressions(ticker):
    bars = json.loads(_review_fixture(ticker).read_text(encoding="utf-8"))["bars"]
    assert analyze_coil(bars)["grade"] == "A"


def test_fcx_reviewed_grade_c_regression():
    bars = json.loads(_review_fixture("FCX").read_text(encoding="utf-8"))["bars"]
    assert analyze_coil(bars)["grade"] == "C"


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
    assert res["touch_count"] == 3
    assert res["value_at_last_bar"] == pytest.approx(100.0, rel=1e-6)
    metrics = result["metrics"]
    assert metrics["pullback_depths_pct"] == pytest.approx([30.0, 20.0, 10.0], abs=0.2)
    assert metrics["compression_ok"] is True
    assert metrics["proximity_pct"] == pytest.approx(96.0, abs=0.5)
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


def test_rising_coil_grades_A():
    # 4%/yr sits inside the recalibrated A band (< 5%/yr). The fitted slope may
    # deviate slightly from the constructed lid when the unconfirmed right-edge
    # anchor joins the least-squares fit.
    spb = spb_for_end_slope(4.0)
    bars = make_coil_bars(slope_per_bar=spb, depths=(0.40, 0.28, 0.20))
    result = analyze_coil(bars)

    assert result["status"] == "coiling"
    assert result["lifecycle"] == "pre_breakout"
    assert result["resistance"]["slope_pct_per_year"] == pytest.approx(4.0, abs=0.5)
    assert result["grade"] == "A"


def test_rising_coil_grades_B():
    spb = spb_for_end_slope(6.0)
    bars = make_coil_bars(slope_per_bar=spb, depths=(0.45, 0.30, 0.22))
    result = analyze_coil(bars)

    assert result["status"] == "coiling"
    assert result["resistance"]["slope_pct_per_year"] == pytest.approx(6.0, abs=0.4)
    assert result["grade"] == "B"


def test_steep_coil_grades_C():
    # The deep mid-base crash keeps the log-close R^2 below the trend gate —
    # a C-grade coil is choppy, not a smooth exponential rise.
    spb = spb_for_end_slope(8.0)
    bars = make_coil_bars(slope_per_bar=spb, depths=(0.45, 0.55, 0.25))
    result = analyze_coil(bars)

    assert result["status"] == "coiling"
    assert result["resistance"]["slope_pct_per_year"] == pytest.approx(8.0, abs=0.6)
    assert result["grade"] == "C"


def test_too_steep_lid_is_basing_without_grade():
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
    result = analyze_coil(bars)

    assert result["resistance"] is not None
    assert result["resistance"]["slope_pct_per_year"] == pytest.approx(14.0, abs=0.5)
    assert result["status"] == "basing"
    assert result["grade"] is None
    assert any("slope" in note for note in result["notes"])


def test_expanding_pullbacks_fail_compression_gate():
    # Widening pullbacks AND price not pressed at the lid -> not wound yet.
    bars = make_coil_bars(depths=(0.25, 0.35, 0.45), end_frac=0.80)
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
        bars[idx].update(close=110.0, high=110.5, low=108.0)
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
    truncated = analyze_coil(bars, as_of=dates[119])

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


def test_unsorted_and_null_bars_are_handled():
    bars = make_coil_bars()
    clean_expected = analyze_coil(bars)

    shuffled = list(reversed(bars))
    shuffled.insert(5, {"date": "2012-06-15", "open": 1.0, "high": None, "low": 1.0, "close": 1.0})
    result = analyze_coil(shuffled)

    assert result["bar_count"] == clean_expected["bar_count"]
    assert result["status"] == clean_expected["status"]
    assert result["grade"] == clean_expected["grade"]
    assert result["resistance"]["slope_pct_per_year"] == clean_expected["resistance"]["slope_pct_per_year"]


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
        assert set(high) == {"idx", "date", "price", "prominence_pct", "source"}


def test_bg_reviewed_major_highs_match_mountain_rule():
    seed = Path(__file__).parent / "seed_cache" / "BG.json"
    import json

    payload = json.loads(seed.read_text(encoding="utf-8"))
    result = analyze_coil(payload["bars"])

    assert [high["date"] for high in result["major_highs"]] == [
        "2017-05-01",
        "2022-04-01",
        "2026-05-01",
    ]
    assert [high["price"] for high in result["major_highs"]] == pytest.approx(
        [83.75, 128.4, 133.93],
        abs=0.02,
    )


def test_lh_reviewed_highs_require_post_peak_quarterly_fall():
    seed = Path(__file__).parent / "seed_cache" / "LH.json"
    import json

    payload = json.loads(seed.read_text(encoding="utf-8"))
    result = analyze_coil(payload["bars"])

    # The 2020-02 high and crash occur inside one Q1 candle; later quarters
    # recover immediately, so it is not a confirmed 3M mountain. The two
    # reviewed active tops remain.
    assert [high["date"] for high in result["major_highs"]] == [
        "2021-12-01",
        "2026-02-01",
    ]
    assert [high["price"] for high in result["major_highs"]] == pytest.approx(
        [272.48, 292.02],
        abs=0.02,
    )


def test_msci_reviewed_highs_include_comparable_secondary_line_anchor():
    seed = Path(__file__).parent / "seed_cache" / "MSCI.json"
    import json

    payload = json.loads(seed.read_text(encoding="utf-8"))
    result = analyze_coil(payload["bars"])

    # The 2021 mountain is the only strict major, but one point cannot define
    # a lid slope. The strongest comparable retest is the December 2024 high.
    assert [high["date"] for high in result["major_highs"]] == [
        "2021-11-01",
        "2024-12-01",
    ]
    assert [high["price"] for high in result["major_highs"]] == pytest.approx(
        [679.85, 642.45],
        abs=0.02,
    )
