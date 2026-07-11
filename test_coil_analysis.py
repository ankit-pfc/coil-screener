"""Behavior tests for the deterministic coil-structure analyzer.

Synthetic monthly series are built from piecewise-linear close paths with
exact lid touches (bar high == line value) and exact pullback lows, so the
expected slope, grade, touch count, and depth sequence are known by
construction.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from coil_analysis import (
    DEFAULT_CONFIG,
    SwingPoint,
    _cluster_touches,
    _pivot_high_indexes,
    _tickers_from_csv,
    analyze_coil,
    detect_swing_highs,
    grade_for_slope,
    select_major_highs,
)


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


@pytest.mark.parametrize(
    ("slope", "expected"),
    [
        (-5.0, None),
        (-2.0, "B"),
        (-0.5, "A"),
        (0.0, "A"),
        (1.9, "A"),
        (2.0, "B"),
        (5.9, "B"),
        (6.0, "C"),
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


def test_rising_coil_grades_B():
    spb = spb_for_end_slope(4.0)
    bars = make_coil_bars(slope_per_bar=spb, depths=(0.40, 0.28, 0.20))
    result = analyze_coil(bars)

    assert result["status"] == "coiling"
    assert result["resistance"]["slope_pct_per_year"] == pytest.approx(4.0, abs=0.2)
    assert result["grade"] == "B"


def test_steep_coil_grades_C():
    # The deep mid-base crash keeps the log-close R^2 below the trend gate —
    # a C-grade coil is choppy, not a smooth exponential rise.
    spb = spb_for_end_slope(8.0)
    bars = make_coil_bars(slope_per_bar=spb, depths=(0.45, 0.55, 0.25))
    result = analyze_coil(bars)

    assert result["status"] == "coiling"
    assert result["resistance"]["slope_pct_per_year"] == pytest.approx(8.0, abs=0.3)
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

    assert result["schema_version"] == 1
    assert result["status"] in {"no_structure", "basing", "coiling", "breaking_out", "broken_out"}
    assert result["bar_count"] > 100
    assert isinstance(result["major_highs"], list)
    for high in result["major_highs"]:
        assert set(high) == {"idx", "date", "price", "prominence_pct", "source"}
