from __future__ import annotations

from pathlib import Path

import pytest

import lifetime_benchmark_charts as charts


def _monthly_bars(count: int = 7) -> list[dict]:
    bars = []
    for offset in range(count):
        month = offset + 1
        price = 100.0 + offset
        bars.append(
            {
                "date": f"2026-{month:02d}-01",
                "open": price,
                "high": price + 4.0,
                "low": price - 4.0,
                "close": price + 1.0,
                "volume": 1_000_000.0,
            }
        )
    return bars


def test_scored_structure_geometry_uses_exact_normalized_line() -> None:
    structure = {
        "anchor_points": [
            {"date": "2020-03-01", "price": 100.0},
            {"date": "2021-03-01", "price": 108.0},
        ],
        "geometry_start_ordinal": charts._quarter_ordinal("2020-03-01"),
        "geometry_end_ordinal": charts._quarter_ordinal("2022-03-01"),
        "slope_per_quarter": 2.0,
        "intercept": -16060.0,
        "band": {"lower_pct": 5.0, "upper_pct": 10.0},
    }

    geometry = charts._structure_geometry(
        structure,
        charts._quarter_ordinal("2020-03-01"),
        charts._quarter_ordinal("2022-03-01"),
    )

    assert geometry is not None
    xs, ys, lower, upper = geometry
    assert ys == pytest.approx([100.0 + 2.0 * index for index in range(len(xs))])
    assert lower == pytest.approx([value * 0.95 for value in ys])
    assert upper == pytest.approx([value * 1.10 for value in ys])


def test_partial_quarter_is_detected_and_renderer_smokes(tmp_path: Path) -> None:
    bars = _monthly_bars()
    quarters = charts._aggregate_quarterly_display_bars(bars)
    assert charts._partial_quarter_ordinal(quarters, "2026-07-27") == charts._quarter_ordinal(
        "2026-07-01"
    )

    output = tmp_path / "partial.png"
    charts.render_benchmark_chart(
        "TEST",
        bars,
        {"as_of": "2026-07-27", "bar_count": len(bars)},
        {"as_of": "2026-07-27", "top_episodes": [], "structures": []},
        output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
