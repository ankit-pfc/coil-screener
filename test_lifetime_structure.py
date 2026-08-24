from __future__ import annotations

import calendar
from pathlib import Path

import pytest

from lifetime_structure import analyze_lifetime_references
from review_snapshots import load_review_snapshot


SOURCE = "amrut_portfolio_exemplars_2026-08-21.csv"


def _quarter_date(start_year: int, offset: int) -> str:
    year = start_year + offset // 4
    quarter = offset % 4
    month = 3 + quarter * 3
    return f"{year:04d}-{month:02d}-01"


def _bars_with_peaks(
    peaks: dict[int, float],
    *,
    quarter_count: int,
    baseline: float = 40.0,
    start_year: int = 1999,
) -> tuple[list[dict], str]:
    bars = []
    for offset in range(quarter_count):
        high = peaks.get(offset, baseline)
        previous_peak = peaks.get(offset - 1)
        if previous_peak is not None:
            low = min(baseline * 0.72, previous_peak * 0.48)
            close = baseline * 0.86
        else:
            low = high * 0.78
            close = high * 0.88
        bars.append(
            {
                "date": _quarter_date(start_year, offset),
                "open": close * 0.96,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    final = bars[-1]["date"]
    year = int(final[:4])
    month = int(final[5:7])
    as_of = f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
    return bars, as_of


def _primary(analysis: dict) -> dict:
    return next(item for item in analysis["structures"] if item["kind"] == "line")


def test_projected_descending_family_locks_earliest_credible_pair() -> None:
    bars, as_of = _bars_with_peaks(
        {0: 100.0, 16: 90.0, 32: 80.0}, quarter_count=40
    )

    analysis = analyze_lifetime_references(bars, as_of=as_of)
    primary = _primary(analysis)

    assert primary["construction_anchor_ids"] == ["top-1999-03", "top-2003-03"]
    assert primary["supporting_touch_ids"] == ["top-2007-03"]
    assert primary["fit"]["later_touch_count"] == 1


def test_point_in_time_replay_does_not_borrow_future_confirmation() -> None:
    bars, final_as_of = _bars_with_peaks(
        {0: 100.0, 16: 90.0, 32: 80.0}, quarter_count=40
    )
    early_as_of = "2004-03-31"

    early = analyze_lifetime_references(bars, as_of=early_as_of)
    final = analyze_lifetime_references(bars, as_of=final_as_of)

    assert early["structures"] == []
    assert _primary(final)["supporting_touch_ids"] == ["top-2007-03"]


def test_singleton_extreme_is_demoted_for_lower_repeated_family() -> None:
    bars, as_of = _bars_with_peaks(
        {0: 100.0, 12: 60.0, 24: 59.0, 36: 60.5}, quarter_count=44
    )

    analysis = analyze_lifetime_references(bars, as_of=as_of)
    ladder = {item["top_id"]: item["status"] for item in analysis["reference_ladder"]}
    primary = _primary(analysis)

    assert ladder["top-1999-03"] == "demoted_singleton"
    assert primary["construction_anchor_ids"] == ["top-2002-03", "top-2005-03"]
    assert primary["supporting_touch_ids"] == ["top-2008-03"]


def test_recent_unpaired_lifetime_high_tracks_without_rotating_old_line() -> None:
    bars, as_of = _bars_with_peaks(
        {0: 60.0, 16: 59.0, 32: 60.5, 39: 82.0}, quarter_count=42
    )

    analysis = analyze_lifetime_references(bars, as_of=as_of)
    ladder = analysis["reference_ladder"]
    primary = _primary(analysis)

    assert ladder[0]["reference_price"] == 82.0
    assert ladder[0]["status"] == "tracking_new_high"
    assert "top-2008-12" not in primary["construction_anchor_ids"]


@pytest.mark.parametrize(
    ("ticker", "expected_anchors"),
    [
        ("1070.HK", ["top-2010-03", "top-2015-06"]),
        ("0836.HK", ["top-2007-12", "top-2021-12"]),
        ("GMDCLTD.NS", ["top-2007-12", "top-2024-03"]),
        ("0981.HK", ["top-2004-03", "top-2020-09"]),
    ],
)
def test_frozen_teaching_examples_surface_expected_outer_family(
    ticker: str, expected_anchors: list[str]
) -> None:
    snapshot_path = (
        Path(__file__).parent
        / "review_snapshots"
        / "amrut_portfolio_exemplars_2026-08-21"
        / f"{ticker}.json"
    )
    if not snapshot_path.exists():
        pytest.skip("frozen Amrut teaching corpus is not installed")

    snapshot = load_review_snapshot(SOURCE, ticker)
    analysis = analyze_lifetime_references(snapshot["monthly_bars"])

    assert _primary(analysis)["construction_anchor_ids"] == expected_anchors


def test_gmdc_retains_nested_price_family_beside_outer_line() -> None:
    snapshot_path = (
        Path(__file__).parent
        / "review_snapshots"
        / "amrut_portfolio_exemplars_2026-08-21"
        / "GMDCLTD.NS.json"
    )
    if not snapshot_path.exists():
        pytest.skip("frozen Amrut teaching corpus is not installed")

    analysis = analyze_lifetime_references(
        load_review_snapshot(SOURCE, "GMDCLTD.NS")["monthly_bars"]
    )

    assert analysis["structures"][0]["kind"] == "line"
    assert any(
        item["kind"] == "resistance_band"
        and item["relationship"] == "nested_below_outer"
        and item["fit"]["touch_count"] >= 3
        for item in analysis["structures"][1:]
    )
