from __future__ import annotations

import calendar
from datetime import date

from bar_integrity import inspect_monthly_bars
from coil_analysis import (
    ANALYSIS_MODE_ALGORITHM_ONLY,
    analyze_coil,
    replay_completed_quarter_prefixes,
)
from test_coil_analysis import lid_override, make_coil_bars


def bar(date: str, value: float = 10.0, volume: float | None = 100.0) -> dict:
    return {
        "date": date,
        "open": value,
        "high": value + 1.0,
        "low": value - 1.0,
        "close": value + 0.5,
        "volume": volume,
    }


def test_strict_integrity_reports_every_blocking_ohlc_failure():
    bars = [
        bar("2020-01-01"),
        {**bar("2020-01-01"), "open": 0.0},
        {**bar("2020-03-01"), "low": 12.0},
        {**bar("2020-04-01"), "high": 9.0},
        {**bar("2020-05-01"), "close": float("nan")},
        {**bar("bad-date")},
    ]

    result = inspect_monthly_bars(bars, adjustment_mode="split_adjusted")
    codes = {issue["code"] for issue in result.report["issues"]}

    assert result.report["status"] == "blocked"
    assert result.report["blocked"] is True
    assert codes >= {
        "duplicate_date",
        "nonpositive_ohlc",
        "low_above_body",
        "high_below_body",
        "missing_or_nonfinite_ohlc",
        "invalid_date",
    }


def test_warnings_sort_gaps_volume_staleness_and_adjustment_without_blocking():
    bars = [
        bar("2020-04-01", 50.0, volume=None),
        bar("2020-01-01", 10.0),
    ]

    result = inspect_monthly_bars(
        bars,
        as_of="2020-12-31",
        adjustment_mode="unknown",
    )
    codes = {issue["code"] for issue in result.report["issues"]}

    assert result.report["status"] == "valid_with_warnings"
    assert result.report["blocked"] is False
    assert [item["date"] for item in result.bars] == ["2020-01-01", "2020-04-01"]
    assert codes >= {
        "out_of_order",
        "missing_periods",
        "missing_volume",
        "corporate_action_like_discontinuity",
        "stale_final_data",
        "unverified_adjustment_mode",
    }


def test_historical_cutoff_uses_month_end_availability_not_day_one_label():
    bars = [bar("2020-01-01"), bar("2020-02-01")]

    mid_month = inspect_monthly_bars(
        bars,
        as_of="2020-02-15",
        adjustment_mode="split_adjusted",
    )
    month_end = inspect_monthly_bars(
        bars,
        as_of="2020-02-29",
        adjustment_mode="split_adjusted",
    )

    assert [item["date"] for item in mid_month.bars] == ["2020-01-01"]
    assert [item["date"] for item in month_end.bars] == [
        "2020-01-01",
        "2020-02-01",
    ]
    assert month_end.report["requested_cutoff"] == "2020-02-29"
    assert month_end.report["effective_bar_cutoff"] == "2020-02-29"
    assert month_end.report["source_fingerprint"].startswith("sha256:")
    assert month_end.report["bar_fingerprint"].startswith("sha256:")


def test_invalid_data_blocks_structure_instead_of_becoming_a_negative_label():
    bars = make_coil_bars()
    bars[40]["low"] = bars[40]["high"] + 1.0

    result = analyze_coil(bars)

    assert result["status"] == "invalid_data"
    assert result["resistance"] is None
    assert result["analysis_metadata"]["classification_blocked"] is True


def test_algorithm_only_never_applies_a_supplied_human_override():
    bars = make_coil_bars()
    override = lid_override(bars, (20, 60, 100))

    algorithm = analyze_coil(
        bars,
        review_override=override,
        mode=ANALYSIS_MODE_ALGORITHM_ONLY,
    )
    effective = analyze_coil(bars, review_override=override)

    assert algorithm["review"] == {
        "reviewed": False,
        "effective": "algorithm",
        "analysis_mode": "algorithm_only",
    }
    assert effective["review"]["effective"] == "human"


def test_prefix_replay_uses_fresh_cutoffs_and_never_predates_confirmation():
    bars = make_coil_bars()

    replay = replay_completed_quarter_prefixes(bars)
    final = replay["snapshots"][-1]["analysis"]
    direct = analyze_coil(bars, mode=ANALYSIS_MODE_ALGORITHM_ONLY)

    assert replay["mode"] == "algorithm_only"
    assert final["status"] == direct["status"]
    assert final["resistance"] == direct["resistance"]
    for snapshot in replay["snapshots"]:
        cutoff = snapshot["as_of"]
        for point in snapshot["analysis"]["points"]:
            assert point["confirmed_at"] is None or point["confirmed_at"] <= cutoff
            if point["confirmed_at"] is not None:
                confirmed = date.fromisoformat(point["confirmed_at"])
                assert confirmed.day == calendar.monthrange(
                    confirmed.year, confirmed.month
                )[1]
            assert point["peak_date"] == point["date"]


def test_every_v231_replay_prefix_matches_literal_prefix_and_keeps_major_events():
    bars = make_coil_bars()
    replay = replay_completed_quarter_prefixes(
        bars,
        adjustment_mode="split_adjusted",
        today=date(2021, 1, 1),
    )
    event_ledger: dict[str, tuple[str | None, float]] = {}

    def major_events(analysis: dict) -> dict[str, tuple[str | None, float]]:
        return {
            str(point["date"]): (
                point.get("confirmed_at"),
                float(point["price"]),
            )
            for point in analysis["major_highs"]
        }

    for snapshot in replay["snapshots"]:
        cutoff = date.fromisoformat(snapshot["as_of"])
        literal_prefix = []
        for item in bars:
            parsed = date.fromisoformat(item["date"])
            available = date(
                parsed.year,
                parsed.month,
                calendar.monthrange(parsed.year, parsed.month)[1],
            )
            if available <= cutoff:
                literal_prefix.append(item)
        literal = analyze_coil(
            literal_prefix,
            as_of=snapshot["as_of"],
            mode=ANALYSIS_MODE_ALGORITHM_ONLY,
            adjustment_mode="split_adjusted",
        )
        current = major_events(snapshot["analysis"])

        assert current == major_events(literal)
        for peak_date, frozen_event in event_ledger.items():
            assert current.get(peak_date) == frozen_event
        event_ledger.update(current)


def test_prefix_replay_does_not_manufacture_a_future_quarter_end():
    partial_march = [
        {
            "date": "2026-03-01",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1_000.0,
        }
    ]

    during_month = replay_completed_quarter_prefixes(
        partial_march,
        adjustment_mode="split_adjusted",
        today=date(2026, 3, 15),
    )
    after_month = replay_completed_quarter_prefixes(
        partial_march,
        adjustment_mode="split_adjusted",
        today=date(2026, 4, 1),
    )

    assert during_month["snapshots"] == []
    assert [item["as_of"] for item in after_month["snapshots"]] == ["2026-03-31"]


def test_every_confirmed_structural_point_has_a_confirmation_timestamp():
    result = analyze_coil(make_coil_bars(), mode=ANALYSIS_MODE_ALGORITHM_ONLY)

    assert result["points"]
    for point in result["points"]:
        if point["confirmed"]:
            assert point["confirmed_at"] is not None
            assert point["confirmed_at_idx"] is not None
