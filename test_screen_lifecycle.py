from __future__ import annotations

import pandas as pd
import pytest

import coil_analysis
import history_cache
import reviews as reviews_module
import screen_monthly
from screen_monthly import run_lifecycle_screen


def test_split_only_adjustment_uses_reported_events_not_adj_close():
    daily = pd.DataFrame(
        {
            "Open": [98.0, 50.0, 54.0],
            "High": [102.0, 53.0, 57.0],
            "Low": [97.0, 49.0, 53.0],
            "Close": [100.0, 52.0, 55.0],
            "Adj Close": [80.0, 40.0, 43.0],
            "Volume": [1_000.0, 2_000.0, 2_200.0],
            "Stock Splits": [0.0, 2.0, 0.0],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-04"]),
    )

    monthly = screen_monthly._split_adjust_and_aggregate_monthly(daily)

    assert monthly.attrs["adjustment_mode"] == "split_adjusted"
    assert monthly.loc["2020-01-01", "Close"] == 50.0
    assert monthly.loc["2020-01-01", "Volume"] == 2_000.0
    assert monthly.loc["2020-02-01", "Open"] == 50.0
    assert monthly.loc["2020-02-01", "Close"] == 55.0


def test_split_adjustment_fails_closed_without_event_provenance():
    daily = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [10.0],
        },
        index=pd.to_datetime(["2020-01-02"]),
    )

    with pytest.raises(ValueError, match="missing Stock Splits"):
        screen_monthly._split_adjust_and_aggregate_monthly(daily)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), "bad", -2.0])
def test_split_adjustment_rejects_invalid_action_values(invalid):
    daily = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [10.0],
            "Stock Splits": [invalid],
        },
        index=pd.to_datetime(["2020-01-02"]),
    )

    with pytest.raises(ValueError, match="invalid split factor"):
        screen_monthly._split_adjust_and_aggregate_monthly(daily)


def test_monthly_fetch_requests_raw_daily_actions_and_marks_provenance(monkeypatch):
    seen = {}
    daily = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [10.5],
            "Volume": [1_000.0],
            "Stock Splits": [0.0],
        },
        index=pd.to_datetime(["2020-01-02"]),
    )

    def download(ticker, **kwargs):
        seen["ticker"] = ticker
        seen.update(kwargs)
        return daily

    monkeypatch.setattr(screen_monthly.yf, "download", download)

    monthly = screen_monthly.fetch_monthly_history("TEST")

    assert seen["ticker"] == "TEST"
    assert seen["interval"] == "1d"
    assert seen["auto_adjust"] is False
    assert seen["actions"] is True
    assert monthly is not None
    assert monthly.attrs["adjustment_mode"] == "split_adjusted"


def _analysis(
    lifecycle: str,
    score: float,
    grade: str | None = None,
    reviewed: bool = False,
    proximity: float = 98.0,
    position: str = "within_lid_band",
):
    return {
        "lifecycle": lifecycle,
        "status": lifecycle,
        "grade": grade,
        "coil_score": score,
        "as_of": "2026-06-01",
        "active_lid": {
            "grade": grade,
            "slope_pct_per_year": -1.2,
            "span_years": 9.0,
            "touch_count": 3,
        },
        "resistance": None,
        "metrics": {
            "proximity_pct": proximity,
            "current_price_position": position,
        },
        "review": {"reviewed": reviewed},
    }


def _no_structure_analysis():
    """A lid-less result: metrics is absent entirely, not an empty dict."""
    return {
        "lifecycle": "no_structure",
        "status": "no_structure",
        "grade": None,
        "coil_score": 0.0,
        "as_of": "2026-06-01",
        "active_lid": None,
        "resistance": None,
        "metrics": None,
        "review": {"reviewed": False},
    }


def _payload(symbol: str):
    return {
        "ticker": symbol,
        "bars": [{"ticker": symbol, "close": 42.5}],
        "features": None,
        "freshness": {
            "status": "fresh",
            "last_bar_date": "2026-06-01",
            "fetched_at": "2026-07-11T00:00:00.000Z",
            "source": "test",
        },
    }


def test_lifecycle_screen_groups_ranks_reviews_and_reports_failures(monkeypatch):
    analyses = {
        "A_LOW": _analysis("pre_breakout", 60.0, "A"),
        "A_HIGH": _analysis("pre_breakout", 80.0, "A", reviewed=True),
        "B_HIGH": _analysis("pre_breakout", 99.0, "B"),
        "FORM": _analysis("forming", 70.0),
        "POST": _analysis("post_breakout", 90.0),
    }
    seen_overrides = []

    def history(symbol, *args, **kwargs):
        if symbol == "FAIL":
            return None
        return {
            "ticker": symbol,
            "bars": [{"ticker": symbol, "close": 42.5}],
            # Deliberately absent: structural analysis is not gated on legacy
            # compute_features returning a numeric score.
            "features": None,
            "freshness": {
                "status": "fresh",
                "last_bar_date": "2026-06-01",
                "fetched_at": "2026-07-11T00:00:00.000Z",
                "source": "test",
            },
        }

    def analyze(bars, review_override=None, **kwargs):
        symbol = bars[0]["ticker"]
        seen_overrides.append((symbol, review_override))
        return analyses[symbol]

    monkeypatch.setattr(history_cache, "get_history_payload", history)
    monkeypatch.setattr(coil_analysis, "analyze_coil", analyze)
    run = run_lifecycle_screen(
        ["B_HIGH", "POST", "A_LOW", "FAIL", "FORM", "A_HIGH"],
        fetch_monthly=lambda _: None,
        review_override_for=lambda symbol: {"review_id": 7}
        if symbol == "A_HIGH"
        else None,
    )

    assert [row["ticker"] for row in run["results"]] == [
        "A_HIGH",
        "A_LOW",
        "B_HIGH",
        "FORM",
        "POST",
    ]
    assert run["results"][0]["reviewed"] is True
    assert run["results"][0]["last_close"] == 42.5
    assert run["results"][0]["age_years"] == 0.0833
    assert run["results"][0]["score_total"] is None
    assert ("A_HIGH", {"review_id": 7}) in seen_overrides
    assert run["bucket_counts"]["pre_breakout"] == 3
    assert run["bucket_counts"]["forming"] == 1
    assert run["bucket_counts"]["post_breakout"] == 1
    assert run["failures"] == [
        {"ticker": "FAIL", "error": "No usable monthly history found."}
    ]
    assert run["algorithm_version"] == coil_analysis.ALGORITHM_VERSION
    assert run["screened_at"].endswith("Z")


def test_algorithm_only_screen_skips_review_callbacks(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        history_cache, "get_history_payload", lambda symbol, *a, **k: _payload(symbol)
    )

    def analyze(bars, **kwargs):
        seen.update(kwargs)
        return _analysis("forming", 40.0)

    monkeypatch.setattr(coil_analysis, "analyze_coil", analyze)

    def forbidden(*args, **kwargs):
        raise AssertionError("algorithm_only must not read review state")

    run = run_lifecycle_screen(
        ["PURE"],
        fetch_monthly=lambda _: None,
        review_override_for=forbidden,
        review_state_for=forbidden,
        analysis_mode="algorithm_only",
    )

    assert len(run["results"]) == 1
    assert run["analysis_variant"] == "v2_3_1"
    assert run["analysis_mode"] == "algorithm_only"
    assert seen["variant"] == "v2_3_1"
    assert seen["mode"] == "algorithm_only"


def test_algorithm_only_dataframe_wrapper_never_opens_review_store(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        reviews_module,
        "get_review_store",
        lambda: (_ for _ in ()).throw(
            AssertionError("algorithm_only must not open the review store")
        ),
    )

    def fake_run(tickers, **kwargs):
        seen.update(kwargs)
        return {
            "results": [],
            "bucket_counts": {},
            "failures": [],
            "algorithm_version": "test",
            "analysis_variant": kwargs["analysis_variant"],
            "analysis_mode": kwargs["analysis_mode"],
            "screened_at": "2026-01-01T00:00:00.000Z",
        }

    monkeypatch.setattr(screen_monthly, "run_lifecycle_screen", fake_run)

    result = screen_monthly.run_screen(["PURE"], analysis_mode="algorithm_only")

    assert result.empty
    assert seen["review_override_for"] is None
    assert seen["analysis_mode"] == "algorithm_only"


def test_blocked_screen_failure_retains_complete_integrity_report(monkeypatch):
    quality = {
        "status": "blocked",
        "blocked": True,
        "bar_fingerprint": "sha256:test",
        "adjustment_mode": "unknown",
        "issues": [
            {
                "severity": "error",
                "code": "low_above_high",
                "date": "2020-01-01",
            }
        ],
    }
    monkeypatch.setattr(
        history_cache, "get_history_payload", lambda symbol, *a, **k: _payload(symbol)
    )
    monkeypatch.setattr(
        coil_analysis,
        "analyze_coil",
        lambda bars, **kwargs: {
            "analysis_metadata": {"data_quality": quality},
            "review": {"reviewed": False},
        },
    )

    run = run_lifecycle_screen(
        ["BAD"],
        fetch_monthly=lambda _: None,
        analysis_mode="algorithm_only",
    )

    assert run["results"] == []
    assert run["failures"][0]["data_quality"] == quality


def test_rows_carry_the_lid_band_position_next_to_the_raw_ratio(monkeypatch):
    """v2.2 ships both: the ratio for diagnostics, the enum for classification.

    Every band value must survive the lift out of ``metrics``, and a lid-less
    result must report ``None`` rather than inventing a placement.
    """
    analyses = {
        "BELOW": _analysis("forming", 40.0, proximity=61.5, position="below_lid_band"),
        "INSIDE": _analysis(
            "pre_breakout", 80.0, "A", proximity=96.0, position="within_lid_band"
        ),
        "ABOVE": _analysis(
            "post_breakout", 30.0, proximity=143.2, position="above_lid_band"
        ),
        "NONE": _no_structure_analysis(),
    }

    monkeypatch.setattr(
        history_cache, "get_history_payload", lambda symbol, *a, **k: _payload(symbol)
    )
    monkeypatch.setattr(
        coil_analysis, "analyze_coil", lambda bars, **k: analyses[bars[0]["ticker"]]
    )
    run = run_lifecycle_screen(
        ["BELOW", "INSIDE", "ABOVE", "NONE"], fetch_monthly=lambda _: None
    )

    rows = {row["ticker"]: row for row in run["results"]}
    assert {t: rows[t]["current_price_position"] for t in analyses} == {
        "BELOW": "below_lid_band",
        "INSIDE": "within_lid_band",
        "ABOVE": "above_lid_band",
        "NONE": None,
    }
    # The raw ratio is not replaced by the enum; both travel together.
    assert [rows[t]["proximity_pct"] for t in ("BELOW", "INSIDE", "ABOVE", "NONE")] == [
        61.5,
        96.0,
        143.2,
        None,
    ]


def test_csv_export_path_keeps_the_lid_band_position(monkeypatch, tmp_path):
    """Saved runs are DataFrame -> CSV, so the enum must survive the round trip."""
    monkeypatch.setattr(
        history_cache, "get_history_payload", lambda symbol, *a, **k: _payload(symbol)
    )
    monkeypatch.setattr(
        coil_analysis,
        "analyze_coil",
        lambda bars, **k: _analysis(
            "post_breakout", 30.0, proximity=143.2, position="above_lid_band"
        ),
    )
    monkeypatch.setattr(
        reviews_module,
        "get_review_store",
        lambda: type("Store", (), {"get_override": lambda self, *args: None})(),
    )

    df = screen_monthly.run_screen(["BRK"])
    target = tmp_path / "run.csv"
    df.to_csv(target, index=False)
    reloaded = pd.read_csv(target)

    assert reloaded.iloc[0]["current_price_position"] == "above_lid_band"
    assert float(reloaded.iloc[0]["proximity_pct"]) == 143.2


def test_dataframe_screen_wrapper_preserves_v2_results_and_envelope(monkeypatch):
    monkeypatch.setattr(
        screen_monthly,
        "run_lifecycle_screen",
        lambda tickers, **kwargs: {
            "results": [
                {
                    "ticker": "TEST",
                    "lifecycle": "pre_breakout",
                    "grade": "A",
                    "coil_score": 88.0,
                }
            ],
            "bucket_counts": {"pre_breakout": 1},
            "failures": [],
            "algorithm_version": "v2-test",
            "screened_at": "2026-07-11T00:00:00.000Z",
        },
    )
    monkeypatch.setattr(
        reviews_module,
        "get_review_store",
        lambda: type("Store", (), {"get_override": lambda self, *args: None})(),
    )

    result = screen_monthly.run_screen(["TEST"])

    assert result.iloc[0]["lifecycle"] == "pre_breakout"
    assert result.attrs["bucket_counts"] == {"pre_breakout": 1}
    assert result.attrs["algorithm_version"] == "v2-test"
