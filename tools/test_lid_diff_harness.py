"""Behavioural tests for the lid diff harness.

The harness is a measurement instrument, so its extraction and aggregation
functions are tested against realistic ``analyze_coil``-shaped payloads and
snapshot dicts.  The loader is tested against throwaway stub repo trees, which
proves the two-tree isolation contract without importing the real analyzer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import lid_diff_harness as h


# --------------------------------------------------------------------------
# fixtures / builders
# --------------------------------------------------------------------------


def make_result(
    *,
    grade="A",
    lifecycle="pre_breakout",
    status="coiling",
    coil_score=71.5,
    proximity_pct=98.4,
    current_price_position=None,
    anchors=((36, "2017-02-01", 19.82), (95, "2022-01-01", 23.81)),
    slope=2.96,
    lid_value=27.79,
    points=(),
    notes=(),
    structure=True,
):
    """A realistically shaped ``analyze_coil`` result dict."""
    metrics = {
        "proximity_pct": proximity_pct,
        "base_years": 4.75,
        "compression_ok": True,
        "violation_count": 0,
    }
    if current_price_position is not None:
        metrics["current_price_position"] = current_price_position

    active_lid = None
    if structure:
        active_lid = {
            "from": {"idx": anchors[0][0], "date": anchors[0][1], "price": anchors[0][2]},
            "to": {"idx": anchors[-1][0], "date": anchors[-1][1], "price": anchors[-1][2]},
            "anchors": [{"idx": i, "date": d, "price": p} for i, d, p in anchors],
            "slope_pct_per_year": slope,
            "grade": grade,
            "value_at_last_bar": lid_value,
            "touch_count": len(anchors),
            "span_years": 4.75,
            "source": "timeseries",
        }

    return {
        "schema_version": 2,
        "algorithm_version": "2.1.0",
        "as_of": "2026-07-01",
        "bar_count": 150,
        "grade": grade,
        "lifecycle": lifecycle if structure else "no_structure",
        "status": status if structure else "no_structure",
        "coil_score": coil_score,
        "active_lid": active_lid,
        "metrics": metrics if structure else None,
        "points": list(points),
        "major_highs": [
            {"idx": i, "date": d, "price": p, "prominence_pct": 20.0} for i, d, p in anchors
        ]
        if structure
        else [],
        "notes": list(notes),
    }


def point(idx, date, price, role, lid_member):
    return {
        "idx": idx,
        "date": date,
        "price": price,
        "prominence_pct": 18.0,
        "role": role,
        "confirmed": role != h.ROLE_PROVISIONAL_TOP,
        "lid_member": lid_member,
        "source": "timeseries",
        "evidence": {},
    }


def record(ticker, **over):
    """A snapshot record with sane defaults, overridable field by field."""
    base = h.build_record(ticker, make_result())
    base.update(over)
    if "anchors" in over:
        base["anchor_count"] = len(over["anchors"])
    return base


# --------------------------------------------------------------------------
# build_record
# --------------------------------------------------------------------------


def test_build_record_extracts_the_outcome_fields():
    rec = h.build_record(
        "NSC",
        make_result(grade="B", lifecycle="pre_breakout", proximity_pct=99.66, slope=4.1),
        last_close=274.5,
    )
    assert rec["ticker"] == "NSC"
    assert rec["ok"] is True and rec["error"] is None
    assert rec["grade"] == "B"
    assert rec["lifecycle"] == "pre_breakout"
    assert rec["proximity_pct"] == 99.66
    assert rec["slope_pct_per_year"] == 4.1
    assert rec["last_close"] == 274.5
    assert rec["has_structure"] is True
    assert rec["anchor_count"] == 2
    assert rec["anchors"] == [
        {"idx": 36, "date": "2017-02-01", "price": 19.82},
        {"idx": 95, "date": "2022-01-01", "price": 23.81},
    ]


def test_build_record_reports_missing_current_price_position_as_null():
    """Pre-2.2.0 results have no ``current_price_position``; that must not raise."""
    rec = h.build_record("UNP", make_result())
    assert rec["current_price_position"] is None

    rec_after = h.build_record(
        "UNP", make_result(current_price_position="above_lid_band")
    )
    assert rec_after["current_price_position"] == "above_lid_band"


def test_build_record_counts_provisional_top_used_as_a_lid_anchor():
    result = make_result(
        anchors=((36, "2017-02-01", 19.82), (140, "2026-06-01", 30.0)),
        points=(
            point(36, "2017-02-01", 19.82, "major_top", True),
            point(88, "2021-05-01", 22.0, "structural_retest", False),
            point(140, "2026-06-01", 30.0, h.ROLE_PROVISIONAL_TOP, True),
        ),
    )
    rec = h.build_record("CSX", result)
    assert rec["provisional_top_count"] == 1
    assert rec["provisional_anchor_idxs"] == [140]
    assert rec["uses_provisional_anchor"] is True
    assert rec["role_counts"] == {"major_top": 1, "structural_retest": 1, "provisional_top": 1}
    assert rec["role_point_count"] == 3


def test_build_record_provisional_top_that_is_not_an_anchor_is_not_counted_as_one():
    """Emitting the role and anchoring the lid to it are different failures."""
    result = make_result(
        anchors=((36, "2017-02-01", 19.82), (95, "2022-01-01", 23.81)),
        points=(
            point(36, "2017-02-01", 19.82, "major_top", True),
            point(95, "2022-01-01", 23.81, "major_top", True),
            point(140, "2026-06-01", 30.0, h.ROLE_PROVISIONAL_TOP, False),
        ),
    )
    rec = h.build_record("SIE.DE", result)
    assert rec["provisional_top_count"] == 1
    assert rec["provisional_top_lid_member_count"] == 0
    assert rec["provisional_anchor_idxs"] == []
    assert rec["uses_provisional_anchor"] is False


def test_build_record_handles_a_no_structure_result():
    rec = h.build_record("AAPL", make_result(grade=None, structure=False))
    assert rec["has_structure"] is False
    assert rec["anchors"] == []
    assert rec["anchor_count"] == 0
    assert rec["slope_pct_per_year"] is None
    assert rec["proximity_pct"] is None
    assert rec["lid_value_at_last_bar"] is None
    assert rec["lifecycle"] == "no_structure"


def test_error_record_has_the_same_keys_as_a_good_record():
    """Diff/aggregation reads records by key; an error row must not KeyError."""
    good = h.build_record("OK", make_result())
    bad = h.error_record("BOOM", ValueError("no cached bars for BOOM"))
    assert set(bad) == set(good)
    assert bad["ok"] is False
    assert bad["error"] == "ValueError: no cached bars for BOOM"


# --------------------------------------------------------------------------
# last_close_at
# --------------------------------------------------------------------------


def test_last_close_at_respects_the_as_of_cutoff():
    bars = [
        {"date": "2025-08-01", "close": 10.0},
        {"date": "2025-09-01", "close": 11.0},
        {"date": "2025-10-01", "close": 12.0},
    ]
    assert h.last_close_at(bars, None) == 12.0
    assert h.last_close_at(bars, "2025-09-30") == 11.0
    assert h.last_close_at(bars, "2025-09-01") == 11.0
    assert h.last_close_at(bars, "2025-07-31") is None


def test_last_close_at_skips_unusable_closes_and_empty_input():
    bars = [
        {"date": "2025-08-01", "close": 10.0},
        {"date": "2025-09-01", "close": None},
        {"date": "2025-10-01", "close": "not-a-number"},
    ]
    assert h.last_close_at(bars, None) == 10.0
    assert h.last_close_at([], None) is None


# --------------------------------------------------------------------------
# slope_distribution
# --------------------------------------------------------------------------


def test_slope_distribution_buckets_on_the_grade_band_edges():
    dist = h.slope_distribution([-4.0, -3.0, -1.0, 4.99, 5.0, 6.5, 12.0, 13.4])
    assert dist["buckets"]["<-3 (too negative)"] == 1          # -4.0
    assert dist["buckets"]["-3..-1"] == 1                      # -3.0
    assert dist["buckets"]["-1..5 (A band)"] == 2              # -1.0, 4.99
    assert dist["buckets"]["5..6.5 (B band)"] == 1             # 5.0
    assert dist["buckets"]["6.5..12 (C band)"] == 1            # 6.5
    assert dist["buckets"][">=12 (too steep)"] == 2            # 12.0, 13.4
    assert dist["count"] == 8


def test_slope_distribution_ignores_nulls_and_reports_spread():
    dist = h.slope_distribution([3.0, None, 1.0, None, 5.0])
    assert dist["count"] == 3
    assert dist["min"] == 1.0
    assert dist["median"] == 3.0
    assert dist["max"] == 5.0
    assert h.slope_distribution([None, None])["count"] == 0
    assert h.slope_distribution([None])["median"] is None


# --------------------------------------------------------------------------
# diff_records
# --------------------------------------------------------------------------


def test_diff_records_flags_only_the_fields_that_actually_moved():
    before = record("LOSER", grade="A", lifecycle="pre_breakout", proximity_pct=99.5)
    after = record("LOSER", grade=None, lifecycle="post_breakout", proximity_pct=133.0)
    row = h.diff_records(before, after)
    assert row["grade_before"] == "A" and row["grade_after"] is None
    assert row["lifecycle_before"] == "pre_breakout" and row["lifecycle_after"] == "post_breakout"
    assert row["proximity_pct_before"] == 99.5 and row["proximity_pct_after"] == 133.0
    assert set(row["changed_fields"]) == {"grade", "lifecycle", "proximity_pct"}
    assert row["anchors_changed"] is False
    assert row["changed"] is True


def test_diff_records_detects_anchor_moves_and_ignores_identical_ones():
    before = record("ENB.TO")
    same = h.diff_records(before, record("ENB.TO"))
    assert same["anchors_changed"] is False
    assert same["changed"] is False

    moved = h.diff_records(
        before,
        record(
            "ENB.TO",
            anchors=[
                {"idx": 36, "date": "2017-02-01", "price": 19.82},
                {"idx": 110, "date": "2023-05-01", "price": 25.0},
            ],
        ),
    )
    assert moved["anchors_changed"] is True
    assert "anchors" in moved["changed_fields"]


def test_diff_records_handles_a_ticker_missing_from_one_side():
    row = h.diff_records(None, record("NEW"))
    assert row["ticker"] == "NEW"
    assert row["present_before"] is False and row["present_after"] is True
    assert row["anchors_before"] is None
    assert row["changed"] is True


def test_diff_records_does_not_flag_a_row_purely_for_the_new_position_field():
    """current_price_position is new metadata derived from proximity; on its own
    it must not mark every one of the 79 tickers as changed."""
    before = record("SAME", current_price_position=None)
    after = record("SAME", current_price_position="within_lid_band")
    row = h.diff_records(before, after)
    assert row["current_price_position_before"] is None
    assert row["current_price_position_after"] == "within_lid_band"
    assert row["changed_fields"] == []
    assert row["changed"] is False


# --------------------------------------------------------------------------
# build_diff aggregates
# --------------------------------------------------------------------------


@pytest.fixture
def scenario():
    """Six tickers on both sides plus one after-only, covering every bucket."""
    before = {
        "repo": "/frozen/baseline",
        "as_of": None,
        "algorithm_versions": ["2.1.0"],
        "records": {
            # ungraded with structure -> blows up on the after side
            "BOOM": record("BOOM", grade=None, lifecycle="forming", slope_pct_per_year=6.0,
                           proximity_pct=60.0),
            # ungraded -> graded
            "GAINER": record("GAINER", grade=None, lifecycle="forming", slope_pct_per_year=8.0,
                             proximity_pct=70.0),
            # graded -> rejected above the band, lid retained
            "LOSER": record("LOSER", grade="A", lifecycle="pre_breakout", slope_pct_per_year=3.0,
                            proximity_pct=99.5, provisional_top_count=1,
                            provisional_anchor_idxs=[140], uses_provisional_anchor=True),
            # structure disappears entirely
            "LOSTSTRUCT": record("LOSTSTRUCT", grade=None, lifecycle="forming",
                                 slope_pct_per_year=2.0, proximity_pct=88.0,
                                 provisional_top_count=1, provisional_anchor_idxs=[120],
                                 uses_provisional_anchor=True),
            # keeps its grade but stops anchoring to the live edge
            "PROVONLY": record("PROVONLY", grade="A", lifecycle="pre_breakout",
                               slope_pct_per_year=4.0, proximity_pct=98.0,
                               provisional_top_count=1, provisional_anchor_idxs=[130],
                               uses_provisional_anchor=True),
            # untouched
            "SAME": record("SAME", grade="A", lifecycle="pre_breakout", slope_pct_per_year=0.5,
                           proximity_pct=96.0),
        },
    }
    after = {
        "repo": "/live/tree",
        "as_of": None,
        "algorithm_versions": ["2.2.0"],
        "records": {
            "BOOM": h.error_record("BOOM", ValueError("no cached bars for BOOM")),
            "GAINER": record("GAINER", grade="B", lifecycle="pre_breakout",
                             slope_pct_per_year=5.5, proximity_pct=95.0,
                             current_price_position="within_lid_band",
                             anchors=[{"idx": 20, "date": "2016-01-01", "price": 9.0},
                                      {"idx": 95, "date": "2022-01-01", "price": 23.81}]),
            "LOSER": record("LOSER", grade=None, lifecycle="post_breakout",
                            slope_pct_per_year=1.5, proximity_pct=133.0,
                            current_price_position="above_lid_band",
                            anchors=[{"idx": 36, "date": "2017-02-01", "price": 19.82},
                                     {"idx": 110, "date": "2023-05-01", "price": 25.0}],
                            notes=["last close 133% of lid — above the +20% band"]),
            "LOSTSTRUCT": record("LOSTSTRUCT", grade=None, lifecycle="no_structure",
                                 status="no_structure", has_structure=False, anchors=[],
                                 slope_pct_per_year=None, proximity_pct=None,
                                 lid_value_at_last_bar=None),
            "PROVONLY": record("PROVONLY", grade="A", lifecycle="pre_breakout",
                               slope_pct_per_year=4.2, proximity_pct=97.0,
                               current_price_position="within_lid_band",
                               anchors=[{"idx": 36, "date": "2017-02-01", "price": 19.82},
                                        {"idx": 118, "date": "2023-11-01", "price": 26.4}]),
            "SAME": record("SAME", grade="A", lifecycle="pre_breakout", slope_pct_per_year=0.5,
                           proximity_pct=96.0, current_price_position="within_lid_band"),
            "ONLYAFTER": record("ONLYAFTER", grade="C", lifecycle="pre_breakout",
                                slope_pct_per_year=7.0, proximity_pct=92.0,
                                current_price_position="within_lid_band"),
        },
    }
    return before, after


def test_build_diff_side_totals(scenario):
    diff = h.build_diff(*scenario)
    b, a = diff["before"], diff["after"]

    assert b["tickers"] == 6 and a["tickers"] == 7
    assert b["ok"] == 6 and a["ok"] == 6
    assert b["errors"] == [] and a["errors"] == ["BOOM"]
    assert b["graded"] == 3            # LOSER, PROVONLY, SAME
    assert a["graded"] == 4            # GAINER, PROVONLY, SAME, ONLYAFTER
    assert b["with_structure"] == 6
    assert a["with_structure"] == 5    # BOOM and LOSTSTRUCT lost theirs
    assert b["provisional_emitted"] == 3
    assert b["provisional_as_anchor"] == 3
    assert a["provisional_emitted"] == 0
    assert a["provisional_as_anchor"] == 0


def test_build_diff_transition_buckets(scenario):
    diff = h.build_diff(*scenario)
    s = diff["summary"]

    assert s["tickers_compared"] == 6
    assert s["only_in_before"] == []
    assert s["only_in_after"] == ["ONLYAFTER"]

    assert s["newly_rejected_count"] == 1
    assert [e["ticker"] for e in s["newly_rejected"]] == ["LOSER"]
    assert s["newly_rejected"][0]["current_price_position"] == "above_lid_band"
    assert s["newly_rejected"][0]["grade_before"] == "A"
    assert s["newly_rejected"][0]["has_structure_after"] is True
    assert s["newly_rejected_by_position"] == {"above_lid_band": 1}

    assert s["newly_graded_count"] == 1
    assert s["newly_graded"][0] == {
        "ticker": "GAINER",
        "grade_after": "B",
        "lifecycle_before": "forming",
        "proximity_pct_after": 95.0,
        "current_price_position": "within_lid_band",
    }

    assert s["lost_structure"] == ["BOOM", "LOSTSTRUCT"]
    assert s["lost_structure_count"] == 2
    assert s["gained_structure_count"] == 0
    assert s["grade_changed"] == ["GAINER", "LOSER"]
    assert sorted(s["lifecycle_changed"]) == ["BOOM", "GAINER", "LOSER", "LOSTSTRUCT"]
    assert s["anchors_changed_count"] == 5   # everything except SAME
    assert "SAME" not in s["anchors_changed"]
    assert s["changed"] == 6                 # SAME is the only unchanged row


def test_build_diff_grade_and_slope_distributions(scenario):
    diff = h.build_diff(*scenario)
    assert diff["before"]["grade_counts"] == {"A": 3, "None": 3}
    assert diff["after"]["grade_counts"] == {"A": 2, "B": 1, "C": 1, "None": 3}

    bd = diff["before"]["slope_distribution"]
    ad = diff["after"]["slope_distribution"]
    assert bd["count"] == 6 and ad["count"] == 5
    assert bd["median"] == 3.5
    assert bd["min"] == 0.5 and bd["max"] == 8.0
    assert ad["median"] == 4.2
    assert ad["min"] == 0.5 and ad["max"] == 7.0
    assert bd["buckets"]["-1..5 (A band)"] == 4
    assert ad["buckets"]["-1..5 (A band)"] == 3


def test_build_diff_survives_an_empty_before_snapshot():
    after = {"records": {"X": record("X")}}
    diff = h.build_diff({"records": {}}, after)
    assert diff["summary"]["tickers_compared"] == 0
    assert diff["summary"]["only_in_after"] == ["X"]
    assert diff["before"]["slope_distribution"]["count"] == 0
    assert diff["summary"]["newly_graded_count"] == 0


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_format_anchor_key_disambiguates_from_the_before_after_arrow():
    assert h.format_anchor_key(None) == "-"
    assert h.format_anchor_key([]) == "[]"
    rendered = h.format_anchor_key([["2017-02-01", 19.82], ["2022-01-01", 23.81]])
    assert rendered == "[2017-02-01@19.82, 2022-01-01@23.81]"
    assert "->" not in rendered


def test_render_report_carries_the_headline_numbers(scenario):
    report = h.render_report(h.build_diff(*scenario))
    assert "| provisional_top used as lid anchor | 3 | 0 |" in report
    assert "| graded (grade not null) | 3 | 4 |" in report
    assert "newly rejected (graded -> ungraded): **1**" in report
    assert "`current_price_position=above_lid_band`: 1" in report
    assert "lost structure entirely: **2** (BOOM, LOSTSTRUCT)" in report
    # unchanged rows are omitted from the per-ticker table
    assert "| LOSER |" in report
    assert "| SAME |" not in report


def test_render_report_handles_a_no_change_diff():
    snapshot = {"records": {"SAME": record("SAME")}}
    report = h.render_report(h.build_diff(snapshot, json.loads(json.dumps(snapshot))))
    assert "_No ticker changed._" in report


# --------------------------------------------------------------------------
# replay assertions
# --------------------------------------------------------------------------


def test_evaluate_expectation_equality_null_and_range():
    rec = record("KN", grade="A", proximity_pct=99.4, current_price_position=None)

    passing = h.evaluate_expectation(
        rec, {"grade": "A", "current_price_position": None, "proximity_pct": [95, 105]}
    )
    assert all(c["pass"] for c in passing)

    failing = h.evaluate_expectation(
        rec, {"grade": "B", "proximity_pct": [110, 120]}
    )
    assert [c["pass"] for c in failing] == [False, False]
    assert failing[0]["actual"] == "A"
    assert failing[1]["expected"] == "[110, 120]"


def test_evaluate_expectation_rejects_a_missing_field():
    checks = h.evaluate_expectation(record("KN"), {"not_a_field": "x"})
    assert checks[0]["pass"] is False
    assert checks[0]["actual"] is None


def test_seeded_replay_assertion_covers_kn_at_the_reference_date():
    kn = [c for c in h.REPLAY_ASSERTIONS if c["ticker"] == "KN"]
    assert len(kn) == 1
    assert kn[0]["as_of"] == "2025-09-30"
    assert kn[0]["expect"]["grade"] == "A"


# --------------------------------------------------------------------------
# repo tree loading / run mode, against throwaway stub trees
# --------------------------------------------------------------------------


STUB_COIL_ANALYSIS = '''
ALGORITHM_VERSION = "{version}"


class _Config:
    pass


DEFAULT_CONFIG = _Config()


def analyze_coil(bars, config=DEFAULT_CONFIG, as_of=None, review_override=None):
    ticker = bars[0]["ticker"]
    if ticker == "BOOM":
        raise ZeroDivisionError("synthetic analyzer failure")
    used = [b for b in bars if as_of is None or b["date"] <= as_of]
    return {{
        "schema_version": 2,
        "algorithm_version": ALGORITHM_VERSION,
        "as_of": used[-1]["date"],
        "bar_count": len(used),
        "grade": "A" if ticker == "GOOD" else None,
        "lifecycle": "pre_breakout",
        "status": "coiling",
        "coil_score": 50.0,
        "active_lid": {{
            "from": {{"idx": 0, "date": bars[0]["date"], "price": 10.0}},
            "to": {{"idx": 1, "date": bars[1]["date"], "price": 11.0}},
            "anchors": [
                {{"idx": 0, "date": bars[0]["date"], "price": 10.0}},
                {{"idx": 1, "date": bars[1]["date"], "price": 11.0}},
            ],
            "slope_pct_per_year": 3.0,
            "grade": "A",
            "value_at_last_bar": 11.5,
            "touch_count": 2,
            "span_years": 1.0,
        }},
        "metrics": {{"proximity_pct": 95.0}},
        "points": [],
        "major_highs": [],
        "notes": [],
    }}
'''

STUB_HISTORY_CACHE = '''
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"


def read_cache(symbol):
    path = CACHE_DIR / (symbol + ".json")
    if not path.exists():
        return None
    return json.loads(path.read_text())
'''


def write_stub_repo(root: Path, version: str, tickers, cache_name: str = "cache") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "coil_analysis.py").write_text(STUB_COIL_ANALYSIS.format(version=version))
    (root / "history_cache.py").write_text(STUB_HISTORY_CACHE)
    cache = root / cache_name
    cache.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        bars = [
            {"ticker": ticker, "date": "2024-01-01", "close": 10.0},
            {"ticker": ticker, "date": "2025-01-01", "close": 11.0},
            {"ticker": ticker, "date": "2026-01-01", "close": 12.0},
        ]
        (cache / f"{ticker}.json").write_text(json.dumps({"ticker": ticker, "bars": bars}))
    return cache


def test_list_tickers_reads_dotted_symbols_and_skips_scratch_files(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("AAPL.json", "0005.HK.json", "000660.KS.json", "AAPL.json.tmp.json"):
        (cache / name).write_text("{}")
    (cache / "notes.txt").write_text("ignored")
    assert h.list_tickers(cache) == ["0005.HK", "000660.KS", "AAPL"]


def test_run_snapshot_analyzes_every_cached_ticker_and_survives_one_failure(tmp_path):
    repo = tmp_path / "tree"
    write_stub_repo(repo, "9.9.9", ["GOOD", "BOOM", "MEH"])

    snapshot = h.run_snapshot(repo)

    assert snapshot["ticker_count"] == 3
    assert sorted(snapshot["records"]) == ["BOOM", "GOOD", "MEH"]
    assert snapshot["algorithm_versions"] == ["9.9.9"]

    boom = snapshot["records"]["BOOM"]
    assert boom["ok"] is False
    assert "ZeroDivisionError" in boom["error"]
    assert snapshot["records"]["GOOD"]["ok"] is True
    assert snapshot["records"]["GOOD"]["grade"] == "A"
    assert snapshot["records"]["GOOD"]["last_close"] == 12.0
    assert snapshot["records"]["MEH"]["grade"] is None


def test_run_snapshot_passes_as_of_through_and_honours_a_cache_override(tmp_path):
    repo = tmp_path / "tree"
    write_stub_repo(repo, "9.9.9", ["GOOD"])
    alt = write_stub_repo(tmp_path / "altsrc", "9.9.9", ["OTHER"], cache_name="altcache")

    default_run = h.run_snapshot(repo, as_of="2025-06-30")
    assert list(default_run["records"]) == ["GOOD"]
    assert default_run["as_of"] == "2025-06-30"
    # as_of truncated the series: 2 bars, last close from 2025-01-01
    assert default_run["records"]["GOOD"]["bar_count"] == 2
    assert default_run["records"]["GOOD"]["last_close"] == 11.0

    override_run = h.run_snapshot(repo, cache_dir=alt)
    assert list(override_run["records"]) == ["OTHER"]
    assert override_run["cache"] == str(alt.resolve())


def test_two_repo_trees_load_without_colliding(tmp_path):
    """The whole point of --repo: baseline and live measured by one harness."""
    old = tmp_path / "old"
    new = tmp_path / "new"
    write_stub_repo(old, "2.1.0", ["GOOD"])
    write_stub_repo(new, "2.2.0", ["GOOD"])

    before = h.run_snapshot(old)
    after = h.run_snapshot(new)

    assert before["algorithm_versions"] == ["2.1.0"]
    assert after["algorithm_versions"] == ["2.2.0"]
    # re-running the first tree must still report the first tree's version
    assert h.run_snapshot(old)["algorithm_versions"] == ["2.1.0"]


def test_load_tree_rejects_a_directory_that_is_not_a_repo(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="coil_analysis.py"):
        h.load_tree(empty)


def test_run_replay_reports_pass_and_fail_against_a_stub_tree(tmp_path):
    repo = tmp_path / "tree"
    write_stub_repo(repo, "9.9.9", ["GOOD", "BOOM"])

    report = h.run_replay(
        repo,
        assertions=[
            {"ticker": "GOOD", "as_of": "2025-06-30", "expect": {"grade": "A", "bar_count": 2}},
            {"ticker": "GOOD", "as_of": None, "expect": {"grade": "C"}},
            {"ticker": "BOOM", "as_of": None, "expect": {"grade": "A"}},
            {"ticker": "ABSENT", "as_of": None, "expect": {"grade": "A"}},
        ],
    )

    assert report["total"] == 4
    assert report["passed"] == 1
    assert report["failed"] == 3
    assert [c["pass"] for c in report["cases"]] == [True, False, False, False]
    assert "ZeroDivisionError" in report["cases"][2]["record"]["error"]
    assert "no cached bars for ABSENT" in report["cases"][3]["record"]["error"]

    text = h.render_replay(report)
    assert "[PASS] GOOD as_of=2025-06-30" in text
    assert "1/4 passed, 3 failed" in text


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_cli_run_then_diff_round_trips_through_disk(tmp_path, capsys):
    old = tmp_path / "old"
    new = tmp_path / "new"
    write_stub_repo(old, "2.1.0", ["GOOD", "MEH"])
    write_stub_repo(new, "2.2.0", ["GOOD", "MEH"])
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    report = tmp_path / "report.md"
    diff_json = tmp_path / "diff.json"

    assert h.main(["run", "--repo", str(old), "--out", str(before)]) == 0
    assert h.main(["run", "--repo", str(new), "--out", str(after)]) == 0
    assert json.loads(before.read_text())["ticker_count"] == 2

    assert h.main(
        [
            "diff",
            "--before", str(before),
            "--after", str(after),
            "--report", str(report),
            "--json", str(diff_json),
        ]
    ) == 0
    assert "# Lid diff report" in report.read_text()
    payload = json.loads(diff_json.read_text())
    assert payload["before"]["algorithm_versions"] == ["2.1.0"]
    assert payload["after"]["algorithm_versions"] == ["2.2.0"]
    assert payload["summary"]["tickers_compared"] == 2
    capsys.readouterr()


def test_cli_replay_exit_code_is_nonzero_on_failure(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "tree"
    write_stub_repo(repo, "9.9.9", ["MEH"])
    monkeypatch.setattr(
        h, "REPLAY_ASSERTIONS", [{"ticker": "MEH", "as_of": None, "expect": {"grade": "A"}}]
    )
    assert h.main(["replay", "--repo", str(repo)]) == 1

    monkeypatch.setattr(
        h, "REPLAY_ASSERTIONS", [{"ticker": "MEH", "as_of": None, "expect": {"grade": None}}]
    )
    assert h.main(["replay", "--repo", str(repo)]) == 0
    capsys.readouterr()
