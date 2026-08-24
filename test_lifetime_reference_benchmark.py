from __future__ import annotations

from pathlib import Path

import pytest

import lifetime_reference_benchmark as benchmark


def _monthly_bars(count: int = 96, start_year: int = 2000) -> list[dict]:
    bars = []
    for offset in range(count):
        year = start_year + offset // 12
        month = offset % 12 + 1
        price = 40.0 + (offset % 24) * 0.8
        bars.append(
            {
                "date": f"{year:04d}-{month:02d}-01",
                "open": price,
                "high": price * 1.08,
                "low": price * 0.75,
                "close": price * 0.82,
                "volume": 1_000_000.0,
                "human_label": "must be stripped",
            }
        )
    return bars


def test_corpus_selection_counts_and_withholds_blind_identity_overlap() -> None:
    loaded = benchmark.load_benchmark_cases()

    assert loaded["counts"] == {
        "exact_selected": 24,
        "portfolio_selected": 9,
        "shadow_quality_accepted": 67,
        "shadow_executed_safe": 43,
        "shadow_withheld_blind_overlap": 24,
        "blind_snapshots_executed": 0,
    }
    sealed = benchmark._sealed_ticker_identities()
    assert not ({case.ticker for case in loaded["shadow"]} & sealed)
    assert sum(
        case.manifest_item.get("cohort_role") == "control_reference"
        for case in loaded["shadow"]
    ) == 18
    assert sum(
        bool((case.manifest_item.get("prior_review") or {}).get("flag"))
        for case in loaded["shadow"]
    ) == 22
    assert sum(
        bool((case.manifest_item.get("tuning_anchor") or {}).get("flag"))
        for case in loaded["shadow"]
    ) == 8
    assert sum(
        case.manifest_item.get("cohort_role") == "prospective_international"
        and not bool((case.manifest_item.get("prior_review") or {}).get("flag"))
        and not bool((case.manifest_item.get("tuning_anchor") or {}).get("flag"))
        for case in loaded["shadow"]
    ) == 20


def test_blind_manifest_is_fail_closed() -> None:
    with pytest.raises(benchmark.BenchmarkSafetyError, match="sealed"):
        benchmark._assert_allowed_manifest(
            {
                "corpus_id": benchmark.SEALED_CORPUS_ID,
                "review_policy": {"detector_outputs_hidden": True},
            },
            benchmark.SEALED_SOURCE,
        )


def test_sanitize_physically_truncates_and_strips_non_ohlcv_fields() -> None:
    bars = _monthly_bars(4)
    sanitized = benchmark.sanitize_ohlcv_prefix(bars, "2000-02-15")

    assert [item["date"] for item in sanitized] == ["2000-01-01", "2000-02-01"]
    assert set(sanitized[0]) == {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }


def test_detector_pair_passes_only_sanitized_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def fake_current(bars, *, as_of, review_override):
        rows = list(bars)
        seen.append(("current", rows, as_of, review_override))
        return {
            "algorithm_version": benchmark.CURRENT_ALGORITHM_VERSION,
            "bar_count": len(rows),
            "review": {"reviewed": False, "effective": "algorithm"},
            "pattern_anatomy": {"boundary": None},
        }

    def fake_lifetime(bars, *, as_of):
        rows = list(bars)
        seen.append(("lifetime", rows, as_of, None))
        return {"source": "timeseries", "algorithm_version": "test"}

    monkeypatch.setattr(benchmark, "analyze_coil", fake_current)
    monkeypatch.setattr(benchmark, "analyze_lifetime_references", fake_lifetime)

    result = benchmark.run_detector_pair(_monthly_bars(3), "2000-02-15")

    assert result["current"]["run_status"] == "ok"
    assert result["lifetime"]["run_status"] == "ok"
    assert [item[0] for item in seen] == ["current", "lifetime"]
    for _, rows, as_of, _ in seen:
        assert as_of == "2000-02-15"
        assert len(rows) == 2
        assert all("human_label" not in row for row in rows)
    assert seen[0][3] is None


def test_current_detector_safety_violation_is_not_recorded_as_data_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_current(bars, *, as_of, review_override):
        return {
            "bar_count": len(bars),
            "review": {"reviewed": True, "effective": "human"},
            "pattern_anatomy": {"boundary": {"family": "human_review"}},
        }

    monkeypatch.setattr(benchmark, "analyze_coil", fake_current)

    with pytest.raises(
        benchmark.BenchmarkSafetyError, match="current detector became human-effective"
    ):
        benchmark._safe_current_run(_monthly_bars(3), "2000-03-31")


def test_lifetime_detector_provenance_violation_is_not_recorded_as_data_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_lifetime(bars, *, as_of):
        return {"source": "human_review", "algorithm_version": "test"}

    monkeypatch.setattr(benchmark, "analyze_lifetime_references", fake_lifetime)

    with pytest.raises(
        benchmark.BenchmarkSafetyError,
        match="lifetime detector provenance is not timeseries",
    ):
        benchmark._safe_lifetime_run(_monthly_bars(3), "2000-03-31")


def test_calendar_ols_normalization_and_direction_threshold() -> None:
    line = benchmark._ols_line(
        [
            {"date": "2000-03-01", "price": 100.0},
            {"date": "2005-03-01", "price": 80.0},
        ],
        cutoff_date="2005-03-01",
        line_id="test",
        role="primary_lid",
        boundary_kind="line",
        selection="primary",
        relationship="standalone",
    )

    assert line is not None
    assert line["slope_per_quarter"] == pytest.approx(-1.0)
    assert line["value_at_cutoff"] == pytest.approx(80.0)
    assert line["direction"] == "falling"


def test_line_comparison_distinguishes_descriptive_and_strict_alignment() -> None:
    expected = benchmark._ols_line(
        [
            {"date": "2000-03-01", "price": 100.0},
            {"date": "2005-03-01", "price": 100.0},
        ],
        cutoff_date="2010-03-01",
        line_id="expected",
        role="primary_lid",
        boundary_kind="line",
        selection="primary",
        relationship="standalone",
    )
    predicted = benchmark._ols_line(
        [
            {"date": "2000-03-01", "price": 106.0},
            {"date": "2005-03-01", "price": 106.0},
        ],
        cutoff_date="2010-03-01",
        line_id="predicted",
        role="primary_lid",
        boundary_kind="line",
        selection="primary",
        relationship="standalone",
    )

    comparison = benchmark.compare_lines(expected, predicted)

    assert comparison["descriptive_rms_within_10pct"] is True
    assert comparison["primary_geometry_match"] is False
    assert comparison["reference_anchor_rms_error_pct"] == pytest.approx(6.0)


def test_lifetime_band_normalization_preserves_emitted_flat_median() -> None:
    analysis = {
        "top_episodes": [
            {"id": "outer-a", "date": "2000-03-01", "price": 140.0},
            {"id": "outer-b", "date": "2005-03-01", "price": 120.0},
            {"id": "band-a", "date": "2002-03-01", "price": 80.0},
            {"id": "band-b", "date": "2008-03-01", "price": 120.0},
        ],
        "structures": [
            {
                "id": "outer",
                "kind": "line",
                "relationship": "outer_reference",
                "construction_anchor_ids": ["outer-a", "outer-b"],
                "supporting_touch_ids": [],
                "line": {
                    "from": {
                        "date": "2000-03-01",
                        "time_ordinal": benchmark.quarter_ordinal("2000-03-01"),
                        "price": 140.0,
                    },
                    "slope_per_quarter": -1.0,
                },
                "fit": {},
            },
            {
                "id": "median-band",
                "kind": "resistance_band",
                "relationship": "nested_below_outer",
                "construction_anchor_ids": ["band-a", "band-b"],
                "supporting_touch_ids": [],
                "line": {
                    "from": {
                        "date": "2002-03-01",
                        "time_ordinal": benchmark.quarter_ordinal("2002-03-01"),
                        "price": 100.0,
                    },
                    "slope_per_quarter": 0.0,
                    "projected": {
                        "time_ordinal": benchmark.quarter_ordinal("2010-03-01"),
                        "price": 100.0,
                    },
                },
                "band": {"lower_pct": 5.0, "upper_pct": 5.0},
                "fit": {},
            },
        ],
    }

    normalized = benchmark.normalize_lifetime_structures(
        analysis, "2010-03-01"
    )
    band = next(item for item in normalized if item["id"] == "median-band")

    assert band["slope_per_quarter"] == 0.0
    assert band["value_at_cutoff"] == 100.0
    assert band["direction"] == "flat"
    assert band["normalization_source"] == "detector_emitted_band_centerline"
    assert band["anchor_prices"] == [80.0, 120.0]


def test_structure_set_pairing_requires_compatible_semantics() -> None:
    expected = benchmark._ols_line(
        [
            {"date": "2000-03-01", "price": 100.0},
            {"date": "2005-03-01", "price": 100.0},
        ],
        cutoff_date="2010-03-01",
        line_id="expected",
        role="primary_lid",
        boundary_kind="line",
        selection="primary",
        relationship="outer_reference",
    )
    assert expected is not None

    for field, value in (
        ("role", "secondary_lid"),
        ("boundary_kind", "resistance_band"),
        ("relationship", "nested_below_outer"),
    ):
        incompatible = {**expected, "id": f"wrong-{field}", field: value}
        comparison = benchmark.compare_structure_sets([expected], [incompatible])
        assert comparison["within_10pct_tp"] == 0
        assert comparison["within_10pct_fp"] == 1
        assert comparison["within_10pct_fn"] == 1

    compatible = {**expected, "id": "compatible"}
    comparison = benchmark.compare_structure_sets([expected], [compatible])
    assert comparison["within_10pct_tp"] == 1
    assert comparison["pairs"][0]["predicted_id"] == "compatible"


def test_structure_set_pairing_maximizes_cardinality_before_rms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rms_by_pair = {
        ("expected-a", "predicted-a"): 1.0,
        ("expected-a", "predicted-b"): 2.0,
        ("expected-b", "predicted-a"): 1.5,
        ("expected-b", "predicted-b"): 20.0,
    }

    def fake_compare(expected, predicted):
        rms = rms_by_pair[(expected["id"], predicted["id"])]
        return {
            "direction_match": True,
            "reference_anchor_rms_error_pct": rms,
            "primary_geometry_match": rms <= 5.0,
        }

    monkeypatch.setattr(benchmark, "compare_lines", fake_compare)
    expected = [{"id": "expected-a"}, {"id": "expected-b"}]
    predicted = [{"id": "predicted-a"}, {"id": "predicted-b"}]

    comparison = benchmark.compare_structure_sets(expected, predicted)

    assert comparison["within_10pct_tp"] == 2
    assert {
        (pair["expected_id"], pair["predicted_id"])
        for pair in comparison["pairs"]
    } == {
        ("expected-a", "predicted-b"),
        ("expected-b", "predicted-a"),
    }


def test_date_matching_maximizes_cardinality_before_timing_error() -> None:
    matching = benchmark.match_dates(
        ["2000-03-01", "2000-06-01"],
        ["2000-03-01", "2000-09-01"],
        tolerance=1,
    )

    assert matching["tp"] == 2
    assert matching["fp"] == 0
    assert matching["fn"] == 0


def test_replay_hides_reference_until_two_anchors_are_available() -> None:
    reference = {
        "lines": [
            {
                "id": "primary",
                "role": "primary_lid",
                "constructionAnchors": [
                    {"date": "2000-03-01", "price": 100.0},
                    {"date": "2005-03-01", "price": 100.0},
                ],
            }
        ]
    }

    early, early_reason = benchmark._trim_reference_for_replay(
        reference, "2004-12-01"
    )
    mature, mature_reason = benchmark._trim_reference_for_replay(
        reference, "2005-03-01"
    )

    assert early == []
    assert early_reason == "fewer_than_two_reference_anchors_available"
    assert len(mature) == 1
    assert mature_reason is None


def test_replay_reference_is_not_expected_before_first_recognizable_date() -> None:
    reference = {
        "firstRecognizableDate": "2005-03-01",
        "lines": [
            {
                "constructionAnchors": [
                    {"date": "2000-03-01", "price": 100.0},
                    {"date": "2004-03-01", "price": 100.0},
                ]
            }
        ],
    }

    assert not benchmark._reference_line_expected(
        reference, "pre_recognizable", "2004-12-31"
    )
    assert benchmark._reference_line_expected(
        reference, "first_recognizable", "2005-03-01"
    )
    hidden, reason = benchmark._trim_reference_for_replay(
        reference,
        "2004-12-01",
        reference_line_expected=False,
    )
    assert hidden == []
    assert reason == "reference_line_not_expected_at_checkpoint"


def test_month_start_milestone_replay_excludes_that_months_complete_candle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_prefixes: dict[str, list[dict]] = {}

    def fake_detector_pair(bars, decision_as_of):
        prefix = benchmark.sanitize_ohlcv_prefix(bars, decision_as_of)
        seen_prefixes[decision_as_of] = prefix
        empty_run = {
            "run_status": "ok",
            "error": None,
            "analysis": {},
            "raw_output_sha256": benchmark.sha256_json({}),
        }
        return {
            "bars": prefix,
            "current": empty_run,
            "lifetime": empty_run,
        }

    monkeypatch.setattr(benchmark, "run_detector_pair", fake_detector_pair)
    case = benchmark.BenchmarkCase(
        cohort="test",
        geometry_quality="exact_clicks",
        corpus_id="test",
        source="test.csv",
        ticker="TEST",
        manifest_sha256="manifest",
        manifest_item={},
        snapshot={"monthly_bars": _monthly_bars(30)},
        decision_as_of="2002-06-15",
    )

    rows, _ = benchmark._replay_case(
        case,
        {"firstWatchDate": "2002-03-01", "lines": []},
        full_decision_as_of="2002-06-15",
    )

    milestone = next(row for row in rows if row["checkpoint_role"] == "first_watch")
    assert milestone["checkpoint_target_date"] == "2002-03-01"
    assert milestone["decision_as_of"] == "2002-02-28"
    assert milestone["cutoff_date"] == "2002-02-01"
    assert seen_prefixes["2002-02-28"][-1]["date"] == "2002-02-01"
    assert all(
        bar["date"] != "2002-03-01" for bar in seen_prefixes["2002-02-28"]
    )


def test_frozen_bdc_run_remains_algorithm_only_and_point_in_time() -> None:
    loaded = benchmark.load_benchmark_cases()
    case = next(item for item in loaded["exact"] if item.ticker == "BDC")
    pair = benchmark.run_detector_pair(
        case.snapshot["monthly_bars"], case.decision_as_of
    )

    assert pair["current"]["run_status"] == "ok"
    assert pair["current"]["analysis"]["review"] == {
        "reviewed": False,
        "effective": "algorithm",
    }
    assert pair["bars"][-1]["date"] <= case.decision_as_of
    assert pair["lifetime"]["analysis"]["source"] == "timeseries"


def test_output_directory_default_is_separate_from_teaching_v1() -> None:
    expected = (
        Path(__file__).parent
        / "docs"
        / "lifetime-reference-benchmark"
        / "v1"
    )
    assert expected != (
        Path(__file__).parent
        / "docs"
        / "lifetime-reference-validation"
        / "v1"
    )


def test_committed_benchmark_bundle_verifies_when_present() -> None:
    output_dir = (
        Path(__file__).parent
        / "docs"
        / "lifetime-reference-benchmark"
        / "v1"
    )
    if not (output_dir / "manifest.json").exists():
        pytest.skip("generated benchmark bundle is not installed")

    verification = benchmark.verify_benchmark_artifacts(output_dir)

    assert verification == {
        "verified": True,
        "labelled_rows": 33,
        "shadow_rows": 43,
        "replay_rows": 113,
        "setup_json_files": 76,
        "sealed_ticker_overlap": 0,
    }
