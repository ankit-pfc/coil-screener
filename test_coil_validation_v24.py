from __future__ import annotations

import copy

import pytest

import coil_validation_v24 as v24
from coil_analysis import (
    ANALYSIS_MODE_ALGORITHM_ONLY,
    ANALYSIS_VARIANT_V2_4_VALIDATION,
    analyze_coil,
)
from test_coil_analysis import make_coil_bars, month_dates


def mature_coil_bars() -> list[dict]:
    bars = make_coil_bars()
    dates = month_dates(156)
    for idx in range(120, 156):
        close = 92.0 + (idx - 120) * 0.12
        bars.append(
            {
                "date": dates[idx],
                "open": close - 0.5,
                "high": min(99.0, close + 1.0),
                "low": close - 2.0,
                "close": close,
                "volume": 900_000.0,
            }
        )
    return bars


def analyze(bars: list[dict]) -> dict:
    return analyze_coil(
        bars,
        variant=ANALYSIS_VARIANT_V2_4_VALIDATION,
        mode=ANALYSIS_MODE_ALGORITHM_ONLY,
        adjustment_mode="split_adjusted",
    )


def test_registered_config_sweep_and_fingerprint_are_deterministic():
    config = v24.ValidationConfig(
        zone_candidate_prominence_pct=15.0,
        zone_similarity_pct=7.5,
        touch_tolerance_pct=2.5,
        max_qualifying_lid_slope_pct_per_year=6.5,
    )

    assert v24.config_fingerprint(config) == v24.config_fingerprint(config)
    assert v24.config_fingerprint(config).startswith("sha256:")
    with pytest.raises(ValueError, match="outside the registered lean sweep"):
        v24.ValidationConfig(zone_similarity_pct=6.0)
    with pytest.raises(ValueError, match="frozen for the lean validation pilot"):
        v24.ValidationConfig(min_structure_years=1.0)


def test_v24_is_explicitly_algorithm_only():
    with pytest.raises(ValueError, match="algorithm-only"):
        analyze_coil(
            make_coil_bars(),
            variant=ANALYSIS_VARIANT_V2_4_VALIDATION,
            mode="effective",
        )


def test_latest_rising_quarter_is_pending_without_a_confirmation_crash():
    quarters = [
        {
            "quarter_key": (2020, idx + 1),
            "date": f"2020-{month:02d}-01",
            "last_month": month,
            "open": 10.0,
            "high": high,
            "low": 9.0,
            "close": 10.0,
            "volume": 1_000.0,
            "high_month_idx": idx,
            "close_month_idx": idx,
            "peak_date": f"2020-{month:02d}-01",
        }
        for idx, (month, high) in enumerate(((3, 10.0), (6, 11.0)))
    ]

    candidates, eligible = v24._top_candidates(quarters, v24.DEFAULT_CONFIG)

    assert eligible == []
    assert candidates[-1]["role"] == "pending_top"
    assert candidates[-1]["confirmed_at"] is None


def test_confirmation_freezes_first_prefix_instead_of_using_later_decline():
    quarters = [
        {
            "quarter_key": (2000 + idx // 4, idx % 4 + 1),
            "date": f"{2000 + idx // 4}-{(idx % 4 + 1) * 3:02d}-01",
            "last_month": (idx % 4 + 1) * 3,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000.0,
            "high_month_idx": idx * 3,
            "close_month_idx": idx * 3 + 2,
            "peak_date": f"{2000 + idx // 4}-{(idx % 4 + 1) * 3:02d}-01",
        }
        for idx, (high, low, close) in enumerate(
            (
                (80.0, 70.0, 75.0),
                (100.0, 90.0, 95.0),
                (94.0, 80.0, 85.0),
                (93.0, 60.0, 70.0),
                (80.0, 50.0, 60.0),
            )
        )
    ]

    early, _ = v24._top_candidates(quarters[:4], v24.DEFAULT_CONFIG)
    later, _ = v24._top_candidates(quarters, v24.DEFAULT_CONFIG)
    early_peak = next(item for item in early if item["quarter_index"] == 1)
    later_peak = next(item for item in later if item["quarter_index"] == 1)

    assert early_peak["role"] == "pending_top"
    assert early_peak["confirmed_at"] is None
    assert later_peak["structural_eligible"] is True
    assert later_peak["confirmed_at"] == "2001-03-31"


def test_unconfirmed_peak_rejection_freezes_when_a_higher_boundary_arrives():
    quarters = [
        {
            "quarter_key": (2000 + idx // 4, idx % 4 + 1),
            "date": f"{2000 + idx // 4}-{(idx % 4 + 1) * 3:02d}-01",
            "last_month": (idx % 4 + 1) * 3,
            "open": high - 2.0,
            "high": high,
            "low": high - 4.0,
            "close": high - 2.0,
            "volume": 1_000.0,
            "high_month_idx": idx * 3,
            "close_month_idx": idx * 3 + 2,
            "peak_date": f"{2000 + idx // 4}-{(idx % 4 + 1) * 3:02d}-01",
        }
        for idx, high in enumerate((90.0, 100.0, 95.0, 110.0, 90.0))
    ]

    before, _ = v24._top_candidates(quarters[:3], v24.DEFAULT_CONFIG)
    rejected, _ = v24._top_candidates(quarters[:4], v24.DEFAULT_CONFIG)
    after, _ = v24._top_candidates(quarters, v24.DEFAULT_CONFIG)
    before_peak = next(item for item in before if item["quarter_index"] == 1)
    rejected_peak = next(item for item in rejected if item["quarter_index"] == 1)
    after_peak = next(item for item in after if item["quarter_index"] == 1)

    assert before_peak["role"] == "pending_top"
    assert rejected_peak["role"] == "rejected_high"
    assert rejected_peak["confirmed_at"] == "2000-12-31"
    assert rejected_peak == after_peak


def test_later_same_zone_retest_does_not_replace_an_earlier_top():
    quarters = [
        {
            "quarter_key": (2000 + idx // 4, idx % 4 + 1),
            "date": f"{2000 + idx // 4}-{(idx % 4 + 1) * 3:02d}-01",
            "last_month": (idx % 4 + 1) * 3,
            "open": high - 5.0,
            "high": high,
            "low": high - 10.0,
            "close": high - 6.0,
            "volume": 1_000.0,
            "high_month_idx": idx * 3,
            "close_month_idx": idx * 3 + 2,
            "peak_date": f"{2000 + idx // 4}-{(idx % 4 + 1) * 3:02d}-01",
        }
        for idx, high in enumerate((75.0, 100.0, 70.0, 80.0, 102.0, 70.0))
    ]

    candidates, eligible = v24._top_candidates(quarters, v24.DEFAULT_CONFIG)

    assert [item["quarter_index"] for item in eligible] == [1, 4]
    assert [item["peak_date"] for item in eligible] == [
        quarters[1]["peak_date"],
        quarters[4]["peak_date"],
    ]
    assert all(item["role"] != "pending_top" for item in candidates)


def test_continuous_multi_quarter_shoulder_is_one_plateau():
    highs = [70.0, 100.0, 100.0, 100.0, 70.0, 60.0]

    pivots = v24._pivot_indexes(highs)
    clustered = v24._cluster_plateaus(pivots, highs, v24.DEFAULT_CONFIG)

    assert pivots == [1, 3]
    assert clustered == [1]


def test_hypothesis_support_counts_only_quarter_separated_contacts():
    quarters = [{"close": 90.0, "low": 80.0} for _ in range(9)]

    def point(idx: int) -> dict:
        return {
            "id": f"p{idx}",
            "quarter_index": idx,
            "price": 100.0,
            "strict_major": True,
        }

    crowded, rejected = v24._hypotheses(
        quarters,
        [point(0), point(2), point(4)],
        v24.DEFAULT_CONFIG,
    )
    independent, _ = v24._hypotheses(
        quarters,
        [point(0), point(4), point(8)],
        v24.DEFAULT_CONFIG,
    )

    assert rejected
    assert len(crowded) == 1
    assert crowded[0]["contact_ids"] == ["p0", "p4"]
    assert crowded[0]["contact_count"] == 2
    assert independent[0]["contact_ids"] == ["p0", "p4", "p8"]
    assert independent[0]["contact_count"] == 3


def test_omitted_variant_remains_exactly_v231_effective():
    bars = make_coil_bars()

    implicit = analyze_coil(bars)
    explicit = analyze_coil(bars, variant="v2_3_1", mode="effective")

    assert implicit == explicit


def test_sub_ten_year_structure_abstains_without_legacy_geometry():
    result = analyze(make_coil_bars())

    assert result["algorithm_version"] == "2.4.0-validation"
    assert result["analysis_metadata"]["variant"] == "v2_4_validation"
    assert result["structure_validity"] == "watch_immature"
    assert result["readiness"] == "watch_immature"
    assert result["confidence"] == "medium"
    assert result["abstained"] is True
    assert result["lid_hypotheses"]
    assert result["resistance_band"] is not None
    assert result["major_highs"] == []
    assert result["active_lid"] is None
    assert result["resistance"] is None


def test_mature_structure_publishes_ranked_hypotheses_band_and_compatibility():
    result = analyze(mature_coil_bars())

    assert result["structure_validity"] == "qualified"
    assert result["readiness"] == "pre_breakout"
    assert result["confidence"] == "high"
    assert result["abstained"] is False
    assert result["lifecycle"] == "pre_breakout"
    assert result["status"] == "coiling"
    assert result["grade"] == "A"
    assert result["resistance"] is not None
    assert result["active_lid"] is not None
    assert result["coil_score"] == 0.0

    hypotheses = result["lid_hypotheses"]
    assert [item["rank"] for item in hypotheses] == list(
        range(1, len(hypotheses) + 1)
    )
    assert all(item["id"].startswith("lid_") for item in hypotheses)
    assert all(not item["rejection_reasons"] for item in hypotheses)
    band = result["resistance_band"]
    assert band["lower"] < band["centre"] < band["upper"]
    assert band["hypothesis_ids"][0] == hypotheses[0]["id"]

    confirmed = [
        item for item in result["top_candidates"] if item["role"] != "pending_top"
    ]
    assert confirmed
    assert all(item["peak_date"] < item["confirmed_at"] for item in confirmed)
    assert all(item["rejection_reasons"] or item["structural_eligible"] for item in confirmed)
    assert any(item["secondary_compatibility_eligible"] for item in confirmed)
    assert all(point["confirmed_at"] for point in result["points"])
    assert "completed_period_volume_contraction" in result["metrics"]


def test_equal_support_divergent_lids_abstain_without_speculative_winner(monkeypatch):
    base = {
        "contact_count": 2,
        "strict_major_count": 2,
        "fit_error_pct": 0.25,
    }
    conflicting = [
        {**base, "id": "lid_a", "projected_lid": 100.0},
        {**base, "id": "lid_b", "projected_lid": 120.0},
    ]
    monkeypatch.setattr(v24, "_hypotheses", lambda *args, **kwargs: (conflicting, []))

    result = v24.analyze_coil_v24(
        mature_coil_bars(), adjustment_mode="split_adjusted"
    )

    assert result["structure_validity"] == "uncertain_structure"
    assert result["readiness"] == "uncertain_structure"
    assert result["abstained"] is True
    assert result["resistance_band"] is None
    assert result["major_highs"] == []
    assert result["active_lid"] is None
    assert result["pattern_assessment"]["failed_rules"] == [
        "no_unresolved_competing_lid"
    ]


def test_constant_price_scaling_preserves_normalized_geometry():
    bars = mature_coil_bars()
    scaled = copy.deepcopy(bars)
    for bar in scaled:
        for field in ("open", "high", "low", "close"):
            bar[field] *= 10.0

    original = analyze(bars)
    transformed = analyze(scaled)

    assert transformed["structure_validity"] == original["structure_validity"]
    assert transformed["readiness"] == original["readiness"]
    assert [item["peak_date"] for item in transformed["top_candidates"]] == [
        item["peak_date"] for item in original["top_candidates"]
    ]
    assert transformed["lid_hypotheses"][0]["contact_count"] == original[
        "lid_hypotheses"
    ][0]["contact_count"]
    assert transformed["lid_hypotheses"][0]["slope_pct_per_year"] == pytest.approx(
        original["lid_hypotheses"][0]["slope_pct_per_year"], abs=1e-4
    )
    assert transformed["resistance_band"]["centre"] == pytest.approx(
        original["resistance_band"]["centre"] * 10.0, abs=1e-3
    )


@pytest.mark.parametrize(
    ("quarter_closes", "partial_close", "expected"),
    [
        ([105.0], None, "breaking_out"),
        ([105.0, 106.0], None, "post_breakout"),
        ([105.0, 100.0], None, "retest"),
        ([105.0, 90.0], None, "failed_breakout"),
        ([], 105.0, "breakout_provisional"),
    ],
)
def test_validation_lifecycle_states_are_separate_from_structure(
    quarter_closes, partial_close, expected
):
    primary = {
        "contacts": [
            {
                "quarter_index": 0,
                "price": 100.0,
                "confirmed_quarter_index": 0,
            },
            {
                "quarter_index": 4,
                "price": 100.0,
                "confirmed_quarter_index": 0,
            },
        ],
        "contact_count": 2,
        "intercept": 100.0,
        "slope_per_quarter": 0.0,
    }
    quarters = [
        {
            "open": 95.0,
            "high": max(101.0, close),
            "low": min(94.0, close),
            "close": close,
            "volume": 1_000.0,
        }
        for close in [95.0, 96.0, *quarter_closes]
    ]
    partial_quarter = None
    if partial_close is not None:
        partial_quarter = {
            "open": 98.0,
            "high": partial_close + 1.0,
            "low": 97.0,
            "close": partial_close,
            "volume": 500.0,
        }
    signals = {
        "proximity_pct": quarters[-1]["close"] if quarters else 95.0,
        "independent_signal_count": 1,
    }

    assert v24._validation_state(
        quarters,
        partial_quarter,
        primary,
        {
            "lower": 96.5,
            "centre": 100.0,
            "upper": 103.5,
            "touch_tolerance_pct": 3.5,
        },
        signals,
        True,
    ) == expected


def test_partial_quarter_cannot_override_a_confirmed_post_breakout_state():
    primary = {
        "contacts": [
            {"quarter_index": 0, "confirmed_quarter_index": 0},
            {"quarter_index": 1, "confirmed_quarter_index": 0},
        ],
        "intercept": 100.0,
        "slope_per_quarter": 0.0,
    }
    quarters = [
        {"close": close}
        for close in (95.0, 105.0, 106.0)
    ]
    partial_quarter = {"close": 90.0}

    state = v24._validation_state(
        quarters,
        partial_quarter,
        primary,
        {
            "lower": 96.5,
            "centre": 100.0,
            "upper": 103.5,
            "touch_tolerance_pct": 3.5,
        },
        {"proximity_pct": 90.0, "independent_signal_count": 0},
        True,
    )

    assert state == "post_breakout"


def test_pressing_the_lid_alone_does_not_promote_pre_breakout_readiness():
    primary = {
        "contacts": [
            {"quarter_index": 0, "confirmed_quarter_index": 0},
            {"quarter_index": 1, "confirmed_quarter_index": 0},
        ],
        "intercept": 100.0,
        "slope_per_quarter": 0.0,
    }
    quarters = [{"close": 95.0}, {"close": 95.0}]

    state = v24._validation_state(
        quarters,
        None,
        primary,
        {
            "lower": 96.5,
            "centre": 100.0,
            "upper": 103.5,
            "touch_tolerance_pct": 3.5,
        },
        {"proximity_pct": 95.0, "independent_signal_count": 0},
        True,
    )

    assert state == "forming"


def test_lifecycle_waits_for_every_equivalent_leader_confirmation():
    primary = {
        "contacts": [
            {"quarter_index": 0, "confirmed_quarter_index": 0},
            {"quarter_index": 1, "confirmed_quarter_index": 0},
        ],
        "intercept": 100.0,
        "slope_per_quarter": 0.0,
    }
    later_leader = {
        "contacts": [
            {"quarter_index": 0, "confirmed_quarter_index": 0},
            {"quarter_index": 3, "confirmed_quarter_index": 3},
        ],
        "intercept": 100.0,
        "slope_per_quarter": 0.0,
    }
    quarters = [{"close": close} for close in (95.0, 105.0, 106.0, 95.0)]

    state = v24._validation_state(
        quarters,
        None,
        primary,
        {
            "lower": 96.5,
            "centre": 100.0,
            "upper": 103.5,
            "touch_tolerance_pct": 3.5,
        },
        {"proximity_pct": 95.0, "independent_signal_count": 0},
        True,
        [primary, later_leader],
    )

    assert state == "forming"


def test_historical_incomplete_quarter_is_not_trailing_partial_context():
    historical_gap = {
        "date": "2020-02-01",
        "last_month": 2,
        "quarter_key": (2020, 1),
    }
    latest_complete = {
        "date": "2020-06-01",
        "last_month": 6,
        "quarter_key": (2020, 2),
    }
    latest_partial = {
        "date": "2020-08-01",
        "last_month": 8,
        "quarter_key": (2020, 3),
    }

    assert v24._trailing_partial_quarter(
        [historical_gap, latest_complete], as_of="2020-06-30"
    ) is None
    assert v24._trailing_partial_quarter(
        [historical_gap, latest_complete, latest_partial], as_of="2020-08-31"
    ) == latest_partial


def test_future_bars_do_not_backfill_v24_top_confirmation():
    bars = mature_coil_bars()
    cutoffs = month_dates(len(bars))
    for end in range(36, len(bars) + 1, 3):
        cutoff = cutoffs[end - 1]
        direct = analyze_coil(
            bars,
            as_of=v24._month_end(cutoff),
            variant=ANALYSIS_VARIANT_V2_4_VALIDATION,
            mode=ANALYSIS_MODE_ALGORITHM_ONLY,
            adjustment_mode="split_adjusted",
        )
        for top in direct["top_candidates"]:
            if top["confirmed_at"] is not None:
                assert top["confirmed_at"] <= v24._month_end(cutoff)
        prefix = analyze_coil(
            bars[:end],
            as_of=v24._month_end(cutoff),
            variant=ANALYSIS_VARIANT_V2_4_VALIDATION,
            mode=ANALYSIS_MODE_ALGORITHM_ONLY,
            adjustment_mode="split_adjusted",
        )
        direct_events = [
            (item["peak_date"], item["confirmed_at"], item["role"])
            for item in direct["top_candidates"]
        ]
        prefix_events = [
            (item["peak_date"], item["confirmed_at"], item["role"])
            for item in prefix["top_candidates"]
        ]
        assert direct_events == prefix_events
