from coil_analysis import ALGORITHM_VERSION
from review_snapshots import (
    load_blind_review_context,
    load_review_context,
    load_review_manifest,
    review_snapshot_identity,
)


V22_SOURCE = "screen_2026-07-27_v2.2.0.csv"
V23_SOURCE = "screen_2026-08-05_v2.3.0.csv"
V23_TICKERS = ["REG", "BG", "1299.HK", "AZN.L"]


def test_v23_release_is_a_frozen_four_candidate_cohort():
    manifest = load_review_manifest(V23_SOURCE)
    assert manifest["source_run"]["algorithm_version"] == ALGORITHM_VERSION
    assert manifest["source_run"]["selection_policy"] == "grade_is_not_null"
    assert manifest["ordered_universe"] == V23_TICKERS

    for ticker in V23_TICKERS:
        identity = review_snapshot_identity(V23_SOURCE, ticker)
        assert identity["reviewable"] is True
        assert identity["screen_snapshot"]["grade"] == "A"
        context = load_review_context(V23_SOURCE, ticker)
        assert context["analysis_status"] == "frozen_algorithm_only"
        assert context["analysis"]["algorithm_version"] == ALGORITHM_VERSION
        assert context["model_snapshot"]["algorithm_version"] == ALGORITHM_VERSION


def test_prior_release_price_evidence_survives_an_algorithm_upgrade():
    blind = load_blind_review_context(V22_SOURCE, "REG")
    assert blind["reviewable"] is True
    assert blind["monthly_bars"]

    revealed = load_review_context(V22_SOURCE, "REG")
    assert revealed["analysis"] is None
    assert revealed["analysis_status"] == "frozen_model_unavailable"
    assert revealed["model_snapshot"]["algorithm_version"] == "2.2.0"
