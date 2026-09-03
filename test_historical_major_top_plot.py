"""Contract tests for the protected full-lifetime major-top plot transport."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
import review_snapshots
import reviews as reviews_module
from reviews import ReviewStore
from test_fresh_review_sessions import TOKEN, _create, _headers, _make_corpus


@pytest.fixture
def top_plot_client(tmp_path, monkeypatch) -> tuple[TestClient, Path]:
    store = ReviewStore(sqlite_path=tmp_path / "reviews.db")
    monkeypatch.setattr(reviews_module, "_store", store)
    snapshot_root = tmp_path / "review_snapshots"
    monkeypatch.setattr(review_snapshots, "REVIEW_SNAPSHOT_ROOT", snapshot_root)
    return TestClient(app_module.app), snapshot_root


def _make_top_plot_corpus(root: Path, source: str) -> None:
    _make_corpus(root, source, ["AAA"])
    folder = root / Path(source).stem

    snapshot_path = folder / "AAA.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["screen_snapshot"]["review_mode"] = "detector_shape_audit"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "protected_bootstrap": True,
            "reviewer_name": "Amrut",
            "review_policy": {
                "detector_outputs_hidden": True,
                "model_reveal_allowed": False,
                "production_effect": "none",
                "candidate_rules_visible": False,
                "coordinator_key_visible": False,
            },
        }
    )
    manifest["items"][0]["cutoff_date"] = "2019-12-31"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_major_top_plot_returns_only_lifetime_top_transport(top_plot_client):
    client, root = top_plot_client
    source = "historical_top_plot.csv"
    _make_top_plot_corpus(root, source)
    session = _create(client, source, ["AAA"])["session"]

    response = client.get(
        f"/api/review-sessions/{session['id']}/items/AAA/major-tops",
        headers=_headers(TOKEN),
    )

    assert response.status_code == 200, response.text
    analysis = response.json()["analysis"]
    assert analysis["kind"] == "coilingview.historical-major-top-plot"
    assert analysis["history_scope"] == "observed_lifetime"
    assert analysis["cutoff_date"] == "2019-12-31"
    assert analysis["history_end"] <= analysis["cutoff_date"]
    assert analysis["completed_through"] <= analysis["cutoff_date"]
    assert analysis["major_tops"]
    assert set(analysis) == {
        "schema_version",
        "kind",
        "ticker",
        "interval",
        "history_scope",
        "history_start",
        "history_end",
        "cutoff_date",
        "completed_through",
        "completed_quarter_count",
        "major_tops",
        "algorithm_version",
        "sample_id",
        "bars_hash",
    }
    forbidden = {
        "coil",
        "recognized",
        "score",
        "grade",
        "lifecycle",
        "forecast",
        "sampling_stratum",
    }
    assert forbidden.isdisjoint(analysis)


def test_saved_run_metadata_and_prelock_item_are_neutral(top_plot_client, tmp_path, monkeypatch):
    client, root = top_plot_client
    source = "historical_top_plot.csv"
    _make_top_plot_corpus(root, source)
    (tmp_path / source).write_text(
        "ticker,review_mode,score_total,lifecycle,grade\n"
        "AAA,detector_shape_audit,99,coiling,A\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)

    saved = client.get(f"/api/saved-runs/{source}")
    assert saved.status_code == 200, saved.text
    payload = saved.json()
    assert payload["protected_bootstrap"] is True
    assert payload["reviewer_name"] == "Amrut"
    assert payload["protected_policy"]["detector_outputs_hidden"] is True
    assert payload["results"] == [
        {
            "ticker": "AAA",
            "review_mode": "detector_shape_audit",
            "age_years": None,
            "last_close": None,
            "score_total": None,
            "score_long_coil": None,
            "score_tight_resistance": None,
            "score_ascending_compression": None,
            "pos_in_10y_range": None,
            "dist_to_10y_high_pct": None,
            "range_ratio_24_120": None,
            "range_ratio_24_60": None,
            "low_36m_above_10y_low_pct": None,
            "slope_high_60m": None,
            "slope_low_60m": None,
            "trend_r2_60m": None,
            "peak_age_months": None,
            "old_peak_similarity": None,
        }
    ]

    session = _create(client, source, ["AAA"])["session"]
    item = session["items"][0]
    assert item["snapshot"]["review_mode"] == "detector_shape_audit"
    assert "lifecycle" not in item["snapshot"]
    assert "grade" not in item["snapshot"]


def test_major_top_plot_fails_closed_for_another_review_mode(top_plot_client):
    client, root = top_plot_client
    source = "ordinary_review.csv"
    _make_corpus(root, source, ["AAA"])
    session = _create(client, source, ["AAA"])["session"]

    response = client.get(
        f"/api/review-sessions/{session['id']}/items/AAA/major-tops",
        headers=_headers(TOKEN),
    )

    assert response.status_code == 404
