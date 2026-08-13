"""Endpoint contract + history-cache + saved-run loader tests.

Network (yfinance) is never hit: ``fetch_monthly_history`` is monkeypatched and
the history disk cache is redirected to a tmp dir per test.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
import history_cache
import reviews as reviews_module
from reviews import ReviewStore
from screen_monthly import compute_features


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    """Redirect the history disk cache to an isolated tmp dir."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(history_cache, "CACHE_DIR", cache_dir)
    return cache_dir


@pytest.fixture(autouse=True)
def tmp_review_store(tmp_path, monkeypatch):
    """Isolate the review store in a per-test SQLite file."""
    store = ReviewStore(sqlite_path=tmp_path / "reviews.db")
    monkeypatch.setattr(reviews_module, "_store", store)
    return store


def _synthetic_monthly(rows: int = 130) -> pd.DataFrame:
    """A flat coil: enough bars (>=120) for compute_features to return a result."""
    idx = pd.date_range("2010-01-01", periods=rows, freq="MS")
    base = 100.0
    high = [base + 5 + (i % 3) for i in range(rows)]
    low = [base - 5 - (i % 3) for i in range(rows)]
    close = [base + (i % 2) for i in range(rows)]
    result = pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": [1_000_000 + i for i in range(rows)],
        },
        index=idx,
    )
    result.attrs["adjustment_mode"] = "split_adjusted"
    result.attrs["adjustment_source"] = "yfinance_stock_splits"
    result.attrs["source_interval"] = "1d"
    result.attrs["adjustment_transform_version"] = "yfinance-stock-splits-v1"
    return result


# --------------------------------------------------------------------------- #
# Simple endpoint contracts
# --------------------------------------------------------------------------- #
def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["capture_only"] is False
    assert body["persistence"]["ready"] is True


def test_default_tickers(client):
    resp = client.get("/api/default-tickers")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["tickers"], list)
    assert len(body["tickers"]) > 0
    assert all(isinstance(t, str) for t in body["tickers"])


def test_international_universe_is_accepted_by_screen_api(client, monkeypatch):
    seen = {}

    def fake_run(tickers, **kwargs):
        seen["tickers"] = list(tickers)
        seen["force_refresh"] = kwargs["force_refresh"]
        return {
            "results": [],
            "bucket_counts": {},
            "failures": [],
            "algorithm_version": "test-v2.1",
            "screened_at": "2026-07-14T00:00:00.000Z",
        }

    monkeypatch.setattr(app_module, "run_lifecycle_screen", fake_run)
    response = client.post(
        "/api/screen",
        json={"universe": "international", "limit": 7, "force_refresh": True},
    )

    assert response.status_code == 200
    assert len(seen["tickers"]) == 7
    assert all("." in ticker for ticker in seen["tickers"])
    assert seen["force_refresh"] is True


def test_saved_runs_lists_current_algorithm_run_first(client):
    resp = client.get("/api/saved-runs")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert isinstance(runs, list)
    assert len(runs) > 0
    # contract: each run has a name + size, and history remains available
    names = [r["name"] for r in runs]
    assert all(n.endswith(".csv") for n in names)
    assert all(isinstance(r["size_bytes"], int) for r in runs)
    assert "demo_curated_coils_results.csv" in names
    # The frontend loads runs[0], which must match the running algorithm.
    assert names[0] == "screen_2026-08-05_v2.3.0.csv"


def test_saved_runs_current_release_pinned_above_newer_csv(monkeypatch, tmp_path):
    """The current run stays first even when another CSV has a newer mtime —
    guards the Railway cold-start case where checkout flattens all mtimes."""
    import os
    import time

    root = tmp_path / "runs"
    root.mkdir()
    current = root / app_module.DEMO_DEFAULT_RUN
    other = root / "sp500_plus_amrut_results.csv"
    current.write_text("ticker\nREG\n", encoding="utf-8")
    other.write_text("ticker\nAAPL\n", encoding="utf-8")
    # Make the historical CSV strictly newer than the current run.
    now = time.time()
    os.utime(current, (now - 100, now - 100))
    os.utime(other, (now, now))
    monkeypatch.setattr(app_module, "PROJECT_ROOT", root)

    runs = app_module.saved_runs()["runs"]
    assert runs[0]["name"] == app_module.DEMO_DEFAULT_RUN
    assert runs[1]["name"] == "sp500_plus_amrut_results.csv"


def test_v23_saved_run_contains_only_graded_candidates(client):
    body = client.get("/api/saved-runs/screen_2026-08-05_v2.3.0.csv").json()
    assert body["count"] == 4
    assert [row["ticker"] for row in body["results"]] == [
        "REG",
        "BG",
        "1299.HK",
        "AZN.L",
    ]
    assert {row["algorithm_version"] for row in body["results"]} == {"2.3.0"}
    assert all(row["grade"] for row in body["results"])


# --------------------------------------------------------------------------- #
# Saved-run loader: success + 404 + 400 (path traversal) branches
# --------------------------------------------------------------------------- #
def test_saved_run_success(client):
    resp = client.get("/api/saved-runs/demo_curated_coils_results.csv")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "demo_curated_coils_results.csv"
    assert body["count"] == len(body["results"])
    assert body["count"] > 0
    assert "ticker" in body["results"][0]


def test_curated_saved_run_is_materially_enriched_with_v2_fields(client):
    body = client.get("/api/saved-runs/demo_curated_coils_results.csv").json()
    required = {
        "lifecycle",
        "grade",
        "lid_grade",
        "coil_score",
        "lid_slope_pct_per_year",
        "span_years",
        "touches",
        "reviewed",
        "data_date",
        "freshness",
    }
    assert required <= body["results"][0].keys()
    assert {row["lifecycle"] for row in body["results"]} >= {
        "pre_breakout",
        "forming",
        "post_breakout",
    }


def test_normalize_saved_results_declares_the_band_without_inventing_one():
    """Legacy CSVs predate the v2.2 band: null, present, and never overwritten."""
    legacy, current = app_module.normalize_saved_results(
        [
            {"ticker": "OLD", "score_total": 0.5},
            {
                "ticker": "NEW",
                "proximity_pct": 143.2,
                "current_price_position": "above_lid_band",
            },
        ]
    )

    assert "current_price_position" in legacy
    assert legacy["current_price_position"] is None
    assert legacy["proximity_pct"] is None
    # A run saved under v2.2 already carries the enum; normalization must not
    # blank it back out.
    assert current["current_price_position"] == "above_lid_band"
    assert current["proximity_pct"] == 143.2


def test_saved_run_missing_returns_404(client):
    resp = client.get("/api/saved-runs/does_not_exist.csv")
    assert resp.status_code == 404


def test_saved_run_non_csv_returns_404(client):
    resp = client.get("/api/saved-runs/README.txt")
    assert resp.status_code == 404


def test_saved_run_forward_slash_guard_returns_400():
    # A "/" in the filename must hit the explicit guard (400), never read outside
    # the project root. Starlette won't route %2F as a single path param, so we
    # exercise the handler's guard branch directly.
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        app_module.saved_run("../../etc/passwd")
    assert exc.value.status_code == 400


def test_saved_run_backslash_traversal_returns_400(client):
    # %5C decodes to "\\" which DOES reach the handler -> 400 guard branch
    resp = client.get("/api/saved-runs/..%5Csecret.csv")
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# History endpoint: cache miss (live fetch) then cache hit (no fetch)
# --------------------------------------------------------------------------- #
def test_history_cache_miss_then_hit(client, monkeypatch, tmp_cache):
    calls = {"n": 0}

    def fake_fetch(symbol: str):
        calls["n"] += 1
        return _synthetic_monthly()

    monkeypatch.setattr(app_module, "fetch_monthly_history", fake_fetch)

    # MISS: triggers live fetch, writes cache, returns full contract shape
    r1 = client.get("/api/history/TEST")
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["ticker"] == "TEST"
    assert set(body1.keys()) == {"ticker", "bars", "features", "freshness"}
    assert body1["freshness"]["status"] == "fresh"
    assert len(body1["bars"]) > 0
    bar = body1["bars"][0]
    assert set(bar.keys()) == {"date", "open", "high", "low", "close", "volume"}
    assert body1["features"] is not None
    assert calls["n"] == 1
    # cache file written
    assert (tmp_cache / "TEST.json").exists()

    # HIT: served from disk, fetch NOT called again, identical body
    r2 = client.get("/api/history/TEST")
    assert r2.status_code == 200
    assert r2.json() == body1
    assert calls["n"] == 1


def test_history_max_bars_trims_response(client, monkeypatch):
    monkeypatch.setattr(app_module, "fetch_monthly_history", lambda s: _synthetic_monthly(130))
    resp = client.get("/api/history/TRIM", params={"max_bars": 24})
    assert resp.status_code == 200
    bars = resp.json()["bars"]
    assert len(bars) == 24
    # trimming keeps the most-recent bars
    assert bars[-1]["date"] == "2020-10-01"


def test_history_returns_full_listing_history_by_default(client, monkeypatch):
    monkeypatch.setattr(
        app_module, "fetch_monthly_history", lambda s: _synthetic_monthly(240)
    )
    resp = client.get("/api/history/FULL")
    assert resp.status_code == 200
    assert len(resp.json()["bars"]) == 240


def test_history_legacy_cache_refresh_failure_is_explicit_fallback(client, monkeypatch, tmp_cache):
    """Metadata-free writable entries are stale and only fallback after refresh."""
    tmp_cache.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": "SEED",
        "bars": [
            {"date": "2024-01-01", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}
        ],
        "features": None,
    }
    (tmp_cache / "SEED.json").write_text(json.dumps(payload), encoding="utf-8")

    def boom(symbol: str):
        raise RuntimeError("provider offline")

    monkeypatch.setattr(app_module, "fetch_monthly_history", boom)

    resp = client.get("/api/history/SEED")
    assert resp.status_code == 200
    body = resp.json()
    assert body["bars"] == payload["bars"]
    assert body["freshness"]["status"] == "stale_fallback"
    assert body["freshness"]["origin"] == "writable_cache"
    assert body["freshness"]["refresh_error"] == "provider offline"


def test_history_404_when_no_data(client, monkeypatch):
    monkeypatch.setattr(app_module, "fetch_monthly_history", lambda s: None)
    resp = client.get("/api/history/NODATA")
    assert resp.status_code == 404


def test_seed_cache_served_when_writable_cache_empty(monkeypatch, tmp_path):
    """With an empty writable cache, read_cache falls back to the tracked seed."""
    seed_dir = tmp_path / "seed_cache"
    seed_dir.mkdir()
    payload = {"ticker": "SEEDED", "bars": [{"date": "2024-01-01", "close": 1.5}], "features": None}
    (seed_dir / "SEEDED.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(history_cache, "SEED_DIR", seed_dir)
    # tmp_cache autouse points CACHE_DIR at an empty (nonexistent) tmp dir.
    assert history_cache.read_cache("SEEDED") == payload


def test_writable_cache_takes_precedence_over_seed(monkeypatch, tmp_path, tmp_cache):
    """A fresh write in cache/ shadows a stale seed of the same ticker."""
    tmp_cache.mkdir(parents=True, exist_ok=True)
    seed_dir = tmp_path / "seed_cache"
    seed_dir.mkdir()
    (seed_dir / "DUP.json").write_text(
        json.dumps({"ticker": "DUP", "bars": [{"date": "2000-01-01", "close": 1.0}], "features": None}),
        encoding="utf-8",
    )
    monkeypatch.setattr(history_cache, "SEED_DIR", seed_dir)
    fresh = history_cache.write_cache("DUP", [{"date": "2024-06-01", "close": 99.0}], None)
    assert history_cache.read_cache("DUP") == fresh
    assert history_cache.read_cache("DUP")["bars"][0]["close"] == 99.0


def test_bundled_seed_covers_demo_universe():
    """The 11 curated demo names ship a valid, non-empty seed payload."""
    demo = ["MSCI", "UNP", "SPG", "VTR", "NSC", "REG", "NUE", "CSX", "BG", "FCX", "LH"]
    for sym in demo:
        payload = history_cache._read_payload(history_cache.SEED_DIR / f"{sym}.json")
        assert payload is not None, f"missing seed for {sym}"
        assert payload["ticker"] == sym
        assert len(payload["bars"]) > 0


# --------------------------------------------------------------------------- #
# /api/coil/{ticker}: deterministic structure analysis over cached bars
# --------------------------------------------------------------------------- #
def _seed_coil_cache(tmp_cache, ticker: str, bars: list[dict]) -> None:
    tmp_cache.mkdir(parents=True, exist_ok=True)
    payload = {"ticker": ticker, "bars": bars, "features": None}
    (tmp_cache / f"{ticker}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_coil_endpoint_analyzes_cached_bars(client, monkeypatch, tmp_cache):
    from test_coil_analysis import make_coil_bars

    _seed_coil_cache(tmp_cache, "COIL", make_coil_bars())

    def boom(symbol: str):
        raise AssertionError("live fetch must not be called on a cache hit")

    monkeypatch.setattr(app_module, "fetch_monthly_history", boom)

    resp = client.get("/api/coil/coil")  # lowercase input normalizes
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "COIL"
    assert body["status"] == "coiling"
    assert body["grade"] == "A"
    assert body["resistance"]["touch_count"] == 3
    assert body["resistance"]["from"].keys() == {"idx", "date", "price"}
    # touch_count includes only confirmed members of the winning price zone.
    # major_highs stays capped at display_max_highs for the legacy overlay, and
    # the cap protects both lid anchors.
    assert len(body["major_highs"]) == 3
    anchor_idxs = {body["resistance"]["from"]["idx"], body["resistance"]["to"]["idx"]}
    assert anchor_idxs <= {point["idx"] for point in body["major_highs"]}
    assert [high["date"] for high in body["major_highs"]] == [
        "2011-09-01",
        "2015-01-01",
        "2018-05-01",
    ]
    assert "2019-12-01" not in [point["date"] for point in body["points"]]
    assert body["schema_version"] == 2
    assert body["lifecycle"] == "pre_breakout"
    assert body["active_lid"]["grade"] == "A"
    assert body["breakout"]["state"] == "sealed"
    assert {p["role"] for p in body["points"]} <= {
        "major_top",
        "structural_retest",
        "provisional_top",
        "breakout_peak",
    }


def test_coil_algorithm_only_never_reads_review_state(
    client, monkeypatch, tmp_cache
):
    from test_coil_analysis import make_coil_bars

    _seed_coil_cache(tmp_cache, "PURE", make_coil_bars())
    monkeypatch.setattr(
        app_module,
        "get_review_store",
        lambda: (_ for _ in ()).throw(
            AssertionError("algorithm_only must not open the review store")
        ),
    )

    resp = client.get(
        "/api/coil/PURE",
        params={"variant": "v2_3_1", "mode": "algorithm_only"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["analysis_metadata"]["variant"] == "v2_3_1"
    assert body["analysis_metadata"]["mode"] == "algorithm_only"
    assert body["review"]["analysis_mode"] == "algorithm_only"


def test_coil_v24_validation_endpoint_is_additive_and_algorithm_only(
    client, monkeypatch, tmp_cache
):
    from test_coil_validation_v24 import mature_coil_bars

    _seed_coil_cache(tmp_cache, "VALIDATE", mature_coil_bars())
    monkeypatch.setattr(app_module, "fetch_monthly_history", lambda symbol: None)
    monkeypatch.setattr(
        app_module,
        "get_review_store",
        lambda: (_ for _ in ()).throw(
            AssertionError("validation detector must not open the review store")
        ),
    )

    resp = client.get(
        "/api/coil/VALIDATE",
        params={"variant": "v2_4_validation", "mode": "algorithm_only"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["analysis_metadata"]["variant"] == "v2_4_validation"
    assert body["analysis_metadata"]["mode"] == "algorithm_only"
    assert body["top_candidates"]
    assert body["lid_hypotheses"]
    assert body["resistance_band"] is not None
    assert body["pattern_assessment"]["structure_state"] == "qualified"


def test_coil_v24_endpoint_scopes_evidence_ids_to_ticker(client, monkeypatch, tmp_cache):
    from test_coil_validation_v24 import mature_coil_bars

    bars = mature_coil_bars()
    for ticker in ("FIRST", "SECOND"):
        _seed_coil_cache(tmp_cache, ticker, bars)
    monkeypatch.setattr(app_module, "fetch_monthly_history", lambda symbol: None)

    responses = [
        client.get(
            f"/api/coil/{ticker}",
            params={"variant": "v2_4_validation", "mode": "algorithm_only"},
        )
        for ticker in ("FIRST", "SECOND")
    ]

    assert all(response.status_code == 200 for response in responses)
    evidence_ids = [
        {candidate["id"] for candidate in response.json()["top_candidates"]}
        for response in responses
    ]
    assert evidence_ids[0]
    assert evidence_ids[0].isdisjoint(evidence_ids[1])


def test_coil_v24_validation_rejects_effective_mode(client, tmp_cache):
    from test_coil_validation_v24 import mature_coil_bars

    _seed_coil_cache(tmp_cache, "VALIDATE", mature_coil_bars())

    resp = client.get(
        "/api/coil/VALIDATE",
        params={"variant": "v2_4_validation", "mode": "effective"},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == (
        "v2_4_validation is available only in algorithm_only mode."
    )


def test_coil_endpoint_as_of_replays_pre_breakout(client, tmp_cache):
    from test_coil_analysis import make_coil_bars, month_dates

    bars = make_coil_bars()
    dates = month_dates(126)
    for k in range(6):
        bars.append(
            {"date": dates[120 + k], "open": 110.0, "high": 110.5, "low": 108.0, "close": 110.0, "volume": 1e6}
        )
    _seed_coil_cache(tmp_cache, "REPLAY", bars)

    full = client.get("/api/coil/REPLAY").json()
    truncated = client.get("/api/coil/REPLAY", params={"as_of": dates[119]}).json()

    assert full["status"] == "broken_out"
    assert truncated["status"] == "coiling"
    assert truncated["grade"] == "A"


def test_coil_endpoint_rejects_bad_as_of(client, tmp_cache):
    from test_coil_analysis import make_coil_bars

    _seed_coil_cache(tmp_cache, "BADQ", make_coil_bars())
    resp = client.get("/api/coil/BADQ", params={"as_of": "20250101"})
    assert resp.status_code == 400


def test_coil_endpoint_404_when_no_data(client, monkeypatch):
    monkeypatch.setattr(app_module, "fetch_monthly_history", lambda s: None)
    resp = client.get("/api/coil/NODATA")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# /api/screen contract (run_screen monkeypatched -> no network)
# --------------------------------------------------------------------------- #
def test_screen_contract(client, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "run_lifecycle_screen",
        lambda tickers, **kwargs: {
            "results": [{"ticker": "TEST", "lifecycle": "pre_breakout"}],
            "bucket_counts": {"pre_breakout": 1},
            "failures": [],
            "algorithm_version": "test-v2",
            "screened_at": "2026-01-01T00:00:00.000Z",
        },
    )

    resp = client.post("/api/screen", json={"tickers": ["TEST"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["tickers"] == ["TEST"]
    assert body["results"][0]["ticker"] == "TEST"
    assert body["bucket_counts"] == {"pre_breakout": 1}
    assert body["failures"] == []


def test_screen_forwards_explicit_analysis_variant_and_mode(client, monkeypatch):
    seen = {}

    def fake_run(tickers, **kwargs):
        seen.update(kwargs)
        return {
            "results": [],
            "bucket_counts": {},
            "failures": [],
            "algorithm_version": "test-v2",
            "analysis_variant": kwargs["analysis_variant"],
            "analysis_mode": kwargs["analysis_mode"],
            "screened_at": "2026-01-01T00:00:00.000Z",
        }

    monkeypatch.setattr(app_module, "run_lifecycle_screen", fake_run)

    resp = client.post(
        "/api/screen",
        json={
            "tickers": ["TEST"],
            "analysisVariant": "v2_3_1",
            "analysisMode": "algorithm_only",
        },
    )

    assert resp.status_code == 200
    assert seen["analysis_variant"] == "v2_3_1"
    assert seen["analysis_mode"] == "algorithm_only"
    assert resp.json()["analysis_mode"] == "algorithm_only"


def test_screen_forwards_v24_validation_variant(client, monkeypatch):
    seen = {}

    def fake_run(tickers, **kwargs):
        seen.update(kwargs)
        return {
            "results": [],
            "bucket_counts": {},
            "failures": [],
            "algorithm_version": "2.4.0-validation",
            "analysis_variant": kwargs["analysis_variant"],
            "analysis_mode": kwargs["analysis_mode"],
            "screened_at": "2026-01-01T00:00:00.000Z",
        }

    monkeypatch.setattr(app_module, "run_lifecycle_screen", fake_run)
    resp = client.post(
        "/api/screen",
        json={
            "tickers": ["TEST"],
            "analysisVariant": "v2_4_validation",
            "analysisMode": "algorithm_only",
        },
    )

    assert resp.status_code == 200
    assert seen["analysis_variant"] == "v2_4_validation"
    assert seen["analysis_mode"] == "algorithm_only"


def test_screen_rejects_v24_effective_mode_before_opening_review_state(client):
    resp = client.post(
        "/api/screen",
        json={
            "tickers": ["TEST"],
            "analysisVariant": "v2_4_validation",
            "analysisMode": "effective",
        },
    )

    assert resp.status_code == 400


def test_screen_rows_expose_the_lid_band_position_end_to_end(
    client, monkeypatch, tmp_cache
):
    """Real analysis -> row -> /api/screen: the band enum survives untouched.

    ``/api/screen`` forwards rows verbatim, so this also pins that the enum
    and the ratio it is derived from stay consistent for a coil pressing its
    lid from below.
    """
    from test_coil_analysis import make_coil_bars

    _seed_coil_cache(tmp_cache, "BAND", make_coil_bars())
    monkeypatch.setattr(app_module, "fetch_monthly_history", lambda s: None)

    body = client.post("/api/screen", json={"tickers": ["BAND"]}).json()

    row = body["results"][0]
    assert row["ticker"] == "BAND"
    assert row["current_price_position"] == "within_lid_band"
    assert 80.0 <= row["proximity_pct"] <= 120.0
