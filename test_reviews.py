"""Review persistence + /api/highs/corrections + review-precedence tests.

The store is exercised against SQLite (the local/test backend). Precedence:
approved points become immediately effective in /api/coil, the algorithm's
own output is retained for comparison, and derived slope/status recompute
from the reviewed anchors (including under ``as_of`` replay).
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
import history_cache
import reviews as reviews_module
from reviews import ReviewStore
from test_coil_analysis import make_coil_bars


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(history_cache, "CACHE_DIR", cache_dir)
    return cache_dir


@pytest.fixture(autouse=True)
def tmp_review_store(tmp_path, monkeypatch):
    store = ReviewStore(sqlite_path=tmp_path / "reviews.db")
    monkeypatch.setattr(reviews_module, "_store", store)
    return store


def _correction(ticker: str, highs: list[dict]) -> dict:
    return {
        "schemaVersion": 1,
        "ticker": ticker,
        "interval": "3M",
        "createdAt": "2026-07-01T00:00:00.000Z",
        "barCount": 120,
        "dateRange": {"from": "2010-01-01", "to": "2019-12-01"},
        "algoHighs": [],
        "manualHighs": highs,
        "diff": {"added": highs, "removed": [], "kept": []},
    }


def _anchors(*, first_price: float = 100.0, second_price: float = 101.0) -> list[dict]:
    return [
        {"idx": 20, "date": "2011-09-01", "price": first_price},
        {"idx": 60, "date": "2015-01-01", "price": second_price},
    ]


def _seed_cache(tmp_cache, ticker: str, bars: list[dict]) -> None:
    tmp_cache.mkdir(parents=True, exist_ok=True)
    payload = {"ticker": ticker, "bars": bars, "features": None}
    (tmp_cache / f"{ticker}.json").write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Store behavior
# --------------------------------------------------------------------------- #
def test_reviews_are_append_only_and_override_is_latest(tmp_review_store):
    first = tmp_review_store.append_review(
        _correction("KN", _anchors())
    )
    second = tmp_review_store.append_review(
        _correction(
            "KN",
            [
                {"idx": 60, "date": "2015-01-01", "price": 101.0},
                {"idx": 100, "date": "2018-05-01", "price": 102.0},
            ],
        )
    )

    history = tmp_review_store.list_reviews("KN")
    assert [entry["id"] for entry in history] == [first["id"], second["id"]]
    # Both original records retained verbatim (append-only).
    assert history[0]["record"]["manualHighs"][0]["date"] == "2011-09-01"
    assert history[1]["record"]["manualHighs"][0]["date"] == "2015-01-01"

    override = tmp_review_store.get_override("KN")
    assert override["review_id"] == second["id"]
    assert override["points"] == [
        {"date": "2015-01-01", "price": 101.0},
        {"date": "2018-05-01", "price": 102.0},
    ]


def test_review_requires_ticker(tmp_review_store):
    with pytest.raises(ValueError):
        tmp_review_store.append_review(_correction("", _anchors()))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record.update(interval="1M"), "3M"),
        (lambda record: record["manualHighs"][0].update(date="09/01/2011"), "ISO"),
        (lambda record: record["manualHighs"][0].update(price=0), "greater than 0"),
        (lambda record: record["manualHighs"][0].update(role="shoulder"), "role"),
        (
            lambda record: record.update(
                manualHighs=[
                    {
                        "date": "2018-05-01",
                        "price": 112.0,
                        "role": "breakout_peak",
                    }
                ]
            ),
            "at least two non-breakout anchors",
        ),
    ],
)
def test_authoritative_review_validation(tmp_review_store, mutate, message):
    record = _correction("KN", _anchors())
    mutate(record)
    with pytest.raises(ValueError, match=message):
        tmp_review_store.append_review(record)


def test_override_roles_are_preserved(tmp_review_store):
    tmp_review_store.append_review(
        _correction(
            "VTRX",
            [
                {"idx": 1, "date": "2016-07-01", "price": 76.8},
                {"idx": 2, "date": "2019-09-01", "price": 75.4},
                {"idx": 3, "date": "2026-05-01", "price": 91.06, "role": "breakout_peak"},
            ],
        )
    )
    override = tmp_review_store.get_override("VTRX")
    assert override["points"][2]["role"] == "breakout_peak"


def test_revoke_clears_override_but_appends_audit_event(tmp_review_store):
    accepted = tmp_review_store.append_review(_correction("KN", _anchors()))

    revoked = tmp_review_store.revoke_override("kn")

    assert revoked["revoked_review_id"] == accepted["id"]
    assert tmp_review_store.get_override("KN") is None
    history = tmp_review_store.list_reviews("KN")
    assert len(history) == 2
    assert history[0]["record"]["manualHighs"] == _anchors()
    assert history[1]["record"] == {
        "schemaVersion": 2,
        "action": "revoke",
        "ticker": "KN",
        "interval": "3M",
        "createdAt": revoked["created_at"],
        "revokedReviewId": accepted["id"],
    }
    assert tmp_review_store.revoke_override("KN") is None


# --------------------------------------------------------------------------- #
# Endpoint contract
# --------------------------------------------------------------------------- #
def test_post_correction_endpoint_persists_and_returns_review(client):
    record = _correction("KN", _anchors())
    resp = client.post("/api/highs/corrections", json=record)
    assert resp.status_code == 200
    review = resp.json()["review"]
    assert review["ticker"] == "KN"
    assert review["points"] == [
        {"date": "2011-09-01", "price": 100.0},
        {"date": "2015-01-01", "price": 101.0},
    ]

    listing = client.get("/api/highs/corrections/KN")
    assert listing.status_code == 200
    body = listing.json()
    assert body["override"]["review_id"] == review["id"]
    assert len(body["reviews"]) == 1


def test_post_correction_without_ticker_is_422(client):
    resp = client.post("/api/highs/corrections", json=_correction("", []))
    assert resp.status_code == 422


def test_delete_correction_revokes_effective_override_and_keeps_history(client):
    posted = client.post(
        "/api/highs/corrections", json=_correction("KN", _anchors())
    )
    assert posted.status_code == 200

    deleted = client.delete("/api/highs/corrections/kn", params={"interval": "3M"})

    assert deleted.status_code == 200
    assert deleted.json()["override"] is None
    listing = client.get("/api/highs/corrections/KN").json()
    assert listing["override"] is None
    assert [entry["record"].get("action", "accept") for entry in listing["reviews"]] == [
        "accept",
        "revoke",
    ]
    assert client.delete("/api/highs/corrections/KN").status_code == 404


# --------------------------------------------------------------------------- #
# Review precedence in /api/coil
# --------------------------------------------------------------------------- #
def test_reviewed_points_override_algorithm_structure(client, tmp_cache):
    bars = make_coil_bars()  # algorithm finds touches at bars 20/60/100
    _seed_cache(tmp_cache, "RVW", bars)

    baseline = client.get("/api/coil/RVW").json()
    assert baseline["review"]["reviewed"] is False
    assert baseline["review"]["effective"] == "algorithm"

    # The reviewer keeps only the first and last touches.
    client.post(
        "/api/highs/corrections",
        json=_correction(
            "RVW",
            [
                {"idx": 20, "date": "2011-09-01", "price": 100.0},
                {"idx": 100, "date": "2018-05-01", "price": 100.0},
            ],
        ),
    )

    reviewed = client.get("/api/coil/RVW").json()
    assert reviewed["review"]["reviewed"] is True
    assert reviewed["review"]["effective"] == "human"
    assert reviewed["review"]["review_id"] is not None
    # The algorithm's own structure is retained for comparison.
    assert reviewed["review"]["algorithm"] is not None
    assert len(reviewed["review"]["algorithm"]["major_highs"]) == 3
    # The effective lid is fit through the reviewed anchors.
    anchors = reviewed["active_lid"]["anchors"]
    assert [a["date"] for a in anchors] == ["2011-09-01", "2018-05-01"]
    assert reviewed["resistance"]["slope_pct_per_year"] == pytest.approx(0.0, abs=0.1)
    assert reviewed["grade"] == "A"


def test_reviewed_breakout_peak_is_excluded_from_lid_fit(client, tmp_cache):
    from test_coil_analysis import month_dates

    bars = make_coil_bars()
    dates = month_dates(126)
    for k in range(6):
        bars.append(
            {"date": dates[120 + k], "open": 110.0, "high": 112.0, "low": 108.0, "close": 110.0, "volume": 1e6}
        )
    _seed_cache(tmp_cache, "RVB", bars)

    client.post(
        "/api/highs/corrections",
        json=_correction(
            "RVB",
            [
                {"idx": 20, "date": "2011-09-01", "price": 100.0},
                {"idx": 100, "date": "2018-05-01", "price": 100.0},
                {"idx": 124, "date": "2020-05-01", "price": 112.0, "role": "breakout_peak"},
            ],
        ),
    )

    reviewed = client.get("/api/coil/RVB").json()
    anchors = reviewed["active_lid"]["anchors"]
    assert [a["date"] for a in anchors] == ["2011-09-01", "2018-05-01"]
    roles = {p["date"][:7]: p["role"] for p in reviewed["points"]}
    assert roles["2020-05"] == "breakout_peak"
    # The flat reviewed lid stays; the escape is classified, not erased.
    assert reviewed["status"] == "broken_out"
    assert reviewed["lifecycle"] == "post_breakout"


def test_review_points_after_as_of_are_ignored_in_replay(client, tmp_cache):
    from test_coil_analysis import month_dates

    bars = make_coil_bars()
    dates = month_dates(126)
    for k in range(6):
        bars.append(
            {"date": dates[120 + k], "open": 110.0, "high": 112.0, "low": 108.0, "close": 110.0, "volume": 1e6}
        )
    _seed_cache(tmp_cache, "RVA", bars)

    client.post(
        "/api/highs/corrections",
        json=_correction(
            "RVA",
            [
                {"idx": 20, "date": "2011-09-01", "price": 100.0},
                {"idx": 100, "date": "2018-05-01", "price": 100.0},
                {"idx": 124, "date": "2020-05-01", "price": 112.0, "role": "breakout_peak"},
            ],
        ),
    )

    replay = client.get("/api/coil/RVA", params={"as_of": dates[119]}).json()
    assert replay["review"]["effective"] == "human"
    # The 2020 breakout peak does not exist yet at the replay date.
    assert all(not p["date"].startswith("2020") for p in replay["points"])
    assert replay["status"] == "coiling"
