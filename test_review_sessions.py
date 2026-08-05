"""Review sessions + unified decision endpoint + staleness tests.

Covers the dedicated screened-stock review workflow: session create/resume by
snapshot fingerprint, ordered queues with computed statuses, session-specific
skips, shared snapshot-aware decisions (approve/correct), lid-member line
fitting, stale-override re-review flags, and legacy correction translation.
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import app as app_module
import history_cache
import reviews as reviews_module
from coil_analysis import ALGORITHM_VERSION
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


def _seed_cache(tmp_cache, ticker: str, bars: list[dict]) -> None:
    tmp_cache.mkdir(parents=True, exist_ok=True)
    payload = {"ticker": ticker, "bars": bars, "features": None}
    (tmp_cache / f"{ticker}.json").write_text(json.dumps(payload), encoding="utf-8")


def _session_payload(
    tickers: list[str], *, source: str = "screen:current", data_date: str = "2019-12-01"
) -> dict:
    return {
        "source": source,
        "items": [
            {"ticker": ticker, "snapshot": {"data_date": data_date, "grade": "A"}}
            for ticker in tickers
        ],
        "snapshot": {"count": len(tickers)},
    }


def _decision(
    ticker: str,
    *,
    decision: str = "corrected",
    highs: list[dict] | None = None,
    session_id: int | None = None,
    as_of: str = "2019-12-01",
    algorithm_version: str = ALGORITHM_VERSION,
) -> dict:
    return {
        "schemaVersion": 3,
        "sessionId": session_id,
        "ticker": ticker,
        "interval": "3M",
        "asOf": as_of,
        "algorithmVersion": algorithm_version,
        "decision": decision,
        "algorithm": {"majorHighs": [], "slopePctPerYear": 0.0},
        "reviewedHighs": highs or [],
        "createdAt": "2026-07-01T00:00:00.000Z",
    }


def _lid_highs() -> list[dict]:
    return [
        {"date": "2011-09-01", "price": 100.0, "lidMember": True},
        {"date": "2015-01-01", "price": 100.5, "lidMember": False},
        {"date": "2018-05-01", "price": 100.0, "lidMember": True},
    ]


def _open_quarter_bars() -> list[dict]:
    """A coil whose series runs two months into an unfinished quarter.

    ``make_coil_bars`` ends on 2019-12 (Q4 closed). Appending 2020-01 and
    2020-02 leaves Q1 2020 open, so those two months are live-price territory
    and not reviewable structure.
    """
    from test_coil_analysis import month_dates

    bars = make_coil_bars()
    dates = month_dates(122)
    for k in range(2):
        bars.append(
            {
                "date": dates[120 + k],
                "open": 96.0,
                "high": 97.0,
                "low": 95.0,
                "close": 96.0,
                "volume": 1e6,
            }
        )
    return bars


# --------------------------------------------------------------------------- #
# Sessions: create / resume / order / counts
# --------------------------------------------------------------------------- #
def test_create_session_preserves_ranked_order_and_counts(client):
    resp = client.post("/api/review-sessions", json=_session_payload(["BBB", "AAA", "CCC"]))
    assert resp.status_code == 200
    body = resp.json()
    assert body["created"] is True
    session = body["session"]
    assert [item["ticker"] for item in session["items"]] == ["BBB", "AAA", "CCC"]
    assert [item["position"] for item in session["items"]] == [0, 1, 2]
    assert session["counts"] == {"pending": 3, "reviewed": 0, "skipped": 0, "total": 3}
    assert session["next_pending_ticker"] == "BBB"


def test_same_snapshot_resumes_existing_session(client):
    first = client.post("/api/review-sessions", json=_session_payload(["AAA", "BBB"])).json()
    second = client.post("/api/review-sessions", json=_session_payload(["AAA", "BBB"])).json()
    assert second["created"] is False
    assert second["session"]["id"] == first["session"]["id"]


def test_new_data_or_order_starts_new_session(client):
    first = client.post("/api/review-sessions", json=_session_payload(["AAA", "BBB"])).json()
    reordered = client.post("/api/review-sessions", json=_session_payload(["BBB", "AAA"])).json()
    newer_data = client.post(
        "/api/review-sessions", json=_session_payload(["AAA", "BBB"], data_date="2020-03-01")
    ).json()
    assert reordered["created"] is True
    assert newer_data["created"] is True
    ids = {first["session"]["id"], reordered["session"]["id"], newer_data["session"]["id"]}
    assert len(ids) == 3


def test_session_requires_tickers(client):
    resp = client.post(
        "/api/review-sessions", json={"source": "screen:current", "items": []}
    )
    assert resp.status_code == 400


def test_get_unknown_session_is_404(client):
    assert client.get("/api/review-sessions/9999").status_code == 404


def test_export_unknown_session_is_404(client):
    assert client.get("/api/review-sessions/9999/export").status_code == 404


def test_session_export_is_versioned_and_keeps_all_session_revisions(client):
    session = client.post(
        "/api/review-sessions", json=_session_payload(["AAA", "BBB"])
    ).json()["session"]
    session_id = session["id"]

    first = client.post(
        "/api/highs/reviews",
        json=_decision("AAA", decision="approved", session_id=session_id),
    ).json()["review"]
    second = client.post(
        "/api/highs/reviews",
        json=_decision("AAA", highs=_lid_highs(), session_id=session_id),
    ).json()["review"]
    client.patch(
        f"/api/review-sessions/{session_id}/items/BBB", json={"status": "skipped"}
    )

    response = client.get(f"/api/review-sessions/{session_id}/export")
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 2
    assert body["kind"] == "coilingview.review-session-feedback"
    assert body["algorithm_version"] == ALGORITHM_VERSION
    assert body["session"]["fingerprint"] == session["fingerprint"]
    assert body["session"]["counts"] == {
        "pending": 0,
        "reviewed": 1,
        "skipped": 1,
        "total": 2,
    }
    assert [record["id"] for record in body["records"]] == [
        first["id"],
        second["id"],
    ]
    assert [record["record"]["decision"] for record in body["records"]] == [
        "approved",
        "corrected",
    ]
    assert all(record["event_id"] for record in body["records"])
    assert body["session"]["items"][0]["review_id"] == second["id"]


# --------------------------------------------------------------------------- #
# Session items: skip / unskip (session-specific)
# --------------------------------------------------------------------------- #
def test_skip_and_unskip_persist_per_session(client):
    session = client.post(
        "/api/review-sessions", json=_session_payload(["AAA", "BBB"])
    ).json()["session"]

    skipped = client.patch(
        f"/api/review-sessions/{session['id']}/items/AAA", json={"status": "skipped"}
    )
    assert skipped.status_code == 200
    assert skipped.json()["item"]["status"] == "skipped"

    fetched = client.get(f"/api/review-sessions/{session['id']}").json()["session"]
    assert fetched["counts"]["skipped"] == 1
    assert fetched["next_pending_ticker"] == "BBB"

    restored = client.patch(
        f"/api/review-sessions/{session['id']}/items/aaa", json={"status": "pending"}
    )
    assert restored.json()["item"]["status"] == "pending"


def test_reviewed_status_cannot_be_patched(client):
    session = client.post(
        "/api/review-sessions", json=_session_payload(["AAA"])
    ).json()["session"]
    resp = client.patch(
        f"/api/review-sessions/{session['id']}/items/AAA", json={"status": "reviewed"}
    )
    assert resp.status_code == 422


def test_patch_unknown_item_is_404(client):
    session = client.post(
        "/api/review-sessions", json=_session_payload(["AAA"])
    ).json()["session"]
    resp = client.patch(
        f"/api/review-sessions/{session['id']}/items/ZZZ", json={"status": "skipped"}
    )
    assert resp.status_code == 404


def test_skips_are_session_specific_while_decisions_are_shared(client):
    session_a = client.post(
        "/api/review-sessions", json=_session_payload(["AAA", "BBB"])
    ).json()["session"]
    session_b = client.post(
        "/api/review-sessions", json=_session_payload(["AAA", "BBB"], source="screen:saved")
    ).json()["session"]

    client.patch(
        f"/api/review-sessions/{session_a['id']}/items/AAA", json={"status": "skipped"}
    )
    client.post("/api/highs/reviews", json=_decision("BBB", decision="approved"))

    fetched_a = client.get(f"/api/review-sessions/{session_a['id']}").json()["session"]
    fetched_b = client.get(f"/api/review-sessions/{session_b['id']}").json()["session"]
    status_a = {item["ticker"]: item["status"] for item in fetched_a["items"]}
    status_b = {item["ticker"]: item["status"] for item in fetched_b["items"]}
    assert status_a == {"AAA": "skipped", "BBB": "reviewed"}
    assert status_b == {"AAA": "pending", "BBB": "reviewed"}


# --------------------------------------------------------------------------- #
# Unified decision endpoint
# --------------------------------------------------------------------------- #
def test_approval_records_event_and_clears_override(client, tmp_cache, tmp_review_store):
    _seed_cache(tmp_cache, "APV", make_coil_bars())
    # An older correction exists; approving the algorithm must supersede it.
    client.post(
        "/api/highs/reviews",
        json=_decision("APV", highs=_lid_highs()),
    )
    assert tmp_review_store.get_override("APV") is not None

    resp = client.post("/api/highs/reviews", json=_decision("APV", decision="approved"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["review"]["decision"] == "approved"
    assert body["override"] is None
    assert tmp_review_store.get_override("APV") is None
    # The recomputed analysis is the algorithm's own, but marked human-reviewed.
    assert body["analysis"]["review"]["reviewed"] is True
    assert body["analysis"]["review"]["effective"] == "algorithm"
    assert body["analysis"]["review"]["stale"] is False
    # Both decisions remain in the append-only history.
    history = tmp_review_store.list_reviews("APV")
    assert [entry["record"]["decision"] for entry in history] == ["corrected", "approved"]


def test_schema4_keeps_top_verdict_separate_from_not_coil_label(
    client, tmp_cache, tmp_review_store
):
    _seed_cache(tmp_cache, "SEP", make_coil_bars())
    session = client.post(
        "/api/review-sessions", json=_session_payload(["SEP"])
    ).json()["session"]
    record = _decision(
        "SEP", highs=_lid_highs(), session_id=session["id"]
    )
    record.update(
        {
            "schemaVersion": 4,
            "labelPolicyVersion": 1,
            "coilLabel": "not_coil",
            "confidence": "high",
            "note": "The reviewed tops are valid, but the chart is not compressing.",
        }
    )

    response = client.post("/api/highs/reviews", json=record)

    assert response.status_code == 200
    body = response.json()
    assert body["review"]["decision"] == "corrected"
    assert body["review"]["coil_label"] == "not_coil"
    assert body["review"]["event_id"]
    assert body["override"] is not None
    state = tmp_review_store.get_review_state("SEP")
    assert state["coil_label"] == "not_coil"
    assert state["human_grade"] is None
    assert state["confidence"] == "high"
    assert state["note"] == record["note"]
    item = body["session_item"]
    assert item["review_coil_label"] == "not_coil"
    assert item["effective"] == "human"


def test_schema4_coil_label_round_trips_grade_confidence_and_note(
    client, tmp_cache, tmp_review_store
):
    _seed_cache(tmp_cache, "LBL", make_coil_bars())
    record = _decision("LBL", decision="approved")
    record.update(
        {
            "schemaVersion": 4,
            "labelPolicyVersion": 1,
            "coilLabel": "coil",
            "humanGrade": "B",
            "confidence": "low",
            "note": "Borderline slope, but the long-term compression is intact.",
        }
    )

    response = client.post("/api/highs/reviews", json=record)

    assert response.status_code == 200
    review = response.json()["review"]
    assert review["human_grade"] == "B"
    assert review["confidence"] == "low"
    assert review["note"] == record["note"]
    analysis_review = response.json()["analysis"]["review"]
    assert analysis_review["coil_label"] == "coil"
    assert analysis_review["human_grade"] == "B"
    assert analysis_review["confidence"] == "low"
    assert analysis_review["note"] == record["note"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"coilLabel": None, "humanGrade": None}, "coilLabel"),
        ({"coilLabel": "coil", "humanGrade": None}, "humanGrade"),
        (
            {"coilLabel": "not_coil", "humanGrade": "A", "note": "No compression."},
            "only applies",
        ),
        ({"coilLabel": "not_coil", "humanGrade": None, "note": None}, "rationale"),
    ],
)
def test_schema4_rejects_ambiguous_labels(client, changes, message):
    record = _decision("BAD", decision="approved")
    record.update(
        {
            "schemaVersion": 4,
            "labelPolicyVersion": 1,
            "coilLabel": "coil",
            "humanGrade": "A",
            "confidence": "high",
        }
    )
    record.update(changes)
    response = client.post("/api/highs/reviews", json=record)
    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_decision_session_must_contain_ticker(client):
    session = client.post(
        "/api/review-sessions", json=_session_payload(["AAA"])
    ).json()["session"]
    response = client.post(
        "/api/highs/reviews",
        json=_decision("BBB", decision="approved", session_id=session["id"]),
    )
    assert response.status_code == 400
    assert "containing this ticker" in response.json()["detail"]


def test_correction_with_lid_members_fits_line_from_members_only(client, tmp_cache):
    _seed_cache(tmp_cache, "LID", make_coil_bars())
    resp = client.post("/api/highs/reviews", json=_decision("LID", highs=_lid_highs()))
    assert resp.status_code == 200
    body = resp.json()
    assert body["override"]["points"] == [
        {"date": "2011-09-01", "price": 100.0, "lid_member": True},
        {"date": "2015-01-01", "price": 100.5, "lid_member": False},
        {"date": "2018-05-01", "price": 100.0, "lid_member": True},
    ]
    analysis = body["analysis"]
    anchors = analysis["active_lid"]["anchors"]
    assert [a["date"] for a in anchors] == ["2011-09-01", "2018-05-01"]
    # The non-member top is still plotted as a reviewed major top.
    reviewed_dates = {p["date"][:7] for p in analysis["points"]}
    assert "2015-01" in reviewed_dates
    members = {p["date"][:7]: p["lid_member"] for p in analysis["points"]}
    assert members["2011-09"] is True
    assert members["2015-01"] is False
    # Slope/grade recompute immediately from the two flat anchors.
    assert analysis["resistance"]["slope_pct_per_year"] == pytest.approx(0.0, abs=0.1)
    assert analysis["review"]["effective"] == "human"


def test_decision_updates_session_item(client, tmp_cache):
    _seed_cache(tmp_cache, "SES", make_coil_bars())
    session = client.post(
        "/api/review-sessions", json=_session_payload(["SES", "AAA"])
    ).json()["session"]
    resp = client.post(
        "/api/highs/reviews",
        json=_decision("SES", highs=_lid_highs(), session_id=session["id"]),
    )
    item = resp.json()["session_item"]
    assert item["ticker"] == "SES"
    assert item["status"] == "reviewed"
    assert item["review_decision"] == "corrected"
    assert item["effective"] == "human"

    fetched = client.get(f"/api/review-sessions/{session['id']}").json()["session"]
    assert fetched["counts"] == {"pending": 1, "reviewed": 1, "skipped": 0, "total": 2}
    assert fetched["next_pending_ticker"] == "AAA"


def test_editing_a_review_appends_a_new_revision(client, tmp_cache, tmp_review_store):
    _seed_cache(tmp_cache, "EDT", make_coil_bars())
    first = client.post("/api/highs/reviews", json=_decision("EDT", highs=_lid_highs()))
    revised_highs = [
        {"date": "2011-09-01", "price": 100.0, "lidMember": True},
        {"date": "2015-01-01", "price": 100.5, "lidMember": True},
    ]
    second = client.post("/api/highs/reviews", json=_decision("EDT", highs=revised_highs))

    history = tmp_review_store.list_reviews("EDT")
    assert len(history) == 2
    override = tmp_review_store.get_override("EDT")
    assert override["review_id"] == second.json()["review"]["id"]
    assert override["review_id"] != first.json()["review"]["id"]


@pytest.mark.parametrize(
    ("highs", "message"),
    [
        ([{"date": "2011-09-01", "price": 100.0, "lidMember": True}], "two lid members"),
        (
            [
                {"date": "2011-09-01", "price": 100.0, "lidMember": True},
                {"date": "2011-09-01", "price": 101.0, "lidMember": True},
            ],
            "duplicate",
        ),
        (
            [
                {"date": "2011-09-01", "price": 100.0, "lidMember": True},
                {"date": "2018-05-01", "price": 112.0, "role": "breakout_peak", "lidMember": True},
            ],
            "breakout peaks cannot be lid anchors",
        ),
        ([{"date": "2011-09-01", "price": 100.0}], "at least two non-breakout anchors"),
    ],
)
def test_invalid_corrections_are_400(client, highs, message):
    resp = client.post("/api/highs/reviews", json=_decision("BAD", highs=highs))
    assert resp.status_code == 400
    assert message in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# Incomplete-final-quarter anchor rule (v2.2 §7)
# --------------------------------------------------------------------------- #
def test_decision_anchored_to_the_open_quarter_is_rejected(
    client, tmp_cache, tmp_review_store
):
    """Reviewers keep zone authority, but the open quarter is not structure yet."""
    _seed_cache(tmp_cache, "OPN", _open_quarter_bars())

    resp = client.post(
        "/api/highs/reviews",
        json=_decision(
            "OPN",
            highs=[
                {"date": "2011-09-01", "price": 100.0, "lidMember": True},
                {"date": "2020-02-01", "price": 97.0, "lidMember": True},
            ],
        ),
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "2020-02-01" in detail
    assert "incomplete final quarter" in detail
    # Refused before any write: no decision, no override, no audit entry.
    assert tmp_review_store.get_review_state("OPN") is None
    assert tmp_review_store.get_override("OPN") is None
    assert tmp_review_store.list_reviews("OPN") == []


def test_incomplete_quarter_rule_covers_non_anchor_roles(client, tmp_review_store, tmp_cache):
    """The analyzer drops every point in the open quarter, anchor or not.

    A breakout-peak marker there is plotted-only, but it would still vanish
    silently, so it is refused rather than accepted and discarded.
    """
    _seed_cache(tmp_cache, "PEAK", _open_quarter_bars())

    resp = client.post(
        "/api/highs/reviews",
        json=_decision(
            "PEAK",
            highs=[
                {"date": "2011-09-01", "price": 100.0, "lidMember": True},
                {"date": "2018-05-01", "price": 100.0, "lidMember": True},
                {"date": "2020-01-01", "price": 97.0, "role": "breakout_peak"},
            ],
        ),
    )

    assert resp.status_code == 400
    assert "2020-01-01" in resp.json()["detail"]
    assert tmp_review_store.get_override("PEAK") is None


def test_decision_anchored_to_the_last_closed_quarter_is_accepted(
    client, tmp_cache, tmp_review_store, monkeypatch
):
    monkeypatch.setattr(app_module, "fetch_monthly_history", lambda symbol: None)
    _seed_cache(tmp_cache, "CLS", _open_quarter_bars())

    resp = client.post(
        "/api/highs/reviews",
        json=_decision(
            "CLS",
            highs=[
                {"date": "2011-09-01", "price": 100.0, "lidMember": True},
                {"date": "2019-12-01", "price": 100.0, "lidMember": True},
            ],
        ),
    )

    assert resp.status_code == 200
    assert tmp_review_store.get_review_state("CLS")["decision"] == "corrected"
    anchors = resp.json()["analysis"]["active_lid"]["anchors"]
    assert [a["date"] for a in anchors] == ["2011-09-01", "2019-12-01"]


def test_invalid_as_of_is_400(client):
    record = _decision("BAD", decision="approved")
    record["asOf"] = "07/01/2026"
    resp = client.post("/api/highs/reviews", json=record)
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# Staleness: new data or algorithm returns items to pending
# --------------------------------------------------------------------------- #
def test_stale_data_returns_item_to_pending_but_override_stays_effective(client, tmp_cache):
    _seed_cache(tmp_cache, "STL", make_coil_bars())
    session = client.post(
        "/api/review-sessions", json=_session_payload(["STL"], data_date="2019-12-01")
    ).json()["session"]
    client.post(
        "/api/highs/reviews",
        json=_decision("STL", highs=_lid_highs(), session_id=session["id"], as_of="2019-09-01"),
    )

    fetched = client.get(f"/api/review-sessions/{session['id']}").json()["session"]
    item = fetched["items"][0]
    assert item["status"] == "pending"
    assert item["review_stale"] is True
    assert item["review_decision"] == "corrected"
    # The stale human override remains the effective analysis.
    assert item["effective"] == "human"
    coil = client.get("/api/coil/STL").json()
    assert coil["review"]["effective"] == "human"
    assert coil["review"]["stale"] is True


def test_algorithm_version_change_flags_stale(client, tmp_cache):
    _seed_cache(tmp_cache, "ALG", make_coil_bars())
    session = client.post(
        "/api/review-sessions", json=_session_payload(["ALG"])
    ).json()["session"]
    client.post(
        "/api/highs/reviews",
        json=_decision(
            "ALG", decision="approved", session_id=session["id"], algorithm_version="0.9.0"
        ),
    )
    item = client.get(f"/api/review-sessions/{session['id']}").json()["session"]["items"][0]
    assert item["status"] == "pending"
    assert item["review_stale"] is True
    coil = client.get("/api/coil/ALG").json()
    assert coil["review"]["reviewed"] is True
    assert coil["review"]["stale"] is True


def test_algorithm_bump_reopens_every_stored_review_without_touching_it(
    client, tmp_cache, tmp_review_store, monkeypatch
):
    """An algorithm bump re-opens the whole stored corpus, non-destructively.

    Mirrors the production ``reviews.db``: one approval recorded under 2.0.0
    and three corrections under 2.0.0/2.1.0, every row carrying provenance.
    A version change alone must flag all four for re-review while the stored
    decision, its override, and the append-only log survive verbatim — no
    migration, nothing rewritten, nothing deleted.
    """
    monkeypatch.setattr(app_module, "fetch_monthly_history", lambda symbol: None)
    stored = {"SPG": "2.0.0", "UNP": "2.0.0", "ULVRL": "2.1.0", "ENBTO": "2.1.0"}
    for ticker in stored:
        _seed_cache(tmp_cache, ticker, make_coil_bars())
    session = client.post(
        "/api/review-sessions", json=_session_payload(list(stored))
    ).json()["session"]

    client.post(
        "/api/highs/reviews",
        json=_decision(
            "SPG",
            decision="approved",
            session_id=session["id"],
            algorithm_version=stored["SPG"],
        ),
    )
    for ticker in ("UNP", "ULVRL", "ENBTO"):
        client.post(
            "/api/highs/reviews",
            json=_decision(
                ticker,
                highs=_lid_highs(),
                session_id=session["id"],
                algorithm_version=stored[ticker],
            ),
        )

    items = {
        item["ticker"]: item
        for item in client.get(f"/api/review-sessions/{session['id']}")
        .json()["session"]["items"]
    }
    for ticker, version in stored.items():
        state = tmp_review_store.get_review_state(ticker)
        # Nothing rewrote the stored provenance to the current version.
        assert state["algorithm_version"] == version
        assert reviews_module.decision_is_stale(
            state,
            current_as_of=state["as_of"],
            current_algorithm_version=ALGORITHM_VERSION,
        )
        assert items[ticker]["review_stale"] is True
        assert items[ticker]["status"] == "pending"
        coil = client.get(f"/api/coil/{ticker}").json()["review"]
        assert coil["reviewed"] is True
        assert coil["stale"] is True
        # The decision itself is intact and still readable.
        history = tmp_review_store.list_reviews(ticker)
        assert len(history) == 1
        assert history[0]["record"]["algorithmVersion"] == version

    # Stale corrections stay effective: re-review is requested, not enforced.
    for ticker in ("UNP", "ULVRL", "ENBTO"):
        override = tmp_review_store.get_override(ticker)
        assert [point["date"] for point in override["points"]] == [
            "2011-09-01",
            "2015-01-01",
            "2018-05-01",
        ]
        assert client.get(f"/api/coil/{ticker}").json()["review"]["effective"] == "human"
    assert client.get("/api/coil/SPG").json()["review"]["effective"] == "algorithm"


def test_review_recorded_at_the_current_algorithm_version_is_not_stale(
    client, tmp_cache, tmp_review_store
):
    """Control for the bump test: same path, current version, no staleness.

    Without this, a ``decision_is_stale`` that always returned True would
    still pass the bump test.
    """
    _seed_cache(tmp_cache, "CUR", make_coil_bars())
    client.post(
        "/api/highs/reviews",
        json=_decision("CUR", highs=_lid_highs(), algorithm_version=ALGORITHM_VERSION),
    )

    state = tmp_review_store.get_review_state("CUR")
    assert state["algorithm_version"] == ALGORITHM_VERSION
    assert not reviews_module.decision_is_stale(
        state,
        current_as_of=state["as_of"],
        current_algorithm_version=ALGORITHM_VERSION,
    )
    assert client.get("/api/coil/CUR").json()["review"]["stale"] is False


def test_fresh_decision_marks_item_reviewed_and_not_stale(client, tmp_cache):
    _seed_cache(tmp_cache, "FRS", make_coil_bars())
    session = client.post(
        "/api/review-sessions", json=_session_payload(["FRS"])
    ).json()["session"]
    client.post(
        "/api/highs/reviews",
        json=_decision("FRS", decision="approved", session_id=session["id"]),
    )
    item = client.get(f"/api/review-sessions/{session['id']}").json()["session"]["items"][0]
    assert item["status"] == "reviewed"
    assert item["review_stale"] is False


# --------------------------------------------------------------------------- #
# Legacy compatibility + revocation
# --------------------------------------------------------------------------- #
def test_legacy_correction_translates_to_corrected_decision(client, tmp_review_store):
    legacy = {
        "schemaVersion": 1,
        "ticker": "LGC",
        "interval": "3M",
        "createdAt": "2026-07-01T00:00:00.000Z",
        "barCount": 120,
        "algoHighs": [],
        "manualHighs": [
            {"idx": 20, "date": "2011-09-01", "price": 100.0},
            {"idx": 100, "date": "2018-05-01", "price": 100.0},
        ],
        "diff": {"added": [], "removed": [], "kept": []},
    }
    resp = client.post("/api/highs/corrections", json=legacy)
    assert resp.status_code == 200

    state = tmp_review_store.get_review_state("LGC")
    assert state["decision"] == "corrected"
    # No provenance on legacy corrections: never flagged stale.
    assert state["as_of"] is None
    session = client.post(
        "/api/review-sessions", json=_session_payload(["LGC"])
    ).json()["session"]
    assert session["items"][0]["status"] == "reviewed"
    assert session["items"][0]["review_stale"] is False


def test_revocation_returns_ticker_to_pending_everywhere(client, tmp_cache, tmp_review_store):
    _seed_cache(tmp_cache, "RVK", make_coil_bars())
    session = client.post(
        "/api/review-sessions", json=_session_payload(["RVK"])
    ).json()["session"]
    client.post(
        "/api/highs/reviews",
        json=_decision("RVK", highs=_lid_highs(), session_id=session["id"]),
    )
    client.delete("/api/highs/corrections/RVK", params={"interval": "3M"})

    assert tmp_review_store.get_review_state("RVK") is None
    item = client.get(f"/api/review-sessions/{session['id']}").json()["session"]["items"][0]
    assert item["status"] == "pending"
    assert item["review_decision"] is None


def test_screen_rows_carry_review_metadata(client, tmp_cache):
    _seed_cache(tmp_cache, "ROW", make_coil_bars())
    client.post("/api/highs/reviews", json=_decision("ROW", highs=_lid_highs()))
    run = client.post("/api/screen", json={"tickers": ["ROW"]}).json()
    row = run["results"][0]
    assert row["reviewed"] is True
    assert row["review_status"] == "corrected"
    assert row["review_stale"] is False
    assert row["review_effective"] == "human"
    assert row["review_id"] is not None
    assert row["review_as_of"] == "2019-12-01"
    assert row["review_algorithm_version"] == ALGORITHM_VERSION


# --------------------------------------------------------------------------- #
# Store migration: pre-session SQLite databases gain the new columns
# --------------------------------------------------------------------------- #
def test_existing_database_is_migrated_in_place(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE high_reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ticker TEXT NOT NULL, interval TEXT NOT NULL, created_at TEXT NOT NULL, "
            "record TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE high_overrides (ticker TEXT NOT NULL, interval TEXT NOT NULL, "
            "review_id INTEGER NOT NULL, points TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "PRIMARY KEY (ticker, interval))"
        )
        conn.execute(
            "INSERT INTO high_overrides VALUES ('OLD', '3M', 1, "
            "'[{\"date\": \"2011-09-01\", \"price\": 100.0}]', '2026-01-01T00:00:00Z')"
        )
        conn.commit()

    store = ReviewStore(sqlite_path=db_path)
    override = store.get_override("OLD")
    assert override["points"] == [{"date": "2011-09-01", "price": 100.0}]
    assert override["as_of"] is None
    assert override["algorithm_version"] is None
