"""Safety tests for the protected, capture-only fresh-review workflow."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
import review_snapshots
import reviews as reviews_module
from coil_analysis import ALGORITHM_VERSION
from reviews import ReviewStore
from test_coil_analysis import make_coil_bars

TOKEN = "amrut-review-capability-token-0001"
TOKEN_2 = "amrut-review-capability-token-0002"


@pytest.fixture
def fresh_client(tmp_path, monkeypatch) -> tuple[TestClient, ReviewStore, Path]:
    store = ReviewStore(sqlite_path=tmp_path / "reviews.db")
    monkeypatch.setattr(reviews_module, "_store", store)
    snapshot_root = tmp_path / "review_snapshots"
    monkeypatch.setattr(review_snapshots, "REVIEW_SNAPSHOT_ROOT", snapshot_root)
    return TestClient(app_module.app), store, snapshot_root


def _make_corpus(
    root: Path,
    source: str,
    tickers: list[str],
    *,
    quarantined: set[str] | None = None,
) -> None:
    blocked = quarantined or set()
    folder = root / Path(source).stem
    folder.mkdir(parents=True, exist_ok=True)
    manifest_items: list[dict] = []
    for position, ticker in enumerate(tickers, start=1):
        bars = make_coil_bars()
        if ticker in blocked:
            bars[-1] = {**bars[-1], "low": 0.0}
        quality = {
            "status": "quarantined" if ticker in blocked else "clear_by_defined_checks",
            "reviewable": ticker not in blocked,
            "hard_failure_count": 1 if ticker in blocked else 0,
            "heuristic_flag_count": 0,
            "date_order_failures": [],
            "numeric_price_failures": [],
            "nonpositive_price_fields": (
                [{"date": bars[-1]["date"], "field": "low"}]
                if ticker in blocked
                else []
            ),
            "ohlc_invariant_failures": [],
            "extreme_wicks": [],
            "extreme_discontinuities": [],
        }
        snapshot = {
            "schema_version": 1,
            "kind": "coilingview.saved-run-review-snapshot",
            "source": source,
            "ticker": ticker,
            "run": {
                "algorithm_version": ALGORITHM_VERSION,
                "code_sha": "test-sha",
                "source_csv_sha256": "a" * 64,
                "universe_position": position,
            },
            "screen_snapshot": {
                "ticker": ticker,
                "lifecycle": "pre_breakout",
                "grade": "A",
                "data_date": bars[-1]["date"],
            },
            "monthly_bars": bars,
            "data_quality": quality,
            "corpus_labels": {
                "cohort_role": (
                    "prospective_international" if position > 1 else "control_reference"
                ),
                "prior_review": {"flag": position == 1},
                "tuning_anchor": {"flag": False, "evidence": []},
            },
            "provenance": {"source_cache_file": f"cache/{ticker}.json"},
            "source_cache_metadata": {},
        }
        (folder / f"{ticker}.json").write_text(
            json.dumps(snapshot), encoding="utf-8"
        )
        manifest_items.append({"ticker": ticker, "universe_position": position})
    manifest = {
        "schema_version": 1,
        "kind": "coilingview.saved-run-review-corpus-manifest",
        "source_run": {
            "filename": source,
            "algorithm_version": ALGORITHM_VERSION,
            "row_count": len(tickers),
            "sha256": "a" * 64,
        },
        "corpus_id": f"test-{Path(source).stem}",
        "items": manifest_items,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _create(
    client: TestClient,
    source: str,
    tickers: list[str],
    *,
    token: str = TOKEN,
    admin_key: str | None = None,
) -> dict:
    response = client.post(
        "/api/review-sessions",
        json={
            "source": source,
            "reviewerName": "Amrut",
            "accessToken": token,
            "requireFreshReview": True,
            "items": [{"ticker": ticker} for ticker in tickers],
            "snapshot": {"purpose": "fresh algo review"},
        },
        headers=(
            {"X-Review-Admin-Key": admin_key} if admin_key is not None else None
        ),
    )
    assert response.status_code == 200, response.text
    return response.json()


def _headers(token: str = TOKEN) -> dict[str, str]:
    return {"X-Review-Token": token}


BASE_RATIONALE = (
    "The blind chart review satisfies the complete base-pattern checklist."
)
EXCEPTION_BASE_RATIONALE = (
    "The blind review found top geometry outside the standard base-pattern rules."
)


def _base_learning_capture(
    *,
    base_path: str = "base_pattern",
    failed_base_rules: list[str] | None = None,
) -> dict:
    return {
        "reviewerName": "Amrut",
        "sequencePolicyVersion": 1,
        "baseAssessmentLocked": True,
        "basePath": base_path,
        "failedBaseRules": failed_base_rules or [],
        "baseRationale": BASE_RATIONALE,
        "exceptionVerdict": (
            "not_evaluated" if base_path == "base_pattern" else "uncertain"
        ),
        "exceptionRationale": (
            ""
            if base_path == "base_pattern"
            else "The exception needs a full model-evidence comparison to resolve."
        ),
        "ruleProposal": None,
        "commentary": "Blind base assessment recorded before model reveal.",
    }


def _lock_base(
    client: TestClient,
    session: dict,
    ticker: str,
    *,
    learning_capture: dict | None = None,
    token: str = TOKEN,
) -> tuple[dict, dict]:
    capture = learning_capture or _base_learning_capture()
    item = next(entry for entry in session["items"] if entry["ticker"] == ticker)
    draft = client.put(
        f"/api/review-sessions/{session['id']}/items/{ticker}/draft",
        headers=_headers(token),
        json={
            "expectedRevision": item["draft_revision"],
            "payload": {
                "schemaVersion": 5,
                "learningCapture": capture,
            },
        },
    )
    assert draft.status_code == 200, draft.text
    revision = draft.json()["item"]["draft_revision"]
    lock = client.post(
        f"/api/review-sessions/{session['id']}/items/{ticker}/base-lock",
        headers=_headers(token),
        json={
            "expectedDraftRevision": revision,
            "baseClassification": {
                "locked": True,
                "basePath": capture["basePath"],
                "failedBaseRules": capture["failedBaseRules"],
                "rationale": capture["baseRationale"],
            },
        },
    )
    assert lock.status_code == 200, lock.text
    context = client.get(
        f"/api/review-sessions/{session['id']}/items/{ticker}/context",
        headers=_headers(token),
    )
    assert context.status_code == 200, context.text
    return lock.json()["item"], context.json()["context"]


def _base_finalize(
    client: TestClient,
    session: dict,
    ticker: str,
    *,
    key: str = "finalize-key-0001",
    learning_capture: dict | None = None,
) -> dict:
    capture = learning_capture or _base_learning_capture()
    item, context = _lock_base(
        client,
        session,
        ticker,
        learning_capture=capture,
    )
    return _finalize_payload(
        session,
        ticker,
        item=item,
        context=context,
        learning_capture=capture,
        key=key,
    )


def _finalize_payload(
    session: dict,
    ticker: str,
    *,
    item: dict,
    context: dict,
    learning_capture: dict,
    key: str,
) -> dict:
    return {
        "schemaVersion": 5,
        "labelPolicyVersion": 2,
        "sessionId": session["id"],
        "ticker": ticker,
        "interval": "3M",
        "asOf": context["monthly_bars"][-1]["date"],
        "algorithmVersion": ALGORITHM_VERSION,
        "decision": "approved",
        "coilLabel": "coil",
        "humanGrade": "A",
        "confidence": "high",
        "note": "Meets the sequential base-screen criteria.",
        "learningCapture": learning_capture,
        "reviewedHighs": [],
        "createdAt": "2026-07-28T00:00:00Z",
        "idempotencyKey": key,
        "expectedDraftRevision": item["draft_revision"],
        "sampleId": item["sample_id"],
    }


def _nested_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for child in value.values()
            for key in _nested_keys(child)
        }
    if isinstance(value, list):
        return {
            key
            for child in value
            for key in _nested_keys(child)
        }
    return set()


def test_token_scope_and_new_capability_create_distinct_session(fresh_client):
    client, _, root = fresh_client
    source = "protected.csv"
    _make_corpus(root, source, ["AAA"])
    first = _create(client, source, ["AAA"])
    resumed = _create(client, source, ["AAA"])
    separate = _create(client, source, ["AAA"], token=TOKEN_2)

    assert first["created"] is True
    assert resumed["created"] is False
    assert resumed["session"]["id"] == first["session"]["id"]
    assert separate["created"] is True
    assert separate["session"]["id"] != first["session"]["id"]
    path = f"/api/review-sessions/{first['session']['id']}"
    assert client.get(path).status_code == 403
    assert client.get(path, headers=_headers("wrong-token-that-is-long-enough")).status_code == 403
    assert client.get(path, headers=_headers()).status_code == 200
    assert "capability_token_hash" not in json.dumps(first)
    assert TOKEN not in json.dumps(first)


def test_blind_gate_redacts_then_atomically_reveals_model_context(fresh_client):
    client, _, root = fresh_client
    source = "blind-gate.csv"
    _make_corpus(root, source, ["AAA"])
    session = _create(client, source, ["AAA"])["session"]
    session_item = session["items"][0]
    context_path = f"/api/review-sessions/{session['id']}/items/AAA/context"
    blind = client.get(context_path, headers=_headers()).json()["context"]
    forbidden = {
        "analysis",
        "analysis_status",
        "quarterly_bars",
        "model_snapshot",
        "screen_snapshot",
        "corpus_labels",
        "source_features",
        "lifecycle",
        "grade",
        "coil_score",
        "lid_grade",
        "lid_slope_pct_per_year",
        "touch_count",
        "touches",
        "review_algorithm_version",
        "review_coil_label",
        "review_human_grade",
        "effective",
    }
    assert forbidden.isdisjoint(_nested_keys(session))
    assert forbidden.isdisjoint(_nested_keys(blind))
    assert blind["model_revealed"] is False
    assert "monthly_bars" in blind
    assert (
        client.get(
            f"/api/review-sessions/{session['id']}/export",
            headers=_headers(),
        ).status_code
        == 409
    )

    unlocked_payload = _finalize_payload(
        session,
        "AAA",
        item=session_item,
        context=blind,
        learning_capture=_base_learning_capture(),
        key="must-lock-first",
    )
    finalize_path = f"/api/review-sessions/{session['id']}/items/AAA/finalize"
    assert (
        client.post(
            finalize_path, headers=_headers(), json=unlocked_payload
        ).status_code
        == 409
    )

    lock_path = f"/api/review-sessions/{session['id']}/items/AAA/base-lock"
    classification = {
        "locked": True,
        "basePath": "base_pattern",
        "failedBaseRules": [],
        "rationale": BASE_RATIONALE,
    }
    assert (
        client.post(
            lock_path,
            headers=_headers(),
            json={
                "expectedDraftRevision": 1,
                "baseClassification": classification,
            },
        ).status_code
        == 409
    )
    draft = client.put(
        f"/api/review-sessions/{session['id']}/items/AAA/draft",
        headers=_headers(),
        json={
            "expectedRevision": 0,
            "payload": {
                "schemaVersion": 5,
                "learningCapture": _base_learning_capture(),
            },
        },
    )
    assert draft.status_code == 200
    revision = draft.json()["item"]["draft_revision"]
    mismatch = {
        **classification,
        "rationale": (
            "A different substantive blind rationale must never unlock evidence."
        ),
    }
    assert (
        client.post(
            lock_path,
            headers=_headers(),
            json={
                "expectedDraftRevision": revision,
                "baseClassification": mismatch,
            },
        ).status_code
        == 400
    )
    assert (
        client.get(context_path, headers=_headers()).json()["context"][
            "model_revealed"
        ]
        is False
    )

    locked = client.post(
        lock_path,
        headers=_headers(),
        json={
            "expectedDraftRevision": revision,
            "baseClassification": classification,
        },
    )
    assert locked.status_code == 200
    assert locked.json()["item"]["base_classification_locked"] is True
    repeated = client.post(
        lock_path,
        headers=_headers(),
        json={
            "expectedDraftRevision": revision,
            "baseClassification": classification,
        },
    )
    assert repeated.status_code == 200
    different = client.post(
        lock_path,
        headers=_headers(),
        json={
            "expectedDraftRevision": revision,
            "baseClassification": {
                "locked": True,
                "basePath": "exception_territory",
                "failedBaseRules": ["top_geometry"],
                "rationale": EXCEPTION_BASE_RATIONALE,
            },
        },
    )
    assert different.status_code == 409

    revealed = client.get(context_path, headers=_headers()).json()["context"]
    assert revealed["model_revealed"] is True
    assert {"analysis", "quarterly_bars", "model_snapshot"} <= revealed.keys()
    refreshed = client.get(
        f"/api/review-sessions/{session['id']}", headers=_headers()
    ).json()["session"]["items"][0]
    assert refreshed["snapshot"]["screen_snapshot"]["grade"] == "A"
    changed_capture = _base_learning_capture(
        base_path="exception_territory",
        failed_base_rules=["top_geometry"],
    )
    changed_capture["baseRationale"] = EXCEPTION_BASE_RATIONALE
    assert (
        client.put(
            f"/api/review-sessions/{session['id']}/items/AAA/draft",
            headers=_headers(),
            json={
                "expectedRevision": revision,
                "payload": {
                    "schemaVersion": 5,
                    "learningCapture": changed_capture,
                },
            },
        ).status_code
        == 409
    )


def test_fresh_create_requires_complete_exact_manifest_order(fresh_client):
    client, _, root = fresh_client
    source = "ordered.csv"
    tickers = ["AAA", "BBB", "CCC"]
    _make_corpus(root, source, tickers)

    def attempt(items: list[str]):
        return client.post(
            "/api/review-sessions",
            json={
                "source": source,
                "reviewerName": "Amrut",
                "accessToken": TOKEN,
                "requireFreshReview": True,
                "items": [{"ticker": ticker} for ticker in items],
            },
        )

    assert attempt(["AAA", "BBB"]).status_code == 400
    assert attempt(["BBB", "AAA", "CCC"]).status_code == 400
    assert attempt(["AAA", "BBB", "BBB"]).status_code == 400
    assert attempt(tickers).status_code == 200


def test_admin_can_rotate_and_revoke_reviewer_capability(
    fresh_client, monkeypatch
):
    client, _, root = fresh_client
    source = "token-admin.csv"
    _make_corpus(root, source, ["AAA"])
    admin_key = "server-side-review-admin-secret"
    monkeypatch.setenv("REVIEW_SESSION_CREATE_KEY", admin_key)
    session = _create(
        client,
        source,
        ["AAA"],
        admin_key=admin_key,
    )["session"]
    path = f"/api/review-sessions/{session['id']}"
    rotate_path = f"{path}/access-token/rotate"
    assert client.post(rotate_path).status_code == 403
    assert (
        client.post(
            rotate_path, headers={"X-Review-Admin-Key": "wrong"}
        ).status_code
        == 403
    )
    rotated = client.post(
        rotate_path,
        headers={"X-Review-Admin-Key": admin_key},
    )
    assert rotated.status_code == 200
    assert rotated.headers["cache-control"] == "no-store"
    rotated_token = rotated.json()["accessToken"]
    assert len(rotated_token) >= 32
    assert client.get(path, headers=_headers()).status_code == 403
    assert client.get(path, headers=_headers(rotated_token)).status_code == 200
    assert rotated_token not in client.get(
        path, headers=_headers(rotated_token)
    ).text

    revoked = client.post(
        f"{path}/access-token/revoke",
        headers={"X-Review-Admin-Key": admin_key},
    )
    assert revoked.status_code == 200
    assert revoked.headers["cache-control"] == "no-store"
    assert client.get(path, headers=_headers(rotated_token)).status_code == 403


def test_concurrent_create_or_resume_returns_one_session(fresh_client):
    _, store, _ = fresh_client
    items = [{"ticker": "AAA", "snapshot": {"data_date": "2019-12-01"}}]

    def create():
        return store.create_session(
            "screen:concurrent",
            items,
            snapshot={"count": 1},
            algorithm_version=ALGORITHM_VERSION,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: create(), range(2)))
    assert {result[0]["id"] for result in results} == {results[0][0]["id"]}
    assert sum(1 for _, created in results if created) == 1


def test_all_items_are_fresh_and_ignore_global_review_state(fresh_client):
    client, store, root = fresh_client
    source = "all-79.csv"
    tickers = [f"T{index:03d}" for index in range(79)]
    _make_corpus(root, source, tickers)
    store.record_decision(
        {
            "schemaVersion": 3,
            "ticker": tickers[0],
            "interval": "3M",
            "decision": "corrected",
            "reviewedHighs": [
                {"date": "2011-09-01", "price": 100.0},
                {"date": "2018-05-01", "price": 100.0},
            ],
        }
    )
    created = _create(client, source, tickers)
    session = created["session"]
    assert session["counts"] == {
        "pending": 79,
        "reviewed": 0,
        "skipped": 0,
        "total": 79,
    }
    assert all(item["status"] == "pending" for item in session["items"])
    first = session["items"][0]
    assert "review_id" not in first
    assert "effective" not in first
    assert "corpus_labels" not in first["snapshot"]
    assert "screen_snapshot" not in first["snapshot"]
    assert first["snapshot"]["cohort_position"] == 1
    assert first["base_classification_locked"] is False


def test_durable_draft_revision_conflict_and_reload(fresh_client):
    client, _, root = fresh_client
    source = "draft.csv"
    _make_corpus(root, source, ["AAA"])
    session = _create(client, source, ["AAA"])["session"]
    path = f"/api/review-sessions/{session['id']}/items/AAA/draft"
    capture = _base_learning_capture()
    saved = client.put(
        path,
        headers=_headers(),
        json={
            "expectedRevision": 0,
            "payload": {
                "schemaVersion": 5,
                "learningCapture": capture,
                "modelGuess": {"grade": "must-not-echo"},
            },
        },
    )
    assert saved.status_code == 200
    assert saved.json()["item"]["draft_revision"] == 1
    assert saved.json()["item"]["draft"]["learningCapture"]["basePath"] == (
        "base_pattern"
    )
    assert "modelGuess" not in saved.text
    stale = client.put(
        path,
        headers=_headers(),
        json={"expectedRevision": 0, "payload": {"step": 2}},
    )
    assert stale.status_code == 409
    reloaded = client.get(
        f"/api/review-sessions/{session['id']}", headers=_headers()
    ).json()["session"]["items"][0]
    assert reloaded["draft_revision"] == 1
    assert reloaded["draft"]["learningCapture"]["baseRationale"] == BASE_RATIONALE
    assert "modelGuess" not in json.dumps(reloaded)


def test_missing_or_changed_frozen_snapshot_fails_closed(fresh_client):
    client, _, root = fresh_client
    source = "integrity.csv"
    _make_corpus(root, source, ["AAA"])
    snapshot_path = root / "integrity" / "AAA.json"
    original = snapshot_path.read_text(encoding="utf-8")
    snapshot_path.unlink()
    missing = client.post(
        "/api/review-sessions",
        json={
            "source": source,
            "reviewerName": "Amrut",
            "accessToken": TOKEN,
            "requireFreshReview": True,
            "items": [{"ticker": "AAA"}],
        },
    )
    assert missing.status_code == 400

    snapshot_path.write_text(original, encoding="utf-8")
    session = _create(client, source, ["AAA"])["session"]
    changed = json.loads(original)
    changed["corpus_labels"]["cohort_role"] = "tampered"
    snapshot_path.write_text(json.dumps(changed), encoding="utf-8")
    context = client.get(
        f"/api/review-sessions/{session['id']}/items/AAA/context",
        headers=_headers(),
    )
    assert context.status_code == 409


def test_capture_is_idempotent_and_never_changes_authoritative_truth(fresh_client):
    client, store, root = fresh_client
    source = "safety.csv"
    _make_corpus(root, source, ["AAA"])
    authoritative = store.record_decision(
        {
            "schemaVersion": 3,
            "ticker": "AAA",
            "interval": "3M",
            "decision": "corrected",
            "reviewedHighs": [
                {"date": "2011-09-01", "price": 100.0},
                {"date": "2018-05-01", "price": 100.0},
            ],
        }
    )
    before_state = store.get_review_state("AAA")
    before_override = store.get_override("AAA")
    session = _create(client, source, ["AAA"])["session"]
    path = f"/api/review-sessions/{session['id']}/items/AAA/finalize"
    payload = _base_finalize(client, session, "AAA")
    first = client.post(path, headers=_headers(), json=payload)
    retry = client.post(path, headers=_headers(), json=payload)
    assert first.status_code == retry.status_code == 200
    assert first.json()["review"] == retry.json()["review"]
    assert first.json()["review"]["capture_only"] is True
    assert store.get_review_state("AAA") == before_state
    assert store.get_override("AAA") == before_override
    assert store.get_override("AAA")["review_id"] == authoritative["review"]["id"]
    capture_record = store.list_reviews("AAA")[-1]["record"]
    assert capture_record["captureOnly"] is True
    public_history = client.get("/api/highs/corrections/AAA")
    assert public_history.status_code == 200
    assert all(
        not entry["record"].get("captureOnly")
        for entry in public_history.json()["reviews"]
    )
    assert "learningCapture" not in public_history.text

    changed = {**payload, "note": "Different request under the same idempotency key."}
    assert client.post(path, headers=_headers(), json=changed).status_code == 409


def test_evidence_membership_price_and_sample_identity_are_server_validated(
    fresh_client,
):
    client, _, root = fresh_client
    source = "evidence.csv"
    _make_corpus(root, source, ["AAA"])
    session = _create(client, source, ["AAA"])["session"]
    capture = _base_learning_capture(
        base_path="exception_territory",
        failed_base_rules=["top_geometry"],
    )
    capture["baseRationale"] = EXCEPTION_BASE_RATIONALE
    item, context = _lock_base(
        client,
        session,
        "AAA",
        learning_capture=capture,
    )
    quarterly = context["quarterly_bars"]
    payload = _finalize_payload(
        session,
        "AAA",
        item=item,
        context=context,
        learning_capture=capture,
        key="evidence-key-0001",
    )
    payload.update(
        {
            "learningCapture": {
                "reviewerName": "Amrut",
                "sequencePolicyVersion": 1,
                "baseAssessmentLocked": True,
                "basePath": "exception_territory",
                "failedBaseRules": ["top_geometry"],
                "baseRationale": EXCEPTION_BASE_RATIONALE,
                "exceptionVerdict": "applies",
                "exceptionRationale": "",
                "ruleProposal": {
                    "name": "Alternate geometry exception",
                    "patternKind": "other",
                    "applicability": "Use only after the base geometry rule has clearly failed.",
                    "exclusions": "Exclude short histories and charts with unresolved data quality.",
                    "detectionLogic": "Detect two ordered structural candles around the alternate geometry.",
                    "confirmation": "Require a later close to confirm the proposed structural sequence.",
                    "proposedAction": "hold_for_human_review",
                    "impactedStages": ["screening", "top_detection"],
                    "validationPlan": "Backtest on unseen candidates and record false positives separately.",
                    "evidence": [
                        {
                            "id": "e1",
                            "sequence": 1,
                            "date": quarterly[0]["date"],
                            "price": quarterly[0]["high"] + 1,
                            "priceField": "high",
                            "role": "major_top",
                            "label": "First top",
                            "note": "",
                        },
                        {
                            "id": "e2",
                            "sequence": 2,
                            "date": quarterly[1]["date"],
                            "price": quarterly[1]["high"],
                            "priceField": "high",
                            "role": "slope_anchor",
                            "label": "Second anchor",
                            "note": "",
                        },
                    ],
                },
                "commentary": "Exception evidence captured for future validation.",
            }
        }
    )
    path = f"/api/review-sessions/{session['id']}/items/AAA/finalize"
    assert client.post(path, headers=_headers(), json=payload).status_code == 400
    wrong_sample = {**payload, "sampleId": "0" * 64}
    assert client.post(path, headers=_headers(), json=wrong_sample).status_code == 409
    payload["learningCapture"]["ruleProposal"]["evidence"][0]["price"] = quarterly[0][
        "high"
    ]
    assert client.post(path, headers=_headers(), json=payload).status_code == 200


def test_corrected_highs_must_snap_to_frozen_quarterly_candle_highs(fresh_client):
    client, _, root = fresh_client
    source = "corrected-highs.csv"
    _make_corpus(root, source, ["AAA"])
    session = _create(client, source, ["AAA"])["session"]
    item, context = _lock_base(client, session, "AAA")
    first, second = context["quarterly_bars"][:2]
    payload = _finalize_payload(
        session,
        "AAA",
        item=item,
        context=context,
        learning_capture=_base_learning_capture(),
        key="corrected-high-key",
    )
    payload.update(
        {
            "decision": "corrected",
            "note": "Corrected anchors after reviewing the frozen quarterly candles.",
            "reviewedHighs": [
                {
                    "date": first["date"],
                    "price": first["high"],
                    "lidMember": True,
                },
                {
                    "date": "2099-12-01",
                    "price": second["high"],
                    "lidMember": True,
                },
            ],
        }
    )
    path = f"/api/review-sessions/{session['id']}/items/AAA/finalize"
    assert client.post(path, headers=_headers(), json=payload).status_code == 400
    payload["reviewedHighs"][1]["date"] = second["date"]
    payload["reviewedHighs"][0]["price"] = first["high"] + 1
    assert client.post(path, headers=_headers(), json=payload).status_code == 400
    payload["reviewedHighs"][0]["price"] = first["high"]
    assert client.post(path, headers=_headers(), json=payload).status_code == 200


def test_quarantined_context_can_only_be_closed_with_explained_skip(fresh_client):
    client, _, root = fresh_client
    source = "quarantine.csv"
    _make_corpus(root, source, ["BAD"], quarantined={"BAD"})
    session = _create(client, source, ["BAD"])["session"]
    context_path = f"/api/review-sessions/{session['id']}/items/BAD/context"
    context = client.get(context_path, headers=_headers()).json()["context"]
    assert context["reviewable"] is False
    assert "analysis" not in context
    assert context["model_revealed"] is False
    assert context["monthly_bars"][-1]["low"] == 0.0
    finalize_path = f"/api/review-sessions/{session['id']}/items/BAD/finalize"
    payload = _finalize_payload(
        session,
        "BAD",
        item=session["items"][0],
        context=context,
        learning_capture=_base_learning_capture(),
        key="quarantined-finalize",
    )
    assert (
        client.post(
            finalize_path,
            headers=_headers(),
            json=payload,
        ).status_code
        == 400
    )
    item_path = f"/api/review-sessions/{session['id']}/items/BAD"
    assert (
        client.patch(
            item_path, headers=_headers(), json={"status": "skipped"}
        ).status_code
        == 400
    )
    skipped = client.patch(
        item_path,
        headers=_headers(),
        json={
            "status": "skipped",
            "reason": "Quarantined: frozen July low is zero; no repair was applied.",
        },
    )
    assert skipped.status_code == 200
    assert skipped.json()["item"]["skip_reason"].startswith("Quarantined")


def test_impossible_calendar_date_is_quarantined_before_analysis(fresh_client):
    client, _, root = fresh_client
    source = "bad-date.csv"
    _make_corpus(root, source, ["BAD"])
    path = root / "bad-date" / "BAD.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["monthly_bars"][-1]["date"] = "2026-13-40"
    snapshot["screen_snapshot"]["data_date"] = "2026-13-40"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    session = _create(client, source, ["BAD"])["session"]
    context = client.get(
        f"/api/review-sessions/{session['id']}/items/BAD/context",
        headers=_headers(),
    ).json()["context"]
    assert context["reviewable"] is False
    assert "analysis" not in context
    assert context["model_revealed"] is False
    assert any(
        issue["code"] == "invalid_date"
        for issue in context["data_quality_validation"]["server_checks"][
            "blocking_issues"
        ]
    )


def test_session_finalization_freezes_stable_linked_only_export(fresh_client):
    client, store, root = fresh_client
    source = "export.csv"
    _make_corpus(root, source, ["AAA"])
    store.record_decision(
        {
            "schemaVersion": 3,
            "ticker": "AAA",
            "interval": "3M",
            "decision": "approved",
            "reviewedHighs": [],
        }
    )
    session = _create(client, source, ["AAA"])["session"]
    assert (
        client.post(
            f"/api/review-sessions/{session['id']}/finalize",
            headers=_headers(),
        ).status_code
        == 409
    )
    item_path = f"/api/review-sessions/{session['id']}/items/AAA/finalize"
    assert (
        client.post(
                item_path,
                headers=_headers(),
                json=_base_finalize(client, session, "AAA"),
        ).status_code
        == 200
    )
    finalize_path = f"/api/review-sessions/{session['id']}/finalize"
    first = client.post(finalize_path, headers=_headers())
    second = client.post(finalize_path, headers=_headers())
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    frozen = first.json()["export"]
    canonical = json.dumps(
        frozen,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == first.json()[
        "sha256"
    ]
    assert "export_sha256" not in frozen
    assert frozen["session"]["counts"] == {
        "pending": 0,
        "reviewed": 1,
        "skipped": 0,
        "total": 1,
    }
    assert frozen["session"]["next_pending_ticker"] is None
    assert frozen["session"]["items"][0]["base_classification"] == {
        "basePath": "base_pattern",
        "failedBaseRules": [],
        "locked": True,
        "rationale": BASE_RATIONALE,
    }
    assert frozen["session"]["items"][0]["base_classification_locked_at"]
    assert len(frozen["records"]) == 1
    assert frozen["records"][0]["record"]["captureOnly"] is True
    export_path = f"/api/review-sessions/{session['id']}/export"
    export_response = client.get(export_path, headers=_headers())
    assert export_response.json() == frozen
    assert export_response.headers["cache-control"] == "no-store"
    assert export_response.headers["x-export-canonicalization"] == (
        "coilingview-canonical-json-v1"
    )
    assert hashlib.sha256(export_response.content).hexdigest() == (
        export_response.headers["x-export-sha256"]
    )
    assert export_response.headers["x-export-sha256"] == first.json()["sha256"]
    assert (
        client.put(
            f"/api/review-sessions/{session['id']}/items/AAA/draft",
            headers=_headers(),
            json={"expectedRevision": 0, "payload": {"late": True}},
        ).status_code
        == 409
    )


def test_finalized_export_keeps_unlocked_skips_redacted(fresh_client):
    client, _, root = fresh_client
    source = "skipped-export.csv"
    _make_corpus(root, source, ["AAA"])
    session = _create(client, source, ["AAA"])["session"]
    item_path = f"/api/review-sessions/{session['id']}/items/AAA"
    skipped = client.patch(
        item_path,
        headers=_headers(),
        json={
            "status": "skipped",
            "reason": "Reviewer could not reach a reliable blind classification.",
        },
    )
    assert skipped.status_code == 200
    finalized = client.post(
        f"/api/review-sessions/{session['id']}/finalize",
        headers=_headers(),
    )
    assert finalized.status_code == 200
    item = finalized.json()["export"]["session"]["items"][0]
    assert item["base_classification"] is None
    assert "screen_snapshot" not in item["snapshot"]
    assert "corpus_labels" not in item["snapshot"]
    assert {"data_quality", "data_quality_validation", "frozen"} <= (
        item["snapshot"].keys()
    )


def test_capture_only_deployment_blocks_legacy_writes_and_admin_gates_creation(
    fresh_client, monkeypatch
):
    client, store, root = fresh_client
    source = "share.csv"
    _make_corpus(root, source, ["AAA"])
    monkeypatch.setenv("CAPTURE_ONLY_MODE", "true")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("CAPTURE_ONLY_ALLOW_EPHEMERAL", "true")
    monkeypatch.delenv("REVIEW_SESSION_CREATE_KEY", raising=False)
    assert (
        client.post(
            "/api/review-sessions",
            json={
                "source": source,
                "reviewerName": "Amrut",
                "accessToken": TOKEN,
                "requireFreshReview": True,
                "items": [{"ticker": "AAA"}],
            },
        ).status_code
        == 503
    )
    monkeypatch.setenv("REVIEW_SESSION_CREATE_KEY", "internal-create-secret")

    assert (
        client.post(
            "/api/highs/reviews",
            json={
                "schemaVersion": 3,
                "ticker": "AAA",
                "interval": "3M",
                "decision": "approved",
                "reviewedHighs": [],
            },
        ).status_code
        == 403
    )
    assert client.delete("/api/highs/corrections/AAA").status_code == 403
    assert client.post("/api/screen", json={"tickers": ["AAA"]}).status_code == 403
    for path in (
        "/api/default-tickers",
        "/api/saved-runs",
        "/api/history/AAA",
        "/api/coil/AAA",
        "/api/highs/corrections/AAA",
        "/api/vision/runs",
        "/docs",
        "/redoc",
        "/openapi.json",
    ):
        assert client.get(path).status_code == 403, path
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["capture_only"] is True
    assert client.get("/").status_code == 200
    assert (
        client.post(
            "/api/vision/review",
            json={"run_id": "x", "ticker": "AAA"},
        ).status_code
        == 403
    )
    assert store.get_review_state("AAA") is None

    base_request = {
        "source": source,
        "reviewerName": "Amrut",
        "accessToken": TOKEN,
        "requireFreshReview": True,
        "items": [{"ticker": "AAA"}],
    }
    assert client.post("/api/review-sessions", json=base_request).status_code == 403
    assert (
        client.post(
            "/api/review-sessions",
            json=base_request,
            headers={"X-Review-Admin-Key": "wrong"},
        ).status_code
        == 403
    )
    legacy_request = {
        "source": "screen:current",
        "items": [{"ticker": "AAA"}],
    }
    assert (
        client.post(
            "/api/review-sessions",
            json=legacy_request,
            headers={"X-Review-Admin-Key": "internal-create-secret"},
        ).status_code
        == 403
    )

    created = _create(
        client,
        source,
        ["AAA"],
        admin_key="internal-create-secret",
    )
    session_id = created["session"]["id"]
    session_path = f"/api/review-sessions/{session_id}"
    assert client.get(session_path).status_code == 403
    assert (
        client.get(
            session_path,
            headers=_headers("wrong-token-that-is-long-enough"),
        ).status_code
        == 403
    )
    assert (
        client.get(
            session_path, headers=_headers()
        ).status_code
        == 200
    )
    assert (
        client.put(
            f"/api/review-sessions/{session_id}/items/AAA/draft",
            headers=_headers(),
            json={"expectedRevision": 0, "payload": {"allowed": True}},
        ).status_code
        == 200
    )


def test_capture_only_readiness_fails_closed_without_durable_storage(
    fresh_client, monkeypatch
):
    client, _, _ = fresh_client
    for name in (
        "DATABASE_URL",
        "REVIEW_DB_PATH",
        "RAILWAY_VOLUME_MOUNT_PATH",
        "RAILWAY_ENVIRONMENT_ID",
        "RAILWAY_ENVIRONMENT_NAME",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_STATIC_URL",
        "APP_ENV",
        "ENVIRONMENT",
        "CAPTURE_ONLY_ALLOW_EPHEMERAL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CAPTURE_ONLY_MODE", "true")

    health = client.get("/api/health")
    assert health.status_code == 503
    body = health.json()
    assert body["status"] == "not_ready"
    assert body["capture_only"] is True
    assert body["persistence"]["durable"] is False
    assert client.get("/api/default-tickers").status_code == 503
    with pytest.raises(RuntimeError, match="persistence is not ready"):
        with TestClient(app_module.app):
            pass


def test_ephemeral_escape_is_explicit_test_only_and_disabled_on_railway(
    fresh_client, monkeypatch
):
    client, _, _ = fresh_client
    for name in (
        "DATABASE_URL",
        "REVIEW_DB_PATH",
        "RAILWAY_VOLUME_MOUNT_PATH",
        "RAILWAY_ENVIRONMENT_ID",
        "RAILWAY_ENVIRONMENT_NAME",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_STATIC_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CAPTURE_ONLY_MODE", "true")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("CAPTURE_ONLY_ALLOW_EPHEMERAL", "true")
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["persistence"]["development_escape"] is True

    monkeypatch.setenv("RAILWAY_SERVICE_ID", "railway-service")
    blocked = client.get("/api/health")
    assert blocked.status_code == 503
    assert blocked.json()["persistence"]["development_escape"] is False


def test_explicit_sqlite_path_satisfies_capture_readiness(
    fresh_client, monkeypatch
):
    client, store, _ = fresh_client
    assert store.sqlite_path is not None
    monkeypatch.setenv("CAPTURE_ONLY_MODE", "true")
    monkeypatch.setenv("REVIEW_DB_PATH", str(store.sqlite_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("RAILWAY_VOLUME_MOUNT_PATH", raising=False)
    monkeypatch.delenv("CAPTURE_ONLY_ALLOW_EPHEMERAL", raising=False)
    health = client.get("/api/health")
    assert health.status_code == 200
    persistence = health.json()["persistence"]
    assert persistence["ready"] is True
    assert persistence["durable"] is True
    assert persistence["backend"] == "sqlite_explicit"


def test_relative_sqlite_path_is_not_accepted_as_durable_capture_storage(
    fresh_client, monkeypatch
):
    client, _, _ = fresh_client
    monkeypatch.setenv("CAPTURE_ONLY_MODE", "true")
    monkeypatch.setenv("REVIEW_DB_PATH", "relative/reviews.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("RAILWAY_VOLUME_MOUNT_PATH", raising=False)
    monkeypatch.delenv("CAPTURE_ONLY_ALLOW_EPHEMERAL", raising=False)
    health = client.get("/api/health")
    assert health.status_code == 503
    assert health.json()["persistence"]["reason"] == (
        "REVIEW_DB_PATH must be absolute"
    )


def test_admin_sqlite_backup_is_consistent_and_never_cacheable(
    fresh_client, monkeypatch, tmp_path
):
    client, _, root = fresh_client
    source = "backup.csv"
    _make_corpus(root, source, ["AAA"])
    admin_key = "server-side-backup-admin-secret"
    monkeypatch.setenv("REVIEW_SESSION_CREATE_KEY", admin_key)
    _create(client, source, ["AAA"], admin_key=admin_key)
    path = "/api/admin/review-storage/backup"
    assert client.get(path).status_code == 403
    assert (
        client.get(path, headers={"X-Review-Admin-Key": "wrong"}).status_code
        == 403
    )
    response = client.get(
        path, headers={"X-Review-Admin-Key": admin_key}
    )
    assert response.status_code == 200
    assert response.content.startswith(b"SQLite format 3\x00")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "attachment;" in response.headers["content-disposition"]
    assert ".sqlite3" in response.headers["content-disposition"]

    downloaded = tmp_path / "downloaded-review-backup.sqlite3"
    downloaded.write_bytes(response.content)
    with sqlite3.connect(downloaded) as conn:
        session_count = conn.execute(
            "SELECT COUNT(*) FROM review_sessions"
        ).fetchone()[0]
    assert session_count == 1
