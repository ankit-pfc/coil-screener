from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest
from fastapi.testclient import TestClient

import app as app_module
import reviews as reviews_module
from reviews import ReviewStore
from scripts.verify_amrut_release import (
    DEFAULT_MANIFEST,
    HTTPResponse,
    ReleaseVerifier,
    VerificationConfig,
    VerificationFailure,
)

TOKEN = "unit-test-reviewer-token-that-is-never-printed"


class FakeClient:
    def __init__(
        self,
        session: dict[str, Any],
        context: dict[str, Any],
        *,
        health: dict[str, Any] | None = None,
    ) -> None:
        self.session = session
        self.context = context
        self.health = health or {
            "status": "ok",
            "persistence": {
                "ready": True,
                "required": True,
                "configured": True,
                "durable": True,
                "backend": "sqlite_railway_volume",
                "development_escape": False,
            },
        }
        self.calls: list[tuple[str, str, Mapping[str, str], Any]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
    ) -> HTTPResponse:
        supplied_headers = dict(headers or {})
        self.calls.append((method, path, supplied_headers, json_body))
        if method == "GET" and path == "/api/health":
            return HTTPResponse(200, self.health)
        if method == "GET" and path == "/api/review-sessions/7":
            if supplied_headers.get("X-Review-Token") != TOKEN:
                return HTTPResponse(403, {"detail": "denied"})
            return HTTPResponse(200, {"session": self.session})
        if method == "GET" and path.endswith("/context"):
            if supplied_headers.get("X-Review-Token") != TOKEN:
                return HTTPResponse(403, {"detail": "denied"})
            return HTTPResponse(200, {"context": self.context})
        if method == "GET" and path in {
            "/api/saved-runs",
            "/api/highs/corrections/REG",
        }:
            return HTTPResponse(403, {"detail": "capture-only"})
        if method == "POST":
            return HTTPResponse(403, {"detail": "capture-only"})
        raise AssertionError(f"unexpected request: {method} {path}")


class InProcessHTTPClient:
    def __init__(self, client: TestClient) -> None:
        self.client = client

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
    ) -> HTTPResponse:
        response = self.client.request(
            method,
            path,
            headers=dict(headers or {}),
            json=json_body,
        )
        return HTTPResponse(
            status=response.status_code,
            payload=response.json() if response.content else None,
        )


def _fixture() -> tuple[
    VerificationConfig,
    dict[str, Any],
    dict[str, Any],
]:
    manifest_path = Path(DEFAULT_MANIFEST)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    items = []
    for index, expected in enumerate(manifest["items"]):
        items.append(
            {
                "ticker": expected["ticker"],
                "position": index,
                "status": "pending",
                "reviewable": expected["data_quality"]["reviewable"],
                "base_classification_locked": False,
                "bars_hash": expected["backend_bars_identity_sha256"],
                "sample_id": hashlib.sha256(
                    f"sample-{index}".encode("utf-8")
                ).hexdigest(),
                "draft": None,
                "draft_revision": 0,
                "snapshot": {
                    "data_quality": expected["data_quality"],
                    "reviewable": expected["data_quality"]["reviewable"],
                    "frozen": {
                        "source": manifest["source_run"]["filename"],
                        "snapshot_sha256": expected["snapshot_sha256"],
                    },
                },
            }
        )
    session = {
        "id": 7,
        "source": manifest["source_run"]["filename"],
        "require_fresh_review": True,
        "finalized_at": None,
        "items": items,
        "counts": {
            "pending": 79,
            "reviewed": 0,
            "skipped": 0,
            "total": 79,
        },
        "snapshot": {
            "frozen_manifest": {
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "item_count": 79,
                "source_run": manifest["source_run"],
            }
        },
    }
    first = items[0]
    context = {
        "ticker": first["ticker"],
        "reviewable": first["reviewable"],
        "model_revealed": False,
        "bars_hash": first["bars_hash"],
        "sample_id": first["sample_id"],
        "monthly_bars": [
            {
                "date": "2026-07-01",
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
            }
        ],
        "data_quality": first["snapshot"]["data_quality"],
    }
    config = VerificationConfig(
        session_id=7,
        reviewer_token=TOKEN,
        manifest_path=manifest_path,
        expected_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        expected_source_run_sha256=manifest["source_run"]["sha256"],
    )
    return config, session, context


def test_default_release_verification_is_non_mutating_and_passes():
    config, session, context = _fixture()
    client = FakeClient(session, context)

    report = ReleaseVerifier(client, config).run()

    assert len(report.checks) == 6
    assert {call[0] for call in client.calls} == {"GET", "POST"}
    assert not any(call[0] in {"PUT", "PATCH", "DELETE"} for call in client.calls)
    post_calls = [call for call in client.calls if call[0] == "POST"]
    assert all(
        call[3]
        in (
            {},
            {"universe": "__release_verifier_invalid__"},
            {"interval": "__release_verifier_invalid__"},
        )
        for call in post_calls
    )


def test_prelock_context_model_leak_fails_closed():
    config, session, context = _fixture()
    context["analysis"] = {"grade": "A"}

    with pytest.raises(VerificationFailure, match="leaks pre-lock model fields"):
        ReleaseVerifier(FakeClient(session, context), config).run()


def test_prelock_session_model_leak_fails_closed():
    config, session, context = _fixture()
    session["items"][0]["snapshot"]["screen_snapshot"] = {"grade": "A"}

    with pytest.raises(VerificationFailure, match="leaks pre-lock model fields"):
        ReleaseVerifier(FakeClient(session, context), config).run()


def test_manifest_order_or_item_hash_drift_fails_closed():
    config, session, context = _fixture()
    first = deepcopy(session["items"][0])
    session["items"][0] = session["items"][1]
    session["items"][1] = first

    with pytest.raises(VerificationFailure, match="order differs"):
        ReleaseVerifier(FakeClient(session, context), config).run()

    config, session, context = _fixture()
    session["items"][0]["bars_hash"] = "0" * 64
    with pytest.raises(VerificationFailure, match="bars hash differs"):
        ReleaseVerifier(FakeClient(session, context), config).run()


@pytest.mark.parametrize(
    "health",
    [
        {
            "status": "ok",
            "capture_only": False,
            "persistence": {
                "ready": True,
                "durable": True,
                "backend": "postgresql",
                "configured": True,
                "development_escape": False,
            },
        },
        {
            "status": "ok",
            "capture_only": True,
            "persistence": {
                "ready": True,
                "durable": False,
                "backend": "sqlite",
                "configured": True,
                "development_escape": False,
            },
        },
        {
            "status": "ok",
            "capture_only": True,
            "persistence": {
                "ready": True,
                "durable": True,
                "configured": False,
                "backend": "sqlite_railway_volume",
                "development_escape": False,
            },
        },
        {
            "status": "ok",
            "capture_only": True,
            "persistence": {
                "ready": True,
                "durable": True,
                "configured": True,
                "backend": "postgresql",
                "development_escape": False,
            },
        },
        {
            "status": "ok",
            "capture_only": True,
            "persistence": {
                "ready": True,
                "durable": True,
                "configured": True,
                "backend": "sqlite_railway_volume",
                "development_escape": True,
            },
        },
    ],
)
def test_health_must_explicitly_prove_capture_only_and_durability(health):
    config, session, context = _fixture()
    with pytest.raises(VerificationFailure):
        ReleaseVerifier(
            FakeClient(session, context, health=health),
            config,
        ).run()


def test_capability_token_is_omitted_from_config_repr_and_reports():
    config, session, context = _fixture()
    assert TOKEN not in repr(config)
    report = ReleaseVerifier(FakeClient(session, context), config).run()
    assert TOKEN not in json.dumps(report.as_dict())


def test_verifier_matches_real_capture_only_backend_contract(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "durable-review-test.db"
    store = ReviewStore(sqlite_path=database_path)
    monkeypatch.setattr(reviews_module, "_store", store)
    monkeypatch.setenv("CAPTURE_ONLY_MODE", "true")
    monkeypatch.setenv("REVIEW_SESSION_CREATE_KEY", "test-admin-key")
    monkeypatch.setenv("REVIEW_DB_PATH", str(database_path))
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("RAILWAY_VOLUME_MOUNT_PATH", str(tmp_path))
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_ID", "test-environment")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name in (
        "RAILWAY_ENVIRONMENT_NAME",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_STATIC_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    manifest_path = Path(DEFAULT_MANIFEST)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    client = TestClient(app_module.app)
    created = client.post(
        "/api/review-sessions",
        headers={"X-Review-Admin-Key": "test-admin-key"},
        json={
            "source": manifest["source_run"]["filename"],
            "reviewerName": "Amrut",
            "accessToken": TOKEN,
            "requireFreshReview": True,
            "items": [
                {"ticker": item["ticker"]}
                for item in manifest["items"]
            ],
        },
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["session"]["id"]
    config = VerificationConfig(
        session_id=session_id,
        reviewer_token=TOKEN,
        manifest_path=manifest_path,
        expected_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        expected_source_run_sha256=manifest["source_run"]["sha256"],
    )

    report = ReleaseVerifier(InProcessHTTPClient(client), config).run()

    assert {check.name for check in report.checks} >= {
        "health",
        "capability boundary",
        "frozen session",
        "pre-lock session redaction",
        "pre-lock context redaction",
        "capture-only API boundary",
    }
