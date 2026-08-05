#!/usr/bin/env python3
"""Fail-closed verification for the protected Amrut review deployment.

The default verification is non-mutating.  It checks the deployed health and
security boundary, validates the exact frozen 79-item queue against the local
manifest, and proves that one still-blind item does not expose model output.

Reviewer tokens are accepted by argument or environment variable but are
never included in reports, exceptions, request diagnostics, or object reprs.
For routine use, prefer ``AMRUT_REVIEW_TOKEN`` over the command-line option so
the token does not appear in process listings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import ssl
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "review_snapshots"
    / "screen_2026-08-05_v2.3.0"
    / "manifest.json"
)
EXPECTED_ITEM_COUNT = 4
EXPECTED_REVIEWABLE_COUNT = 4
EXPECTED_QUARANTINE_COUNT = 0
INVALID_TOKEN = "release-verifier-invalid-capability-token"
MUTATION_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TWO_DRAFT_REVISIONS"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These are outputs or model-derived inputs that the server must not return
# until the reviewer has explicitly locked the base-pattern assessment.
FORBIDDEN_BLIND_KEYS = frozenset(
    {
        "algorithm_analysis",
        "analysis",
        "coil_score",
        "corpus_labels",
        "grade",
        "lid_grade",
        "lid_line",
        "lid_slope_pct_per_year",
        "lifecycle",
        "major_highs",
        "model_grade",
        "model_snapshot",
        "model_status",
        "quarterly_bars",
        "reviewed_highs",
        "score_total",
        "screen_snapshot",
        "source_features",
    }
)

# Invalid payloads are intentional: if the capture-only middleware were
# accidentally disabled, FastAPI would reject these before any handler could
# perform work.  A correctly configured share deployment rejects them earlier
# with 403.
SAFE_CAPTURE_BOUNDARY_PROBES = (
    (
        "legacy review mutation",
        "POST",
        "/api/highs/reviews",
        {},
    ),
    (
        "screen execution",
        "POST",
        "/api/screen",
        {"universe": "__release_verifier_invalid__"},
    ),
    (
        "vision review mutation",
        "POST",
        "/api/vision/reviews",
        {},
    ),
    (
        "vision execution",
        "POST",
        "/api/vision/run",
        {"interval": "__release_verifier_invalid__"},
    ),
)

SAFE_CAPTURE_READ_PROBES = (
    ("saved-run listing", "/api/saved-runs"),
    (
        "authoritative correction/model-history read",
        "/api/highs/corrections/REG",
    ),
)


class VerificationFailure(RuntimeError):
    """A release invariant was not satisfied."""


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    payload: Any


class HTTPClient(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
    ) -> HTTPResponse: ...


class _RejectRedirects(HTTPRedirectHandler):
    """Do not forward the capability header to another URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class URLHTTPClient:
    """Small JSON client that never includes request headers in errors."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        insecure_tls: bool = False,
    ) -> None:
        self._base_url = _normalize_base_url(base_url)
        self._timeout_seconds = timeout_seconds
        handlers: list[Any] = [_RejectRedirects()]
        if insecure_tls:
            handlers.append(
                HTTPSHandler(context=ssl._create_unverified_context())
            )
        self._opener = build_opener(*handlers)

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
    ) -> HTTPResponse:
        url = f"{self._base_url}{path}"
        body: bytes | None = None
        request_headers = {
            "Accept": "application/json",
            "User-Agent": "coilingview-amrut-release-verifier/1",
        }
        if headers:
            request_headers.update(headers)
        if json_body is not None:
            body = json.dumps(
                json_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        request = Request(
            url,
            data=body,
            headers=request_headers,
            method=method.upper(),
        )
        try:
            with self._opener.open(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                return HTTPResponse(
                    status=int(response.status),
                    payload=_decode_json(response.read()),
                )
        except HTTPError as exc:
            return HTTPResponse(
                status=int(exc.code),
                payload=_decode_json(exc.read()),
            )
        except (URLError, TimeoutError, OSError) as exc:
            # Do not include the Request repr: it can grow to include headers
            # in alternate urllib implementations.
            raise VerificationFailure(
                f"{method.upper()} {path} could not reach the deployment "
                f"({type(exc).__name__})."
            ) from None


@dataclass(frozen=True)
class VerificationConfig:
    session_id: int
    reviewer_token: str = field(repr=False)
    manifest_path: Path
    expected_manifest_sha256: str
    expected_source_run_sha256: str
    expected_corpus_tree_sha256: str | None = None
    expected_item_count: int = EXPECTED_ITEM_COUNT
    expected_reviewable_count: int = EXPECTED_REVIEWABLE_COUNT
    expected_quarantine_count: int = EXPECTED_QUARANTINE_COUNT
    probe_draft_persistence: bool = False
    mutation_confirmation: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Check:
    name: str
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    checks: tuple[Check, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "passed",
            "check_count": len(self.checks),
            "checks": [
                {"name": check.name, "detail": check.detail}
                for check in self.checks
            ],
        }


class ReleaseVerifier:
    def __init__(self, client: HTTPClient, config: VerificationConfig) -> None:
        self.client = client
        self.config = config
        self._checks: list[Check] = []
        self._manifest = _load_and_validate_manifest(config)

    def run(self) -> VerificationReport:
        self._check_health()
        self._check_capability_boundary()
        session = self._load_authorized_session()
        blind_item = self._check_session(session)
        self._check_blind_context(blind_item)
        self._check_capture_only_boundary()
        if self.config.probe_draft_persistence:
            self._probe_draft_persistence(session, blind_item)
        return VerificationReport(checks=tuple(self._checks))

    def _pass(self, name: str, detail: str) -> None:
        self._checks.append(Check(name=name, detail=detail))

    def _check_health(self) -> None:
        response = self.client.request("GET", "/api/health")
        _require_status(response, 200, "health")
        health = _require_mapping(response.payload, "health response")
        if health.get("status") != "ok":
            raise VerificationFailure("Health status is not ok.")

        persistence = health.get("persistence")
        capture_only = _first_present(
            health,
            "capture_only",
            "captureOnly",
        )
        if capture_only is None and isinstance(persistence, Mapping):
            # The backend describes whether durable persistence is "required"
            # from the same capture-only flag. This remains an explicit,
            # fail-closed signal rather than an inference from the backend type.
            capture_only = persistence.get("required")
        if capture_only is not True:
            raise VerificationFailure(
                "Health does not explicitly report capture-only mode."
            )

        if isinstance(persistence, Mapping):
            ready = _first_present(persistence, "ready", "persistence_ready")
            configured = _first_present(
                persistence,
                "configured",
                "persistence_configured",
            )
            durable = _first_present(
                persistence,
                "durable",
                "persistence_durable",
            )
            backend = _first_present(
                persistence,
                "backend",
                "persistence_backend",
            )
            development_escape = _first_present(
                persistence,
                "development_escape",
                "developmentEscape",
            )
        else:
            ready = _first_present(
                health,
                "persistence_ready",
                "persistenceReady",
            )
            configured = _first_present(
                health,
                "persistence_configured",
                "persistenceConfigured",
            )
            durable = _first_present(
                health,
                "persistence_durable",
                "persistenceDurable",
            )
            backend = _first_present(
                health,
                "persistence_backend",
                "persistenceBackend",
            )
            development_escape = _first_present(
                health,
                "persistence_development_escape",
                "persistenceDevelopmentEscape",
            )
        if ready is not True or configured is not True or durable is not True:
            raise VerificationFailure(
                "Health does not report configured, ready, durable persistence."
            )
        if backend != "sqlite_railway_volume":
            raise VerificationFailure(
                "Health does not report the isolated Railway SQLite volume."
            )
        if development_escape is not False:
            raise VerificationFailure(
                "Health does not explicitly disable ephemeral development storage."
            )
        self._pass(
            "health",
            "capture-only; durable Railway-volume SQLite persistence is ready",
        )

    def _check_capability_boundary(self) -> None:
        path = f"/api/review-sessions/{self.config.session_id}"
        missing = self.client.request("GET", path)
        wrong = self.client.request(
            "GET",
            path,
            headers={"X-Review-Token": INVALID_TOKEN},
        )
        if missing.status != 403:
            raise VerificationFailure(
                f"Missing reviewer token returned {missing.status}, expected 403."
            )
        if wrong.status != 403:
            raise VerificationFailure(
                f"Wrong reviewer token returned {wrong.status}, expected 403."
            )
        self._pass(
            "capability boundary",
            "missing and incorrect reviewer tokens are denied",
        )

    def _load_authorized_session(self) -> dict[str, Any]:
        response = self.client.request(
            "GET",
            f"/api/review-sessions/{self.config.session_id}",
            headers=self._auth_headers(),
        )
        _require_status(response, 200, "authorized session")
        payload = _require_mapping(response.payload, "session response")
        session = _require_mapping(payload.get("session"), "session")
        if session.get("id") != self.config.session_id:
            raise VerificationFailure("Authorized response returned another session.")
        return dict(session)

    def _check_session(self, session: Mapping[str, Any]) -> dict[str, Any]:
        if session.get("require_fresh_review") is not True:
            raise VerificationFailure(
                "Session is not marked as a fresh-review session."
            )
        expected_source = _require_nonempty_string(
            _require_mapping(
                self._manifest.get("source_run"),
                "manifest source-run metadata",
            ).get("filename"),
            "manifest source filename",
        )
        if session.get("source") != expected_source:
            raise VerificationFailure(
                "Session source does not match the frozen manifest."
            )
        items = _require_list(session.get("items"), "session items")
        expected_items = _require_list(
            self._manifest.get("items"),
            "manifest items",
        )
        if len(items) != self.config.expected_item_count:
            raise VerificationFailure(
                f"Session has {len(items)} items; "
                f"expected {self.config.expected_item_count}."
            )
        expected_order = [
            _require_nonempty_string(item.get("ticker"), "manifest ticker")
            for item in expected_items
            if isinstance(item, Mapping)
        ]
        actual_order = [
            _require_nonempty_string(item.get("ticker"), "session ticker")
            for item in items
            if isinstance(item, Mapping)
        ]
        if len(actual_order) != len(items):
            raise VerificationFailure("A session item is not a JSON object.")
        if actual_order != expected_order:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(actual_order, expected_order))
                    if pair[0] != pair[1]
                ),
                min(len(actual_order), len(expected_order)),
            )
            raise VerificationFailure(
                f"Session order differs from the manifest at item {mismatch + 1}."
            )

        expected_reviewable = 0
        actual_reviewable = 0
        seen_sample_ids: set[str] = set()
        blind_candidates: list[dict[str, Any]] = []
        for index, (raw_item, raw_expected) in enumerate(
            zip(items, expected_items),
            start=1,
        ):
            item = _require_mapping(raw_item, f"session item {index}")
            expected = _require_mapping(raw_expected, f"manifest item {index}")
            expected_is_reviewable = bool(
                _require_mapping(
                    expected.get("data_quality"),
                    f"manifest item {index} data quality",
                ).get("reviewable")
            )
            expected_reviewable += int(expected_is_reviewable)
            if item.get("reviewable") is not expected_is_reviewable:
                raise VerificationFailure(
                    f"{item.get('ticker', 'item')} reviewability differs "
                    "from the manifest."
                )
            actual_reviewable += int(bool(item.get("reviewable")))

            bars_hash = _require_sha256(
                item.get("bars_hash"),
                f"{item.get('ticker', 'item')} bars hash",
            )
            expected_bars_hash = _require_sha256(
                expected.get("backend_bars_identity_sha256"),
                f"{item.get('ticker', 'item')} manifest bars hash",
            )
            if not secrets.compare_digest(bars_hash, expected_bars_hash):
                raise VerificationFailure(
                    f"{item.get('ticker', 'item')} bars hash differs "
                    "from the manifest."
                )

            sample_id = _require_sha256(
                item.get("sample_id"),
                f"{item.get('ticker', 'item')} sample id",
            )
            if sample_id in seen_sample_ids:
                raise VerificationFailure("Session sample IDs are not unique.")
            seen_sample_ids.add(sample_id)

            snapshot = _require_mapping(
                item.get("snapshot"),
                f"{item.get('ticker', 'item')} snapshot",
            )
            frozen = _require_mapping(
                snapshot.get("frozen"),
                f"{item.get('ticker', 'item')} frozen identity",
            )
            expected_snapshot_hash = _require_sha256(
                expected.get("snapshot_sha256"),
                f"{item.get('ticker', 'item')} manifest snapshot hash",
            )
            actual_snapshot_hash = _require_sha256(
                frozen.get("snapshot_sha256"),
                f"{item.get('ticker', 'item')} snapshot hash",
            )
            if not secrets.compare_digest(
                actual_snapshot_hash,
                expected_snapshot_hash,
            ):
                raise VerificationFailure(
                    f"{item.get('ticker', 'item')} snapshot hash differs "
                    "from the manifest."
                )

            if _is_prelock_candidate(item):
                _assert_blind_payload(item, f"pre-lock session item {index}")
                blind_candidates.append(dict(item))

        expected_quarantine = len(items) - expected_reviewable
        actual_quarantine = len(items) - actual_reviewable
        if expected_reviewable != self.config.expected_reviewable_count:
            raise VerificationFailure(
                "Local manifest reviewable count does not match the "
                "release expectation."
            )
        if expected_quarantine != self.config.expected_quarantine_count:
            raise VerificationFailure(
                "Local manifest quarantine count does not match the "
                "release expectation."
            )
        if actual_reviewable != self.config.expected_reviewable_count:
            raise VerificationFailure(
                f"Session has {actual_reviewable} reviewable items; "
                f"expected {self.config.expected_reviewable_count}."
            )
        if actual_quarantine != self.config.expected_quarantine_count:
            raise VerificationFailure(
                f"Session has {actual_quarantine} quarantined items; "
                f"expected {self.config.expected_quarantine_count}."
            )

        self._check_session_hashes(session)
        counts = _require_mapping(session.get("counts"), "session counts")
        if counts.get("total") != self.config.expected_item_count:
            raise VerificationFailure("Session count summary is inconsistent.")
        summarized = sum(
            int(counts.get(status, 0))
            for status in ("pending", "reviewed", "skipped")
        )
        if summarized != self.config.expected_item_count:
            raise VerificationFailure(
                "Pending/reviewed/skipped counts do not total 79."
            )
        if not blind_candidates:
            raise VerificationFailure(
                "No pending, unlocked item is available for the blind-response gate."
            )
        self._pass(
            "frozen session",
            (
                f"{len(items)} items in manifest order; "
                f"{actual_reviewable} reviewable and "
                f"{actual_quarantine} quarantined; hashes match"
            ),
        )
        self._pass(
            "pre-lock session redaction",
            "pending unlocked item contains no known model fields",
        )
        return blind_candidates[0]

    def _check_session_hashes(self, session: Mapping[str, Any]) -> None:
        snapshot = _require_mapping(session.get("snapshot"), "session snapshot")
        frozen_manifest = _require_mapping(
            snapshot.get("frozen_manifest"),
            "session frozen-manifest metadata",
        )
        manifest_hash = _require_sha256(
            frozen_manifest.get("sha256"),
            "deployed manifest hash",
        )
        if not secrets.compare_digest(
            manifest_hash,
            self.config.expected_manifest_sha256,
        ):
            raise VerificationFailure("Deployed manifest hash is unexpected.")

        source_run = frozen_manifest.get("source_run")
        deployed_source_hash = _first_present(
            frozen_manifest,
            "source_run_sha256",
            "sourceRunSha256",
        )
        if isinstance(source_run, Mapping):
            deployed_source_hash = source_run.get("sha256")
        if deployed_source_hash is not None:
            source_hash = _require_sha256(
                deployed_source_hash,
                "deployed source-run hash",
            )
            if not secrets.compare_digest(
                source_hash,
                self.config.expected_source_run_sha256,
            ):
                raise VerificationFailure(
                    "Deployed source-run hash is unexpected."
                )
        # A matching manifest hash still binds the exact source-run hash even
        # when the pre-lock API deliberately redacts source-run metadata. The
        # local manifest/hash check above proves that expected binding.
        if frozen_manifest.get("item_count") != self.config.expected_item_count:
            raise VerificationFailure(
                "Deployed manifest item count does not match the expected "
                f"{self.config.expected_item_count}."
            )

        expected_tree = self.config.expected_corpus_tree_sha256
        if expected_tree is not None:
            deployed_tree = _first_present(
                frozen_manifest,
                "corpus_tree_sha256",
                "tree_sha256",
            )
            deployed_tree = _require_sha256(
                deployed_tree,
                "deployed corpus-tree hash",
            )
            if not secrets.compare_digest(deployed_tree, expected_tree):
                raise VerificationFailure(
                    "Deployed corpus-tree hash is unexpected."
                )

    def _check_blind_context(self, item: Mapping[str, Any]) -> None:
        ticker = _require_nonempty_string(item.get("ticker"), "blind ticker")
        response = self.client.request(
            "GET",
            (
                f"/api/review-sessions/{self.config.session_id}/items/"
                f"{quote(ticker, safe='')}/context"
            ),
            headers=self._auth_headers(),
        )
        _require_status(response, 200, f"{ticker} pre-lock context")
        payload = _require_mapping(response.payload, "context response")
        context = _require_mapping(payload.get("context"), "context")
        if context.get("model_revealed") not in (False, None):
            raise VerificationFailure(
                f"{ticker} context reports the model as revealed before lock."
            )
        _assert_blind_payload(context, f"{ticker} pre-lock context")
        monthly_bars = _require_list(
            context.get("monthly_bars"),
            f"{ticker} frozen monthly bars",
        )
        if not monthly_bars:
            raise VerificationFailure(
                f"{ticker} pre-lock context has no frozen monthly bars."
            )
        context_bars_hash = _require_sha256(
            context.get("bars_hash"),
            f"{ticker} context bars hash",
        )
        if not secrets.compare_digest(
            context_bars_hash,
            _require_sha256(item.get("bars_hash"), f"{ticker} session bars hash"),
        ):
            raise VerificationFailure(
                f"{ticker} context bars hash differs from the session."
            )
        self._pass(
            "pre-lock context redaction",
            f"{ticker} exposes frozen candles and identity, but no model output",
        )

    def _check_capture_only_boundary(self) -> None:
        for name, path in SAFE_CAPTURE_READ_PROBES:
            response = self.client.request("GET", path)
            if response.status != 403:
                raise VerificationFailure(
                    f"{name} returned {response.status}, expected capture-only 403."
                )
        for name, method, path, body in SAFE_CAPTURE_BOUNDARY_PROBES:
            response = self.client.request(
                method,
                path,
                json_body=body,
            )
            if response.status != 403:
                raise VerificationFailure(
                    f"{name} returned {response.status}, expected capture-only 403."
                )
        self._pass(
            "capture-only API boundary",
            (
                "saved-run and model-history reads plus legacy review, screen, "
                "and vision writes are denied"
            ),
        )

    def _probe_draft_persistence(
        self,
        session: Mapping[str, Any],
        item: Mapping[str, Any],
    ) -> None:
        if self.config.mutation_confirmation != MUTATION_CONFIRMATION:
            raise VerificationFailure(
                "Draft persistence probe requires the exact mutation confirmation."
            )
        if session.get("finalized_at") is not None:
            raise VerificationFailure(
                "Draft persistence probe cannot run on a finalized session."
            )
        ticker = _require_nonempty_string(item.get("ticker"), "draft probe ticker")
        revision = item.get("draft_revision")
        if not isinstance(revision, int) or revision < 0:
            raise VerificationFailure(
                f"{ticker} does not expose a valid draft revision."
            )
        original = item.get("draft")
        if original is not None and not isinstance(original, Mapping):
            raise VerificationFailure(
                f"{ticker} has an unsupported draft payload."
            )
        restored_payload = dict(original or {})
        marker = secrets.token_hex(16)
        probe_payload = dict(restored_payload)
        probe_payload["_releaseVerifier"] = {
            "nonce": marker,
            "purpose": "explicit durable-storage release probe",
        }
        path = (
            f"/api/review-sessions/{self.config.session_id}/items/"
            f"{quote(ticker, safe='')}/draft"
        )
        saved = self.client.request(
            "PUT",
            path,
            headers=self._auth_headers(),
            json_body={
                "expectedRevision": revision,
                "payload": probe_payload,
            },
        )
        _require_status(saved, 200, f"{ticker} draft probe write")
        saved_payload = _require_mapping(saved.payload, "draft probe response")
        saved_item = _require_mapping(saved_payload.get("item"), "saved draft item")
        saved_revision = saved_item.get("draft_revision")
        if saved_revision != revision + 1:
            raise VerificationFailure("Draft probe revision did not advance.")
        verification_error: VerificationFailure | None = None
        try:
            reloaded = self._load_authorized_session()
            reloaded_item = next(
                (
                    candidate
                    for candidate in _require_list(
                        reloaded.get("items"),
                        "reloaded session items",
                    )
                    if isinstance(candidate, Mapping)
                    and candidate.get("ticker") == ticker
                ),
                None,
            )
            if not isinstance(reloaded_item, Mapping):
                raise VerificationFailure(
                    "Draft probe item disappeared after reload."
                )
            persisted_marker = (
                reloaded_item.get("draft", {})
                if isinstance(reloaded_item.get("draft"), Mapping)
                else {}
            ).get("_releaseVerifier")
            if (
                not isinstance(persisted_marker, Mapping)
                or persisted_marker.get("nonce") != marker
            ):
                raise VerificationFailure(
                    "Draft probe did not persist across reload."
                )
        except VerificationFailure as exc:
            verification_error = exc
        finally:
            # Always try to remove the marker. Optimistic concurrency keeps
            # this cleanup from overwriting an intervening reviewer save.
            restored = self.client.request(
                "PUT",
                path,
                headers=self._auth_headers(),
                json_body={
                    "expectedRevision": saved_revision,
                    "payload": restored_payload,
                },
            )
            _require_status(restored, 200, f"{ticker} draft probe restore")
            restored_item = _require_mapping(
                _require_mapping(
                    restored.payload,
                    "draft restore response",
                ).get("item"),
                "restored draft item",
            )
            if restored_item.get("draft") != restored_payload:
                raise VerificationFailure(
                    "Draft probe could not restore prior content."
                )
        if verification_error is not None:
            raise verification_error
        self._pass(
            "explicit draft persistence probe",
            (
                f"{ticker} persisted across reload and prior content was restored; "
                "draft revision advanced twice"
            ),
        )

    def _auth_headers(self) -> dict[str, str]:
        return {"X-Review-Token": self.config.reviewer_token}


def _load_and_validate_manifest(
    config: VerificationConfig,
) -> dict[str, Any]:
    try:
        manifest_bytes = config.manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationFailure(
            f"Frozen manifest could not be read ({type(exc).__name__})."
        ) from None
    manifest = _require_mapping(manifest, "local manifest")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if not secrets.compare_digest(
        manifest_hash,
        config.expected_manifest_sha256,
    ):
        raise VerificationFailure(
            "Local manifest bytes do not match the expected manifest hash."
        )
    source_run = _require_mapping(
        manifest.get("source_run"),
        "local source-run metadata",
    )
    local_source_hash = _require_sha256(
        source_run.get("sha256"),
        "local source-run hash",
    )
    if not secrets.compare_digest(
        local_source_hash,
        config.expected_source_run_sha256,
    ):
        raise VerificationFailure(
            "Local manifest source-run hash does not match the expectation."
        )
    items = _require_list(manifest.get("items"), "local manifest items")
    if len(items) != config.expected_item_count:
        raise VerificationFailure(
            f"Local manifest has {len(items)} items; "
            f"expected {config.expected_item_count}."
        )
    return dict(manifest)


def _is_prelock_candidate(item: Mapping[str, Any]) -> bool:
    if item.get("status") != "pending":
        return False
    for key in (
        "base_assessment_locked",
        "baseAssessmentLocked",
        "base_classification_locked",
        "baseClassificationLocked",
        "base_locked",
        "baseLocked",
    ):
        if item.get(key) is True:
            return False
    base_lock = _first_present(item, "base_lock", "baseLock")
    return base_lock in (None, False)


def _assert_blind_payload(value: Any, label: str) -> None:
    leaks = sorted(set(_find_forbidden_keys(value)))
    if leaks:
        raise VerificationFailure(
            f"{label} leaks pre-lock model fields: {', '.join(leaks)}."
        )


def _find_forbidden_keys(value: Any, path: str = "$") -> Iterable[str]:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = _snake_case(key)
            child_path = f"{path}.{key}"
            if normalized in FORBIDDEN_BLIND_KEYS:
                yield child_path
            yield from _find_forbidden_keys(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _find_forbidden_keys(child, f"{path}[{index}]")


def _snake_case(value: str) -> str:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return separated.replace("-", "_").lower()


def _normalize_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise VerificationFailure("Base URL must be an http(s) URL.")
    if parsed.username or parsed.password:
        raise VerificationFailure("Base URL must not contain credentials.")
    if parsed.query or parsed.fragment:
        raise VerificationFailure(
            "Base URL must not contain a query string or fragment."
        )
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _decode_json(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _require_status(
    response: HTTPResponse,
    expected: int,
    label: str,
) -> None:
    if response.status != expected:
        raise VerificationFailure(
            f"{label} returned {response.status}, expected {expected}."
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationFailure(f"{label} is not a JSON object.")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise VerificationFailure(f"{label} is not a JSON array.")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerificationFailure(f"{label} is missing.")
    return value.strip()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value.lower()):
        raise VerificationFailure(f"{label} is not a SHA-256 value.")
    return value.lower()


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _env_or_none(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only release gate for the protected Amrut review deployment."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=_env_or_none("AMRUT_RELEASE_BASE_URL"),
        help="Deployment origin (or AMRUT_RELEASE_BASE_URL).",
    )
    parser.add_argument(
        "--session-id",
        type=int,
        default=_env_or_none("AMRUT_REVIEW_SESSION_ID"),
        help="Protected review session ID (or AMRUT_REVIEW_SESSION_ID).",
    )
    parser.add_argument(
        "--reviewer-token",
        default=_env_or_none("AMRUT_REVIEW_TOKEN"),
        help=(
            "Reviewer capability token. Prefer AMRUT_REVIEW_TOKEN so it does "
            "not appear in process listings."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Local frozen manifest used to prove exact order and item hashes.",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        default=_env_or_none("AMRUT_EXPECTED_MANIFEST_SHA256"),
        help="Expected manifest SHA-256 (or AMRUT_EXPECTED_MANIFEST_SHA256).",
    )
    parser.add_argument(
        "--expected-source-run-sha256",
        default=_env_or_none("AMRUT_EXPECTED_SOURCE_RUN_SHA256"),
        help="Expected source CSV SHA-256 (or AMRUT_EXPECTED_SOURCE_RUN_SHA256).",
    )
    parser.add_argument(
        "--expected-corpus-tree-sha256",
        default=_env_or_none("AMRUT_EXPECTED_CORPUS_TREE_SHA256"),
        help=(
            "Optional corpus-tree SHA-256; when supplied, deployed metadata "
            "must expose and match it."
        ),
    )
    parser.add_argument(
        "--expected-item-count",
        type=int,
        default=int(
            _env_or_none("AMRUT_EXPECTED_ITEM_COUNT")
            or EXPECTED_ITEM_COUNT
        ),
    )
    parser.add_argument(
        "--expected-reviewable-count",
        type=int,
        default=int(
            _env_or_none("AMRUT_EXPECTED_REVIEWABLE_COUNT")
            or EXPECTED_REVIEWABLE_COUNT
        ),
    )
    parser.add_argument(
        "--expected-quarantine-count",
        type=int,
        default=int(
            _env_or_none("AMRUT_EXPECTED_QUARANTINE_COUNT")
            or EXPECTED_QUARANTINE_COUNT
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--insecure-tls",
        action="store_true",
        help="Disable TLS verification for local testing only.",
    )
    parser.add_argument(
        "--probe-draft-persistence",
        action="store_true",
        help=(
            "EXPLICIT MUTATION: write, reload, and restore one pending draft. "
            "This advances its revision twice."
        ),
    )
    parser.add_argument(
        "--confirm-draft-mutation",
        help=(
            "Required with --probe-draft-persistence; exact value: "
            f"{MUTATION_CONFIRMATION}"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable, secret-free report.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> VerificationConfig:
    missing: list[str] = []
    if not args.base_url:
        missing.append("--base-url / AMRUT_RELEASE_BASE_URL")
    if args.session_id is None:
        missing.append("--session-id / AMRUT_REVIEW_SESSION_ID")
    if not args.reviewer_token:
        missing.append("--reviewer-token / AMRUT_REVIEW_TOKEN")
    if not args.expected_manifest_sha256:
        missing.append(
            "--expected-manifest-sha256 / AMRUT_EXPECTED_MANIFEST_SHA256"
        )
    if not args.expected_source_run_sha256:
        missing.append(
            "--expected-source-run-sha256 / AMRUT_EXPECTED_SOURCE_RUN_SHA256"
        )
    if missing:
        raise VerificationFailure(
            "Missing required configuration: " + "; ".join(missing) + "."
        )
    if args.session_id < 1:
        raise VerificationFailure("Session ID must be a positive integer.")
    if len(args.reviewer_token) < 24:
        raise VerificationFailure("Reviewer token is too short.")
    manifest_hash = _require_sha256(
        args.expected_manifest_sha256,
        "expected manifest hash",
    )
    source_hash = _require_sha256(
        args.expected_source_run_sha256,
        "expected source-run hash",
    )
    tree_hash = (
        _require_sha256(
            args.expected_corpus_tree_sha256,
            "expected corpus-tree hash",
        )
        if args.expected_corpus_tree_sha256
        else None
    )
    if args.expected_reviewable_count + args.expected_quarantine_count != (
        args.expected_item_count
    ):
        raise VerificationFailure(
            "Expected reviewable and quarantine counts must total item count."
        )
    if args.timeout_seconds <= 0:
        raise VerificationFailure("Timeout must be positive.")
    if args.probe_draft_persistence and (
        args.confirm_draft_mutation != MUTATION_CONFIRMATION
    ):
        raise VerificationFailure(
            "Draft persistence probe requires the exact mutation confirmation."
        )
    return VerificationConfig(
        session_id=args.session_id,
        reviewer_token=args.reviewer_token,
        manifest_path=args.manifest.resolve(),
        expected_manifest_sha256=manifest_hash,
        expected_source_run_sha256=source_hash,
        expected_corpus_tree_sha256=tree_hash,
        expected_item_count=args.expected_item_count,
        expected_reviewable_count=args.expected_reviewable_count,
        expected_quarantine_count=args.expected_quarantine_count,
        probe_draft_persistence=args.probe_draft_persistence,
        mutation_confirmation=args.confirm_draft_mutation,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        client = URLHTTPClient(
            args.base_url,
            timeout_seconds=args.timeout_seconds,
            insecure_tls=args.insecure_tls,
        )
        report = ReleaseVerifier(client, config).run()
    except VerificationFailure as exc:
        message = str(exc)
        token = getattr(args, "reviewer_token", None)
        if token:
            message = message.replace(token, "[REDACTED]")
        if args.json:
            print(
                json.dumps(
                    {"status": "failed", "error": message},
                    sort_keys=True,
                )
            )
        else:
            print(f"FAILED: {message}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.as_dict(), sort_keys=True))
    else:
        print(f"PASSED: {len(report.checks)} release checks")
        for check in report.checks:
            print(f"- {check.name}: {check.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
