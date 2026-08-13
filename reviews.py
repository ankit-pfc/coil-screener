"""Approved human-review persistence for major-high corrections and sessions.

The frontend's manual-high flow (see frontend/src/lib/manualHighs.ts) posts a
``CorrectionRecord`` to ``/api/highs/corrections``. This module treats each
accepted record as an approved internal review:

- ``high_reviews`` is APPEND-ONLY: every correction, decision, and revocation
  is kept verbatim so the original algorithm output and the human decision
  remain available for comparison and calibration.
- ``high_overrides`` holds the CURRENT effective points per (ticker,
  interval). Approving a correction upserts this row, and ``/api/coil``
  applies it immediately: the reviewed points become the active structure and
  slope/status recompute from them as new bars arrive.
- ``review_state`` holds the LATEST decision per (ticker, interval) regardless
  of kind — an approval leaves the algorithm effective but still records that
  a human signed off, together with the ``as_of`` data date and algorithm
  version the sign-off applies to. Newer data or a newer algorithm makes the
  decision stale (re-review required) without disabling a corrected override.
- ``review_sessions`` / ``review_session_items`` model one reviewer pass over
  an ordered screener result. Sessions resume by snapshot fingerprint; skips
  are session-specific while decisions are shared across sessions.

Reviewers keep full authority over which zone or level they anchor to. The
one structural restriction they share with the algorithm (v2.2) is that a
reviewed point may not sit in the incomplete final quarter; see
``reject_incomplete_quarter_points``, which the review endpoints call before
anything is persisted.

Storage backend is PostgreSQL when ``DATABASE_URL`` is set. Otherwise it uses
SQLite, preferring ``REVIEW_DB_PATH`` or Railway's mounted-volume path when
available and falling back to the project directory for local development.
Both run the same logical schema; records are stored as JSON text for backend
parity.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import sqlite3
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent


def _default_sqlite_path() -> Path:
    explicit = os.environ.get("REVIEW_DB_PATH")
    if explicit:
        return Path(explicit).expanduser()
    volume_mount = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    if volume_mount:
        return Path(volume_mount) / "reviews.db"
    return PROJECT_ROOT / "reviews.db"


DEFAULT_SQLITE_PATH = _default_sqlite_path()
AUTHORITATIVE_INTERVAL = "3M"
SUPPORTED_ROLES = frozenset(
    {"major_top", "structural_retest", "provisional_top", "breakout_peak"}
)
DECISION_APPROVED = "approved"
DECISION_CORRECTED = "corrected"
ITEM_STATUS_PENDING = "pending"
ITEM_STATUS_SKIPPED = "skipped"
ITEM_STATUS_REVIEWED = "reviewed"


class ReviewConflictError(ValueError):
    """A protected-session write lost an optimistic or immutable-state check."""


class ReviewAccessError(PermissionError):
    """A protected session was addressed without its capability token."""


def hash_review_token(token: str) -> str:
    """One-way representation of a high-entropy review capability."""
    value = str(token or "")
    if not value:
        raise ValueError("review access token is required")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _draft_learning_capture(draft: Any) -> Optional[dict[str, Any]]:
    if not isinstance(draft, dict):
        return None
    payload = draft.get("payload") if isinstance(draft.get("payload"), dict) else draft
    capture = payload.get("learningCapture", payload.get("learning_capture"))
    return capture if isinstance(capture, dict) else None


def _draft_matches_base_classification(
    draft: Any,
    classification: dict[str, Any],
    *,
    reviewer_name: Optional[str],
) -> bool:
    capture = _draft_learning_capture(draft)
    if capture is None or capture.get("baseAssessmentLocked") is not True:
        return False
    if str(capture.get("reviewerName", "")).strip() != str(reviewer_name or ""):
        return False
    if capture.get("sequencePolicyVersion") != 1:
        return False
    if capture.get("basePath") != classification.get("basePath"):
        return False
    raw_failed = capture.get("failedBaseRules")
    if not isinstance(raw_failed, list):
        return False
    if raw_failed != classification.get("failedBaseRules"):
        return False
    if str(capture.get("baseRationale", "")).strip() != str(
        classification.get("rationale", "")
    ).strip():
        return False
    blind_assessment = classification.get("blindAssessment")
    if blind_assessment is not None:
        payload = draft.get("payload") if isinstance(draft.get("payload"), dict) else draft
        if payload.get("blindAssessment") != blind_assessment:
            return False
    return True


def _redacted_prelock_snapshot(
    snapshot: dict[str, Any], *, position: int
) -> dict[str, Any]:
    frozen = snapshot.get("frozen") if isinstance(snapshot.get("frozen"), dict) else {}
    return {
        "cohort_position": position + 1,
        "reviewable": bool(snapshot.get("reviewable", True)),
        "data_quality": snapshot.get("data_quality"),
        "data_quality_validation": snapshot.get("data_quality_validation"),
        "frozen": {
            key: frozen.get(key)
            for key in (
                "source",
                "data_date",
                "sample_id",
                "bars_hash",
                "snapshot_sha256",
            )
        },
    }


def _redacted_prelock_draft(draft: Any) -> Optional[dict[str, Any]]:
    capture = _draft_learning_capture(draft)
    if capture is None:
        return None
    payload = draft.get("payload") if isinstance(draft.get("payload"), dict) else draft
    return {
        "schemaVersion": 5,
        "learningCapture": {
            key: capture.get(key)
            for key in (
                "reviewerName",
                "sequencePolicyVersion",
                "baseAssessmentLocked",
                "basePath",
                "failedBaseRules",
                "baseRationale",
                "commentary",
            )
        },
        "blindAssessment": payload.get("blindAssessment"),
        "reviewedHighs": payload.get("reviewedHighs", []),
    }


class ReviewedPoint(BaseModel):
    """One authoritative chart point accepted by a human reviewer."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    date: str
    price: float = Field(gt=0)
    role: Literal[
        "major_top", "structural_retest", "provisional_top", "breakout_peak"
    ] = "major_top"
    lid_member: Optional[bool] = Field(default=None, alias="lidMember")

    @field_validator("date")
    @classmethod
    def valid_iso_date(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("review point dates must be ISO YYYY-MM-DD") from exc
        if parsed.isoformat() != value:
            raise ValueError("review point dates must be ISO YYYY-MM-DD")
        return value

    @field_validator("price")
    @classmethod
    def finite_price(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("review point prices must be finite")
        return value


def _valid_iso_timestamp(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamps must be ISO-8601") from exc
    return value


class CorrectionRecord(BaseModel):
    """Validated production correction; Vision output is advisory only."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_version: int = Field(default=2, alias="schemaVersion")
    ticker: str
    interval: Literal["3M"] = AUTHORITATIVE_INTERVAL
    created_at: Optional[str] = Field(default=None, alias="createdAt")
    manual_highs: list[ReviewedPoint] = Field(alias="manualHighs")

    @field_validator("ticker")
    @classmethod
    def valid_ticker(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("Correction record must include a ticker.")
        return symbol

    @field_validator("created_at")
    @classmethod
    def valid_created_at(cls, value: Optional[str]) -> Optional[str]:
        return _valid_iso_timestamp(value)

    @model_validator(mode="after")
    def enough_lid_anchors(self) -> "CorrectionRecord":
        anchors = [point for point in self.manual_highs if point.role != "breakout_peak"]
        if len(anchors) < 2:
            raise ValueError("authoritative reviews require at least two non-breakout anchors")
        return self


class ReviewDecisionRecord(BaseModel):
    """One reviewer decision: sign off on the algorithm or correct it.

    ``reviewed_highs`` may carry explicit ``lidMember`` flags — the two line
    anchors chosen in the review UI. Legacy corrections without membership
    keep fitting all eligible (non-breakout) points downstream.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_version: Literal[3, 4] = Field(default=3, alias="schemaVersion")
    label_policy_version: Optional[Literal[1]] = Field(
        default=None, alias="labelPolicyVersion"
    )
    session_id: Optional[int] = Field(default=None, alias="sessionId")
    ticker: str
    interval: Literal["3M"] = AUTHORITATIVE_INTERVAL
    as_of: Optional[str] = Field(default=None, alias="asOf")
    algorithm_version: Optional[str] = Field(default=None, alias="algorithmVersion")
    decision: Literal["approved", "corrected"]
    coil_label: Optional[Literal["coil", "not_coil", "uncertain"]] = Field(
        default=None, alias="coilLabel"
    )
    human_grade: Optional[Literal["A", "B", "C"]] = Field(
        default=None, alias="humanGrade"
    )
    confidence: Optional[Literal["high", "low"]] = None
    note: Optional[str] = Field(default=None, max_length=2000)
    algorithm: Optional[dict[str, Any]] = None
    reviewed_highs: list[ReviewedPoint] = Field(default_factory=list, alias="reviewedHighs")
    created_at: Optional[str] = Field(default=None, alias="createdAt")

    @field_validator("ticker")
    @classmethod
    def valid_ticker(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("Review decision must include a ticker.")
        return symbol

    @field_validator("as_of")
    @classmethod
    def valid_as_of(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        try:
            parsed = date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("asOf must be ISO YYYY-MM-DD") from exc
        if parsed.isoformat() != value:
            raise ValueError("asOf must be ISO YYYY-MM-DD")
        return value

    @field_validator("created_at")
    @classmethod
    def valid_created_at(cls, value: Optional[str]) -> Optional[str]:
        return _valid_iso_timestamp(value)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def valid_correction_anchors(self) -> "ReviewDecisionRecord":
        if self.schema_version == 4:
            if self.label_policy_version != 1:
                raise ValueError("schema-4 reviews require labelPolicyVersion 1")
            if self.coil_label is None:
                raise ValueError("schema-4 reviews require a coilLabel")
            if self.coil_label == "coil" and self.human_grade is None:
                raise ValueError("coil labels require a humanGrade")
            if self.coil_label != "coil" and self.human_grade is not None:
                raise ValueError("humanGrade only applies when coilLabel is 'coil'")
            if (
                self.decision == DECISION_CORRECTED
                or self.coil_label == "not_coil"
            ) and self.note is None:
                raise ValueError(
                    "corrected tops and not-coil labels require a rationale note"
                )
        if self.decision != DECISION_CORRECTED:
            if self.reviewed_highs:
                raise ValueError("approved reviews cannot include reviewed highs")
            return self
        seen: set[str] = set()
        for point in self.reviewed_highs:
            if point.date in seen:
                raise ValueError(f"duplicate reviewed high for {point.date}")
            seen.add(point.date)
        members = [p for p in self.reviewed_highs if p.lid_member]
        if members:
            if any(p.role == "breakout_peak" for p in members):
                raise ValueError("breakout peaks cannot be lid anchors")
            if len(members) < 2:
                raise ValueError("corrected reviews require at least two lid members")
        else:
            anchors = [p for p in self.reviewed_highs if p.role != "breakout_peak"]
            if len(anchors) < 2:
                raise ValueError(
                    "authoritative reviews require at least two non-breakout anchors"
                )
        return self


def validate_correction_record(record: dict[str, Any]) -> CorrectionRecord:
    """Validate a raw frontend record and expose one stable ValueError API."""
    try:
        return CorrectionRecord.model_validate(record)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def validate_decision_record(record: dict[str, Any]) -> ReviewDecisionRecord:
    """Validate a raw decision payload and expose one stable ValueError API."""
    try:
        return ReviewDecisionRecord.model_validate(record)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def _authoritative_interval(interval: str) -> str:
    value = str(interval or AUTHORITATIVE_INTERVAL).strip().upper()
    if value != AUTHORITATIVE_INTERVAL:
        raise ValueError("only authoritative 3M reviews are supported")
    return value


def _parse_iso_month(value: Any) -> Optional[date]:
    """Lenient ISO-date parse: shape is enforced by the point validators."""
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _calendar_quarter(moment: date) -> tuple[int, int]:
    """(year, quarter), matching the analyzer's quarterly aggregation key."""
    return moment.year, (moment.month - 1) // 3 + 1


def incomplete_final_quarter(
    last_bar_date: Optional[str], *, today: Optional[date] = None
) -> Optional[tuple[int, int]]:
    """Calendar (year, quarter) of a trailing incomplete quarter, else ``None``.

    Mirrors ``coil_analysis._quarter_is_complete``: the last quarter of a
    monthly series is complete only once its calendar-final monthly candle has
    actually closed. Reaching March / June / September / December is not
    enough while that same calendar month is still in progress. An unknown or
    unparseable data date yields ``None`` — there is nothing to measure against.
    """
    parsed = _parse_iso_month(last_bar_date)
    if parsed is None:
        return None
    live_date = today or date.today()
    month_is_still_open = (parsed.year, parsed.month) >= (
        live_date.year,
        live_date.month,
    )
    if parsed.month % 3 == 0 and not month_is_still_open:
        return None
    return _calendar_quarter(parsed)


def reject_incomplete_quarter_points(
    dates: Iterable[Any], *, last_bar_date: Optional[str]
) -> None:
    """Reviewers may not anchor inside the incomplete final quarter (v2.2).

    Human reviews stay authoritative over *which* zone or level they anchor
    to — no zone eligibility is applied here, because overriding the zone
    choice is the point of the review workspace. The one restriction
    reviewers share with the algorithm is the structure invariant: nothing in
    the unfinished quarter is structure, because its high still moves with
    live price. ``coil_analysis._structure_from_review_override`` drops such
    points silently, so the request is refused here instead and the reviewer
    is told which date is not reviewable yet.
    """
    quarter = incomplete_final_quarter(last_bar_date)
    if quarter is None:
        return
    offending = [
        str(raw)
        for raw in dates
        if (parsed := _parse_iso_month(raw)) is not None
        and _calendar_quarter(parsed) == quarter
    ]
    if offending:
        year, index = quarter
        raise ValueError(
            f"reviewed points cannot fall in the incomplete final quarter "
            f"{year}Q{index} (data through {last_bar_date}): "
            f"{', '.join(offending)}"
        )

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS high_reviews (
        id {pk},
        ticker TEXT NOT NULL,
        interval TEXT NOT NULL,
        created_at TEXT NOT NULL,
        record TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS high_overrides (
        ticker TEXT NOT NULL,
        interval TEXT NOT NULL,
        review_id INTEGER NOT NULL,
        points TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (ticker, interval)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_state (
        ticker TEXT NOT NULL,
        interval TEXT NOT NULL,
        review_id INTEGER NOT NULL,
        decision TEXT NOT NULL,
        as_of TEXT,
        algorithm_version TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (ticker, interval)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_sessions (
        id {pk},
        fingerprint TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL,
        snapshot TEXT NOT NULL
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS review_sessions_fingerprint
        ON review_sessions (fingerprint)
    """,
    """
    CREATE TABLE IF NOT EXISTS review_session_items (
        session_id INTEGER NOT NULL,
        position INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        snapshot TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (session_id, ticker)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_capture_idempotency (
        session_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        review_id INTEGER NOT NULL,
        event_id TEXT NOT NULL,
        response TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (session_id, ticker, idempotency_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_candidate_nominations (
        session_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        universe TEXT NOT NULL,
        rationale TEXT NOT NULL,
        history_as_of TEXT,
        bars_hash TEXT,
        revision INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (session_id, ticker)
    )
    """,
]

# Pre-session deployments created high_overrides without decision provenance.
_MIGRATION_COLUMNS = {
    "high_overrides": [("as_of", "TEXT"), ("algorithm_version", "TEXT")],
    "review_state": [
        ("coil_label", "TEXT"),
        ("human_grade", "TEXT"),
        ("confidence", "TEXT"),
        ("note", "TEXT"),
        ("label_policy_version", "INTEGER"),
        ("event_id", "TEXT"),
    ],
    "review_sessions": [
        ("reviewer_name", "TEXT"),
        ("capability_token_hash", "TEXT"),
        ("require_fresh_review", "INTEGER NOT NULL DEFAULT 0"),
        ("finalized_at", "TEXT"),
        ("final_export", "TEXT"),
        ("final_export_hash", "TEXT"),
        ("token_revision", "INTEGER NOT NULL DEFAULT 1"),
        ("token_rotated_at", "TEXT"),
        ("token_revoked_at", "TEXT"),
    ],
    "review_session_items": [
        ("draft", "TEXT"),
        ("draft_revision", "INTEGER NOT NULL DEFAULT 0"),
        ("draft_updated_at", "TEXT"),
        ("completed_review_id", "INTEGER"),
        ("completed_event_id", "TEXT"),
        ("sample_id", "TEXT"),
        ("bars_hash", "TEXT"),
        ("reviewable", "INTEGER NOT NULL DEFAULT 1"),
        ("skip_reason", "TEXT"),
        ("completed_at", "TEXT"),
        ("base_classification", "TEXT"),
        ("base_classification_locked_at", "TEXT"),
    ],
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _legacy_event_id(
    ticker: str,
    interval: str,
    created_at: str,
    record: dict[str, Any],
) -> str:
    """Content-derived identity for events created before server UUIDs existed."""
    basis = json.dumps(
        {
            "ticker": ticker,
            "interval": interval,
            "created_at": created_at,
            "record": record,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"legacy-sha256:{hashlib.sha256(basis.encode('utf-8')).hexdigest()}"


def _point_payload(point: ReviewedPoint) -> dict[str, Any]:
    payload: dict[str, Any] = {"date": point.date, "price": point.price}
    if point.role != "major_top":
        payload["role"] = point.role
    if point.lid_member is not None:
        payload["lid_member"] = bool(point.lid_member)
    return payload


def _points_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The effective reviewed points: the user's manual highs, with roles.

    ``manualHighs`` is the post-review truth (kept + added highs). Each point
    keeps an optional ``role`` (defaulting to ``major_top`` downstream) so a
    reviewer can mark structural retests and breakout peaks explicitly.
    """
    validated = validate_correction_record(record)
    return [_point_payload(high) for high in validated.manual_highs]


def session_fingerprint(
    source: str,
    items: list[dict[str, Any]],
    *,
    snapshot: Optional[dict[str, Any]] = None,
    algorithm_version: Optional[str] = None,
) -> str:
    """Stable identity of one screened cohort and its model/config snapshot."""
    basis = {
        "source": source,
        "algorithm_version": algorithm_version,
        "snapshot": snapshot or {},
        "items": [
            [
                str(item.get("ticker", "")).strip().upper(),
                item.get("snapshot") or {},
            ]
            for item in items
        ],
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode("utf-8")).hexdigest()


def decision_is_stale(
    state: Optional[dict[str, Any]],
    *,
    current_as_of: Optional[str],
    current_algorithm_version: Optional[str],
) -> bool:
    """A decision is stale when newer data or a newer algorithm has arrived.

    Legacy decisions without provenance (``as_of``/``algorithm_version`` both
    unset) never flag stale — they keep their pre-session behavior.
    """
    if not state:
        return False
    as_of = state.get("as_of")
    algo = state.get("algorithm_version")
    if as_of and current_as_of and str(current_as_of) > str(as_of):
        return True
    if algo and current_algorithm_version and str(current_algorithm_version) != str(algo):
        return True
    return False


def annotate_review(
    analysis: dict[str, Any],
    state: Optional[dict[str, Any]],
    *,
    algorithm_version: Optional[str] = None,
) -> None:
    """Attach the latest decision + staleness to an analysis ``review`` block.

    ``analyze_coil`` only sees corrected overrides; approvals leave the
    algorithm effective and are known solely to ``review_state``. Staleness
    compares the decision's provenance against the analysis's own data date
    and algorithm version, so stale corrected overrides stay effective but
    are flagged for re-review.
    """
    if not state:
        return
    review = analysis.setdefault("review", {})
    stale = decision_is_stale(
        state,
        current_as_of=analysis.get("as_of"),
        current_algorithm_version=algorithm_version or analysis.get("algorithm_version"),
    )
    review.update(
        {
            "reviewed": True,
            "decision": state["decision"],
            "review_id": state["review_id"],
            "review_as_of": state["as_of"],
            "review_algorithm_version": state["algorithm_version"],
            "coil_label": state.get("coil_label"),
            "human_grade": state.get("human_grade"),
            "confidence": state.get("confidence"),
            "note": state.get("note"),
            "label_policy_version": state.get("label_policy_version"),
            "event_id": state.get("event_id"),
            "stale": stale,
        }
    )
    review.setdefault("effective", "algorithm")


class ReviewStore:
    """Append-only review log + current-override table over SQLite/Postgres."""

    def __init__(self, database_url: Optional[str] = None, sqlite_path: Optional[Path] = None):
        self._database_url = database_url
        self._sqlite_path = (
            Path(sqlite_path) if sqlite_path else _default_sqlite_path()
        )
        self._is_postgres = bool(database_url)
        self._ph = "%s" if self._is_postgres else "?"
        self._ensure_schema()

    def _connect(self):
        if self._is_postgres:
            import psycopg2  # deferred: only needed when DATABASE_URL is set

            return psycopg2.connect(self._database_url)
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self._sqlite_path, timeout=30.0)

    def _begin_protected_write(self, conn) -> None:
        """Serialize SQLite protected-session writes before read/compare/update."""
        if not self._is_postgres:
            conn.execute("BEGIN IMMEDIATE")

    @property
    def is_postgres(self) -> bool:
        return self._is_postgres

    @property
    def sqlite_path(self) -> Optional[Path]:
        return None if self._is_postgres else self._sqlite_path

    def persistence_probe(self) -> None:
        """Verify the configured review store can be opened and queried."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM review_sessions")
            cursor.fetchone()

    def backup_sqlite(self, destination: Path) -> None:
        """Create a transactionally consistent SQLite backup, including WAL state."""
        if self._is_postgres:
            raise ValueError("SQLite backup is unavailable for PostgreSQL stores")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as source, sqlite3.connect(target) as backup:
            source.backup(backup)

    def _ensure_schema(self) -> None:
        pk = "SERIAL PRIMARY KEY" if self._is_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
        with self._connect() as conn:
            cursor = conn.cursor()
            for statement in _DDL:
                cursor.execute(statement.format(pk=pk))
            for table, columns in _MIGRATION_COLUMNS.items():
                self._ensure_columns(cursor, table, columns)
            conn.commit()

    def _ensure_columns(self, cursor, table: str, columns: list[tuple[str, str]]) -> None:
        if self._is_postgres:
            for name, sql_type in columns:
                cursor.execute(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {sql_type}"
                )
            return
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        for name, sql_type in columns:
            if name not in existing:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")

    # ------------------------------------------------------------------ #
    # Append-only log helpers
    # ------------------------------------------------------------------ #
    def _insert_review_row(
        self, cursor, ticker: str, interval: str, created_at: str, record: dict[str, Any]
    ) -> int:
        if self._is_postgres:
            cursor.execute(
                f"INSERT INTO high_reviews (ticker, interval, created_at, record) "
                f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}) RETURNING id",
                (ticker, interval, created_at, json.dumps(record)),
            )
            return int(cursor.fetchone()[0])
        cursor.execute(
            f"INSERT INTO high_reviews (ticker, interval, created_at, record) "
            f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph})",
            (ticker, interval, created_at, json.dumps(record)),
        )
        return int(cursor.lastrowid)

    def _upsert_override(
        self,
        cursor,
        ticker: str,
        interval: str,
        review_id: int,
        points: list[dict[str, Any]],
        *,
        as_of: Optional[str],
        algorithm_version: Optional[str],
    ) -> str:
        updated_at = _utcnow()
        cursor.execute(
            f"DELETE FROM high_overrides WHERE ticker = {self._ph} AND interval = {self._ph}",
            (ticker, interval),
        )
        cursor.execute(
            f"INSERT INTO high_overrides "
            f"(ticker, interval, review_id, points, updated_at, as_of, algorithm_version) "
            f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph})",
            (ticker, interval, review_id, json.dumps(points), updated_at, as_of, algorithm_version),
        )
        return updated_at

    def _upsert_state(
        self,
        cursor,
        ticker: str,
        interval: str,
        review_id: int,
        decision: str,
        *,
        as_of: Optional[str],
        algorithm_version: Optional[str],
        coil_label: Optional[str] = None,
        human_grade: Optional[str] = None,
        confidence: Optional[str] = None,
        note: Optional[str] = None,
        label_policy_version: Optional[int] = None,
        event_id: Optional[str] = None,
    ) -> str:
        updated_at = _utcnow()
        cursor.execute(
            f"DELETE FROM review_state WHERE ticker = {self._ph} AND interval = {self._ph}",
            (ticker, interval),
        )
        cursor.execute(
            f"INSERT INTO review_state "
            f"(ticker, interval, review_id, decision, as_of, algorithm_version, "
            f"coil_label, human_grade, confidence, note, label_policy_version, "
            f"event_id, updated_at) "
            f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, "
            f"{self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, "
            f"{self._ph}, {self._ph}, {self._ph})",
            (
                ticker,
                interval,
                review_id,
                decision,
                as_of,
                algorithm_version,
                coil_label,
                human_grade,
                confidence,
                note,
                label_policy_version,
                event_id,
                updated_at,
            ),
        )
        return updated_at

    def append_review(self, record: dict[str, Any]) -> dict[str, Any]:
        """Persist one legacy correction record and make its points effective.

        Schema-v1 corrections translate into ``corrected`` review decisions:
        the verbatim record stays in the append-only log while ``review_state``
        gains a corrected entry (without data/algorithm provenance).
        """
        validated = validate_correction_record(record)
        ticker = validated.ticker
        interval = _authoritative_interval(validated.interval)
        created_at = str(validated.created_at or _utcnow())
        points = _points_from_record(record)

        with self._connect() as conn:
            cursor = conn.cursor()
            review_id = self._insert_review_row(cursor, ticker, interval, created_at, record)
            self._upsert_override(
                cursor, ticker, interval, review_id, points,
                as_of=None, algorithm_version=None,
            )
            self._upsert_state(
                cursor, ticker, interval, review_id, DECISION_CORRECTED,
                as_of=None, algorithm_version=None,
            )
            conn.commit()
        return {
            "id": review_id,
            "ticker": ticker,
            "interval": interval,
            "created_at": created_at,
            "points": points,
        }

    def record_decision(self, record: dict[str, Any]) -> dict[str, Any]:
        """Persist one reviewer decision and update the effective state.

        ``approved`` records the sign-off and clears any older human override
        so the algorithm becomes effective. ``corrected`` upserts the override
        so the reviewed points become the live analysis immediately. Both are
        append-only in ``high_reviews``.
        """
        validated = validate_decision_record(record)
        ticker = validated.ticker
        interval = _authoritative_interval(validated.interval)
        created_at = str(validated.created_at or _utcnow())
        override: Optional[dict[str, Any]] = None
        event_id = str(uuid.uuid4())
        stored_record = dict(record)
        stored_record["eventId"] = event_id

        with self._connect() as conn:
            cursor = conn.cursor()
            if validated.session_id is not None:
                cursor.execute(
                    f"SELECT 1 FROM review_session_items "
                    f"WHERE session_id = {self._ph} AND ticker = {self._ph}",
                    (validated.session_id, ticker),
                )
                if cursor.fetchone() is None:
                    raise ValueError(
                        "sessionId must reference a session containing this ticker"
                    )
            review_id = self._insert_review_row(
                cursor, ticker, interval, created_at, stored_record
            )
            self._upsert_state(
                cursor, ticker, interval, review_id, validated.decision,
                as_of=validated.as_of, algorithm_version=validated.algorithm_version,
                coil_label=validated.coil_label,
                human_grade=validated.human_grade,
                confidence=validated.confidence,
                note=validated.note,
                label_policy_version=validated.label_policy_version,
                event_id=event_id,
            )
            if validated.decision == DECISION_CORRECTED:
                points = [_point_payload(p) for p in validated.reviewed_highs]
                updated_at = self._upsert_override(
                    cursor, ticker, interval, review_id, points,
                    as_of=validated.as_of, algorithm_version=validated.algorithm_version,
                )
                override = {
                    "review_id": review_id,
                    "points": points,
                    "updated_at": updated_at,
                    "as_of": validated.as_of,
                    "algorithm_version": validated.algorithm_version,
                }
            else:
                cursor.execute(
                    f"DELETE FROM high_overrides "
                    f"WHERE ticker = {self._ph} AND interval = {self._ph}",
                    (ticker, interval),
                )
            conn.commit()
        return {
            "review": {
                "id": review_id,
                "ticker": ticker,
                "interval": interval,
                "decision": validated.decision,
                "session_id": validated.session_id,
                "as_of": validated.as_of,
                "algorithm_version": validated.algorithm_version,
                "event_id": event_id,
                "coil_label": validated.coil_label,
                "human_grade": validated.human_grade,
                "confidence": validated.confidence,
                "note": validated.note,
                "created_at": created_at,
            },
            "override": override,
        }

    def revoke_override(
        self, ticker: str, interval: str = AUTHORITATIVE_INTERVAL
    ) -> Optional[dict[str, Any]]:
        """Revoke the current override while retaining an append-only event.

        The review row and override deletion share one transaction. A repeated
        revoke is idempotent at the store boundary and returns ``None``.
        Revocation clears ``review_state`` too: the ticker is unreviewed again.
        """
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("Revocation must include a ticker.")
        review_interval = _authoritative_interval(interval)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT review_id FROM high_overrides "
                f"WHERE ticker = {self._ph} AND interval = {self._ph}",
                (symbol, review_interval),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            revoked_review_id = int(row[0])
            created_at = _utcnow()
            event = {
                "schemaVersion": 2,
                "action": "revoke",
                "ticker": symbol,
                "interval": review_interval,
                "createdAt": created_at,
                "revokedReviewId": revoked_review_id,
            }
            revocation_id = self._insert_review_row(
                cursor, symbol, review_interval, created_at, event
            )
            cursor.execute(
                f"DELETE FROM high_overrides "
                f"WHERE ticker = {self._ph} AND interval = {self._ph}",
                (symbol, review_interval),
            )
            cursor.execute(
                f"DELETE FROM review_state "
                f"WHERE ticker = {self._ph} AND interval = {self._ph}",
                (symbol, review_interval),
            )
            conn.commit()
        return {
            "id": revocation_id,
            "ticker": symbol,
            "interval": review_interval,
            "created_at": created_at,
            "revoked_review_id": revoked_review_id,
        }

    def get_override(self, ticker: str, interval: str = "3M") -> Optional[dict[str, Any]]:
        """Current effective reviewed points for a ticker, or None."""
        symbol = ticker.strip().upper()
        interval = _authoritative_interval(interval)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT review_id, points, updated_at, as_of, algorithm_version "
                f"FROM high_overrides "
                f"WHERE ticker = {self._ph} AND interval = {self._ph}",
                (symbol, interval),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "review_id": int(row[0]),
            "points": json.loads(row[1]),
            "updated_at": str(row[2]),
            "as_of": row[3],
            "algorithm_version": row[4],
        }

    def get_review_state(self, ticker: str, interval: str = "3M") -> Optional[dict[str, Any]]:
        """Latest decision (approval or correction) for a ticker, or None."""
        symbol = ticker.strip().upper()
        interval = _authoritative_interval(interval)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT review_id, decision, as_of, algorithm_version, updated_at, "
                f"coil_label, human_grade, confidence, note, label_policy_version, event_id "
                f"FROM review_state "
                f"WHERE ticker = {self._ph} AND interval = {self._ph}",
                (symbol, interval),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "review_id": int(row[0]),
            "decision": str(row[1]),
            "as_of": row[2],
            "algorithm_version": row[3],
            "updated_at": str(row[4]),
            "coil_label": row[5],
            "human_grade": row[6],
            "confidence": row[7],
            "note": row[8],
            "label_policy_version": row[9],
            "event_id": row[10],
        }

    def list_reviews(
        self,
        ticker: str,
        interval: str = "3M",
        *,
        include_capture_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Full append-only review history for a ticker, oldest first."""
        symbol = ticker.strip().upper()
        interval = _authoritative_interval(interval)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, created_at, record FROM high_reviews "
                f"WHERE ticker = {self._ph} AND interval = {self._ph} ORDER BY id",
                (symbol, interval),
            )
            rows = cursor.fetchall()
        history: list[dict[str, Any]] = []
        for row in rows:
            record = json.loads(row[2])
            if not include_capture_only and bool(record.get("captureOnly")):
                continue
            history.append(
                {
                    "id": int(row[0]),
                    "created_at": str(row[1]),
                    "record": record,
                }
            )
        return history

    # ------------------------------------------------------------------ #
    # Review sessions
    # ------------------------------------------------------------------ #
    def create_session(
        self,
        source: str,
        items: list[dict[str, Any]],
        *,
        snapshot: Optional[dict[str, Any]] = None,
        algorithm_version: Optional[str] = None,
        reviewer_name: Optional[str] = None,
        capability_token_hash: Optional[str] = None,
        require_fresh_review: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        """Create a session for one screener snapshot, or resume the existing one.

        ``items`` is the backend-ranked result order: ``[{"ticker", "snapshot"}]``.
        Identity is the snapshot fingerprint, so re-opening the same screen
        resumes the same queue while new data or a re-run starts a new session.
        """
        name = str(source or "").strip()
        if not name:
            raise ValueError("Review sessions require a source.")
        reviewer = str(reviewer_name or "").strip() or None
        if require_fresh_review and reviewer is None:
            raise ValueError("fresh review sessions require an assigned reviewer")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            ticker = str(item.get("ticker", "")).strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            normalized.append(
                {
                    "ticker": ticker,
                    "snapshot": item.get("snapshot") or {},
                    "sample_id": item.get("sample_id"),
                    "bars_hash": item.get("bars_hash"),
                    "reviewable": bool(item.get("reviewable", True)),
                }
            )
        if not normalized:
            raise ValueError("Review sessions require at least one ticker.")

        session_snapshot_data = dict(snapshot or {})
        if require_fresh_review or reviewer is not None:
            session_snapshot_data["_review_policy"] = {
                "require_fresh_review": bool(require_fresh_review),
                "reviewer_name": reviewer,
            }
        fingerprint_snapshot = dict(session_snapshot_data)
        if require_fresh_review:
            fingerprint_snapshot["_capability_identity"] = capability_token_hash
        fingerprint = session_fingerprint(
            name,
            normalized,
            snapshot=fingerprint_snapshot,
            algorithm_version=algorithm_version,
        )
        with self._connect() as conn:
            self._begin_protected_write(conn)
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, capability_token_hash, reviewer_name, "
                f"require_fresh_review FROM review_sessions "
                f"WHERE fingerprint = {self._ph}",
                (fingerprint,),
            )
            row = cursor.fetchone()
            if row is not None:
                session_id = int(row[0])
                stored_hash = row[1]
                if stored_hash and not (
                    capability_token_hash
                    and hmac.compare_digest(
                        str(stored_hash), str(capability_token_hash)
                    )
                ):
                    raise ReviewAccessError("review session access denied")
                created = False
            else:
                created_at = _utcnow()
                session_snapshot = _canonical_json(session_snapshot_data)
                if self._is_postgres:
                    cursor.execute(
                        f"INSERT INTO review_sessions "
                        f"(fingerprint, source, created_at, snapshot, reviewer_name, "
                        f"capability_token_hash, require_fresh_review) "
                        f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, "
                        f"{self._ph}, {self._ph}, {self._ph}) "
                        f"ON CONFLICT (fingerprint) DO NOTHING RETURNING id",
                        (
                            fingerprint,
                            name,
                            created_at,
                            session_snapshot,
                            reviewer,
                            capability_token_hash,
                            int(require_fresh_review),
                        ),
                    )
                    inserted = cursor.fetchone()
                    if inserted is None:
                        cursor.execute(
                            f"SELECT id, capability_token_hash, reviewer_name, "
                            f"require_fresh_review FROM review_sessions "
                            f"WHERE fingerprint = {self._ph}",
                            (fingerprint,),
                        )
                        winning = cursor.fetchone()
                        if winning is None:
                            raise ReviewConflictError(
                                "session creation conflict could not be resolved"
                            )
                        session_id = int(winning[0])
                        stored_hash = winning[1]
                        if stored_hash and not (
                            capability_token_hash
                            and hmac.compare_digest(
                                str(stored_hash), str(capability_token_hash)
                            )
                        ):
                            raise ReviewAccessError("review session access denied")
                        created = False
                    else:
                        session_id = int(inserted[0])
                        created = True
                else:
                    cursor.execute(
                        f"INSERT INTO review_sessions "
                        f"(fingerprint, source, created_at, snapshot, reviewer_name, "
                        f"capability_token_hash, require_fresh_review) "
                        f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, "
                        f"{self._ph}, {self._ph}, {self._ph})",
                        (
                            fingerprint,
                            name,
                            created_at,
                            session_snapshot,
                            reviewer,
                            capability_token_hash,
                            int(require_fresh_review),
                        ),
                    )
                    session_id = int(cursor.lastrowid)
                    created = True
                if created:
                    for position, item in enumerate(normalized):
                        cursor.execute(
                            f"INSERT INTO review_session_items "
                            f"(session_id, position, ticker, status, snapshot, updated_at, "
                            f"sample_id, bars_hash, reviewable) "
                            f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, "
                            f"{self._ph}, {self._ph}, {self._ph}, {self._ph}, "
                            f"{self._ph})",
                            (
                                session_id,
                                position,
                                item["ticker"],
                                ITEM_STATUS_PENDING,
                                _canonical_json(item["snapshot"]),
                                created_at,
                                item["sample_id"],
                                item["bars_hash"],
                                int(item["reviewable"]),
                            ),
                        )
            conn.commit()
        session = self.get_session(session_id, algorithm_version=algorithm_version)
        assert session is not None
        return session, created

    def get_session_security(self, session_id: int) -> Optional[dict[str, Any]]:
        """Return non-secret access metadata for one session."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, source, reviewer_name, capability_token_hash, "
                f"require_fresh_review, finalized_at, token_revision, "
                f"token_revoked_at "
                f"FROM review_sessions WHERE id = {self._ph}",
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": int(row[0]),
            "source": str(row[1]),
            "reviewer_name": row[2],
            "protected": bool(row[3]),
            "require_fresh_review": bool(row[4]),
            "finalized_at": row[5],
            "token_revision": int(row[6] or 1),
            "token_revoked_at": row[7],
        }

    def get_session_snapshot(self, session_id: int) -> Optional[dict[str, Any]]:
        """Return the server-owned session snapshot for internal analysis wiring.

        Fresh-session API responses intentionally expose only a redacted view of
        this object. Detector configuration must nevertheless be loaded from the
        immutable persisted snapshot rather than silently falling back to a
        process default.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT snapshot FROM review_sessions WHERE id = {self._ph}",
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0]) if row[0] else {}

    def authorize_session(
        self, session_id: int, access_token: Optional[str]
    ) -> Optional[dict[str, Any]]:
        """Object-level authorization for protected review sessions."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, source, reviewer_name, capability_token_hash, "
                f"require_fresh_review, finalized_at, token_revision, "
                f"token_revoked_at "
                f"FROM review_sessions WHERE id = {self._ph}",
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        stored_hash = row[3]
        if row[7] is not None:
            raise ReviewAccessError("review session access denied")
        if stored_hash:
            supplied_hash = (
                hash_review_token(access_token) if access_token else ""
            )
            if not supplied_hash or not hmac.compare_digest(
                str(stored_hash), supplied_hash
            ):
                raise ReviewAccessError("review session access denied")
        return {
            "id": int(row[0]),
            "source": str(row[1]),
            "reviewer_name": row[2],
            "protected": bool(stored_hash),
            "require_fresh_review": bool(row[4]),
            "finalized_at": row[5],
            "token_revision": int(row[6] or 1),
            "token_revoked_at": row[7],
        }

    def list_candidate_nominations(
        self, session_id: int
    ) -> list[dict[str, Any]]:
        """Return the broader-universe stocks nominated outside the frozen queue."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT ticker, universe, rationale, history_as_of, bars_hash, "
                f"revision, created_at, updated_at "
                f"FROM review_candidate_nominations "
                f"WHERE session_id = {self._ph} ORDER BY created_at, ticker",
                (session_id,),
            )
            rows = cursor.fetchall()
        return [
            {
                "ticker": str(row[0]),
                "universe": str(row[1]),
                "rationale": str(row[2]),
                "history_as_of": row[3],
                "bars_hash": row[4],
                "revision": int(row[5]),
                "created_at": str(row[6]),
                "updated_at": str(row[7]),
            }
            for row in rows
        ]

    def save_candidate_nomination(
        self,
        session_id: int,
        ticker: str,
        *,
        universe: str,
        rationale: str,
        history_as_of: Optional[str],
        bars_hash: Optional[str],
        expected_revision: int,
    ) -> Optional[dict[str, Any]]:
        """Create or revise a missed-candidate nomination without changing the queue."""
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("candidate ticker is required")
        if universe not in {"sp500", "international"}:
            raise ValueError("candidate universe is not supported")
        note = rationale.strip()
        if len(note) > 2000:
            raise ValueError("candidate rationale exceeds 2000 characters")
        if expected_revision < 0:
            raise ValueError("expected revision must be non-negative")

        with self._connect() as conn:
            self._begin_protected_write(conn)
            cursor = conn.cursor()
            suffix = " FOR UPDATE" if self._is_postgres else ""
            cursor.execute(
                f"SELECT require_fresh_review, finalized_at "
                f"FROM review_sessions WHERE id = {self._ph}{suffix}",
                (session_id,),
            )
            session_row = cursor.fetchone()
            if session_row is None:
                return None
            if not bool(session_row[0]):
                raise ValueError(
                    "candidate nominations require a fresh-review session"
                )
            if session_row[1] is not None:
                raise ReviewConflictError("finalized sessions are immutable")
            cursor.execute(
                f"SELECT 1 FROM review_session_items "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph}",
                (session_id, symbol),
            )
            if cursor.fetchone() is not None:
                raise ValueError(
                    "screened stocks must stay in the frozen review queue"
                )
            cursor.execute(
                f"SELECT revision, created_at "
                f"FROM review_candidate_nominations "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph}{suffix}",
                (session_id, symbol),
            )
            existing = cursor.fetchone()
            now = _utcnow()
            if existing is None:
                if expected_revision != 0:
                    raise ReviewConflictError(
                        "candidate revision conflict: current revision is 0"
                    )
                revision = 1
                cursor.execute(
                    f"INSERT INTO review_candidate_nominations "
                    f"(session_id, ticker, universe, rationale, history_as_of, "
                    f"bars_hash, revision, created_at, updated_at) "
                    f"VALUES ({', '.join([self._ph] * 9)})",
                    (
                        session_id,
                        symbol,
                        universe,
                        note,
                        history_as_of,
                        bars_hash,
                        revision,
                        now,
                        now,
                    ),
                )
            else:
                current_revision = int(existing[0])
                if current_revision != expected_revision:
                    raise ReviewConflictError(
                        "candidate revision conflict: "
                        f"current revision is {current_revision}"
                    )
                revision = current_revision + 1
                cursor.execute(
                    f"UPDATE review_candidate_nominations SET "
                    f"universe = {self._ph}, rationale = {self._ph}, "
                    f"history_as_of = {self._ph}, bars_hash = {self._ph}, "
                    f"revision = {self._ph}, updated_at = {self._ph} "
                    f"WHERE session_id = {self._ph} AND ticker = {self._ph} "
                    f"AND revision = {self._ph}",
                    (
                        universe,
                        note,
                        history_as_of,
                        bars_hash,
                        revision,
                        now,
                        session_id,
                        symbol,
                        current_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ReviewConflictError(
                        "candidate nomination lost a revision race"
                    )
            conn.commit()
        return next(
            (
                entry
                for entry in self.list_candidate_nominations(session_id)
                if entry["ticker"] == symbol
            ),
            None,
        )

    def delete_candidate_nomination(
        self,
        session_id: int,
        ticker: str,
        *,
        expected_revision: int,
    ) -> Optional[bool]:
        """Remove a nomination before finalization with optimistic concurrency."""
        symbol = ticker.strip().upper()
        with self._connect() as conn:
            self._begin_protected_write(conn)
            cursor = conn.cursor()
            suffix = " FOR UPDATE" if self._is_postgres else ""
            cursor.execute(
                f"SELECT require_fresh_review, finalized_at "
                f"FROM review_sessions WHERE id = {self._ph}{suffix}",
                (session_id,),
            )
            session_row = cursor.fetchone()
            if session_row is None:
                return None
            if not bool(session_row[0]):
                raise ValueError(
                    "candidate nominations require a fresh-review session"
                )
            if session_row[1] is not None:
                raise ReviewConflictError("finalized sessions are immutable")
            cursor.execute(
                f"SELECT revision FROM review_candidate_nominations "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph}{suffix}",
                (session_id, symbol),
            )
            row = cursor.fetchone()
            if row is None:
                return False
            current_revision = int(row[0])
            if current_revision != expected_revision:
                raise ReviewConflictError(
                    "candidate revision conflict: "
                    f"current revision is {current_revision}"
                )
            cursor.execute(
                f"DELETE FROM review_candidate_nominations "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph} "
                f"AND revision = {self._ph}",
                (session_id, symbol, current_revision),
            )
            if cursor.rowcount != 1:
                raise ReviewConflictError(
                    "candidate nomination lost a revision race"
                )
            conn.commit()
        return True

    def rotate_session_token(
        self, session_id: int, token_hash: str
    ) -> Optional[dict[str, Any]]:
        """Replace one fresh session capability and invalidate the old token."""
        if not token_hash:
            raise ValueError("new token hash is required")
        with self._connect() as conn:
            self._begin_protected_write(conn)
            cursor = conn.cursor()
            suffix = " FOR UPDATE" if self._is_postgres else ""
            cursor.execute(
                f"SELECT require_fresh_review, token_revision "
                f"FROM review_sessions WHERE id = {self._ph}{suffix}",
                (session_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            if not bool(row[0]):
                raise ValueError("token rotation requires a fresh-review session")
            revision = int(row[1] or 1) + 1
            rotated_at = _utcnow()
            cursor.execute(
                f"UPDATE review_sessions SET capability_token_hash = {self._ph}, "
                f"token_revision = {self._ph}, token_rotated_at = {self._ph}, "
                f"token_revoked_at = NULL WHERE id = {self._ph}",
                (token_hash, revision, rotated_at, session_id),
            )
            conn.commit()
        return {
            "session_id": session_id,
            "token_revision": revision,
            "rotated_at": rotated_at,
        }

    def revoke_session_token(self, session_id: int) -> Optional[dict[str, Any]]:
        """Immediately disable the current capability while retaining its hash."""
        with self._connect() as conn:
            self._begin_protected_write(conn)
            cursor = conn.cursor()
            suffix = " FOR UPDATE" if self._is_postgres else ""
            cursor.execute(
                f"SELECT require_fresh_review, token_revision "
                f"FROM review_sessions WHERE id = {self._ph}{suffix}",
                (session_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            if not bool(row[0]):
                raise ValueError("token revocation requires a fresh-review session")
            revision = int(row[1] or 1) + 1
            revoked_at = _utcnow()
            cursor.execute(
                f"UPDATE review_sessions SET token_revision = {self._ph}, "
                f"token_revoked_at = {self._ph} WHERE id = {self._ph}",
                (revision, revoked_at, session_id),
            )
            conn.commit()
        return {
            "session_id": session_id,
            "token_revision": revision,
            "revoked_at": revoked_at,
        }

    def _get_fresh_session(
        self, session_id: int, *, algorithm_version: Optional[str]
    ) -> Optional[dict[str, Any]]:
        """Session-owned view that deliberately ignores global review_state."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, fingerprint, source, created_at, snapshot, "
                f"reviewer_name, require_fresh_review, finalized_at, "
                f"final_export_hash, token_revision, token_rotated_at, "
                f"token_revoked_at "
                f"FROM review_sessions WHERE id = {self._ph}",
                (session_id,),
            )
            session_row = cursor.fetchone()
            if session_row is None or not bool(session_row[6]):
                return None
            cursor.execute(
                f"SELECT position, ticker, status, snapshot, updated_at, draft, "
                f"draft_revision, draft_updated_at, completed_review_id, "
                f"completed_event_id, sample_id, bars_hash, reviewable, "
                f"skip_reason, completed_at, base_classification, "
                f"base_classification_locked_at "
                f"FROM review_session_items WHERE session_id = {self._ph} "
                f"ORDER BY position",
                (session_id,),
            )
            item_rows = cursor.fetchall()
            review_ids = [
                int(row[8]) for row in item_rows if row[8] is not None
            ]
            records: dict[int, dict[str, Any]] = {}
            if review_ids:
                marks = ", ".join([self._ph] * len(review_ids))
                cursor.execute(
                    f"SELECT id, record FROM high_reviews WHERE id IN ({marks})",
                    tuple(review_ids),
                )
                records = {
                    int(row[0]): json.loads(row[1]) for row in cursor.fetchall()
                }
            cursor.execute(
                f"SELECT ticker, universe, rationale, history_as_of, bars_hash, "
                f"revision, created_at, updated_at "
                f"FROM review_candidate_nominations "
                f"WHERE session_id = {self._ph} ORDER BY created_at, ticker",
                (session_id,),
            )
            candidate_rows = cursor.fetchall()

        items: list[dict[str, Any]] = []
        counts = {
            ITEM_STATUS_PENDING: 0,
            ITEM_STATUS_REVIEWED: 0,
            ITEM_STATUS_SKIPPED: 0,
        }
        next_pending: Optional[str] = None
        for row in item_rows:
            (
                position,
                ticker,
                stored_status,
                item_snapshot,
                updated_at,
                raw_draft,
                draft_revision,
                draft_updated_at,
                completed_review_id,
                completed_event_id,
                sample_id,
                bars_hash,
                reviewable,
                skip_reason,
                completed_at,
                raw_base_classification,
                base_classification_locked_at,
            ) = row
            base_classification = (
                json.loads(raw_base_classification)
                if raw_base_classification
                else None
            )
            base_locked = bool(
                base_classification is not None
                and base_classification_locked_at is not None
            )
            full_snapshot = json.loads(item_snapshot) if item_snapshot else {}
            full_draft = json.loads(raw_draft) if raw_draft else None
            status = (
                ITEM_STATUS_REVIEWED
                if completed_review_id is not None
                else str(stored_status)
            )
            record = (
                records.get(int(completed_review_id), {})
                if completed_review_id is not None
                else {}
            )
            item: dict[str, Any] = {
                "ticker": str(ticker),
                "position": int(position),
                "status": status,
                "stored_status": str(stored_status),
                "snapshot": (
                    full_snapshot
                    if base_locked
                    else _redacted_prelock_snapshot(
                        full_snapshot, position=int(position)
                    )
                ),
                "draft": (
                    full_draft
                    if base_locked
                    else _redacted_prelock_draft(full_draft)
                ),
                "draft_revision": int(draft_revision or 0),
                "draft_updated_at": draft_updated_at,
                "sample_id": sample_id,
                "bars_hash": bars_hash,
                "reviewable": bool(reviewable),
                "skip_reason": skip_reason,
                "completed_at": completed_at,
                "base_classification_locked": base_locked,
                "base_classification_locked_at": base_classification_locked_at,
                "base_classification": base_classification,
                "updated_at": str(updated_at),
            }
            if base_locked:
                item.update(
                    {
                        "review_stale": False,
                        "review_id": (
                            int(completed_review_id)
                            if completed_review_id is not None
                            else None
                        ),
                        "review_decision": record.get("decision"),
                        "review_as_of": record.get("asOf"),
                        "review_algorithm_version": record.get(
                            "algorithmVersion"
                        ),
                        "review_coil_label": record.get("coilLabel"),
                        "review_human_grade": record.get("humanGrade"),
                        "review_confidence": record.get("confidence"),
                        "review_note": record.get("note"),
                        "review_label_policy_version": record.get(
                            "labelPolicyVersion"
                        ),
                        "review_event_id": completed_event_id,
                        "effective": "algorithm",
                    }
                )
            counts[status] = counts.get(status, 0) + 1
            if status == ITEM_STATUS_PENDING and next_pending is None:
                next_pending = str(ticker)
            items.append(item)

        raw_session_snapshot = (
            json.loads(session_row[4]) if session_row[4] else {}
        )
        manifest_snapshot = raw_session_snapshot.get("frozen_manifest")
        safe_manifest = (
            {
                key: manifest_snapshot.get(key)
                for key in (
                    "schema_version",
                    "kind",
                    "sha256",
                    "item_count",
                    "trust_status",
                )
            }
            if isinstance(manifest_snapshot, dict)
            else None
        )
        return {
            "id": int(session_row[0]),
            "fingerprint": str(session_row[1]),
            "source": str(session_row[2]),
            "created_at": str(session_row[3]),
            "snapshot": {
                "frozen_source": raw_session_snapshot.get("frozen_source"),
                "frozen_item_count": raw_session_snapshot.get(
                    "frozen_item_count"
                ),
                "frozen_manifest": safe_manifest,
            },
            "reviewer_name": session_row[5],
            "require_fresh_review": True,
            "finalized_at": session_row[7],
            "export_sha256": session_row[8],
            "token_revision": int(session_row[9] or 1),
            "token_rotated_at": session_row[10],
            "items": items,
            "candidate_nominations": [
                {
                    "ticker": str(row[0]),
                    "universe": str(row[1]),
                    "rationale": str(row[2]),
                    "history_as_of": row[3],
                    "bars_hash": row[4],
                    "revision": int(row[5]),
                    "created_at": str(row[6]),
                    "updated_at": str(row[7]),
                }
                for row in candidate_rows
            ],
            "counts": {**counts, "total": len(items)},
            "next_pending_ticker": next_pending,
        }

    def get_session(
        self, session_id: int, *, algorithm_version: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Session queue with computed per-item statuses and progress counts.

        Item status is derived, not authoritative: a fresh shared decision
        marks the item reviewed in every session; a stale decision returns it
        to pending (flagged ``review_stale``); otherwise the session-local
        pending/skipped stored status applies. Data staleness compares the
        decision's ``as_of`` against the item snapshot's ``data_date``.
        """
        security = self.get_session_security(session_id)
        if security is None:
            return None
        if security["require_fresh_review"]:
            return self._get_fresh_session(
                session_id, algorithm_version=algorithm_version
            )

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, fingerprint, source, created_at, snapshot "
                f"FROM review_sessions WHERE id = {self._ph}",
                (session_id,),
            )
            session_row = cursor.fetchone()
            if session_row is None:
                return None
            cursor.execute(
                f"SELECT position, ticker, status, snapshot, updated_at "
                f"FROM review_session_items WHERE session_id = {self._ph} ORDER BY position",
                (session_id,),
            )
            item_rows = cursor.fetchall()
            tickers = [str(row[1]) for row in item_rows]
            states: dict[str, dict[str, Any]] = {}
            overrides: set[str] = set()
            if tickers:
                marks = ", ".join([self._ph] * len(tickers))
                cursor.execute(
                    f"SELECT ticker, review_id, decision, as_of, algorithm_version, "
                    f"updated_at, coil_label, human_grade, confidence, note, "
                    f"label_policy_version, event_id "
                    f"FROM review_state WHERE interval = {self._ph} AND ticker IN ({marks})",
                    (AUTHORITATIVE_INTERVAL, *tickers),
                )
                for row in cursor.fetchall():
                    states[str(row[0])] = {
                        "review_id": int(row[1]),
                        "decision": str(row[2]),
                        "as_of": row[3],
                        "algorithm_version": row[4],
                        "updated_at": str(row[5]),
                        "coil_label": row[6],
                        "human_grade": row[7],
                        "confidence": row[8],
                        "note": row[9],
                        "label_policy_version": row[10],
                        "event_id": row[11],
                    }
                cursor.execute(
                    f"SELECT ticker FROM high_overrides "
                    f"WHERE interval = {self._ph} AND ticker IN ({marks})",
                    (AUTHORITATIVE_INTERVAL, *tickers),
                )
                overrides = {str(row[0]) for row in cursor.fetchall()}

        items: list[dict[str, Any]] = []
        counts = {ITEM_STATUS_PENDING: 0, ITEM_STATUS_REVIEWED: 0, ITEM_STATUS_SKIPPED: 0}
        next_pending: Optional[str] = None
        for position, ticker, stored_status, item_snapshot, updated_at in item_rows:
            snapshot_data = json.loads(item_snapshot) if item_snapshot else {}
            state = states.get(str(ticker))
            stale = decision_is_stale(
                state,
                current_as_of=snapshot_data.get("data_date"),
                current_algorithm_version=algorithm_version,
            )
            if state and not stale:
                status = ITEM_STATUS_REVIEWED
            else:
                status = str(stored_status)
            item = {
                "ticker": str(ticker),
                "position": int(position),
                "status": status,
                "stored_status": str(stored_status),
                "review_stale": stale,
                "review_id": state["review_id"] if state else None,
                "review_decision": state["decision"] if state else None,
                "review_as_of": state["as_of"] if state else None,
                "review_algorithm_version": state["algorithm_version"] if state else None,
                "review_coil_label": state["coil_label"] if state else None,
                "review_human_grade": state["human_grade"] if state else None,
                "review_confidence": state["confidence"] if state else None,
                "review_note": state["note"] if state else None,
                "review_label_policy_version": (
                    state["label_policy_version"] if state else None
                ),
                "review_event_id": state["event_id"] if state else None,
                "effective": "human" if str(ticker) in overrides else "algorithm",
                "snapshot": snapshot_data,
                "updated_at": str(updated_at),
            }
            counts[status] = counts.get(status, 0) + 1
            if status == ITEM_STATUS_PENDING and next_pending is None:
                next_pending = str(ticker)
            items.append(item)

        return {
            "id": int(session_row[0]),
            "fingerprint": str(session_row[1]),
            "source": str(session_row[2]),
            "created_at": str(session_row[3]),
            "snapshot": json.loads(session_row[4]) if session_row[4] else {},
            "items": items,
            "counts": {**counts, "total": len(items)},
            "next_pending_ticker": next_pending,
        }

    def _build_fresh_export(
        self,
        cursor,
        session_id: int,
        *,
        exported_at: str,
        algorithm_version: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """Build a capture corpus using only session-linked immutable events."""
        cursor.execute(
            f"SELECT id, fingerprint, source, created_at, snapshot, reviewer_name, "
            f"require_fresh_review, finalized_at "
            f"FROM review_sessions WHERE id = {self._ph}",
            (session_id,),
        )
        session_row = cursor.fetchone()
        if session_row is None or not bool(session_row[6]):
            return None
        cursor.execute(
            f"SELECT position, ticker, status, snapshot, updated_at, draft_revision, "
            f"draft_updated_at, completed_review_id, completed_event_id, sample_id, "
            f"bars_hash, reviewable, skip_reason, completed_at, "
            f"base_classification, base_classification_locked_at "
            f"FROM review_session_items WHERE session_id = {self._ph} "
            f"ORDER BY position",
            (session_id,),
        )
        item_rows = cursor.fetchall()
        export_items: list[dict[str, Any]] = []
        counts = {
            ITEM_STATUS_PENDING: 0,
            ITEM_STATUS_REVIEWED: 0,
            ITEM_STATUS_SKIPPED: 0,
        }
        next_pending: Optional[str] = None
        for row in item_rows:
            status = (
                ITEM_STATUS_REVIEWED if row[7] is not None else str(row[2])
            )
            base_classification = (
                json.loads(row[14]) if row[14] else None
            )
            base_locked = bool(
                base_classification is not None and row[15] is not None
            )
            full_snapshot = json.loads(row[3]) if row[3] else {}
            counts[status] = counts.get(status, 0) + 1
            if status == ITEM_STATUS_PENDING and next_pending is None:
                next_pending = str(row[1])
            export_items.append(
                {
                    "position": int(row[0]),
                    "ticker": str(row[1]),
                    "status": status,
                    "snapshot": (
                        full_snapshot
                        if base_locked
                        else _redacted_prelock_snapshot(
                            full_snapshot, position=int(row[0])
                        )
                    ),
                    "updated_at": str(row[4]),
                    "draft_revision": int(row[5] or 0),
                    "draft_updated_at": row[6],
                    "linked_review_id": (
                        int(row[7]) if row[7] is not None else None
                    ),
                    "linked_event_id": row[8],
                    "sample_id": row[9],
                    "bars_hash": row[10],
                    "reviewable": bool(row[11]),
                    "skip_reason": row[12],
                    "completed_at": row[13],
                    "base_classification": base_classification,
                    "base_classification_locked_at": row[15],
                }
            )

        cursor.execute(
            f"SELECT DISTINCT review_id FROM review_capture_idempotency "
            f"WHERE session_id = {self._ph} ORDER BY review_id",
            (session_id,),
        )
        review_ids = [int(row[0]) for row in cursor.fetchall()]
        records: list[dict[str, Any]] = []
        if review_ids:
            marks = ", ".join([self._ph] * len(review_ids))
            cursor.execute(
                f"SELECT id, ticker, interval, created_at, record "
                f"FROM high_reviews WHERE id IN ({marks}) ORDER BY id",
                tuple(review_ids),
            )
            for review_id, ticker, interval, created_at, raw_record in cursor.fetchall():
                record = json.loads(raw_record)
                records.append(
                    {
                        "id": int(review_id),
                        "event_id": record.get("eventId"),
                        "ticker": str(ticker),
                        "interval": str(interval),
                        "created_at": str(created_at),
                        "record": record,
                    }
                )

        cursor.execute(
            f"SELECT ticker, universe, rationale, history_as_of, bars_hash, "
            f"revision, created_at, updated_at "
            f"FROM review_candidate_nominations "
            f"WHERE session_id = {self._ph} ORDER BY created_at, ticker",
            (session_id,),
        )
        candidate_nominations = [
            {
                "ticker": str(row[0]),
                "universe": str(row[1]),
                "rationale": str(row[2]),
                "history_as_of": row[3],
                "bars_hash": row[4],
                "revision": int(row[5]),
                "created_at": str(row[6]),
                "updated_at": str(row[7]),
            }
            for row in cursor.fetchall()
        ]
        session_snapshot = json.loads(session_row[4]) if session_row[4] else {}
        return {
            "schema_version": 5,
            "kind": "coilingview.fresh-review-session-feedback",
            "exported_at": exported_at,
            "algorithm_version": algorithm_version,
            "reviewer": {"name": session_row[5]},
            "frozen_run": session_snapshot.get("frozen_run"),
            "session": {
                "id": int(session_row[0]),
                "fingerprint": str(session_row[1]),
                "source": str(session_row[2]),
                "created_at": str(session_row[3]),
                "finalized_at": session_row[7],
                "snapshot": session_snapshot,
                "items": export_items,
                "counts": {**counts, "total": len(export_items)},
                "next_pending_ticker": next_pending,
            },
            "records": records,
            "candidate_nominations": candidate_nominations,
        }

    def export_session(
        self, session_id: int, *, algorithm_version: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Versioned, portable feedback corpus for one review session.

        The session queue is exported together with:

        - every append-only review event submitted from this session, including
          superseded revisions; and
        - the current review event referenced by each item, even when that
          decision originated in another session and is shared globally.

        This keeps exports self-contained without treating an individual click
        as a model-update unit. Consumers can combine exports, deduplicate by
        session fingerprint + review id, and train/evaluate on the full corpus.
        """
        security = self.get_session_security(session_id)
        if security is None:
            return None
        if security["require_fresh_review"]:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT final_export FROM review_sessions WHERE id = {self._ph}",
                    (session_id,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
                return self._build_fresh_export(
                    cursor,
                    session_id,
                    exported_at=_utcnow(),
                    algorithm_version=algorithm_version,
                )

        session = self.get_session(session_id, algorithm_version=algorithm_version)
        if session is None:
            return None

        current_review_ids = {
            int(item["review_id"])
            for item in session["items"]
            if item.get("review_id") is not None
        }
        records: list[dict[str, Any]] = []
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, ticker, interval, created_at, record "
                "FROM high_reviews ORDER BY id"
            )
            rows = cursor.fetchall()

        for review_id, ticker, interval, created_at, raw_record in rows:
            record = json.loads(raw_record)
            record_session_id = record.get("sessionId", record.get("session_id"))
            belongs_to_session = False
            if record_session_id is not None:
                try:
                    belongs_to_session = int(record_session_id) == int(session_id)
                except (TypeError, ValueError):
                    belongs_to_session = False
            if int(review_id) not in current_review_ids and not belongs_to_session:
                continue
            records.append(
                {
                    "id": int(review_id),
                    "event_id": record.get("eventId")
                    or _legacy_event_id(
                        str(ticker), str(interval), str(created_at), record
                    ),
                    "ticker": str(ticker),
                    "interval": str(interval),
                    "created_at": str(created_at),
                    "record": record,
                }
            )

        return {
            "schema_version": 2,
            "kind": "coilingview.review-session-feedback",
            "exported_at": _utcnow(),
            "algorithm_version": algorithm_version,
            "session": session,
            "records": records,
        }

    def finalize_session(
        self, session_id: int, *, algorithm_version: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Freeze a complete fresh-session export and make every item immutable."""
        with self._connect() as conn:
            self._begin_protected_write(conn)
            cursor = conn.cursor()
            suffix = " FOR UPDATE" if self._is_postgres else ""
            cursor.execute(
                f"SELECT require_fresh_review, finalized_at, final_export, "
                f"final_export_hash FROM review_sessions "
                f"WHERE id = {self._ph}{suffix}",
                (session_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            if not bool(row[0]):
                raise ValueError("only fresh-review sessions can be finalized")
            if row[2]:
                return {
                    "export": json.loads(row[2]),
                    "sha256": str(row[3]),
                    "finalized_at": row[1],
                }

            cursor.execute(
                f"SELECT ticker, status, completed_review_id, skip_reason "
                f"FROM review_session_items WHERE session_id = {self._ph} "
                f"ORDER BY position",
                (session_id,),
            )
            pending: list[str] = []
            for ticker, status, completed_review_id, skip_reason in cursor.fetchall():
                complete = completed_review_id is not None
                valid_skip = (
                    str(status) == ITEM_STATUS_SKIPPED
                    and bool(str(skip_reason or "").strip())
                )
                if not complete and not valid_skip:
                    pending.append(str(ticker))
            if pending:
                raise ReviewConflictError(
                    "session still has pending items: " + ", ".join(pending)
                )

            finalized_at = _utcnow()
            payload = self._build_fresh_export(
                cursor,
                session_id,
                exported_at=finalized_at,
                algorithm_version=algorithm_version,
            )
            assert payload is not None
            payload["session"]["finalized_at"] = finalized_at
            content_hash = hashlib.sha256(
                _canonical_json(payload).encode("utf-8")
            ).hexdigest()
            frozen = _canonical_json(payload)
            cursor.execute(
                f"UPDATE review_sessions SET finalized_at = {self._ph}, "
                f"final_export = {self._ph}, final_export_hash = {self._ph} "
                f"WHERE id = {self._ph} AND finalized_at IS NULL",
                (finalized_at, frozen, content_hash, session_id),
            )
            if cursor.rowcount != 1:
                raise ReviewConflictError("session finalization lost a race")
            conn.commit()
        return {
            "export": json.loads(frozen),
            "sha256": content_hash,
            "finalized_at": finalized_at,
        }

    def save_draft(
        self,
        session_id: int,
        ticker: str,
        *,
        expected_revision: int,
        payload: dict[str, Any],
        algorithm_version: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Durably save one incomplete item draft with optimistic concurrency."""
        symbol = ticker.strip().upper()
        serialized = _canonical_json(payload)
        if len(serialized.encode("utf-8")) > 256_000:
            raise ValueError("review draft exceeds the 256 KB limit")
        with self._connect() as conn:
            self._begin_protected_write(conn)
            cursor = conn.cursor()
            suffix = " FOR UPDATE" if self._is_postgres else ""
            cursor.execute(
                f"SELECT require_fresh_review, finalized_at, reviewer_name "
                f"FROM review_sessions "
                f"WHERE id = {self._ph}{suffix}",
                (session_id,),
            )
            session_row = cursor.fetchone()
            if session_row is None:
                return None
            if not bool(session_row[0]):
                raise ValueError("durable drafts require a fresh-review session")
            if session_row[1] is not None:
                raise ReviewConflictError("finalized sessions are immutable")
            cursor.execute(
                f"SELECT draft_revision, completed_review_id, base_classification "
                f"FROM review_session_items "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph}{suffix}",
                (session_id, symbol),
            )
            item_row = cursor.fetchone()
            if item_row is None:
                return None
            if item_row[1] is not None:
                raise ReviewConflictError("reviewed items are immutable")
            if item_row[2]:
                classification = json.loads(item_row[2])
                if not _draft_matches_base_classification(
                    payload,
                    classification,
                    reviewer_name=session_row[2],
                ):
                    raise ReviewConflictError(
                        "locked base classification cannot be changed"
                    )
            current_revision = int(item_row[0] or 0)
            if current_revision != expected_revision:
                raise ReviewConflictError(
                    f"draft revision conflict: current revision is {current_revision}"
                )
            next_revision = current_revision + 1
            updated_at = _utcnow()
            cursor.execute(
                f"UPDATE review_session_items SET draft = {self._ph}, "
                f"draft_revision = {self._ph}, draft_updated_at = {self._ph}, "
                f"updated_at = {self._ph}, status = {self._ph}, skip_reason = NULL "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph} "
                f"AND draft_revision = {self._ph} AND completed_review_id IS NULL",
                (
                    serialized,
                    next_revision,
                    updated_at,
                    updated_at,
                    ITEM_STATUS_PENDING,
                    session_id,
                    symbol,
                    current_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ReviewConflictError("draft revision conflict")
            conn.commit()
        return self.get_session_item(
            session_id, symbol, algorithm_version=algorithm_version
        )

    def lock_base_classification(
        self,
        session_id: int,
        ticker: str,
        *,
        expected_draft_revision: int,
        classification: dict[str, Any],
        algorithm_version: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Persist the blind Step-1 verdict that unlocks model evidence."""
        symbol = ticker.strip().upper()
        serialized = _canonical_json(classification)
        with self._connect() as conn:
            self._begin_protected_write(conn)
            cursor = conn.cursor()
            suffix = " FOR UPDATE" if self._is_postgres else ""
            cursor.execute(
                f"SELECT require_fresh_review, finalized_at, reviewer_name "
                f"FROM review_sessions WHERE id = {self._ph}{suffix}",
                (session_id,),
            )
            session_row = cursor.fetchone()
            if session_row is None:
                return None
            if not bool(session_row[0]):
                raise ValueError("base locks require a fresh-review session")
            if session_row[1] is not None:
                raise ReviewConflictError("finalized sessions are immutable")
            cursor.execute(
                f"SELECT draft_revision, draft, completed_review_id, reviewable, "
                f"status, base_classification, base_classification_locked_at "
                f"FROM review_session_items "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph}{suffix}",
                (session_id, symbol),
            )
            item_row = cursor.fetchone()
            if item_row is None:
                return None
            existing = json.loads(item_row[5]) if item_row[5] else None
            if existing is not None and item_row[6] is not None:
                if hmac.compare_digest(
                    _canonical_json(existing), serialized
                ):
                    conn.commit()
                    return self.get_session_item(
                        session_id,
                        symbol,
                        algorithm_version=algorithm_version,
                    )
                raise ReviewConflictError(
                    "base classification is already locked"
                )
            if item_row[2] is not None:
                raise ReviewConflictError("reviewed items are immutable")
            if not bool(item_row[3]):
                raise ReviewConflictError(
                    "quarantined samples cannot reveal model evidence"
                )
            if str(item_row[4]) == ITEM_STATUS_SKIPPED:
                raise ReviewConflictError("skipped items cannot be base-locked")
            current_revision = int(item_row[0] or 0)
            if current_revision != expected_draft_revision:
                raise ReviewConflictError(
                    f"draft revision conflict: current revision is {current_revision}"
                )
            draft = json.loads(item_row[1]) if item_row[1] else None
            if not _draft_matches_base_classification(
                draft,
                classification,
                reviewer_name=session_row[2],
            ):
                raise ValueError(
                    "persisted draft does not match the validated base classification"
                )
            locked_at = _utcnow()
            cursor.execute(
                f"UPDATE review_session_items SET base_classification = {self._ph}, "
                f"base_classification_locked_at = {self._ph}, "
                f"updated_at = {self._ph} "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph} "
                f"AND draft_revision = {self._ph} "
                f"AND base_classification IS NULL",
                (
                    serialized,
                    locked_at,
                    locked_at,
                    session_id,
                    symbol,
                    current_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ReviewConflictError("base classification lock lost a race")
            conn.commit()
        return self.get_session_item(
            session_id, symbol, algorithm_version=algorithm_version
        )

    def get_capture_idempotency(
        self, session_id: int, ticker: str, idempotency_key: str
    ) -> Optional[dict[str, Any]]:
        """Return the stored client request hash and response for retry validation."""
        symbol = ticker.strip().upper()
        key = str(idempotency_key or "").strip()
        if not key:
            return None
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT request_hash, response FROM review_capture_idempotency "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph} "
                f"AND idempotency_key = {self._ph}",
                (session_id, symbol, key),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {"request_hash": str(row[0]), "response": json.loads(row[1])}

    def capture_decision(
        self,
        session_id: int,
        ticker: str,
        *,
        expected_draft_revision: int,
        sample_id: str,
        idempotency_key: str,
        record: dict[str, Any],
        request_hash: str | None = None,
    ) -> Optional[dict[str, Any]]:
        """Append one capture-only schema-v5 event and link it to its item.

        This path intentionally never calls ``_upsert_state`` or
        ``_upsert_override``.  The event is research feedback, not an
        authoritative production rule.
        """
        symbol = ticker.strip().upper()
        key = str(idempotency_key or "").strip()
        if not key:
            raise ValueError("idempotencyKey is required")
        request_basis = {
            "session_id": session_id,
            "ticker": symbol,
            "expected_draft_revision": expected_draft_revision,
            "sample_id": sample_id,
            "record": record,
        }
        request_hash = request_hash or hashlib.sha256(
            _canonical_json(request_basis).encode("utf-8")
        ).hexdigest()

        with self._connect() as conn:
            self._begin_protected_write(conn)
            cursor = conn.cursor()
            suffix = " FOR UPDATE" if self._is_postgres else ""
            cursor.execute(
                f"SELECT source, reviewer_name, require_fresh_review, finalized_at "
                f"FROM review_sessions WHERE id = {self._ph}{suffix}",
                (session_id,),
            )
            session_row = cursor.fetchone()
            if session_row is None:
                return None
            cursor.execute(
                f"SELECT draft_revision, completed_review_id, sample_id, bars_hash, "
                f"reviewable, base_classification, base_classification_locked_at "
                f"FROM review_session_items "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph}{suffix}",
                (session_id, symbol),
            )
            item_row = cursor.fetchone()
            if item_row is None:
                return None

            cursor.execute(
                f"SELECT request_hash, response FROM review_capture_idempotency "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph} "
                f"AND idempotency_key = {self._ph}",
                (session_id, symbol, key),
            )
            existing = cursor.fetchone()
            if existing is not None:
                if not hmac.compare_digest(str(existing[0]), request_hash):
                    raise ReviewConflictError(
                        "idempotencyKey was already used for a different request"
                    )
                return json.loads(existing[1])

            if not bool(session_row[2]):
                raise ValueError("capture finalization requires a fresh-review session")
            if session_row[3] is not None:
                raise ReviewConflictError("finalized sessions are immutable")
            reviewer = str(session_row[1] or "")
            capture_reviewer = str(
                (record.get("learningCapture") or {}).get("reviewerName") or ""
            )
            if not reviewer or capture_reviewer != reviewer:
                raise ValueError(
                    "learningCapture.reviewerName must match the assigned reviewer"
                )
            if item_row[1] is not None:
                raise ReviewConflictError("review item has already been finalized")
            if not bool(item_row[4]):
                raise ReviewConflictError(
                    "quarantined data-quality samples cannot be finalized"
                )
            if item_row[5] is None or item_row[6] is None:
                raise ReviewConflictError(
                    "base classification must be locked before finalization"
                )
            classification = json.loads(item_row[5])
            if not _draft_matches_base_classification(
                record,
                classification,
                reviewer_name=session_row[1],
            ):
                raise ValueError(
                    "final review cannot change the locked base classification"
                )
            current_revision = int(item_row[0] or 0)
            if current_revision != expected_draft_revision:
                raise ReviewConflictError(
                    f"draft revision conflict: current revision is {current_revision}"
                )
            if not item_row[2] or not hmac.compare_digest(
                str(item_row[2]), str(sample_id)
            ):
                raise ReviewConflictError(
                    "sampleId does not match the session-owned frozen sample"
                )

            created_at = _utcnow()
            event_id = str(uuid.uuid4())
            stored_record = dict(record)
            stored_record.update(
                {
                    "schemaVersion": 5,
                    "captureOnly": True,
                    "sessionId": session_id,
                    "ticker": symbol,
                    "sampleId": sample_id,
                    "eventId": event_id,
                    "serverCreatedAt": created_at,
                    "frozenContext": {
                        "source": str(session_row[0]),
                        "barsHash": item_row[3],
                        "sampleId": sample_id,
                    },
                }
            )
            review_id = self._insert_review_row(
                cursor, symbol, AUTHORITATIVE_INTERVAL, created_at, stored_record
            )
            cursor.execute(
                f"UPDATE review_session_items SET status = {self._ph}, "
                f"completed_review_id = {self._ph}, completed_event_id = {self._ph}, "
                f"completed_at = {self._ph}, updated_at = {self._ph}, "
                f"skip_reason = NULL "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph} "
                f"AND draft_revision = {self._ph} AND completed_review_id IS NULL",
                (
                    ITEM_STATUS_REVIEWED,
                    review_id,
                    event_id,
                    created_at,
                    created_at,
                    session_id,
                    symbol,
                    current_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ReviewConflictError("review item finalization lost a race")
            response = {
                "review": {
                    "id": review_id,
                    "ticker": symbol,
                    "interval": AUTHORITATIVE_INTERVAL,
                    "decision": stored_record.get("decision"),
                    "session_id": session_id,
                    "as_of": stored_record.get("asOf"),
                    "algorithm_version": stored_record.get("algorithmVersion"),
                    "event_id": event_id,
                    "coil_label": stored_record.get("coilLabel"),
                    "human_grade": stored_record.get("humanGrade"),
                    "confidence": stored_record.get("confidence"),
                    "note": stored_record.get("note"),
                    "created_at": created_at,
                    "capture_only": True,
                    "sample_id": sample_id,
                }
            }
            cursor.execute(
                f"INSERT INTO review_capture_idempotency "
                f"(session_id, ticker, idempotency_key, request_hash, review_id, "
                f"event_id, response, created_at) "
                f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, "
                f"{self._ph}, {self._ph}, {self._ph}, {self._ph})",
                (
                    session_id,
                    symbol,
                    key,
                    request_hash,
                    review_id,
                    event_id,
                    _canonical_json(response),
                    created_at,
                ),
            )
            conn.commit()
        return response

    def set_item_status(
        self,
        session_id: int,
        ticker: str,
        status: str,
        *,
        reason: Optional[str] = None,
        algorithm_version: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Session-specific deferral: only pending/skipped are storable.

        Reviewed status is derived from recorded decisions, never stored.
        Returns the updated computed item, or None when the item is unknown.
        """
        if status not in (ITEM_STATUS_PENDING, ITEM_STATUS_SKIPPED):
            raise ValueError("item status must be 'pending' or 'skipped'")
        symbol = ticker.strip().upper()
        normalized_reason = str(reason or "").strip() or None
        with self._connect() as conn:
            self._begin_protected_write(conn)
            cursor = conn.cursor()
            suffix = " FOR UPDATE" if self._is_postgres else ""
            cursor.execute(
                f"SELECT require_fresh_review, finalized_at FROM review_sessions "
                f"WHERE id = {self._ph}{suffix}",
                (session_id,),
            )
            session_row = cursor.fetchone()
            if session_row is None:
                return None
            fresh = bool(session_row[0])
            if fresh and session_row[1] is not None:
                raise ReviewConflictError("finalized sessions are immutable")
            if fresh and status == ITEM_STATUS_SKIPPED and normalized_reason is None:
                raise ValueError("fresh-review skips require a nonempty reason")
            if fresh:
                cursor.execute(
                    f"SELECT completed_review_id FROM review_session_items "
                    f"WHERE session_id = {self._ph} AND ticker = {self._ph}{suffix}",
                    (session_id, symbol),
                )
                item_row = cursor.fetchone()
                if item_row is None:
                    return None
                if item_row[0] is not None:
                    raise ReviewConflictError("reviewed items are immutable")
            cursor.execute(
                f"UPDATE review_session_items SET status = {self._ph}, "
                f"skip_reason = {self._ph}, updated_at = {self._ph} "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph}",
                (
                    status,
                    normalized_reason if status == ITEM_STATUS_SKIPPED else None,
                    _utcnow(),
                    session_id,
                    symbol,
                ),
            )
            changed = cursor.rowcount
            conn.commit()
        if not changed:
            return None
        return self.get_session_item(session_id, symbol, algorithm_version=algorithm_version)

    def get_session_item(
        self, session_id: int, ticker: str, *, algorithm_version: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """One computed session item (post-decision refresh for API responses)."""
        session = self.get_session(session_id, algorithm_version=algorithm_version)
        if session is None:
            return None
        symbol = ticker.strip().upper()
        for item in session["items"]:
            if item["ticker"] == symbol:
                return item
        return None


_store: Optional[ReviewStore] = None


def get_review_store() -> ReviewStore:
    """Process-wide store: Postgres when DATABASE_URL is set, SQLite otherwise."""
    global _store
    if _store is None:
        _store = ReviewStore(database_url=os.environ.get("DATABASE_URL"))
    return _store
