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

Storage backend is PostgreSQL when ``DATABASE_URL`` is set (production on
Railway), SQLite otherwise (local dev and tests). Both run the same logical
schema; records are stored as JSON text for backend parity.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "reviews.db"
AUTHORITATIVE_INTERVAL = "3M"
SUPPORTED_ROLES = frozenset(
    {"major_top", "structural_retest", "provisional_top", "breakout_peak"}
)
DECISION_APPROVED = "approved"
DECISION_CORRECTED = "corrected"
ITEM_STATUS_PENDING = "pending"
ITEM_STATUS_SKIPPED = "skipped"
ITEM_STATUS_REVIEWED = "reviewed"


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

    schema_version: int = Field(default=3, alias="schemaVersion")
    session_id: Optional[int] = Field(default=None, alias="sessionId")
    ticker: str
    interval: Literal["3M"] = AUTHORITATIVE_INTERVAL
    as_of: Optional[str] = Field(default=None, alias="asOf")
    algorithm_version: Optional[str] = Field(default=None, alias="algorithmVersion")
    decision: Literal["approved", "corrected"]
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

    @model_validator(mode="after")
    def valid_correction_anchors(self) -> "ReviewDecisionRecord":
        if self.decision != DECISION_CORRECTED:
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
]

# Pre-session deployments created high_overrides without decision provenance.
_MIGRATION_COLUMNS = {
    "high_overrides": [("as_of", "TEXT"), ("algorithm_version", "TEXT")],
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


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


def session_fingerprint(source: str, items: list[dict[str, Any]]) -> str:
    """Stable identity of one screener snapshot: source + ordered tickers + data dates."""
    basis = {
        "source": source,
        "items": [
            [
                str(item.get("ticker", "")).strip().upper(),
                (item.get("snapshot") or {}).get("data_date"),
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
            "stale": stale,
        }
    )
    review.setdefault("effective", "algorithm")


class ReviewStore:
    """Append-only review log + current-override table over SQLite/Postgres."""

    def __init__(self, database_url: Optional[str] = None, sqlite_path: Optional[Path] = None):
        self._database_url = database_url
        self._sqlite_path = Path(sqlite_path) if sqlite_path else DEFAULT_SQLITE_PATH
        self._is_postgres = bool(database_url)
        self._ph = "%s" if self._is_postgres else "?"
        self._ensure_schema()

    def _connect(self):
        if self._is_postgres:
            import psycopg2  # deferred: only needed when DATABASE_URL is set

            return psycopg2.connect(self._database_url)
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self._sqlite_path)

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
    ) -> str:
        updated_at = _utcnow()
        cursor.execute(
            f"DELETE FROM review_state WHERE ticker = {self._ph} AND interval = {self._ph}",
            (ticker, interval),
        )
        cursor.execute(
            f"INSERT INTO review_state "
            f"(ticker, interval, review_id, decision, as_of, algorithm_version, updated_at) "
            f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph})",
            (ticker, interval, review_id, decision, as_of, algorithm_version, updated_at),
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

        with self._connect() as conn:
            cursor = conn.cursor()
            review_id = self._insert_review_row(cursor, ticker, interval, created_at, record)
            self._upsert_state(
                cursor, ticker, interval, review_id, validated.decision,
                as_of=validated.as_of, algorithm_version=validated.algorithm_version,
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
                f"SELECT review_id, decision, as_of, algorithm_version, updated_at "
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
        }

    def list_reviews(self, ticker: str, interval: str = "3M") -> list[dict[str, Any]]:
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
        return [
            {"id": int(row[0]), "created_at": str(row[1]), "record": json.loads(row[2])}
            for row in rows
        ]

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
    ) -> tuple[dict[str, Any], bool]:
        """Create a session for one screener snapshot, or resume the existing one.

        ``items`` is the backend-ranked result order: ``[{"ticker", "snapshot"}]``.
        Identity is the snapshot fingerprint, so re-opening the same screen
        resumes the same queue while new data or a re-run starts a new session.
        """
        name = str(source or "").strip()
        if not name:
            raise ValueError("Review sessions require a source.")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            ticker = str(item.get("ticker", "")).strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            normalized.append({"ticker": ticker, "snapshot": item.get("snapshot") or {}})
        if not normalized:
            raise ValueError("Review sessions require at least one ticker.")

        fingerprint = session_fingerprint(name, normalized)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id FROM review_sessions WHERE fingerprint = {self._ph}",
                (fingerprint,),
            )
            row = cursor.fetchone()
            if row is not None:
                session_id = int(row[0])
                created = False
            else:
                created_at = _utcnow()
                session_snapshot = json.dumps(snapshot or {})
                if self._is_postgres:
                    cursor.execute(
                        f"INSERT INTO review_sessions (fingerprint, source, created_at, snapshot) "
                        f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}) RETURNING id",
                        (fingerprint, name, created_at, session_snapshot),
                    )
                    session_id = int(cursor.fetchone()[0])
                else:
                    cursor.execute(
                        f"INSERT INTO review_sessions (fingerprint, source, created_at, snapshot) "
                        f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph})",
                        (fingerprint, name, created_at, session_snapshot),
                    )
                    session_id = int(cursor.lastrowid)
                for position, item in enumerate(normalized):
                    cursor.execute(
                        f"INSERT INTO review_session_items "
                        f"(session_id, position, ticker, status, snapshot, updated_at) "
                        f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph}, {self._ph})",
                        (
                            session_id,
                            position,
                            item["ticker"],
                            ITEM_STATUS_PENDING,
                            json.dumps(item["snapshot"]),
                            created_at,
                        ),
                    )
                created = True
            conn.commit()
        session = self.get_session(session_id, algorithm_version=algorithm_version)
        assert session is not None
        return session, created

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
                    f"SELECT ticker, review_id, decision, as_of, algorithm_version, updated_at "
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

    def set_item_status(
        self,
        session_id: int,
        ticker: str,
        status: str,
        *,
        algorithm_version: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Session-specific deferral: only pending/skipped are storable.

        Reviewed status is derived from recorded decisions, never stored.
        Returns the updated computed item, or None when the item is unknown.
        """
        if status not in (ITEM_STATUS_PENDING, ITEM_STATUS_SKIPPED):
            raise ValueError("item status must be 'pending' or 'skipped'")
        symbol = ticker.strip().upper()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE review_session_items SET status = {self._ph}, updated_at = {self._ph} "
                f"WHERE session_id = {self._ph} AND ticker = {self._ph}",
                (status, _utcnow(), session_id, symbol),
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
