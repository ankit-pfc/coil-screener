from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import tempfile
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import review_snapshots
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from coil_analysis import ALGORITHM_VERSION, analyze_coil
from history_cache import get_history_payload, read_cache
from lifetime_structure import (
    ALGORITHM_VERSION as LIFETIME_TOP_ALGORITHM_VERSION,
    analyze_lifetime_references,
)
from review_capture import (
    BaseClassificationLockRequest,
    CaptureDraftRequest,
    CaptureFinalizeRequest,
    validate_capture_against_context,
)
from review_snapshots import (
    ReviewSnapshotError,
    canonical_json,
    load_blind_review_context,
    load_review_context,
    load_review_manifest,
    review_snapshot_identity,
    verify_manifest_identity,
)
from reviews import (
    ReviewAccessError,
    ReviewConflictError,
    annotate_review,
    get_review_store,
    hash_review_token,
    reject_incomplete_quarter_points,
)
from screen_monthly import (
    DEFAULT_TICKERS,
    build_ticker_list,
    compute_features,
    fetch_monthly_history,
    run_lifecycle_screen,
)
from vision.run import VisionRunConfig, run_vision_pipeline
from vision.storage import VisionRunStore
from starlette.background import BackgroundTask

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
VISION_RUNS_DIR = PROJECT_ROOT / "vision_runs"

# The frontend auto-loads runs[0] from /api/saved-runs as the default view. On a
# fresh deploy (e.g. Railway) every file gets the same checkout mtime, so an
# mtime sort is non-deterministic and may surface an obsolete algorithm run.
# Pin the current versioned candidate cohort first; everything else remains
# available as historical evidence below it.
DEMO_DEFAULT_RUN = "screen_2026-08-05_v2.3.0.csv"
PROTECTED_REFERENCE_RUNS = {
    "amrut_portfolio_exemplars_2026-08-21.csv",
    "amrut_reviewed_exemplars_2026-08-18.csv",
}
PROTECTED_BOOTSTRAP_FIELDS = {
    "ticker",
    "company_name",
    "exchange",
    "currency",
    "review_mode",
    "data_date",
    "freshness",
}
PROTECTED_BOOTSTRAP_NULL_FIELDS = {
    "age_years",
    "last_close",
    "score_total",
    "score_long_coil",
    "score_tight_resistance",
    "score_ascending_compression",
    "pos_in_10y_range",
    "dist_to_10y_high_pct",
    "range_ratio_24_120",
    "range_ratio_24_60",
    "low_36m_above_10y_low_pct",
    "slope_high_60m",
    "slope_low_60m",
    "trend_r2_60m",
    "peak_age_months",
    "old_peak_similarity",
}

app = FastAPI(title="Coil Screening")

_CAPTURE_API_ROUTES = (
    ("GET", re.compile(r"^/api/review-sessions/[0-9]+$")),
    ("GET", re.compile(r"^/api/review-sessions/[0-9]+/export$")),
    (
        "GET",
        re.compile(
            r"^/api/review-sessions/[0-9]+/items/[^/]+/context$"
        ),
    ),
    (
        "GET",
        re.compile(
            r"^/api/review-sessions/[0-9]+/items/[^/]+/major-tops$"
        ),
    ),
    (
        "GET",
        re.compile(r"^/api/review-sessions/[0-9]+/stock-universe$"),
    ),
    (
        "GET",
        re.compile(
            r"^/api/review-sessions/[0-9]+/stock-universe/"
            r"(sp500|international)/[^/]+/context$"
        ),
    ),
    ("POST", re.compile(r"^/api/review-sessions$")),
    (
        "PUT",
        re.compile(
            r"^/api/review-sessions/[0-9]+/items/[^/]+/draft$"
        ),
    ),
    (
        "POST",
        re.compile(
            r"^/api/review-sessions/[0-9]+/items/[^/]+/finalize$"
        ),
    ),
    (
        "POST",
        re.compile(
            r"^/api/review-sessions/[0-9]+/items/[^/]+/base-lock$"
        ),
    ),
    (
        "PATCH",
        re.compile(r"^/api/review-sessions/[0-9]+/items/[^/]+$"),
    ),
    (
        "PUT",
        re.compile(
            r"^/api/review-sessions/[0-9]+/candidate-nominations/[^/]+$"
        ),
    ),
    (
        "DELETE",
        re.compile(
            r"^/api/review-sessions/[0-9]+/candidate-nominations/[^/]+$"
        ),
    ),
    ("POST", re.compile(r"^/api/review-sessions/[0-9]+/finalize$")),
    (
        "POST",
        re.compile(
            r"^/api/review-sessions/[0-9]+/access-token/(rotate|revoke)$"
        ),
    ),
    ("GET", re.compile(r"^/api/admin/review-storage/backup$")),
)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def capture_only_mode() -> bool:
    return _truthy_env("CAPTURE_ONLY_MODE")


def _railway_runtime() -> bool:
    return any(
        os.environ.get(name)
        for name in (
            "RAILWAY_ENVIRONMENT_ID",
            "RAILWAY_ENVIRONMENT_NAME",
            "RAILWAY_PROJECT_ID",
            "RAILWAY_SERVICE_ID",
            "RAILWAY_STATIC_URL",
        )
    )


def review_persistence_readiness(*, probe: bool) -> dict[str, Any]:
    """Describe and, when requested, probe the review store durability."""
    database_url = os.environ.get("DATABASE_URL", "").strip()
    review_db_path = os.environ.get("REVIEW_DB_PATH", "").strip()
    volume_mount = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    railway = _railway_runtime()
    environment = (
        os.environ.get("APP_ENV")
        or os.environ.get("ENVIRONMENT")
        or ""
    ).strip().lower()
    dev_escape = (
        _truthy_env("CAPTURE_ONLY_ALLOW_EPHEMERAL")
        and environment in {"test", "dev", "development", "local"}
        and not railway
    )

    backend = "ephemeral_sqlite"
    durable = False
    configured = False
    reason: str | None = None
    if database_url:
        backend = "postgresql"
        durable = configured = True
    elif volume_mount:
        volume_path = Path(volume_mount).expanduser()
        if not volume_path.is_absolute():
            reason = "RAILWAY_VOLUME_MOUNT_PATH must be absolute"
        else:
            volume = volume_path.resolve()
            if review_db_path:
                configured_path = Path(review_db_path).expanduser().resolve()
                if configured_path == volume or volume in configured_path.parents:
                    backend = "sqlite_railway_volume"
                    durable = configured = True
                else:
                    reason = "REVIEW_DB_PATH is outside RAILWAY_VOLUME_MOUNT_PATH"
            else:
                backend = "sqlite_railway_volume"
                durable = configured = True
    elif review_db_path and not railway:
        configured_path = Path(review_db_path).expanduser()
        if not configured_path.is_absolute():
            reason = "REVIEW_DB_PATH must be absolute"
        else:
            backend = "sqlite_explicit"
            durable = configured = True
    elif review_db_path and railway:
        reason = "Railway SQLite requires RAILWAY_VOLUME_MOUNT_PATH"

    required = capture_only_mode()
    ready = bool(durable or dev_escape or not required)
    if required and not ready and reason is None:
        reason = (
            "capture-only mode requires DATABASE_URL or explicitly mounted "
            "persistent SQLite storage"
        )
    if ready and probe:
        try:
            store = get_review_store()
            if durable and backend == "postgresql" and not store.is_postgres:
                raise RuntimeError("configured PostgreSQL store is not active")
            if durable and backend != "postgresql":
                if store.is_postgres or store.sqlite_path is None:
                    raise RuntimeError("configured SQLite store is not active")
                actual_path = store.sqlite_path.expanduser().resolve()
                if backend == "sqlite_explicit":
                    expected_path = Path(review_db_path).expanduser().resolve()
                    if actual_path != expected_path:
                        raise RuntimeError(
                            "active SQLite path differs from REVIEW_DB_PATH"
                        )
                elif backend == "sqlite_railway_volume":
                    expected_volume = Path(volume_mount).expanduser().resolve()
                    if not (
                        actual_path == expected_volume
                        or expected_volume in actual_path.parents
                    ):
                        raise RuntimeError(
                            "active SQLite path is outside the Railway volume"
                        )
            store.persistence_probe()
        except Exception as exc:  # readiness must fail closed on backend errors
            ready = False
            reason = f"review store probe failed: {type(exc).__name__}"
    return {
        "ready": ready,
        "required": required,
        "configured": configured,
        "durable": durable,
        "backend": backend,
        "development_escape": dev_escape,
        "reason": reason,
    }


@app.middleware("http")
async def capture_only_boundary(request: Request, call_next):
    """Expose only the SPA, readiness, and protected/admin capture APIs."""
    if capture_only_mode():
        method = request.method.upper()
        path = request.url.path
        if path == "/api/health" and method in {"GET", "HEAD"}:
            return await call_next(request)
        if path in {"/docs", "/redoc", "/openapi.json"}:
            return JSONResponse(
                status_code=403,
                content={"detail": "Developer API surfaces are disabled."},
            )
        if path.startswith("/api/"):
            readiness = review_persistence_readiness(probe=False)
            if not readiness["ready"]:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "not_ready",
                        "persistence": readiness,
                    },
                )
            allowed = any(
                method == allowed_method and pattern.fullmatch(path)
                for allowed_method, pattern in _CAPTURE_API_ROUTES
            )
            if not allowed:
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": (
                            "This deployment exposes only protected review APIs."
                        )
                    },
                )
        elif method not in {"GET", "HEAD"}:
            return JSONResponse(
                status_code=403,
                content={"detail": "Only static frontend reads are allowed."},
            )
    return await call_next(request)


@app.on_event("startup")
def enforce_capture_persistence_on_startup() -> None:
    readiness = review_persistence_readiness(probe=True)
    if capture_only_mode() and not readiness["ready"]:
        raise RuntimeError(
            f"capture-only persistence is not ready: {readiness['reason']}"
        )


class ScreenRequest(BaseModel):
    tickers: list[str] = []
    universe: Literal["sp500", "international"] | None = None
    limit: int | None = None
    force_refresh: bool = False


class CorrectionPoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: str
    price: float = Field(gt=0)
    role: Literal[
        "major_top",
        "structural_retest",
        "provisional_top",
        "breakout_peak",
    ] = "major_top"
    idx: int | None = None

    @field_validator("date")
    @classmethod
    def validate_iso_date(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("point dates must use YYYY-MM-DD") from exc
        if parsed.isoformat() != value:
            raise ValueError("point dates must use YYYY-MM-DD")
        return value


class CorrectionRecordRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_version: int = Field(alias="schemaVersion")
    ticker: str
    interval: Literal["3M"]
    created_at: datetime = Field(alias="createdAt")
    manual_highs: list[CorrectionPoint] = Field(alias="manualHighs")
    algo_highs: list[CorrectionPoint] = Field(default_factory=list, alias="algoHighs")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")
        return symbol

    @model_validator(mode="after")
    def require_lid_anchors(self) -> "CorrectionRecordRequest":
        anchors = [p for p in self.manual_highs if p.role != "breakout_peak"]
        if len(anchors) < 2:
            raise ValueError("at least two non-breakout anchors are required")
        return self


class ReviewedHighPoint(CorrectionPoint):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    lid_member: bool | None = Field(default=None, alias="lidMember")


class ReviewDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_version: Literal[3, 4] = Field(default=3, alias="schemaVersion")
    label_policy_version: Literal[1] | None = Field(
        default=None, alias="labelPolicyVersion"
    )
    session_id: int | None = Field(default=None, alias="sessionId")
    ticker: str
    interval: Literal["3M"] = "3M"
    as_of: str | None = Field(default=None, alias="asOf")
    algorithm_version: str | None = Field(default=None, alias="algorithmVersion")
    decision: Literal["approved", "corrected"]
    coil_label: Literal["coil", "not_coil", "uncertain"] | None = Field(
        default=None, alias="coilLabel"
    )
    human_grade: Literal["A", "B", "C"] | None = Field(
        default=None, alias="humanGrade"
    )
    confidence: Literal["high", "low"] | None = None
    note: str | None = Field(default=None, max_length=2000)
    algorithm: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    reviewed_highs: list[ReviewedHighPoint] = Field(
        default_factory=list, alias="reviewedHighs"
    )
    created_at: datetime | None = Field(default=None, alias="createdAt")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")
        return symbol


class ReviewSessionItemRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    ticker: str
    snapshot: dict[str, Any] = Field(default_factory=dict)


class ReviewSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    source: str
    items: list[ReviewSessionItemRequest]
    snapshot: dict[str, Any] = Field(default_factory=dict)
    reviewer_name: str | None = Field(
        default=None, min_length=2, max_length=120, alias="reviewerName"
    )
    access_token: str | None = Field(
        default=None, min_length=24, max_length=512, alias="accessToken"
    )
    require_fresh_review: bool = Field(
        default=False, alias="requireFreshReview"
    )


class ReviewSessionItemPatch(BaseModel):
    status: Literal["pending", "skipped"]
    reason: str | None = Field(default=None, max_length=5000)


class CandidateNominationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    universe: Literal["sp500", "international"]
    rationale: str = Field(default="", max_length=2000)
    expected_revision: int = Field(
        default=0, ge=0, alias="expectedRevision"
    )


class VisionRunRequest(BaseModel):
    tickers: list[str] = []
    saved_run: str | None = None
    interval: Literal["1M", "3M", "6M", "1Y"] = "3M"
    timeframe: Literal["1Y", "2Y", "5Y", "10Y", "All"] = "10Y"
    chart_type: Literal["candles", "bars", "line", "area"] = "candles"
    base_url: str = "http://127.0.0.1:5173"
    limit: int | None = None
    run_id: str | None = None
    headless: bool = True
    max_highs: int = 3
    confidence: float = 0.35
    max_trendlines: int = 5
    touch_tolerance_pct: float = 1.5


class VisionReviewRequest(BaseModel):
    run_id: str
    ticker: str
    interval: Literal["1M", "3M", "6M", "1Y"] = "3M"
    timeframe: Literal["1Y", "2Y", "5Y", "10Y", "All"]
    chart_type: Literal["candles", "bars", "line", "area"]
    decision: Literal["accepted", "rejected", "edited"]
    accepted_highs: list[dict[str, Any]] = []
    notes: str | None = None


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def serialize_frame(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({key: clean_value(value) for key, value in row.items()})
    return records


def normalize_saved_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Supply v2 display fields when loading a legacy numeric-only CSV."""
    normalized = []
    for raw in records:
        row = dict(raw)
        if row.get("review_mode") in {
            "blinded_boundary_negative",
            "detector_shape_audit",
            "reference_exemplar",
        }:
            # This saved run is only a transport for a protected reviewer
            # queue. Reference exemplars carry their frozen detector fields
            # explicitly; do not fabricate missing lifecycle/status/grade
            # values that could be mistaken for evidence.
            normalized.append(row)
            continue
        legacy_score = row.get("score_total")
        row.setdefault("lifecycle", "pre_breakout")
        row.setdefault("status", "coiling")
        row.setdefault("grade", None)
        row.setdefault("lid_grade", None)
        row.setdefault(
            "coil_score",
            round(float(legacy_score) * 100.0, 1) if legacy_score is not None else 0.0,
        )
        row.setdefault("lid_slope_pct_per_year", None)
        row.setdefault("proximity_pct", None)
        # v2.2 lid band. A legacy CSV predates the enum, so the field is
        # present-and-null rather than missing: consumers can tell "not
        # measured" apart from a real band placement.
        row.setdefault("current_price_position", None)
        row.setdefault("span_years", None)
        row.setdefault("touches", None)
        row.setdefault("touch_count", row.get("touches"))
        row.setdefault("reviewed", False)
        row.setdefault("review_status", None)
        row.setdefault("review_id", None)
        row.setdefault("review_as_of", None)
        row.setdefault("review_algorithm_version", None)
        row.setdefault("review_stale", False)
        row.setdefault("review_effective", "algorithm")
        row.setdefault("data_date", None)
        row.setdefault("freshness", "saved")
        normalized.append(row)
    return normalized


def saved_run_protected_metadata(filename: str) -> dict[str, Any]:
    """Read protected-review bootstrap settings from the frozen manifest.

    An explicit ``protected_bootstrap`` marker fails closed: the reviewer and
    policy must travel with it instead of being guessed from a CSV filename.
    Older reference corpora retain their existing protection until their
    immutable manifests are superseded with the explicit metadata.
    """
    manifest_path = (
        review_snapshots.REVIEW_SNAPSHOT_ROOT
        / Path(filename).stem
        / "manifest.json"
    )
    if not manifest_path.is_file():
        return {
            "protected_bootstrap": False,
            "reviewer_name": None,
            "protected_policy": None,
        }

    manifest = load_review_manifest(filename)
    protected_bootstrap = manifest.get("protected_bootstrap")
    if protected_bootstrap is None and filename in PROTECTED_REFERENCE_RUNS:
        # Backward-compatible protection for immutable corpora created before
        # manifest-driven bootstrap metadata existed.
        protected_bootstrap = True
        reviewer_name: Any = "Amrut"
    else:
        reviewer_name = manifest.get("reviewer_name")

    if protected_bootstrap is None:
        protected_bootstrap = False
    if not isinstance(protected_bootstrap, bool):
        raise ReviewSnapshotError(
            "manifest protected_bootstrap must be a boolean"
        )
    if not protected_bootstrap:
        return {
            "protected_bootstrap": False,
            "reviewer_name": None,
            "protected_policy": None,
        }

    if not isinstance(reviewer_name, str) or not reviewer_name.strip():
        raise ReviewSnapshotError(
            "protected review manifest reviewer_name is required"
        )
    protected_policy = manifest.get("review_policy")
    if not isinstance(protected_policy, dict) or not protected_policy:
        raise ReviewSnapshotError(
            "protected review manifest review_policy is required"
        )
    return {
        "protected_bootstrap": True,
        "reviewer_name": reviewer_name.strip(),
        "protected_policy": dict(protected_policy),
    }


def redact_protected_saved_run_bootstrap(
    records: list[dict[str, Any]], *, protected_bootstrap: bool
) -> list[dict[str, Any]]:
    """Expose only neutral ordering before a protected session exists."""
    if not protected_bootstrap:
        return records
    redacted = []
    for row in records:
        neutral = {
            key: value
            for key, value in row.items()
            if key in PROTECTED_BOOTSTRAP_FIELDS
        }
        neutral.update({key: None for key in PROTECTED_BOOTSTRAP_NULL_FIELDS})
        redacted.append(neutral)
    return redacted


@app.get("/api/health")
def health():
    persistence = review_persistence_readiness(probe=True)
    status = "ok" if persistence["ready"] else "not_ready"
    payload = {
        "status": status,
        "capture_only": capture_only_mode(),
        "persistence": persistence,
    }
    if not persistence["ready"]:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/api/default-tickers")
def default_tickers() -> dict[str, list[str]]:
    return {"tickers": DEFAULT_TICKERS}


@app.get("/api/saved-runs")
def saved_runs() -> dict[str, list[dict[str, Any]]]:
    # Curated demo run first (deterministic default), then the rest newest-first.
    files = sorted(
        PROJECT_ROOT.glob("*.csv"),
        key=lambda path: (path.name == DEMO_DEFAULT_RUN, path.stat().st_mtime),
        reverse=True,
    )
    return {
        "runs": [
            {
                "name": file.name,
                "size_bytes": file.stat().st_size,
            }
            for file in files
        ]
    }


@app.get("/api/saved-runs/{filename}")
def saved_run(filename: str) -> dict[str, Any]:
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = PROJECT_ROOT / filename
    if not file_path.exists() or file_path.suffix.lower() != ".csv":
        raise HTTPException(status_code=404, detail="Saved run not found.")

    df = pd.read_csv(file_path)
    records = normalize_saved_results(serialize_frame(df))
    try:
        protected_metadata = saved_run_protected_metadata(filename)
    except ReviewSnapshotError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    records = redact_protected_saved_run_bootstrap(
        records,
        protected_bootstrap=protected_metadata["protected_bootstrap"],
    )
    return {
        "name": filename,
        "count": len(records),
        "results": records,
        **protected_metadata,
    }


@app.post("/api/screen")
def screen(request: ScreenRequest) -> dict[str, Any]:
    tickers = build_ticker_list(
        explicit_tickers=request.tickers,
        universe=request.universe,
        limit=request.limit,
    )
    run = run_lifecycle_screen(
        tickers,
        fetch_monthly=fetch_monthly_history,
        review_override_for=lambda ticker: get_review_store().get_override(ticker, "3M"),
        review_state_for=lambda ticker: get_review_store().get_review_state(ticker, "3M"),
        force_refresh=request.force_refresh,
    )
    return {
        "count": len(run["results"]),
        "tickers": tickers,
        **run,
    }


@app.get("/api/history/{ticker}")
def history(
    ticker: str,
    max_bars: int | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    symbol = ticker.strip().upper()
    payload = get_history_payload(
        symbol,
        fetch_monthly_history,
        compute_features,
        force_refresh=force_refresh,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="No monthly history found.")

    bars = payload["bars"]
    trimmed = bars[-max_bars:] if max_bars and max_bars < len(bars) else bars

    return {
        "ticker": symbol,
        "bars": trimmed,
        "features": payload["features"],
        "freshness": payload["freshness"],
    }


@app.get("/api/coil/{ticker}")
def coil(
    ticker: str,
    as_of: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Deterministic major-top / resistance-slope / coiling analysis.

    ``as_of`` truncates the history first (YYYY-MM-DD, inclusive), which lets
    the UI and validation scripts replay the pre-breakout state of a chart.
    An approved human review for the ticker overrides the algorithm's
    structure; the raw algorithm output is retained under ``review``.
    """
    if as_of is not None and (len(as_of) != 10 or as_of[4] != "-" or as_of[7] != "-"):
        raise HTTPException(status_code=400, detail="as_of must be YYYY-MM-DD.")

    symbol = ticker.strip().upper()
    payload = get_history_payload(
        symbol,
        fetch_monthly_history,
        compute_features,
        force_refresh=force_refresh,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="No monthly history found.")

    store = get_review_store()
    override = store.get_override(symbol)
    analysis = analyze_coil(payload["bars"], as_of=as_of, review_override=override)
    annotate_review(analysis, store.get_review_state(symbol), algorithm_version=ALGORITHM_VERSION)
    analysis["analysis_metadata"]["data_freshness"].update(payload["freshness"])
    return {
        "ticker": symbol,
        **analysis,
        "freshness": payload["freshness"],
    }


def cached_last_bar_date(symbol: str) -> str | None:
    """Data date of the series a reviewer was shown, without a live refresh.

    The review UI loads ``/api/coil`` first, which refreshes and rewrites this
    cache, so the cached tail is the chart the reviewer actually annotated.
    Reading it here keeps anchor validation free of provider I/O. ``None``
    (no cached history) leaves the anchor rule unenforceable, and the
    analyzer's own guard remains the backstop.
    """
    payload = read_cache(symbol.strip().upper()) or {}
    bars = payload.get("bars") or []
    if not bars or not isinstance(bars[-1], dict):
        return None
    last = bars[-1].get("date")
    return str(last) if last else None


@app.post("/api/highs/corrections")
def submit_highs_correction(record: CorrectionRecordRequest) -> dict[str, Any]:
    """Approved internal review: append-only persistence + immediate override.

    Body is the frontend ``CorrectionRecord`` (schemaVersion 1). The user's
    ``manualHighs`` become the effective structure for the ticker; the full
    record is retained verbatim for calibration. Points inside the incomplete
    final quarter are refused (v2.2): the analyzer would drop them, so the
    reviewer is told instead of losing an anchor silently.
    """
    try:
        reject_incomplete_quarter_points(
            [point.date for point in record.manual_highs],
            last_bar_date=cached_last_bar_date(record.ticker),
        )
        review = get_review_store().append_review(
            record.model_dump(mode="json", by_alias=True)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"review": review}


@app.delete("/api/highs/corrections/{ticker}")
def revoke_highs_correction(
    ticker: str, interval: Literal["3M"] = "3M"
) -> dict[str, Any]:
    """Revoke the effective override while retaining an append-only audit event."""
    try:
        revocation = get_review_store().revoke_override(ticker, interval)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if revocation is None:
        raise HTTPException(status_code=404, detail="No active correction found.")
    return {"revocation": revocation, "override": None}


@app.get("/api/highs/corrections/{ticker}")
def highs_corrections(
    ticker: str, interval: Literal["3M"] = "3M"
) -> dict[str, Any]:
    """Current effective override plus the append-only review history."""
    store = get_review_store()
    return {
        "ticker": ticker.strip().upper(),
        "override": store.get_override(ticker, interval),
        "reviews": store.list_reviews(
            ticker, interval, include_capture_only=False
        ),
    }


@app.post("/api/highs/reviews")
def submit_review_decision(request: ReviewDecisionRequest) -> dict[str, Any]:
    """Unified review decision: approve the algorithm or correct it.

    ``approved`` records the sign-off and clears any older human override so
    the algorithm becomes effective; ``corrected`` makes the reviewed highs
    (with optional ``lidMember`` line anchors) the live analysis immediately.
    The response returns the append-only review event, the effective
    override, the recomputed coil analysis, and the updated session item.

    Reviewers keep full authority over which zone they anchor to; the only
    structural restriction (v2.2) is that a reviewed point may not sit in the
    incomplete final quarter, which is refused here rather than dropped
    silently downstream.
    """
    store = get_review_store()
    try:
        reject_incomplete_quarter_points(
            [point.date for point in request.reviewed_highs],
            last_bar_date=cached_last_bar_date(request.ticker),
        )
        outcome = store.record_decision(request.model_dump(mode="json", by_alias=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    symbol = request.ticker
    analysis: dict[str, Any] | None = None
    payload = get_history_payload(symbol, fetch_monthly_history, compute_features)
    if payload is not None:
        analysis = analyze_coil(
            payload["bars"], review_override=store.get_override(symbol)
        )
        annotate_review(
            analysis, store.get_review_state(symbol), algorithm_version=ALGORITHM_VERSION
        )
        analysis["analysis_metadata"]["data_freshness"].update(payload["freshness"])

    session_item = None
    if request.session_id is not None:
        session_item = store.get_session_item(
            request.session_id, symbol, algorithm_version=ALGORITHM_VERSION
        )
    return {
        "review": outcome["review"],
        "override": outcome["override"],
        "analysis": analysis,
        "session_item": session_item,
    }


def require_review_session_access(
    session_id: int,
    x_review_token: str | None = Header(
        default=None, alias="X-Review-Token"
    ),
) -> dict[str, Any]:
    """Central object-level capability check for every session read/write."""
    try:
        security = get_review_store().authorize_session(
            session_id, x_review_token
        )
    except ReviewAccessError as exc:
        raise HTTPException(status_code=403, detail="Review session access denied.") from exc
    if security is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    if capture_only_mode() and not (
        security["require_fresh_review"] and security["protected"]
    ):
        raise HTTPException(
            status_code=403,
            detail="Capture-only deployments expose only protected fresh sessions.",
        )
    return security


def require_review_admin_access(
    x_review_admin_key: str | None = Header(
        default=None, alias="X-Review-Admin-Key"
    ),
) -> bool:
    """Authenticate operational endpoints with the server-held admin key."""
    expected_admin_key = os.environ.get("REVIEW_SESSION_CREATE_KEY", "")
    if not expected_admin_key:
        raise HTTPException(
            status_code=503,
            detail="Review administration is not configured.",
        )
    if not (
        x_review_admin_key
        and hmac.compare_digest(expected_admin_key, x_review_admin_key)
    ):
        raise HTTPException(
            status_code=403,
            detail="Review administration is not authorized.",
        )
    return True


@app.post("/api/review-sessions")
def create_review_session(
    request: ReviewSessionCreateRequest,
    x_review_admin_key: str | None = Header(
        default=None, alias="X-Review-Admin-Key"
    ),
) -> dict[str, Any]:
    """Create a review session for one screener snapshot, or resume it.

    The queue keeps the backend-ranked order it was posted with. Identity is
    a snapshot fingerprint (source + ordered tickers + data dates), so the
    same screen resumes its session while new data starts a fresh one.
    """
    expected_admin_key = os.environ.get("REVIEW_SESSION_CREATE_KEY")
    if capture_only_mode() and not expected_admin_key:
        raise HTTPException(
            status_code=503,
            detail="Capture-only session creation is not configured.",
        )
    if expected_admin_key and not (
        x_review_admin_key
        and hmac.compare_digest(expected_admin_key, x_review_admin_key)
    ):
        raise HTTPException(
            status_code=403, detail="Review session creation is not authorized."
        )
    if capture_only_mode() and not request.require_fresh_review:
        raise HTTPException(
            status_code=403,
            detail="Capture-only deployments may create only fresh protected sessions.",
        )
    try:
        items: list[dict[str, Any]]
        session_snapshot = dict(request.snapshot)
        if request.require_fresh_review:
            if not request.reviewer_name:
                raise ValueError(
                    "fresh review sessions require reviewerName"
                )
            if not request.access_token:
                raise ValueError(
                    "fresh review sessions require a capability accessToken"
                )
            manifest = load_review_manifest(request.source)
            manifest_order = [
                str(ticker).strip().upper()
                for ticker in (
                    manifest.get("ordered_universe")
                    or [
                        item.get("ticker")
                        for item in manifest["items"]
                        if isinstance(item, dict)
                    ]
                )
            ]
            requested_order = [
                item.ticker.strip().upper() for item in request.items
            ]
            if requested_order != manifest_order:
                raise ValueError(
                    "fresh review sessions require the complete manifest universe "
                    "in exact order"
                )
            manifest_items = {
                str(item.get("ticker", "")).strip().upper(): item
                for item in manifest["items"]
                if isinstance(item, dict)
            }
            items = []
            frozen_run: dict[str, Any] | None = None
            for requested_item in request.items:
                identity = review_snapshot_identity(
                    request.source, requested_item.ticker
                )
                if identity["ticker"] not in manifest_items:
                    raise ValueError(
                        f"{identity['ticker']} is not present in the frozen manifest"
                    )
                verify_manifest_identity(
                    identity, manifest_items[identity["ticker"]]
                )
                if frozen_run is None:
                    frozen_run = identity["run"]
                items.append(
                    {
                        "ticker": identity["ticker"],
                        "snapshot": {
                            "screen_snapshot": identity["screen_snapshot"],
                            "corpus_labels": identity["corpus_labels"],
                            "data_quality": identity["data_quality"],
                            "data_quality_validation": identity[
                                "data_quality_validation"
                            ],
                            "reviewable": identity["reviewable"],
                            "frozen": {
                                "source": identity["source"],
                                "data_date": identity["data_date"],
                                "sample_id": identity["sample_id"],
                                "bars_hash": identity["bars_hash"],
                                "snapshot_sha256": identity["snapshot_sha256"],
                                "provenance": identity["provenance"],
                            },
                        },
                        "sample_id": identity["sample_id"],
                        "bars_hash": identity["bars_hash"],
                        "reviewable": identity["reviewable"],
                    }
                )
            session_snapshot.update(
                {
                    "frozen_run": manifest.get("run")
                    or {
                        "corpus_id": manifest.get("corpus_id"),
                        "source_run": manifest.get("source_run"),
                        "algorithm_version": (
                            (manifest.get("source_run") or {}).get(
                                "algorithm_version"
                            )
                            if isinstance(manifest.get("source_run"), dict)
                            else ALGORITHM_VERSION
                        ),
                        "canonicalization": manifest.get("canonicalization"),
                        "generator": manifest.get("generator"),
                    },
                    "frozen_source": request.source,
                    "frozen_item_count": len(items),
                    "frozen_manifest": {
                        "schema_version": manifest.get("schema_version"),
                        "kind": manifest.get("kind"),
                        "sha256": manifest.get("_manifest_sha256"),
                        "item_count": manifest.get(
                            "item_count", len(manifest["items"])
                        ),
                        "trust_status": manifest.get("trust_status"),
                        "source_run": manifest.get("source_run"),
                    },
                }
            )
        else:
            items = [item.model_dump() for item in request.items]
        session, created = get_review_store().create_session(
            request.source,
            items,
            snapshot=session_snapshot,
            algorithm_version=ALGORITHM_VERSION,
            reviewer_name=request.reviewer_name,
            capability_token_hash=(
                hash_review_token(request.access_token)
                if request.access_token
                else None
            ),
            require_fresh_review=request.require_fresh_review,
        )
    except ReviewAccessError as exc:
        raise HTTPException(status_code=403, detail="Review session access denied.") from exc
    except ReviewSnapshotError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session": session, "created": created}


@app.get("/api/review-sessions/{session_id}")
def review_session(
    session_id: int,
    _security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    session = get_review_store().get_session(
        session_id, algorithm_version=ALGORITHM_VERSION
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    return {"session": session}


@lru_cache(maxsize=2)
def _review_stock_universe(
    source: Literal["sp500", "international"],
) -> tuple[str, ...]:
    """Reuse the screener's existing deterministic universe connectors."""
    return tuple(build_ticker_list(universe=source))


def _universe_history_context(
    source: Literal["sp500", "international"],
    ticker: str,
) -> dict[str, Any]:
    symbol = ticker.strip().upper()
    try:
        universe = _review_stock_universe(source)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The stock universe could not be loaded. Try again.",
        ) from exc
    if symbol not in universe:
        raise HTTPException(
            status_code=404,
            detail="Ticker is not part of the selected stock universe.",
        )
    payload = get_history_payload(
        symbol,
        fetch_monthly_history,
        compute_features,
        force_refresh=False,
    )
    if payload is None or not payload.get("bars"):
        raise HTTPException(
            status_code=404,
            detail="No monthly history found for this stock.",
        )
    bars = payload["bars"]
    bars_hash = hashlib.sha256(
        canonical_json({"ticker": symbol, "bars": bars}).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": "coilingview.review-candidate-context",
        "ticker": symbol,
        "universe": source,
        "monthly_bars": bars,
        "history_as_of": bars[-1].get("date"),
        "bars_hash": bars_hash,
        "freshness": payload.get("freshness"),
    }


@app.get("/api/review-sessions/{session_id}/stock-universe")
def review_stock_universe(
    session_id: int,
    source: Literal["sp500", "international"] = "sp500",
    _security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    try:
        tickers = list(_review_stock_universe(source))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The stock universe could not be loaded. Try again.",
        ) from exc
    session = get_review_store().get_session(
        session_id, algorithm_version=ALGORITHM_VERSION
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    screened = {item["ticker"] for item in session["items"]}
    return {
        "source": source,
        "label": (
            "S&P 500"
            if source == "sp500"
            else "International review universe"
        ),
        "tickers": tickers,
        "available_count": sum(ticker not in screened for ticker in tickers),
        "screened_tickers": [
            ticker for ticker in tickers if ticker in screened
        ],
    }


@app.get(
    "/api/review-sessions/{session_id}/stock-universe/"
    "{source}/{ticker}/context"
)
def review_stock_universe_context(
    session_id: int,
    source: Literal["sp500", "international"],
    ticker: str,
    _security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    return {"context": _universe_history_context(source, ticker)}


@app.put(
    "/api/review-sessions/{session_id}/candidate-nominations/{ticker}"
)
def put_review_candidate_nomination(
    session_id: int,
    ticker: str,
    request: CandidateNominationRequest,
    _security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    context = _universe_history_context(request.universe, ticker)
    try:
        nomination = get_review_store().save_candidate_nomination(
            session_id,
            ticker,
            universe=request.universe,
            rationale=request.rationale,
            history_as_of=context["history_as_of"],
            bars_hash=context["bars_hash"],
            expected_revision=request.expected_revision,
        )
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if nomination is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    return {"nomination": nomination}


@app.delete(
    "/api/review-sessions/{session_id}/candidate-nominations/{ticker}"
)
def delete_review_candidate_nomination(
    session_id: int,
    ticker: str,
    expected_revision: int,
    _security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    try:
        removed = get_review_store().delete_candidate_nomination(
            session_id,
            ticker,
            expected_revision=expected_revision,
        )
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if removed is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    if not removed:
        raise HTTPException(
            status_code=404, detail="Candidate nomination not found."
        )
    return {"removed": True, "ticker": ticker.strip().upper()}


@app.post("/api/review-sessions/{session_id}/access-token/rotate")
def rotate_review_session_access_token(
    session_id: int,
    _admin: bool = Depends(require_review_admin_access),
) -> JSONResponse:
    """Issue a new reviewer capability and invalidate the old one immediately."""
    token = secrets.token_urlsafe(32)
    try:
        result = get_review_store().rotate_session_token(
            session_id, hash_review_token(token)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    return JSONResponse(
        content={
            "accessToken": token,
            "tokenRevision": result["token_revision"],
            "rotatedAt": result["rotated_at"],
        },
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
        },
    )


@app.post("/api/review-sessions/{session_id}/access-token/revoke")
def revoke_review_session_access_token(
    session_id: int,
    _admin: bool = Depends(require_review_admin_access),
) -> JSONResponse:
    """Revoke the active reviewer capability without deleting review data."""
    try:
        result = get_review_store().revoke_session_token(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    return JSONResponse(
        content={
            "revoked": True,
            "tokenRevision": result["token_revision"],
            "revokedAt": result["revoked_at"],
        },
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
        },
    )


@app.get("/api/review-sessions/{session_id}/export")
def review_session_export(
    session_id: int,
    security: dict[str, Any] = Depends(require_review_session_access),
) -> Response:
    """Portable, versioned corpus for analysis and offline handoff.

    The response contains the complete queue snapshot, current item states, and
    the append-only structured review records linked to the session. The
    frontend wraps this JSON in a readable Markdown file without weakening the
    machine-readable contract.
    """
    if security["require_fresh_review"] and security["finalized_at"] is None:
        raise HTTPException(
            status_code=409,
            detail="Fresh-session export is available only after finalization.",
        )
    export = get_review_store().export_session(
        session_id, algorithm_version=ALGORITHM_VERSION
    )
    if export is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    canonical = canonical_json(export)
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Response(
        content=canonical,
        media_type="application/json",
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "X-Export-SHA256": content_hash,
            "X-Export-Canonicalization": "coilingview-canonical-json-v1",
        },
    )


@app.get("/api/review-sessions/{session_id}/items/{ticker}/context")
def review_session_item_context(
    session_id: int,
    ticker: str,
    security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    """Return price-only evidence until the persisted blind verdict is locked."""
    if not security["require_fresh_review"]:
        raise HTTPException(
            status_code=400,
            detail="Frozen context is available only for fresh-review sessions.",
        )
    item = get_review_store().get_session_item(
        session_id, ticker, algorithm_version=ALGORITHM_VERSION
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Review session item not found.")
    try:
        if item.get("base_classification_locked"):
            context = load_review_context(security["source"], ticker)
        else:
            context = load_blind_review_context(security["source"], ticker)
    except ReviewSnapshotError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if item.get("sample_id") != context["sample_id"]:
        raise HTTPException(
            status_code=409,
            detail="Frozen context no longer matches the session sample.",
        )
    context["item_snapshot"] = item["snapshot"]
    context["base_classification_locked"] = bool(
        item.get("base_classification_locked")
    )
    context["base_classification_locked_at"] = item.get(
        "base_classification_locked_at"
    )
    return {"context": context}


@app.get("/api/review-sessions/{session_id}/items/{ticker}/major-tops")
def review_session_item_major_tops(
    session_id: int,
    ticker: str,
    security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    """Plot-only lifetime tops over the complete history API payload.

    This deliberately returns neither the coil analyzer nor any shape, grade,
    lifecycle, score, fitted line, or forecasting output.  The displayed
    points are the lifetime detector's conservative local-high episodes after
    a completed-quarter rejection has confirmed them.  The protected snapshot
    establishes ticker/session identity only; it must never truncate the chart.
    Full provider history is refreshed and merged by the same path used by
    ``/api/history/{ticker}``.
    """
    if not security["require_fresh_review"]:
        raise HTTPException(
            status_code=400,
            detail="Frozen major-top plots require a protected session.",
        )
    item = get_review_store().get_session_item(
        session_id, ticker, algorithm_version=ALGORITHM_VERSION
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Review session item not found.")
    if item.get("snapshot", {}).get("review_mode") != "detector_shape_audit":
        raise HTTPException(
            status_code=404,
            detail="This session is not configured for historical major-top plots.",
        )
    try:
        context = load_blind_review_context(security["source"], ticker)
        if item.get("sample_id") != context["sample_id"]:
            raise ReviewSnapshotError(
                "Frozen context no longer matches the session sample."
            )
        history_payload = get_history_payload(
            context["ticker"],
            fetch_monthly_history,
            compute_features,
        )
        if history_payload is None:
            raise ReviewSnapshotError(
                "Complete lifetime history is unavailable for this ticker."
            )
        full_history = history_payload["bars"]
        analysis = analyze_lifetime_references(full_history)
    except ReviewSnapshotError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    major_tops = [
        {
            "id": episode["id"],
            # Markers belong to the quarterly candle even when the exact high
            # occurred in another month inside that quarter.
            "date": episode["quarter_date"],
            "source_date": episode["date"],
            "price": episode["price"],
            "confirmed_at": episode["reaction"]["confirmed_at"],
        }
        for episode in analysis["top_episodes"]
        if episode.get("status") == "confirmed_rejection"
    ]
    history = analysis["history"]
    return {
        "analysis": {
            "schema_version": 1,
            "kind": "coilingview.historical-major-top-plot",
            "ticker": context["ticker"],
            "interval": "3M",
            "history_scope": "full_available_lifetime",
            "history_start": history["start_date"],
            "history_end": history["end_date"],
            "completed_through": history["completed_through"],
            "completed_quarter_count": history["completed_quarter_count"],
            "monthly_bars": full_history,
            "major_tops": major_tops,
            "algorithm_version": LIFETIME_TOP_ALGORITHM_VERSION,
            "sample_id": context["sample_id"],
            "bars_hash": hashlib.sha256(
                canonical_json(full_history).encode("utf-8")
            ).hexdigest(),
            "data_freshness": history_payload.get("freshness"),
        }
    }


@app.put("/api/review-sessions/{session_id}/items/{ticker}/draft")
def put_review_session_item_draft(
    session_id: int,
    ticker: str,
    request: CaptureDraftRequest,
    _security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    try:
        item = get_review_store().save_draft(
            session_id,
            ticker,
            expected_revision=request.expected_revision,
            payload=request.payload,
            algorithm_version=ALGORITHM_VERSION,
        )
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="Review session item not found.")
    return {"item": item}


@app.post("/api/review-sessions/{session_id}/items/{ticker}/base-lock")
def lock_review_session_item_base_classification(
    session_id: int,
    ticker: str,
    request: BaseClassificationLockRequest,
    _security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    """Atomically lock the persisted blind verdict before model reveal."""
    try:
        item = get_review_store().lock_base_classification(
            session_id,
            ticker,
            expected_draft_revision=request.expected_draft_revision,
            classification=request.base_classification.model_dump(
                mode="json", by_alias=True
            ),
            algorithm_version=ALGORITHM_VERSION,
        )
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="Review session item not found.")
    return {"item": item}


@app.post("/api/review-sessions/{session_id}/items/{ticker}/finalize")
def finalize_review_session_item(
    session_id: int,
    ticker: str,
    request: CaptureFinalizeRequest,
    security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    """Validate and append schema-v5 feedback without changing production truth."""
    symbol = ticker.strip().upper()
    if request.session_id != session_id or request.ticker != symbol:
        raise HTTPException(
            status_code=400,
            detail="sessionId/ticker must match the finalize route.",
        )
    if not security["require_fresh_review"]:
        raise HTTPException(
            status_code=400,
            detail="Capture-only finalization requires a fresh-review session.",
        )
    if request.learning_capture.reviewer_name != security["reviewer_name"]:
        raise HTTPException(
            status_code=400,
            detail="reviewerName must match the assigned reviewer.",
        )
    try:
        context = load_review_context(security["source"], symbol)
    except ReviewSnapshotError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if request.sample_id != context["sample_id"]:
        raise HTTPException(
            status_code=409,
            detail="sampleId does not match the frozen review sample.",
        )
    try:
        validate_capture_against_context(request, context)
        record = request.model_dump(
            mode="json",
            by_alias=True,
            exclude={
                "idempotency_key",
                "expected_draft_revision",
                "sample_id",
            },
        )
        client_provenance = record.get("provenance")
        record["algorithm"] = context["analysis"]
        record["provenance"] = {
            "frozen": True,
            "source": context["source"],
            "sampleId": context["sample_id"],
            "barsHash": context["bars_hash"],
            "dataDate": context["monthly_bars"][-1]["date"],
            "algorithmVersion": ALGORITHM_VERSION,
            "reviewOverrideApplied": False,
            "client": client_provenance,
        }
        result = get_review_store().capture_decision(
            session_id,
            symbol,
            expected_draft_revision=request.expected_draft_revision,
            sample_id=request.sample_id,
            idempotency_key=request.idempotency_key,
            record=record,
        )
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Review session item not found.")
    item = get_review_store().get_session_item(
        session_id, symbol, algorithm_version=ALGORITHM_VERSION
    )
    return {**result, "session_item": item}


@app.post("/api/review-sessions/{session_id}/finalize")
def finalize_review_session(
    session_id: int,
    _security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    try:
        result = get_review_store().finalize_session(
            session_id, algorithm_version=ALGORITHM_VERSION
        )
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    return result


@app.patch("/api/review-sessions/{session_id}/items/{ticker}")
def patch_review_session_item(
    session_id: int,
    ticker: str,
    patch: ReviewSessionItemPatch,
    _security: dict[str, Any] = Depends(require_review_session_access),
) -> dict[str, Any]:
    """Session-specific deferral: pending/skipped only; reviewed is derived."""
    try:
        item = get_review_store().set_item_status(
            session_id,
            ticker,
            patch.status,
            reason=patch.reason,
            algorithm_version=ALGORITHM_VERSION,
        )
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="Review session item not found.")
    return {"item": item}


@app.get("/api/admin/review-storage/backup")
def download_review_storage_backup(
    _admin: bool = Depends(require_review_admin_access),
):
    """Stream a transactionally consistent SQLite backup for off-service storage."""
    store = get_review_store()
    if store.is_postgres:
        raise HTTPException(
            status_code=400,
            detail="Use managed PostgreSQL backups for this review store.",
        )
    handle = tempfile.NamedTemporaryFile(
        prefix="coilingview-review-backup-",
        suffix=".sqlite3",
        delete=False,
    )
    backup_path = Path(handle.name)
    handle.close()
    try:
        store.backup_sqlite(backup_path)
    except (OSError, ValueError) as exc:
        backup_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Review backup could not be created.",
        ) from exc
    filename = (
        "coilingview-review-backup-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.sqlite3"
    )
    return FileResponse(
        backup_path,
        media_type="application/vnd.sqlite3",
        filename=filename,
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
        background=BackgroundTask(backup_path.unlink, missing_ok=True),
    )


def vision_store() -> VisionRunStore:
    return VisionRunStore(VISION_RUNS_DIR)


@app.post("/api/vision/run")
def vision_run(request: VisionRunRequest | None = None) -> dict[str, Any]:
    req = request or VisionRunRequest()
    try:
        return run_vision_pipeline(
            VisionRunConfig(
                project_root=PROJECT_ROOT,
                tickers=req.tickers,
                saved_run=req.saved_run,
                interval=req.interval,
                timeframe=req.timeframe,
                chart_type=req.chart_type,
                base_url=req.base_url,
                limit=req.limit,
                run_id=req.run_id,
                headless=req.headless,
                max_highs=req.max_highs,
                confidence=req.confidence,
                max_trendlines=req.max_trendlines,
                touch_tolerance_pct=req.touch_tolerance_pct,
            ),
            store=vision_store(),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/vision/runs")
def vision_runs() -> dict[str, Any]:
    return {"runs": vision_store().list_runs()}


@app.get("/api/vision/runs/{run_id}")
def vision_run_detail(run_id: str) -> dict[str, Any]:
    try:
        return vision_store().read_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Vision run not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/vision/predictions/{ticker}")
def vision_prediction(
    ticker: str,
    interval: Literal["1M", "3M", "6M", "1Y"] = "3M",
    run_id: str = "latest",
    timeframe: Literal["1Y", "2Y", "5Y", "10Y", "All"] | None = None,
    chart_type: Literal["candles", "bars", "line", "area"] | None = None,
) -> dict[str, Any]:
    try:
        return vision_store().read_prediction(
            ticker,
            interval,
            run_id,
            timeframe=timeframe,
            chart_type=chart_type,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Vision prediction not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/vision/reviews")
def vision_review(request: VisionReviewRequest) -> dict[str, Any]:
    store = vision_store()
    try:
        prediction = store.read_prediction(request.ticker, run_id=request.run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Vision prediction not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    expected = {
        "ticker": request.ticker.strip().upper(),
        "interval": request.interval,
        "timeframe": request.timeframe,
        "chart_type": request.chart_type,
    }
    actual = {
        "ticker": str(prediction.get("ticker") or "").strip().upper(),
        "interval": prediction.get("interval"),
        "timeframe": prediction.get("timeframe"),
        "chart_type": prediction.get("chart_type"),
    }
    if actual != expected:
        raise HTTPException(
            status_code=409,
            detail="Vision review context does not match the stored prediction.",
        )
    try:
        review = store.append_review(request.model_dump())
        return {"review": review}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
