from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from coil_analysis import ALGORITHM_VERSION, analyze_coil
from history_cache import get_history_payload, read_cache
from reviews import annotate_review, get_review_store, reject_incomplete_quarter_points
from screen_monthly import (
    DEFAULT_TICKERS,
    build_ticker_list,
    compute_features,
    fetch_monthly_history,
    run_lifecycle_screen,
)
from vision.run import VisionRunConfig, run_vision_pipeline
from vision.storage import VisionRunStore

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
VISION_RUNS_DIR = PROJECT_ROOT / "vision_runs"

# The frontend auto-loads runs[0] from /api/saved-runs as the default view. On a
# fresh deploy (e.g. Railway) every file gets the same checkout mtime, so an
# mtime sort is non-deterministic and may surface a large non-curated run whose
# tickers aren't in seed_cache. Pin the curated demo run first so the demo always
# boots into the seeded coil universe; everything else stays mtime-sorted below.
DEMO_DEFAULT_RUN = "demo_curated_coils_results.csv"

app = FastAPI(title="Coil Screening")


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


class ReviewSessionItemPatch(BaseModel):
    status: Literal["pending", "skipped"]


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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
    return {
        "name": filename,
        "count": len(records),
        "results": records,
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
        "reviews": store.list_reviews(ticker, interval),
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


@app.post("/api/review-sessions")
def create_review_session(request: ReviewSessionCreateRequest) -> dict[str, Any]:
    """Create a review session for one screener snapshot, or resume it.

    The queue keeps the backend-ranked order it was posted with. Identity is
    a snapshot fingerprint (source + ordered tickers + data dates), so the
    same screen resumes its session while new data starts a fresh one.
    """
    try:
        session, created = get_review_store().create_session(
            request.source,
            [item.model_dump() for item in request.items],
            snapshot=request.snapshot,
            algorithm_version=ALGORITHM_VERSION,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session": session, "created": created}


@app.get("/api/review-sessions/{session_id}")
def review_session(session_id: int) -> dict[str, Any]:
    session = get_review_store().get_session(
        session_id, algorithm_version=ALGORITHM_VERSION
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    return {"session": session}


@app.get("/api/review-sessions/{session_id}/export")
def review_session_export(session_id: int) -> dict[str, Any]:
    """Portable, versioned corpus for analysis and offline handoff.

    The response contains the complete queue snapshot, current item states, and
    the append-only structured review records linked to the session. The
    frontend wraps this JSON in a readable Markdown file without weakening the
    machine-readable contract.
    """
    export = get_review_store().export_session(
        session_id, algorithm_version=ALGORITHM_VERSION
    )
    if export is None:
        raise HTTPException(status_code=404, detail="Review session not found.")
    return export


@app.patch("/api/review-sessions/{session_id}/items/{ticker}")
def patch_review_session_item(
    session_id: int, ticker: str, patch: ReviewSessionItemPatch
) -> dict[str, Any]:
    """Session-specific deferral: pending/skipped only; reviewed is derived."""
    try:
        item = get_review_store().set_item_status(
            session_id, ticker, patch.status, algorithm_version=ALGORITHM_VERSION
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="Review session item not found.")
    return {"item": item}


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
