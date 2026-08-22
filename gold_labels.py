"""Whole-pattern gold labels materialized from frozen candle clicks.

The capture contract deliberately stores *which candle* the reviewer clicked,
not a client-computed price, slope, or recognition date.  Materialization is a
server/offline operation over the frozen monthly evidence.  It snaps every
click to a completed quarterly candle and derives all geometry deterministically.

This module is research infrastructure.  A materialized label is detector
truth for evaluation; it is never passed to :func:`coil_analysis.analyze_coil`
and can never become a ``review_override``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from calendar import monthrange
from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from coil_analysis import _aggregate_quarterly_display_bars


CAPTURE_SCHEMA_VERSION = 1
MATERIALIZED_SCHEMA_VERSION = 1
CAPTURE_KIND = "coilingview.whole-pattern-gold-capture"
MATERIALIZED_KIND = "coilingview.whole-pattern-gold-label"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")


def canonical_json(value: Any) -> str:
    """Stable JSON encoding used for label and evidence identities."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _valid_iso_date(value: str, *, field: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must use ISO YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must use ISO YYYY-MM-DD")
    return value


def _ordered_unique_dates(clicks: list["CandleClick"], *, field: str) -> None:
    dates = [click.date for click in clicks]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError(f"{field} clicks must be unique and chronological")


class CandleClick(BaseModel):
    """A reviewer-selected candle; the backend owns its coordinate values."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    date: str
    price_field: Literal["open", "high", "low", "close"] = Field(
        alias="priceField"
    )

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        return _valid_iso_date(value, field="candle click date")


class PatternJudgments(BaseModel):
    """Independent reviewer judgments; no field is inferred from another."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    shape: Literal["coil", "not_coil", "uncertain"]
    maturity: Literal["mature", "immature", "uncertain"]
    lifecycle: Literal[
        "no_structure",
        "forming",
        "pre_breakout",
        "breaking_out",
        "post_breakout",
        "uncertain",
    ]
    readiness: Literal["ready", "not_ready", "uncertain"]
    action: Literal["actionable", "watch", "avoid", "abstain"]
    confidence: Literal["high", "medium", "low"]


class StructureCapture(BaseModel):
    """One line/band in a possibly nested pattern hierarchy."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1, max_length=160)
    parent_id: str | None = Field(default=None, alias="parentId")
    relationship: Literal["standalone", "parent", "child", "parent_child"]
    selection: Literal["primary", "alternate"]
    role: Literal[
        "primary_lid",
        "secondary_lid",
        "support",
        "breakout_level",
    ]
    boundary_kind: Literal["line", "resistance_band"] = Field(
        default="line", alias="boundaryKind"
    )
    confidence: Literal["high", "medium", "low"]
    construction_anchors: list[CandleClick] = Field(
        min_length=2, max_length=8, alias="constructionAnchors"
    )
    recognition_confirmation: CandleClick = Field(
        alias="recognitionConfirmation"
    )
    supporting_touches: list[CandleClick] = Field(
        default_factory=list, max_length=32, alias="supportingTouches"
    )
    excluded_highs: list[CandleClick] = Field(
        default_factory=list, max_length=32, alias="excludedHighs"
    )
    lower_band_touches: list[CandleClick] = Field(
        default_factory=list, max_length=16, alias="lowerBandTouches"
    )

    @field_validator("id", "parent_id")
    @classmethod
    def valid_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _ID_RE.fullmatch(value):
            raise ValueError("structure ids contain unsupported characters")
        return value

    @model_validator(mode="after")
    def coherent_geometry(self) -> "StructureCapture":
        _ordered_unique_dates(
            self.construction_anchors, field="constructionAnchors"
        )
        if self.recognition_confirmation.date < self.construction_anchors[-1].date:
            raise ValueError(
                "recognitionConfirmation cannot precede construction anchors"
            )
        _ordered_unique_dates(self.supporting_touches, field="supportingTouches")
        _ordered_unique_dates(self.excluded_highs, field="excludedHighs")
        _ordered_unique_dates(self.lower_band_touches, field="lowerBandTouches")
        if self.relationship in {"child", "parent_child"} and self.parent_id is None:
            raise ValueError("child and parent_child structures require parentId")
        if self.relationship in {"standalone", "parent"} and self.parent_id is not None:
            raise ValueError(
                "standalone and parent structures cannot declare parentId"
            )
        if self.parent_id == self.id:
            raise ValueError("a structure cannot be its own parent")
        if self.boundary_kind == "resistance_band":
            if len(self.lower_band_touches) < 2:
                raise ValueError(
                    "resistance bands require at least two lowerBandTouches"
                )
        elif self.lower_band_touches:
            raise ValueError("line boundaries cannot contain lowerBandTouches")

        groups = {
            "constructionAnchors": {click.date for click in self.construction_anchors},
            "supportingTouches": {click.date for click in self.supporting_touches},
            "excludedHighs": {click.date for click in self.excluded_highs},
        }
        if groups["constructionAnchors"] & groups["supportingTouches"]:
            raise ValueError(
                "construction anchors and supporting touches must use distinct candles"
            )
        if groups["excludedHighs"] & (
            groups["constructionAnchors"] | groups["supportingTouches"]
        ):
            raise ValueError("excluded highs cannot also be line members")
        return self


class BottomCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1, max_length=160)
    structure_id: str | None = Field(default=None, alias="structureId")
    role: Literal["major_bottom", "undercut", "outlier"]
    click: CandleClick
    confidence: Literal["high", "medium", "low"]

    @field_validator("id", "structure_id")
    @classmethod
    def valid_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _ID_RE.fullmatch(value):
            raise ValueError("bottom ids contain unsupported characters")
        return value


class EventCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1, max_length=160)
    structure_id: str | None = Field(default=None, alias="structureId")
    kind: Literal[
        "breakout",
        "failed_breakout",
        "retest",
        "continuation",
        "invalidation",
    ]
    trigger: CandleClick
    resolution: CandleClick | None = None
    retest_state: Literal["shallow", "deep", "failed"] | None = Field(
        default=None, alias="retestState"
    )
    relative_volume: Literal["confirmed", "not_confirmed", "unavailable"] = Field(
        default="unavailable", alias="relativeVolume"
    )
    action_signal: bool = Field(default=False, alias="actionSignal")
    confidence: Literal["high", "medium", "low"]

    @field_validator("id", "structure_id")
    @classmethod
    def valid_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _ID_RE.fullmatch(value):
            raise ValueError("event ids contain unsupported characters")
        return value

    @model_validator(mode="after")
    def coherent_event(self) -> "EventCapture":
        if self.kind == "failed_breakout" and self.resolution is None:
            raise ValueError("failed_breakout events require a resolution candle")
        if self.resolution is not None and self.resolution.date < self.trigger.date:
            raise ValueError("event resolution cannot precede its trigger")
        if self.kind == "retest" and self.retest_state is None:
            raise ValueError("retest events require retestState")
        if self.kind != "retest" and self.retest_state is not None:
            raise ValueError("retestState applies only to retest events")
        action_capable = self.kind in {"breakout", "continuation"} or (
            self.kind == "retest" and self.retest_state in {"shallow", "deep"}
        )
        if self.action_signal and not action_capable:
            raise ValueError(
                "actionSignal requires a breakout, continuation, or holding retest"
            )
        return self


class PhaseCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1, max_length=160)
    structure_id: str | None = Field(default=None, alias="structureId")
    kind: Literal[
        "prior_trend",
        "base",
        "congestion",
        "compression",
        "breakout",
        "retest",
        "post_breakout",
    ]
    start: CandleClick
    end: CandleClick
    present: bool = True
    confidence: Literal["high", "medium", "low"]

    @field_validator("id", "structure_id")
    @classmethod
    def valid_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _ID_RE.fullmatch(value):
            raise ValueError("phase ids contain unsupported characters")
        return value

    @model_validator(mode="after")
    def chronological(self) -> "PhaseCapture":
        if self.end.date < self.start.date:
            raise ValueError("phase end cannot precede its start")
        return self


class SourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal[
        "blind_review_session",
        "amrut_review_event",
        "amrut_interview_intake",
    ]
    reference: str = Field(min_length=1, max_length=1000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WholePatternGoldCapture(BaseModel):
    """Finalized point-in-time human label before coordinate materialization."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = Field(alias="schemaVersion")
    kind: Literal[CAPTURE_KIND] = CAPTURE_KIND
    episode_id: str = Field(min_length=1, max_length=160, alias="episodeId")
    setup_id: str = Field(min_length=1, max_length=160, alias="setupId")
    evaluation_role: Literal[
        "development", "validation", "blind_benchmark"
    ] = Field(alias="evaluationRole")
    labeler: str = Field(min_length=2, max_length=120)
    labeled_at: datetime = Field(alias="labeledAt")
    cutoff_date: str = Field(alias="cutoffDate")
    decision_as_of: str = Field(alias="decisionAsOf")
    interval: Literal["3M"] = "3M"
    outcome_visible_during_label: bool = Field(
        default=False, alias="outcomeVisibleDuringLabel"
    )
    judgments: PatternJudgments
    active_structure_id: str | None = Field(
        default=None, alias="activeStructureId"
    )
    top_review_complete: bool = Field(alias="topReviewComplete")
    structures: list[StructureCapture] = Field(default_factory=list, max_length=16)
    bottoms: list[BottomCapture] = Field(default_factory=list, max_length=64)
    events: list[EventCapture] = Field(default_factory=list, max_length=64)
    phases: list[PhaseCapture] = Field(default_factory=list, max_length=64)
    source_evidence: list[SourceEvidence] = Field(
        min_length=1, max_length=16, alias="sourceEvidence"
    )
    notes: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("episode_id", "setup_id", "active_structure_id")
    @classmethod
    def valid_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _ID_RE.fullmatch(value):
            raise ValueError("episode/setup/structure ids contain unsupported characters")
        return value

    @field_validator("cutoff_date")
    @classmethod
    def valid_cutoff(cls, value: str) -> str:
        normalized = _valid_iso_date(value, field="cutoffDate")
        if date.fromisoformat(normalized).month % 3 != 0:
            raise ValueError(
                "cutoffDate must identify a quarter-ending monthly bar"
            )
        return normalized

    @field_validator("decision_as_of")
    @classmethod
    def valid_decision_as_of(cls, value: str) -> str:
        return _valid_iso_date(value, field="decisionAsOf")

    @field_validator("labeled_at")
    @classmethod
    def normalize_labeled_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("labeledAt must include a timezone offset")
        return value.astimezone(timezone.utc)

    @field_validator("labeler")
    @classmethod
    def normalize_labeler(cls, value: str) -> str:
        return value.strip()

    @field_validator("notes")
    @classmethod
    def valid_notes(cls, value: list[str]) -> list[str]:
        if any(not note.strip() or len(note) > 2000 for note in value):
            raise ValueError("notes must be nonblank and at most 2000 characters")
        return [note.strip() for note in value]

    @model_validator(mode="after")
    def coherent_pattern(self) -> "WholePatternGoldCapture":
        cutoff = date.fromisoformat(self.cutoff_date)
        quarter_end = date(
            cutoff.year,
            cutoff.month,
            monthrange(cutoff.year, cutoff.month)[1],
        )
        if date.fromisoformat(self.decision_as_of) < quarter_end:
            raise ValueError(
                "decisionAsOf must be on or after the calendar quarter end"
            )
        if date.fromisoformat(self.decision_as_of) > self.labeled_at.date():
            raise ValueError("decisionAsOf cannot be later than labeledAt")

        ids = [structure.id for structure in self.structures]
        if len(ids) != len(set(ids)):
            raise ValueError("structure ids must be unique")
        structures = {structure.id: structure for structure in self.structures}
        children_by_parent: dict[str, list[StructureCapture]] = {}
        for structure in self.structures:
            if structure.parent_id and structure.parent_id not in structures:
                raise ValueError(
                    f"structure {structure.id} references an unknown parentId"
                )
            seen = {structure.id}
            parent_id = structure.parent_id
            while parent_id is not None:
                if parent_id in seen:
                    raise ValueError("structure parent relationships must be acyclic")
                seen.add(parent_id)
                parent_id = structures[parent_id].parent_id
            if structure.parent_id is not None:
                children_by_parent.setdefault(structure.parent_id, []).append(structure)

        for structure in self.structures:
            has_parent = structure.parent_id is not None
            has_children = bool(children_by_parent.get(structure.id))
            expected_relationship = (
                "parent_child"
                if has_parent and has_children
                else (
                    "child"
                    if has_parent
                    else ("parent" if has_children else "standalone")
                )
            )
            if structure.relationship != expected_relationship:
                raise ValueError(
                    f"structure {structure.id} relationship must be "
                    f"{expected_relationship} for its parent/child topology"
                )

        sibling_groups: dict[str | None, list[StructureCapture]] = {}
        for structure in self.structures:
            sibling_groups.setdefault(structure.parent_id, []).append(structure)
        for parent_id, siblings in sibling_groups.items():
            primary = [
                structure
                for structure in siblings
                if structure.selection == "primary"
            ]
            if len(primary) != 1:
                scope = parent_id if parent_id is not None else "top-level"
                raise ValueError(
                    "each parentId group requires exactly one primary structure "
                    f"({scope})"
                )

        if self.judgments.shape == "coil":
            if not structures:
                raise ValueError("coil labels require at least one structure")
            if self.active_structure_id is None:
                raise ValueError(
                    "coil labels require an activeStructureId"
                )
        if self.active_structure_id is not None:
            if self.active_structure_id not in structures:
                raise ValueError("activeStructureId does not exist")
            active_or_ancestor: StructureCapture | None = structures[
                self.active_structure_id
            ]
            while active_or_ancestor is not None:
                if active_or_ancestor.selection != "primary":
                    raise ValueError(
                        "activeStructureId and every ancestor must be primary"
                    )
                active_or_ancestor = (
                    structures[active_or_ancestor.parent_id]
                    if active_or_ancestor.parent_id is not None
                    else None
                )

        referenced_structure_ids = [
            value
            for value in (
                [bottom.structure_id for bottom in self.bottoms]
                + [event.structure_id for event in self.events]
                + [phase.structure_id for phase in self.phases]
            )
            if value is not None
        ]
        unknown = sorted(set(referenced_structure_ids) - set(ids))
        if unknown:
            raise ValueError(
                "bottom/event/phase references unknown structures: "
                + ", ".join(unknown)
            )

        event_state: dict[str | None, bool] = {}
        last_event_date: dict[str | None, str] = {}
        for event in self.events:
            stream = event.structure_id
            prior_date = last_event_date.get(stream)
            if prior_date is not None and event.trigger.date < prior_date:
                scope = stream if stream is not None else "global"
                raise ValueError(
                    f"events for structure {scope} must be chronological"
                )
            if (
                stream is not None
                and event.trigger.date
                < structures[stream].recognition_confirmation.date
            ):
                raise ValueError(
                    f"event {event.id} cannot precede structure recognition"
                )

            breakout_active = event_state.get(stream, False)
            if event.kind == "breakout":
                breakout_active = True
            elif event.kind == "failed_breakout":
                # A failed-breakout record contains its own escape trigger and
                # failure resolution, so it need not duplicate a breakout row.
                breakout_active = False
            elif event.kind in {"retest", "continuation", "invalidation"}:
                if not breakout_active:
                    raise ValueError(
                        f"event {event.id} requires an earlier active breakout "
                        "for the same structure"
                    )
                if event.kind == "invalidation" or (
                    event.kind == "retest" and event.retest_state == "failed"
                ):
                    breakout_active = False
            event_state[stream] = breakout_active
            last_event_date[stream] = (
                event.resolution.date
                if event.resolution is not None
                else event.trigger.date
            )

        for values, field in (
            (self.bottoms, "bottom"),
            (self.events, "event"),
            (self.phases, "phase"),
        ):
            value_ids = [value.id for value in values]
            if len(value_ids) != len(set(value_ids)):
                raise ValueError(f"{field} ids must be unique")
        return self


def _normalize_monthly_bars(
    raw_bars: list[dict[str, Any]], cutoff_date: str
) -> list[dict[str, Any]]:
    if not isinstance(raw_bars, list) or not raw_bars:
        raise ValueError("materialization requires frozen monthly bars")
    clean: list[dict[str, Any]] = []
    prior_date: str | None = None
    seen_months: set[tuple[int, int]] = set()
    for raw in raw_bars:
        if not isinstance(raw, dict):
            raise ValueError("frozen bars must be JSON objects")
        bar_date = _valid_iso_date(str(raw.get("date", "")), field="bar date")
        if bar_date > cutoff_date:
            continue
        if prior_date is not None and bar_date <= prior_date:
            raise ValueError("frozen bars must be unique and chronological")
        prior_date = bar_date
        parsed_bar_date = date.fromisoformat(bar_date)
        month_key = (parsed_bar_date.year, parsed_bar_date.month)
        if month_key in seen_months:
            raise ValueError("frozen bars must contain at most one bar per month")
        seen_months.add(month_key)
        bar = {"date": bar_date}
        for field in ("open", "high", "low", "close"):
            try:
                number = float(raw[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"bar {bar_date} {field} must be numeric") from exc
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"bar {bar_date} {field} must be finite and positive")
            bar[field] = number
        if (
            float(bar["low"]) > min(float(bar["open"]), float(bar["close"]))
            or float(bar["high"])
            < max(float(bar["open"]), float(bar["close"]))
            or float(bar["low"]) > float(bar["high"])
        ):
            raise ValueError(f"bar {bar_date} has impossible OHLC containment")
        volume = raw.get("volume")
        if volume is None:
            bar["volume"] = None
        else:
            try:
                volume_number = float(volume)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bar {bar_date} volume must be numeric") from exc
            if not math.isfinite(volume_number) or volume_number < 0:
                raise ValueError(f"bar {bar_date} volume must be finite and nonnegative")
            bar["volume"] = volume_number
        clean.append(bar)
    if not clean or clean[-1]["date"] != cutoff_date:
        raise ValueError("cutoffDate must equal a frozen monthly bar date")
    return clean


def _completed_quarterly_bars(monthly: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not monthly:
        raise ValueError("no completed quarterly bars exist at cutoffDate")
    cutoff = date.fromisoformat(str(monthly[-1]["date"]))
    cutoff_key = (cutoff.year, (cutoff.month - 1) // 3 + 1)
    if cutoff.month % 3 != 0:
        raise ValueError("cutoffDate quarter is not complete")

    aggregated = _aggregate_quarterly_display_bars(monthly)
    monthly_by_quarter: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for bar in monthly:
        parsed = date.fromisoformat(str(bar["date"]))
        key = (parsed.year, (parsed.month - 1) // 3 + 1)
        monthly_by_quarter.setdefault(key, []).append(bar)

    quarterly: list[dict[str, Any]] = []
    previous_ordinal: int | None = None
    for position, raw_quarter in enumerate(aggregated):
        key = tuple(raw_quarter["_quarter_key"])
        year, quarter_number = int(key[0]), int(key[1])
        source = monthly_by_quarter.get((year, quarter_number), [])
        expected_months = {
            quarter_number * 3 - 2,
            quarter_number * 3 - 1,
            quarter_number * 3,
        }
        actual_months = {
            date.fromisoformat(str(bar["date"])).month for bar in source
        }
        complete = len(source) == 3 and actual_months == expected_months
        if not complete:
            if position == 0 and (year, quarter_number) != cutoff_key:
                # Histories may begin mid-quarter. The leading fragment is not
                # geometry and is discarded; interior/cutoff gaps fail closed.
                continue
            raise ValueError(
                f"quarter {year}-Q{quarter_number} requires exactly one bar "
                "for each constituent calendar month"
            )

        ordinal = year * 4 + quarter_number - 1
        if previous_ordinal is not None and ordinal != previous_ordinal + 1:
            raise ValueError("completed quarterly geometry must be calendar-contiguous")
        previous_ordinal = ordinal

        quarter = {
            key: value
            for key, value in raw_quarter.items()
            if not key.startswith("_")
        }
        volumes = [bar.get("volume") for bar in source]
        if all(volume is not None for volume in volumes):
            quarter["volume"] = sum(float(volume) for volume in volumes)
        else:
            quarter["volume"] = None
        quarterly.append(quarter)

    if not quarterly or tuple(aggregated[-1]["_quarter_key"]) != cutoff_key:
        raise ValueError("cutoffDate quarter is not complete")
    return quarterly


def _materialize_click(
    click: CandleClick,
    lookup: dict[str, tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    match = lookup.get(click.date)
    if match is None:
        raise ValueError(
            f"click {click.date} is not a completed frozen 3M candle at the cutoff"
        )
    idx, bar = match
    value = float(bar[click.price_field])
    return {
        "date": click.date,
        "idx": idx,
        "price": value,
        "price_field": click.price_field,
    }


def _fit_line(
    points: list[dict[str, Any]], *, projection_idx: int
) -> dict[str, Any]:
    mean_idx = sum(float(point["idx"]) for point in points) / len(points)
    mean_price = sum(float(point["price"]) for point in points) / len(points)
    denominator = sum((float(point["idx"]) - mean_idx) ** 2 for point in points)
    if denominator <= 0:
        raise ValueError("line construction anchors must use distinct candles")
    slope = sum(
        (float(point["idx"]) - mean_idx) * (float(point["price"]) - mean_price)
        for point in points
    ) / denominator
    intercept = mean_price - slope * mean_idx
    value_at_cutoff = intercept + slope * projection_idx
    if value_at_cutoff <= 0:
        raise ValueError("derived line is non-positive at the historical cutoff")
    slope_pct_per_year = slope * 4.0 / value_at_cutoff * 100.0
    direction = (
        "rising"
        if slope_pct_per_year > 0.5
        else ("falling" if slope_pct_per_year < -0.5 else "flat")
    )
    residuals = [
        abs(float(point["price"]) - (intercept + slope * float(point["idx"])))
        / max(float(point["price"]), 1e-12)
        * 100.0
        for point in points
    ]
    return {
        "slope_per_bar": round(slope, 8),
        "slope_pct_per_year": round(slope_pct_per_year, 4),
        "intercept": round(intercept, 8),
        "value_at_cutoff": round(value_at_cutoff, 8),
        "projected_idx": projection_idx,
        "direction": direction,
        "fit_error_pct": round(sum(residuals) / len(residuals), 6),
    }


def _relative_volume(
    event_point: dict[str, Any], quarterly: list[dict[str, Any]], *, lookback: int = 8
) -> dict[str, Any]:
    idx = int(event_point["idx"])
    current = quarterly[idx].get("volume")
    prior = [
        float(bar["volume"])
        for bar in quarterly[max(0, idx - lookback) : idx]
        if bar.get("volume") is not None and float(bar["volume"]) > 0
    ]
    if current is None or float(current) <= 0 or len(prior) < 2:
        return {"lookback_bars": lookback, "ratio": None, "available": False}
    ordered = sorted(prior)
    middle = len(ordered) // 2
    baseline = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "lookback_bars": lookback,
        "ratio": round(float(current) / baseline, 4) if baseline > 0 else None,
        "available": baseline > 0,
    }


def materialize_gold_label(
    raw_capture: WholePatternGoldCapture | dict[str, Any],
    monthly_bars: list[dict[str, Any]],
) -> dict[str, Any]:
    """Snap a finalized capture to frozen bars and derive canonical truth.

    The function fails closed on future/incomplete candles, unknown structure
    references, duplicate clicks, or a cutoff that is not the last visible
    monthly candle.  No detector output is accepted as input.
    """
    capture = (
        raw_capture
        if isinstance(raw_capture, WholePatternGoldCapture)
        else WholePatternGoldCapture.model_validate(raw_capture)
    )
    monthly = _normalize_monthly_bars(monthly_bars, capture.cutoff_date)
    quarterly = _completed_quarterly_bars(monthly)
    lookup = {str(bar["date"]): (idx, bar) for idx, bar in enumerate(quarterly)}
    projection_idx = len(quarterly) - 1

    structures: list[dict[str, Any]] = []
    for structure in capture.structures:
        construction = [
            _materialize_click(click, lookup)
            for click in structure.construction_anchors
        ]
        recognition_confirmation = _materialize_click(
            structure.recognition_confirmation, lookup
        )
        supporting = [
            _materialize_click(click, lookup) for click in structure.supporting_touches
        ]
        excluded = [
            _materialize_click(click, lookup) for click in structure.excluded_highs
        ]
        lower_band = [
            _materialize_click(click, lookup) for click in structure.lower_band_touches
        ]
        line = _fit_line(construction, projection_idx=projection_idx)
        line["recognition_date"] = recognition_confirmation["date"]
        band: dict[str, Any] | None = None
        if lower_band:
            lower_line = _fit_line(lower_band, projection_idx=projection_idx)
            labeled_start_idx = min(
                int(point["idx"]) for point in construction + lower_band
            )
            for idx in range(labeled_start_idx, projection_idx + 1):
                upper_value_at_idx = (
                    float(line["intercept"])
                    + float(line["slope_per_bar"]) * idx
                )
                lower_value_at_idx = (
                    float(lower_line["intercept"])
                    + float(lower_line["slope_per_bar"]) * idx
                )
                if lower_value_at_idx >= upper_value_at_idx:
                    raise ValueError(
                        f"resistance band {structure.id} lower line must remain "
                        "below its upper line throughout the labeled interval"
                    )
            upper_value = float(line["value_at_cutoff"])
            lower_value = float(lower_line["value_at_cutoff"])
            band = {
                "lower_line": lower_line,
                "width_at_cutoff_pct": round(
                    abs(upper_value - lower_value) / upper_value * 100.0, 4
                ),
            }
        structures.append(
            {
                "id": structure.id,
                "parent_id": structure.parent_id,
                "relationship": structure.relationship,
                "selection": structure.selection,
                "role": structure.role,
                "boundary_kind": structure.boundary_kind,
                "confidence": structure.confidence,
                "construction_anchors": construction,
                "recognition_confirmation": recognition_confirmation,
                "supporting_touches": supporting,
                "excluded_highs": excluded,
                "lower_band_touches": lower_band,
                "line": line,
                "band": band,
            }
        )

    bottoms = [
        {
            "id": bottom.id,
            "structure_id": bottom.structure_id,
            "role": bottom.role,
            "confidence": bottom.confidence,
            "point": _materialize_click(bottom.click, lookup),
        }
        for bottom in capture.bottoms
    ]
    phases = [
        {
            "id": phase.id,
            "structure_id": phase.structure_id,
            "kind": phase.kind,
            "present": phase.present,
            "confidence": phase.confidence,
            "start": _materialize_click(phase.start, lookup),
            "end": _materialize_click(phase.end, lookup),
        }
        for phase in capture.phases
    ]
    events: list[dict[str, Any]] = []
    for event in capture.events:
        trigger = _materialize_click(event.trigger, lookup)
        resolution = (
            _materialize_click(event.resolution, lookup)
            if event.resolution is not None
            else None
        )
        observed_volume = _relative_volume(trigger, quarterly)
        if (
            event.relative_volume != "unavailable"
            and observed_volume["available"] is not True
        ):
            raise ValueError(
                f"event {event.id} labels relative volume without completed volume evidence"
            )
        events.append(
            {
                "id": event.id,
                "structure_id": event.structure_id,
                "kind": event.kind,
                "trigger": trigger,
                "resolution": resolution,
                "retest_state": event.retest_state,
                "relative_volume_label": event.relative_volume,
                "relative_volume_observed": observed_volume,
                "action_signal": event.action_signal,
                "confidence": event.confidence,
            }
        )

    active = next(
        (
            structure
            for structure in structures
            if structure["id"] == capture.active_structure_id
        ),
        None,
    )
    first_actionable = min(
        (
            str(event["trigger"]["date"])
            for event in events
            if event["action_signal"]
            and (
                event["kind"] in {"breakout", "continuation"}
                or (
                    event["kind"] == "retest"
                    and event["retest_state"] in {"shallow", "deep"}
                )
            )
            and (
                event["structure_id"] is None
                or event["structure_id"] == capture.active_structure_id
            )
        ),
        default=None,
    )
    first_watch = min(
        (
            str(phase["start"]["date"])
            for phase in phases
            if phase["kind"] == "compression"
            and phase["present"]
            and (
                phase["structure_id"] is None
                or phase["structure_id"] == capture.active_structure_id
            )
        ),
        default=None,
    )
    first_recognizable = active["line"]["recognition_date"] if active else None
    ordered_milestones = [
        value for value in (first_recognizable, first_watch, first_actionable) if value
    ]
    if ordered_milestones != sorted(ordered_milestones):
        raise ValueError(
            "derived recognition/watch/action milestones must be chronological"
        )
    normalized_capture = capture.model_dump(mode="json", by_alias=True)
    materialized: dict[str, Any] = {
        "schema_version": MATERIALIZED_SCHEMA_VERSION,
        "kind": MATERIALIZED_KIND,
        "episode_id": capture.episode_id,
        "setup_id": capture.setup_id,
        "evaluation_role": capture.evaluation_role,
        "labeler": capture.labeler,
        "labeled_at": capture.labeled_at.isoformat(),
        "cutoff_date": capture.cutoff_date,
        "decision_as_of": capture.decision_as_of,
        "interval": capture.interval,
        "outcome_visible_during_label": capture.outcome_visible_during_label,
        "judgments": capture.judgments.model_dump(mode="json"),
        "active_structure_id": capture.active_structure_id,
        "top_review_complete": capture.top_review_complete,
        "structures": structures,
        "bottoms": bottoms,
        "events": events,
        "phases": phases,
        "derived_dates": {
            "first_recognizable_date": first_recognizable,
            "first_watch_date": first_watch,
            "first_actionable_date": first_actionable,
        },
        "source_evidence": [
            evidence.model_dump(mode="json", by_alias=False)
            for evidence in capture.source_evidence
        ],
        "notes": capture.notes,
        "capture": normalized_capture,
        "provenance": {
            "capture_sha256": sha256_json(normalized_capture),
            "bars_through_cutoff_sha256": sha256_json(monthly),
            "completed_quarterly_bars_sha256": sha256_json(quarterly),
            "monthly_bar_count": len(monthly),
            "completed_quarterly_bar_count": len(quarterly),
        },
    }
    materialized["label_sha256"] = sha256_json(materialized)
    return materialized


def validate_materialized_gold_label(raw: Any) -> dict[str, Any]:
    """Validate a materialized label's envelope and self-contained hashes.

    Canonical labels should normally be produced by :func:`materialize_gold_label`.
    This function cannot verify derived geometry without frozen bars. Callers
    making truth claims must rematerialize ``raw["capture"]`` with those bars
    and require canonical equality, as the automatic evaluator does.
    """
    if not isinstance(raw, dict):
        raise ValueError("materialized gold label must be a JSON object")
    if raw.get("schema_version") != MATERIALIZED_SCHEMA_VERSION:
        raise ValueError("unsupported materialized gold-label schema")
    if raw.get("kind") != MATERIALIZED_KIND:
        raise ValueError("unexpected materialized gold-label kind")
    expected_hash = raw.get("label_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_hash
    ):
        raise ValueError("materialized gold label requires label_sha256")
    unhashed = dict(raw)
    unhashed.pop("label_sha256", None)
    if expected_hash != sha256_json(unhashed):
        raise ValueError("materialized gold label content hash does not match")
    required = (
        "episode_id",
        "setup_id",
        "evaluation_role",
        "cutoff_date",
        "decision_as_of",
        "judgments",
        "top_review_complete",
        "structures",
        "events",
        "phases",
        "capture",
        "provenance",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError("materialized gold label is missing: " + ", ".join(missing))

    embedded_capture = raw["capture"]
    try:
        capture = WholePatternGoldCapture.model_validate(embedded_capture)
    except (TypeError, ValueError) as exc:
        raise ValueError("materialized gold label contains an invalid capture") from exc
    normalized_capture = capture.model_dump(mode="json", by_alias=True)
    if embedded_capture != normalized_capture:
        raise ValueError("materialized gold label capture is not normalized")

    provenance = raw["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("materialized gold label provenance must be an object")
    if provenance.get("capture_sha256") != sha256_json(normalized_capture):
        raise ValueError("materialized gold label capture hash does not match")

    expected_top_level = {
        "episode_id": capture.episode_id,
        "setup_id": capture.setup_id,
        "evaluation_role": capture.evaluation_role,
        "labeler": capture.labeler,
        "labeled_at": capture.labeled_at.isoformat(),
        "cutoff_date": capture.cutoff_date,
        "decision_as_of": capture.decision_as_of,
        "interval": capture.interval,
        "outcome_visible_during_label": capture.outcome_visible_during_label,
        "judgments": capture.judgments.model_dump(mode="json"),
        "active_structure_id": capture.active_structure_id,
        "top_review_complete": capture.top_review_complete,
        "source_evidence": [
            evidence.model_dump(mode="json", by_alias=False)
            for evidence in capture.source_evidence
        ],
        "notes": capture.notes,
    }
    inconsistent = [
        key for key, value in expected_top_level.items() if raw.get(key) != value
    ]
    if inconsistent:
        raise ValueError(
            "materialized gold label disagrees with its capture: "
            + ", ".join(inconsistent)
        )
    return raw


def validate_materialized_gold_label_against_bars(
    raw: Any,
    monthly_bars: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate envelope integrity and reproduce every derived field."""
    validated = validate_materialized_gold_label(raw)
    rematerialized = materialize_gold_label(validated["capture"], monthly_bars)
    if canonical_json(rematerialized) != canonical_json(validated):
        raise ValueError(
            "materialized gold label differs from frozen-bar rematerialization"
        )
    return rematerialized
