"""Schema-v5 capture-only human-learning contract and evidence validation."""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CaptureReviewedHigh(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    date: str
    price: float = Field(gt=0)
    role: Literal[
        "major_top",
        "structural_retest",
        "provisional_top",
        "breakout_peak",
    ] = "major_top"
    lid_member: bool | None = Field(default=None, alias="lidMember")

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("reviewed-high dates must use YYYY-MM-DD")
        return value

    @field_validator("price")
    @classmethod
    def finite_price(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reviewed-high prices must be finite")
        return value


class BlindResistanceBand(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lower: float = Field(gt=0)
    upper: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered_band(self) -> "BlindResistanceBand":
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("blind resistance-band values must be finite")
        if self.lower > self.upper:
            raise ValueError("blind resistance-band lower must not exceed upper")
        return self


class BlindAssessment(BaseModel):
    """Complete human chart judgment frozen before either detector is shown."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    pattern_label: Literal["coil", "not_coil", "uncertain"] = Field(
        alias="patternLabel"
    )
    lifecycle_label: Literal[
        "no_pattern",
        "watch_immature",
        "forming",
        "pre_breakout",
        "breakout_provisional",
        "breaking_out",
        "failed_breakout",
        "retest",
        "post_breakout",
        "uncertain_structure",
    ] = Field(alias="lifecycleLabel")
    human_tops: list[CaptureReviewedHigh] = Field(
        default_factory=list, max_length=100, alias="humanTops"
    )
    resistance_band: BlindResistanceBand | None = Field(
        default=None, alias="resistanceBand"
    )
    first_chart_displayed_at: datetime = Field(alias="firstChartDisplayedAt")

    @field_validator("first_chart_displayed_at")
    @classmethod
    def timezone_aware_first_display(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("firstChartDisplayedAt must include a timezone")
        return value

    @model_validator(mode="after")
    def complete_blind_geometry(self) -> "BlindAssessment":
        dates = [point.date for point in self.human_tops]
        if len(set(dates)) != len(dates):
            raise ValueError("blind human-top dates must be unique")
        if dates != sorted(dates):
            raise ValueError("blind human tops must be chronological")
        if self.pattern_label == "coil":
            if not self.human_tops:
                raise ValueError("blind coil assessments require a human top")
            if self.resistance_band is None:
                raise ValueError("blind coil assessments require a resistance band")
        return self


class DetectorReviewTiming(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    first_chart_displayed_at: datetime | None = Field(
        default=None, alias="firstChartDisplayedAt"
    )
    blind_assessment_locked_at: datetime | None = Field(
        default=None, alias="blindAssessmentLockedAt"
    )
    detectors_revealed_at: datetime | None = Field(
        default=None, alias="detectorsRevealedAt"
    )
    finalized_at: datetime | None = Field(default=None, alias="finalizedAt")
    blind_active_seconds: int | None = Field(
        default=None, ge=0, alias="blindActiveSeconds"
    )
    assisted_active_seconds: int | None = Field(
        default=None, ge=0, alias="assistedActiveSeconds"
    )
    review_order: Literal["blind_first", "assisted_first"] = Field(
        alias="reviewOrder"
    )
    manual_pattern_label: Literal["coil", "not_coil", "uncertain"] | None = Field(
        default=None, alias="manualPatternLabel"
    )
    manual_lifecycle_label: Literal[
        "no_pattern",
        "watch_immature",
        "forming",
        "pre_breakout",
        "breakout_provisional",
        "breaking_out",
        "failed_breakout",
        "retest",
        "post_breakout",
        "uncertain_structure",
    ] | None = Field(default=None, alias="manualLifecycleLabel")
    manual_confidence: Literal["high", "medium", "low"] | None = Field(
        default=None, alias="manualConfidence"
    )
    assisted_pattern_label: Literal["coil", "not_coil", "uncertain"] | None = Field(
        default=None, alias="assistedPatternLabel"
    )
    assisted_lifecycle_label: Literal[
        "no_pattern",
        "watch_immature",
        "forming",
        "pre_breakout",
        "breakout_provisional",
        "breaking_out",
        "failed_breakout",
        "retest",
        "post_breakout",
        "uncertain_structure",
    ] | None = Field(default=None, alias="assistedLifecycleLabel")
    assisted_confidence: Literal["high", "medium", "low"] | None = Field(
        default=None, alias="assistedConfidence"
    )

    @field_validator(
        "first_chart_displayed_at",
        "blind_assessment_locked_at",
        "detectors_revealed_at",
        "finalized_at",
    )
    @classmethod
    def timezone_aware_timing(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("detector timing timestamps must include a timezone")
        return value


class DetectorReview(BaseModel):
    """Append-only judgment of the isolated v2.4 validation detector."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    algorithm_variant: Literal["v2_4_validation"] = Field(alias="algorithmVariant")
    config_fingerprint: str = Field(
        min_length=71, max_length=71, pattern=r"^sha256:[0-9a-f]{64}$",
        alias="configFingerprint",
    )
    algorithm_mode: Literal["algorithm_only"] = Field(alias="algorithmMode")
    accepted_top_ids: list[str] = Field(default_factory=list, alias="acceptedTopIds")
    rejected_top_ids: list[str] = Field(default_factory=list, alias="rejectedTopIds")
    missing_human_tops: list[str] = Field(
        default_factory=list, alias="missingHumanTops"
    )
    matched_human_tops: list[str] = Field(
        default_factory=list, alias="matchedHumanTops"
    )
    acceptable_hypothesis_ids: list[str] = Field(
        default_factory=list, alias="acceptableHypothesisIds"
    )
    rejected_hypothesis_ids: list[str] = Field(
        default_factory=list, alias="rejectedHypothesisIds"
    )
    pattern_label: Literal["coil", "not_coil", "uncertain"] = Field(
        alias="patternLabel"
    )
    lifecycle_label: Literal[
        "no_pattern",
        "watch_immature",
        "forming",
        "pre_breakout",
        "breakout_provisional",
        "breaking_out",
        "failed_breakout",
        "retest",
        "post_breakout",
        "uncertain_structure",
    ] = Field(alias="lifecycleLabel")
    confidence: Literal["high", "medium", "low"]
    reason_codes: list[str] = Field(min_length=1, max_length=20, alias="reasonCodes")
    timing: DetectorReviewTiming = Field(default_factory=DetectorReviewTiming)

    @field_validator("missing_human_tops", "matched_human_tops")
    @classmethod
    def valid_missing_human_top_dates(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("human-top decisions cannot contain duplicates")
        for value in values:
            parsed = date.fromisoformat(value)
            if parsed.isoformat() != value:
                raise ValueError("human-top decisions must use YYYY-MM-DD dates")
        if values != sorted(values):
            raise ValueError("human-top decisions must be chronological")
        return values

    @model_validator(mode="after")
    def disjoint_decisions(self) -> "DetectorReview":
        pairs = (
            (self.accepted_top_ids, self.rejected_top_ids, "top"),
            (
                self.acceptable_hypothesis_ids,
                self.rejected_hypothesis_ids,
                "hypothesis",
            ),
        )
        for accepted, rejected, name in pairs:
            if len(set(accepted)) != len(accepted) or len(set(rejected)) != len(rejected):
                raise ValueError(f"{name} decision ids must be unique")
            if set(accepted) & set(rejected):
                raise ValueError(f"accepted and rejected {name} ids must be disjoint")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("reasonCodes cannot contain duplicates")
        if set(self.missing_human_tops) & set(self.matched_human_tops):
            raise ValueError("matched and missing human tops must be disjoint")
        return self


class CaptureEvidencePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1, max_length=100)
    sequence: int = Field(ge=1, le=100)
    date: str
    price: float = Field(gt=0)
    price_field: Literal["high", "low", "close"] = Field(alias="priceField")
    role: Literal[
        "left_shoulder",
        "head",
        "right_shoulder",
        "neckline",
        "breakout",
        "major_top",
        "support",
        "slope_anchor",
        "other",
    ]
    label: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("evidence dates must use YYYY-MM-DD")
        return value

    @field_validator("price")
    @classmethod
    def finite_price(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("evidence prices must be finite")
        return value


class CaptureExceptionRule(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(min_length=4, max_length=160)
    pattern_kind: Literal[
        "inverse_head_shoulders",
        "head_shoulders",
        "rounded_base",
        "failed_breakout",
        "regime_change",
        "alternate_lid",
        "other",
    ] = Field(alias="patternKind")
    applicability: str = Field(min_length=20, max_length=5000)
    exclusions: str = Field(default="", max_length=5000)
    detection_logic: str = Field(default="", max_length=10000, alias="detectionLogic")
    confirmation: str = Field(default="", max_length=5000)
    proposed_action: Literal[
        "include_candidate",
        "exclude_candidate",
        "adjust_tops",
        "adjust_lid_or_slope",
        "hold_for_human_review",
    ] = Field(alias="proposedAction")
    impacted_stages: list[
        Literal["screening", "top_detection", "lid_slope", "lifecycle", "grading"]
    ] = Field(default_factory=list, max_length=5, alias="impactedStages")
    validation_plan: str = Field(default="", max_length=5000, alias="validationPlan")
    evidence: list[CaptureEvidencePoint] = Field(min_length=2, max_length=100)

    @field_validator(
        "name",
        "applicability",
    )
    @classmethod
    def nonblank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("rule text fields cannot be blank")
        return normalized

    @model_validator(mode="after")
    def valid_evidence_shape(self) -> "CaptureExceptionRule":
        if len(set(self.impacted_stages)) != len(self.impacted_stages):
            raise ValueError("impactedStages cannot contain duplicates")
        ordered = sorted(self.evidence, key=lambda point: point.sequence)
        if [point.sequence for point in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError("evidence sequence must be contiguous starting at 1")
        if len({point.id for point in ordered}) != len(ordered):
            raise ValueError("evidence ids must be unique")
        if len({point.date for point in ordered}) != len(ordered):
            raise ValueError("each evidence point must use a distinct candle")
        if [point.date for point in ordered] != sorted(point.date for point in ordered):
            raise ValueError("evidence sequence must be chronological")

        if self.pattern_kind == "inverse_head_shoulders":
            by_role = {point.role: point for point in ordered}
            required = ("left_shoulder", "head", "right_shoulder")
            if any(role not in by_role for role in required):
                raise ValueError(
                    "inverse head and shoulders requires left shoulder, head, "
                    "and right shoulder evidence"
                )
            left, head, right = (by_role[role] for role in required)
            if not (left.date < head.date < right.date):
                raise ValueError(
                    "inverse head and shoulders evidence must be chronological"
                )
            if any(point.price_field != "low" for point in (left, head, right)):
                raise ValueError(
                    "inverse head and shoulders shoulders/head must snap to candle lows"
                )
            if not (head.price < left.price and head.price < right.price):
                raise ValueError(
                    "inverse head and shoulders head must be below both shoulders"
                )
        return self


class ReviewLearningCaptureV5(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    reviewer_name: str = Field(min_length=2, max_length=120, alias="reviewerName")
    sequence_policy_version: Literal[1] = Field(alias="sequencePolicyVersion")
    base_assessment_locked: bool = Field(alias="baseAssessmentLocked")
    base_path: Literal["base_pattern", "exception_territory", "uncertain"] = Field(
        alias="basePath"
    )
    failed_base_rules: list[
        Literal[
            "repeated_ceiling",
            "long_duration",
            "compression",
            "near_lid",
            "not_broken_out",
            "trend_shape",
            "top_geometry",
            "data_quality",
            "other",
        ]
    ] = Field(default_factory=list, max_length=9, alias="failedBaseRules")
    base_rationale: str = Field(default="", max_length=5000, alias="baseRationale")
    exception_verdict: Literal[
        "not_evaluated", "applies", "does_not_apply", "uncertain"
    ] = Field(alias="exceptionVerdict")
    exception_rationale: str = Field(
        default="", max_length=5000, alias="exceptionRationale"
    )
    rule_proposal: CaptureExceptionRule | None = Field(
        default=None, alias="ruleProposal"
    )
    commentary: str = Field(default="", max_length=10000)

    @field_validator("reviewer_name")
    @classmethod
    def normalize_reviewer(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def sequential_policy(self) -> "ReviewLearningCaptureV5":
        if not self.base_assessment_locked:
            raise ValueError(
                "base assessment must be locked before model evidence is reviewed"
            )
        if len(set(self.failed_base_rules)) != len(self.failed_base_rules):
            raise ValueError("failedBaseRules cannot contain duplicates")
        if self.base_path == "base_pattern":
            if self.failed_base_rules:
                raise ValueError("base-pattern reviews cannot retain failed base rules")
            if (
                self.exception_verdict != "not_evaluated"
                or self.rule_proposal is not None
            ):
                raise ValueError(
                    "exceptions may be evaluated only after leaving the base path"
                )
            return self

        if len(self.base_rationale.strip()) < 20:
            raise ValueError(
                "non-base reviews require a base rationale of at least 20 characters"
            )
        if self.base_path == "exception_territory" and not self.failed_base_rules:
            raise ValueError(
                "exception territory requires at least one failed base rule"
            )
        if self.exception_verdict == "not_evaluated":
            raise ValueError("non-base reviews require an exception verdict")
        if self.exception_verdict == "applies":
            if self.rule_proposal is None:
                raise ValueError("an applicable exception requires a rule proposal")
        else:
            if len(self.exception_rationale.strip()) < 20:
                raise ValueError(
                    "exception verdict requires a rationale of at least 20 characters"
                )
            if self.rule_proposal is not None:
                raise ValueError(
                    "a reusable rule proposal is allowed only when an exception applies"
                )
        return self

    def derived_coil_label(self) -> Literal["coil", "not_coil", "uncertain"]:
        if self.base_path == "base_pattern" or self.exception_verdict == "applies":
            return "coil"
        if (
            self.base_path == "exception_territory"
            and self.exception_verdict == "does_not_apply"
        ):
            return "not_coil"
        return "uncertain"


class CaptureFinalizeRequest(BaseModel):
    """One immutable capture event plus optimistic/idempotent request metadata."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[5] = Field(alias="schemaVersion")
    label_policy_version: Literal[2] = Field(alias="labelPolicyVersion")
    session_id: int = Field(ge=1, alias="sessionId")
    ticker: str
    interval: Literal["3M"] = "3M"
    as_of: str = Field(alias="asOf")
    algorithm_version: str = Field(alias="algorithmVersion")
    decision: Literal["approved", "corrected"]
    coil_label: Literal["coil", "not_coil", "uncertain"] = Field(alias="coilLabel")
    human_grade: Literal["A", "B", "C"] | None = Field(
        default=None, alias="humanGrade"
    )
    confidence: Literal["high", "low"]
    note: str | None = Field(default=None, max_length=2000)
    learning_capture: ReviewLearningCaptureV5 = Field(alias="learningCapture")
    detector_review: DetectorReview = Field(alias="detectorReview")
    algorithm: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    reviewed_highs: list[CaptureReviewedHigh] = Field(
        default_factory=list, max_length=100, alias="reviewedHighs"
    )
    created_at: datetime | None = Field(default=None, alias="createdAt")
    idempotency_key: str = Field(
        min_length=8, max_length=200, alias="idempotencyKey"
    )
    expected_draft_revision: int = Field(ge=0, alias="expectedDraftRevision")
    sample_id: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$", alias="sampleId"
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")
        return symbol

    @field_validator("as_of")
    @classmethod
    def valid_as_of(cls, value: str) -> str:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("asOf must use YYYY-MM-DD")
        return value

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def coherent_verdict(self) -> "CaptureFinalizeRequest":
        derived = self.learning_capture.derived_coil_label()
        if self.coil_label != derived:
            raise ValueError(
                f"coilLabel must be '{derived}' for the recorded sequential review"
            )
        if self.coil_label == "coil" and self.human_grade is None:
            raise ValueError("coil labels require a humanGrade")
        if self.coil_label != "coil" and self.human_grade is not None:
            raise ValueError("humanGrade applies only to coil labels")
        if (
            self.decision == "corrected" or self.coil_label == "not_coil"
        ) and self.note is None:
            raise ValueError("corrected and not-coil reviews require a note")
        if self.decision == "approved":
            if self.reviewed_highs:
                raise ValueError("approved reviews cannot include reviewedHighs")
            return self

        if len({point.date for point in self.reviewed_highs}) != len(
            self.reviewed_highs
        ):
            raise ValueError("reviewedHighs dates must be unique")
        members = [point for point in self.reviewed_highs if point.lid_member]
        if members:
            if len(members) < 2:
                raise ValueError("corrected reviews require at least two lid members")
            if any(point.role == "breakout_peak" for point in members):
                raise ValueError("breakout peaks cannot be lid members")
        elif (
            len(
                [
                    point
                    for point in self.reviewed_highs
                    if point.role != "breakout_peak"
                ]
            )
            < 2
        ):
            raise ValueError(
                "corrected reviews require at least two non-breakout anchors"
            )
        return self


class CaptureDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    expected_revision: int = Field(ge=0, alias="expectedRevision")
    payload: dict[str, Any]


class BaseClassification(BaseModel):
    """Blind Step-1 verdict persisted before any model evidence is revealed."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    locked: Literal[True]
    base_path: Literal["base_pattern", "exception_territory", "uncertain"] = Field(
        alias="basePath"
    )
    failed_base_rules: list[
        Literal[
            "repeated_ceiling",
            "long_duration",
            "compression",
            "near_lid",
            "not_broken_out",
            "trend_shape",
            "top_geometry",
            "data_quality",
            "other",
        ]
    ] = Field(default_factory=list, max_length=9, alias="failedBaseRules")
    rationale: str = Field(max_length=5000)
    blind_assessment: BlindAssessment = Field(alias="blindAssessment")

    @field_validator("rationale")
    @classmethod
    def normalized_rationale(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def coherent_base_path(self) -> "BaseClassification":
        if len(set(self.failed_base_rules)) != len(self.failed_base_rules):
            raise ValueError("failedBaseRules cannot contain duplicates")
        if self.base_path == "base_pattern" and self.failed_base_rules:
            raise ValueError("base-pattern locks cannot retain failed base rules")
        if self.base_path == "exception_territory" and not self.failed_base_rules:
            raise ValueError(
                "exception-territory locks require at least one failed base rule"
            )
        if self.base_path != "base_pattern" and len(self.rationale) < 20:
            raise ValueError(
                "exception and uncertain classifications need at least 20 rationale characters"
            )
        expected_pattern = {
            "base_pattern": "coil",
            "exception_territory": "not_coil",
            "uncertain": "uncertain",
        }[self.base_path]
        if (
            self.blind_assessment is not None
            and self.blind_assessment.pattern_label != expected_pattern
        ):
            raise ValueError(
                "blindAssessment patternLabel must match the locked base path"
            )
        return self


class BaseClassificationLockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    expected_draft_revision: int = Field(ge=1, alias="expectedDraftRevision")
    base_classification: BaseClassification = Field(alias="baseClassification")


def distinct_quarter_matches(
    human_quarters: list[int], candidate_quarters: list[int]
) -> bool:
    """Maximum one-to-one matching for sorted ±1-quarter interval edges."""
    candidates = sorted(candidate_quarters)
    candidate_index = 0
    for human in sorted(human_quarters):
        while (
            candidate_index < len(candidates)
            and candidates[candidate_index] < human - 1
        ):
            candidate_index += 1
        if (
            candidate_index >= len(candidates)
            or candidates[candidate_index] > human + 1
        ):
            return False
        candidate_index += 1
    return True


def validate_blind_assessment_against_context(
    assessment: BlindAssessment | None, context: dict[str, Any]
) -> None:
    if assessment is None:
        return
    bars = {str(bar["date"]): bar for bar in context.get("quarterly_bars", [])}
    monthly_bars = context.get("monthly_bars", [])
    quarterly_bars = context.get("quarterly_bars", [])
    incomplete_quarter_date = None
    if monthly_bars and quarterly_bars:
        last_month = date.fromisoformat(str(monthly_bars[-1]["date"]))
        if last_month.month % 3 != 0:
            incomplete_quarter_date = str(quarterly_bars[-1]["date"])
    for point in assessment.human_tops:
        bar = bars.get(point.date)
        if bar is None:
            raise ValueError(
                f"blind human top {point.date} is not a frozen 3M candle"
            )
        if point.date == incomplete_quarter_date:
            raise ValueError(
                "blind human tops cannot use the incomplete final quarter"
            )
        expected = float(bar["high"])
        tolerance = max(1e-8, abs(expected) * 1e-8)
        if abs(float(point.price) - expected) > tolerance:
            raise ValueError(
                f"blind human top {point.date} price must equal the frozen "
                f"candle high ({expected})"
            )


def validate_capture_against_context(
    request: CaptureFinalizeRequest,
    context: dict[str, Any],
    *,
    base_classification: dict[str, Any] | None = None,
) -> None:
    """Verify server-owned sample identity and candle-snapped evidence."""
    if not context.get("reviewable"):
        raise ValueError(
            "this frozen sample is quarantined for blocking data-quality issues; "
            "it may only be skipped with an explicit reason"
        )
    if request.sample_id != context["sample_id"]:
        raise ValueError("sampleId does not match the frozen review sample")
    if request.algorithm_version != context["analysis"].get("algorithm_version"):
        raise ValueError("algorithmVersion does not match the frozen analysis")
    if request.as_of != context["monthly_bars"][-1]["date"]:
        raise ValueError("asOf does not match the frozen data date")

    if request.detector_review is not None:
        automatic_comparison = request.detector_review.reason_codes == [
            "automatic_label_comparison"
        ]
        detector = (context.get("detector_outputs") or {}).get(
            request.detector_review.algorithm_variant
        )
        if not isinstance(detector, dict):
            raise ValueError("validation detector output is unavailable")
        metadata = detector.get("analysis_metadata") or {}
        if request.detector_review.algorithm_mode != metadata.get("mode"):
            raise ValueError("detectorReview algorithmMode does not match the output")
        if request.detector_review.config_fingerprint != metadata.get(
            "config_fingerprint"
        ):
            raise ValueError("detectorReview configFingerprint does not match the output")
        corpus_labels = context.get("corpus_labels") or {}
        if corpus_labels.get("benchmark_attempt") == 2:
            timing = request.detector_review.timing
            assigned_order = corpus_labels.get("benchmark_timing_order")
            if timing.review_order != assigned_order:
                raise ValueError(
                    "detectorReview reviewOrder does not match the frozen benchmark assignment"
                )
            if (
                not automatic_comparison
                and (not timing.blind_active_seconds or not timing.assisted_active_seconds)
            ):
                raise ValueError(
                    "repeat benchmark reviews require positive manual and assisted active seconds"
                )
            if not automatic_comparison and any(
                value is None
                for value in (
                    timing.manual_pattern_label,
                    timing.manual_lifecycle_label,
                    timing.manual_confidence,
                    timing.assisted_pattern_label,
                    timing.assisted_lifecycle_label,
                    timing.assisted_confidence,
                )
            ):
                raise ValueError(
                    "repeat benchmark reviews require complete manual and assisted judgments"
                )
        if automatic_comparison:
            manual_decisions = (
                request.detector_review.accepted_top_ids,
                request.detector_review.rejected_top_ids,
                request.detector_review.missing_human_tops,
                request.detector_review.matched_human_tops,
                request.detector_review.acceptable_hypothesis_ids,
                request.detector_review.rejected_hypothesis_ids,
            )
            if any(manual_decisions):
                raise ValueError(
                    "automatic detector comparison cannot contain manual adjudications"
                )
            blind = (base_classification or {}).get("blindAssessment") or {}
            if request.detector_review.pattern_label != blind.get("patternLabel"):
                raise ValueError(
                    "automatic detector comparison must preserve the locked pattern label"
                )
            if request.detector_review.lifecycle_label != blind.get("lifecycleLabel"):
                raise ValueError(
                    "automatic detector comparison must preserve the locked lifecycle label"
                )
        else:
            known_top_ids = {
                str(item.get("id"))
                for item in detector.get("top_candidates", [])
                if isinstance(item, dict) and item.get("id")
            }
            reviewed_top_ids = set(request.detector_review.accepted_top_ids) | set(
                request.detector_review.rejected_top_ids
            )
            if reviewed_top_ids != known_top_ids:
                raise ValueError(
                    "detectorReview must accept or reject every validation top exactly once"
                )
            known_hypothesis_ids = {
                str(item.get("id"))
                for item in detector.get("lid_hypotheses", [])
                if isinstance(item, dict) and item.get("id")
            }
            reviewed_hypothesis_ids = set(
                request.detector_review.acceptable_hypothesis_ids
            ) | set(request.detector_review.rejected_hypothesis_ids)
            if reviewed_hypothesis_ids != known_hypothesis_ids:
                raise ValueError(
                    "detectorReview must accept or reject every lid hypothesis exactly once"
                )
            blind = (base_classification or {}).get("blindAssessment") or {}
            human_dates = {
                str(point.get("date"))
                for point in blind.get("humanTops", [])
                if isinstance(point, dict) and point.get("date")
            }
            candidate_dates = {
                str(item.get("peak_date"))
                for item in detector.get("top_candidates", [])
                if isinstance(item, dict)
                and item.get("peak_date")
                and str(item.get("id")) in request.detector_review.accepted_top_ids
            }
            reviewed_human_dates = set(
                request.detector_review.missing_human_tops
            ) | set(request.detector_review.matched_human_tops)
            if reviewed_human_dates != human_dates:
                raise ValueError(
                    "detectorReview must mark every locked human top matched or missing"
                )

            def quarter_ordinal(value: str) -> int:
                parsed = date.fromisoformat(value)
                return parsed.year * 4 + (parsed.month - 1) // 3

            candidate_quarters = sorted(
                quarter_ordinal(value) for value in candidate_dates
            )
            human_quarters = [
                quarter_ordinal(value)
                for value in request.detector_review.matched_human_tops
            ]

            if not distinct_quarter_matches(human_quarters, candidate_quarters):
                raise ValueError(
                    "matchedHumanTops require distinct accepted detector candidates "
                    "within one completed quarter"
                )

    bars = {str(bar["date"]): bar for bar in context["quarterly_bars"]}
    reviewed_dates = [point.date for point in request.reviewed_highs]
    if reviewed_dates != sorted(reviewed_dates):
        raise ValueError("reviewedHighs must be chronological")
    last_month = date.fromisoformat(context["monthly_bars"][-1]["date"])
    incomplete_quarter_date = (
        context["quarterly_bars"][-1]["date"]
        if last_month.month % 3 != 0 and context["quarterly_bars"]
        else None
    )
    for point in request.reviewed_highs:
        bar = bars.get(point.date)
        if bar is None:
            raise ValueError(
                f"reviewed high date {point.date} is not a frozen 3M candle"
            )
        if point.date == incomplete_quarter_date:
            raise ValueError(
                "reviewed highs cannot use the incomplete final quarter"
            )
        expected = float(bar["high"])
        tolerance = max(1e-8, abs(expected) * 1e-8)
        if abs(float(point.price) - expected) > tolerance:
            raise ValueError(
                f"reviewed high {point.date} price must equal the frozen "
                f"candle high ({expected})"
            )

    proposal = request.learning_capture.rule_proposal
    if proposal is None:
        return
    for point in proposal.evidence:
        bar = bars.get(point.date)
        if bar is None:
            raise ValueError(
                f"evidence date {point.date} is not a frozen 3M candle"
            )
        expected = float(bar[point.price_field])
        tolerance = max(1e-8, abs(expected) * 1e-8)
        if abs(float(point.price) - expected) > tolerance:
            raise ValueError(
                f"evidence {point.id} price must equal the frozen candle "
                f"{point.price_field} ({expected})"
            )
