"""Export finalized fresh reviews into the frozen geometry-lab v1 boundary.

The exporter is deliberately offline and model-blind.  It joins a finalized
schema-v4 review export only to the saved-run snapshot and manifest identities
that were frozen with that review.  Detector output, review metadata, labels,
outcomes, volume, and post-cutoff bars never enter a model-visible artifact.

Frozen geometry-lab v1 has no partition field.  Consequently each populated
issuer split is emitted as its own strict pair under separate
``model-visible/`` and ``evaluation-only/`` roots, then ``development/``,
``validation/``, or ``sealed-holdout/``.  The field named ``daily_bars`` is
retained verbatim for contract compatibility even though the source evidence
currently consists of exact frozen monthly bars.  Frozen v1 also mandates the
literal adjustment claim ``split-dividend-adjusted``, while the backend
snapshots carry no adjustment provenance.  No price transformation is applied:
these artifacts are suitable only for the lab's monthly out-of-distribution
lane, never its daily or weekly lanes.

Opaque IDs use a caller-supplied stable HMAC key.  The key must stay private and
unchanged across exports; rotating it deliberately changes issuer membership
and every derived artifact identity.

Example::

    python geometry_lab_export.py finalized-review.json \
        --output-dir /secure/geometry-export \
        --opaque-id-key-file /secure/coilingview-geometry-id.key
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

from gold_labels import (
    WholePatternGoldCapture,
    canonical_json as canonical_gold_json,
    materialize_gold_label,
    sha256_json as backend_sha256_json,
    validate_materialized_gold_label_against_bars,
)
from review_capture import (
    BaseClassification,
    ReviewLearningCaptureV5,
    validate_api_whole_pattern_capture,
)
from review_snapshots import (
    ReviewSnapshotError,
    load_blind_review_context,
    load_review_manifest,
    review_snapshot_identity,
    verify_manifest_identity,
)


CORPUS_SCHEMA: Final = "geometry-input-corpus.v1"
REFERENCE_SCHEMA: Final = "geometry-reference-key.v1"
ADJUSTMENT: Final = "split-dividend-adjusted"
DEFAULT_SPLIT_SEED: Final = "coilingview-geometry-lab:issuer-split:v1"
LAB_CONTRACT_COMMIT: Final = "82d27bae5b91e7afde14821a7fc684841f2638cd"

PARTITION_DEVELOPMENT: Final = "development"
PARTITION_VALIDATION: Final = "validation"
PARTITION_SEALED_HOLDOUT: Final = "sealed_holdout"
PARTITIONS: Final = (
    PARTITION_DEVELOPMENT,
    PARTITION_VALIDATION,
    PARTITION_SEALED_HOLDOUT,
)

_PARTITION_DIRECTORIES: Final = {
    PARTITION_DEVELOPMENT: "development",
    PARTITION_VALIDATION: "validation",
    PARTITION_SEALED_HOLDOUT: "sealed-holdout",
}
_SPLIT_BUCKETS = 10_000
_DEVELOPMENT_END = 6_000
_VALIDATION_END = 8_000
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_SAMPLE_ID_RE = re.compile(r"^smp_[0-9a-f]{16,64}$")
_ISSUER_ID_RE = re.compile(r"^iss_[0-9a-f]{16,64}$")
_SETUP_ID_RE = re.compile(r"^set_[0-9a-f]{16,64}$")
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.^=_-]{0,31}$")
_FUTURE_CLOCK_TOLERANCE = timedelta(minutes=5)

_REQUIRED_MANIFEST_ITEM_HASHES: Final = (
    "backend_bars_identity_sha256",
    "snapshot_sha256",
    "screen_snapshot_sha256",
)
_LAB_SCHEMA_SHA256: Final = {
    "geometry-embed-request.v1.schema.json": (
        "a4f59e60f0e4c03a0a89e89d14f7dd75f765f0b26832eb5af8a229e7446bdd9f"
    ),
    "geometry-index-manifest.v1.schema.json": (
        "771d2b0ea5e5d4c5f6ce99d15b2999414eda6ede88efa0a94021c1747b977334"
    ),
    "geometry-input-corpus.v1.schema.json": (
        "74f4da763e05b1d9e108e9cce6c08b121cdc46b9b467f8ae7c474936bccbffef"
    ),
    "geometry-neighbors-request.v1.schema.json": (
        "ef0afcbffb2f107722f36dd3ff223d8838a919c409e372b2677cfd3448d482fd"
    ),
    "geometry-reference-key.v1.schema.json": (
        "b73e2270fb5b9d2d6b4956a728ba4f4bbbacc2ea9b5e0266cdbaab95f8116647"
    ),
}
_LAB_FIXTURE_FILE_SHA256: Final = {
    "geometry-input-corpus.v1.json": (
        "cb1f58e5ea04e74b21810f202cb8979c59be3cacb3e358c9bc180d5bea68f1db"
    ),
    "geometry-reference-key.v1.json": (
        "485a7f59ab9187ca51a56272595384fc6496d9e80ca1aa951aab3dc60cfd6a00"
    ),
}
_LAB_FIXTURE_CORPUS_SHA256: Final = (
    "8b9fde4cba88733319be0639a1bf07dc99d1ab7bb26039eeadfdfd0071c7e5b3"
)

PathLike: TypeAlias = str | os.PathLike[str]
ReviewExportSource: TypeAlias = PathLike | Mapping[str, Any]
PartitionName: TypeAlias = Literal["development", "validation", "sealed_holdout"]


class GeometryLabExportError(ValueError):
    """A review or artifact failed a model-blind integrity invariant."""


@dataclass(frozen=True)
class GeometryPartitionArtifacts:
    """One populated strict-v1 partition before it is written to disk."""

    partition: PartitionName
    corpus: dict[str, Any]
    reference_key: dict[str, Any]
    corpus_sha256: str
    source_export_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class WrittenGeometryPartition:
    """Paths and identities for one immutable artifact pair."""

    partition: PartitionName
    corpus_path: Path
    reference_key_path: Path
    corpus_sha256: str
    reference_key_sha256: str
    sample_count: int


@dataclass(frozen=True)
class _Candidate:
    partition: PartitionName
    sample: dict[str, Any]
    reference: dict[str, Any]
    created_at: str
    source_export_sha256: str


def _fail(message: str) -> None:
    raise GeometryLabExportError(message)


def _require_object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{field} must be a JSON object")
    return value


def _require_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{field} must be a JSON array")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    field: str,
) -> None:
    allowed = required | (optional or set())
    missing = sorted(required - set(value))
    unexpected = sorted(set(value) - allowed)
    if missing:
        _fail(f"{field} is missing: {', '.join(missing)}")
    if unexpected:
        _fail(f"{field} has unknown fields: {', '.join(unexpected)}")


def _parse_iso_date(value: Any, *, field: str) -> date:
    if not isinstance(value, str):
        _fail(f"{field} must use ISO YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise GeometryLabExportError(
            f"{field} must use ISO YYYY-MM-DD"
        ) from exc
    if parsed.isoformat() != value:
        _fail(f"{field} must use ISO YYYY-MM-DD")
    return parsed


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be an ISO timestamp with a timezone")
    normalized = value.strip()
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GeometryLabExportError(
            f"{field} must be an ISO timestamp with a timezone"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _lab_timestamp(value: Any, *, field: str) -> str:
    return (
        _parse_timestamp(value, field=field)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _finite_positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be a finite positive number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise GeometryLabExportError(
            f"{field} must be a finite positive number"
        ) from exc
    if not math.isfinite(number) or number <= 0:
        _fail(f"{field} must be a finite positive number")
    return number


def _exact_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{field} must be at least {minimum}")
    return value


def _normalized_id(value: Any, *, pattern: re.Pattern[str], field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field} has an invalid identifier")
    normalized = value.strip()
    if not pattern.fullmatch(normalized):
        _fail(f"{field} has an invalid identifier")
    return normalized


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise GeometryLabExportError(
            "value cannot be encoded as canonical JSON"
        ) from exc
    return (rendered + "\n").encode("utf-8")


def _backend_canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise GeometryLabExportError(
            "review export cannot be encoded as canonical JSON"
        ) from exc


def backend_review_export_sha256(value: Mapping[str, Any]) -> str:
    """Return the schema-v4 backend's no-newline canonical content hash."""

    return hashlib.sha256(_backend_canonical_json(value).encode("utf-8")).hexdigest()


def validate_geometry_input_corpus(value: Any) -> dict[str, Any]:
    """Validate and normalize exactly the frozen lab input-corpus v1 shape."""

    raw = _require_object(value, field="geometry input corpus")
    _strict_keys(
        raw,
        required={
            "schema_version",
            "corpus_id",
            "created_at",
            "adjustment",
            "samples",
        },
        field="geometry input corpus",
    )
    if raw["schema_version"] != CORPUS_SCHEMA:
        _fail("unsupported geometry input corpus schema")
    corpus_id = _normalized_id(
        raw["corpus_id"], pattern=_ARTIFACT_ID_RE, field="corpus_id"
    )
    if raw["adjustment"] != ADJUSTMENT:
        _fail(f"adjustment must equal {ADJUSTMENT}")
    raw_samples = _require_list(raw["samples"], field="samples")
    if not raw_samples:
        _fail("samples must contain at least one item")

    samples: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for sample_index, raw_sample in enumerate(raw_samples):
        sample = _require_object(raw_sample, field=f"samples[{sample_index}]")
        _strict_keys(
            sample,
            required={
                "sample_id",
                "issuer_group_id",
                "setup_group_id",
                "cutoff_date",
                "daily_bars",
            },
            field=f"samples[{sample_index}]",
        )
        sample_id = _normalized_id(
            sample["sample_id"],
            pattern=_SAMPLE_ID_RE,
            field=f"samples[{sample_index}].sample_id",
        )
        if sample_id in sample_ids:
            _fail(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        issuer_group_id = _normalized_id(
            sample["issuer_group_id"],
            pattern=_ISSUER_ID_RE,
            field=f"samples[{sample_index}].issuer_group_id",
        )
        setup_group_id = _normalized_id(
            sample["setup_group_id"],
            pattern=_SETUP_ID_RE,
            field=f"samples[{sample_index}].setup_group_id",
        )
        cutoff = _parse_iso_date(
            sample["cutoff_date"], field=f"samples[{sample_index}].cutoff_date"
        )
        raw_bars = _require_list(
            sample["daily_bars"], field=f"samples[{sample_index}].daily_bars"
        )
        if len(raw_bars) < 2:
            _fail(f"samples[{sample_index}].daily_bars requires at least two bars")
        bars: list[dict[str, Any]] = []
        previous_date: date | None = None
        for bar_index, raw_bar in enumerate(raw_bars):
            bar = _require_object(
                raw_bar,
                field=f"samples[{sample_index}].daily_bars[{bar_index}]",
            )
            _strict_keys(
                bar,
                required={"date", "open", "high", "low", "close"},
                field=f"samples[{sample_index}].daily_bars[{bar_index}]",
            )
            bar_date = _parse_iso_date(
                bar["date"],
                field=f"samples[{sample_index}].daily_bars[{bar_index}].date",
            )
            if previous_date is not None and bar_date <= previous_date:
                _fail("daily bars must be strictly chronological and unique")
            if bar_date > cutoff:
                _fail("bars after cutoff_date are forbidden")
            previous_date = bar_date
            open_ = _finite_positive_float(bar["open"], field="bar open")
            high = _finite_positive_float(bar["high"], field="bar high")
            low = _finite_positive_float(bar["low"], field="bar low")
            close = _finite_positive_float(bar["close"], field="bar close")
            if high < max(open_, low, close):
                _fail("bar high must contain open, low, and close")
            if low > min(open_, high, close):
                _fail("bar low must contain open, high, and close")
            bars.append(
                {
                    "date": bar_date.isoformat(),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                }
            )
        samples.append(
            {
                "sample_id": sample_id,
                "issuer_group_id": issuer_group_id,
                "setup_group_id": setup_group_id,
                "cutoff_date": cutoff.isoformat(),
                "daily_bars": bars,
            }
        )
    return {
        "schema_version": CORPUS_SCHEMA,
        "corpus_id": corpus_id,
        "created_at": _lab_timestamp(raw["created_at"], field="created_at"),
        "adjustment": ADJUSTMENT,
        "samples": samples,
    }


def canonical_geometry_corpus_bytes(value: Any) -> bytes:
    """Return the lab's normalized, compact, trailing-newline corpus bytes."""

    return _canonical_json_bytes(validate_geometry_input_corpus(value))


def canonical_geometry_corpus_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_geometry_corpus_bytes(value)).hexdigest()


def validate_geometry_reference_key(
    value: Any,
    *,
    corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize exactly the frozen lab reference-key v1 shape."""

    raw = _require_object(value, field="geometry reference key")
    _strict_keys(
        raw,
        required={"schema_version", "corpus_sha256", "records"},
        field="geometry reference key",
    )
    if raw["schema_version"] != REFERENCE_SCHEMA:
        _fail("unsupported geometry reference-key schema")
    raw_corpus_hash = raw["corpus_sha256"]
    if not isinstance(raw_corpus_hash, str):
        _fail("reference corpus_sha256 must be a string")
    corpus_hash = raw_corpus_hash
    if not _HEX_64_RE.fullmatch(corpus_hash):
        _fail("reference corpus_sha256 must be lowercase SHA-256")
    raw_records = _require_list(raw["records"], field="reference records")
    if not raw_records:
        _fail("reference records must contain at least one item")
    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        record = _require_object(raw_record, field=f"records[{index}]")
        _strict_keys(
            record,
            required={"sample_id", "ticker", "display_name", "setup_label"},
            optional={"notes"},
            field=f"records[{index}]",
        )
        sample_id = _normalized_id(
            record["sample_id"],
            pattern=_SAMPLE_ID_RE,
            field=f"records[{index}].sample_id",
        )
        if sample_id in record_ids:
            _fail(f"duplicate reference sample_id: {sample_id}")
        record_ids.add(sample_id)
        normalized_record: dict[str, Any] = {"sample_id": sample_id}
        for key, minimum, maximum in (
            ("ticker", 1, 32),
            ("display_name", 1, 160),
            ("setup_label", 1, 160),
        ):
            raw_text = record[key]
            if not isinstance(raw_text, str):
                _fail(f"records[{index}].{key} must be a string")
            text = raw_text.strip()
            if not minimum <= len(text) <= maximum:
                _fail(f"records[{index}].{key} has an invalid length")
            normalized_record[key] = text
        notes_value = record.get("notes", "")
        if not isinstance(notes_value, str):
            _fail(f"records[{index}].notes must be a string")
        notes = notes_value.strip()
        if len(notes) > 2000:
            _fail(f"records[{index}].notes exceeds 2000 characters")
        normalized_record["notes"] = notes
        records.append(normalized_record)

    if corpus is not None:
        normalized_corpus = validate_geometry_input_corpus(corpus)
        actual_hash = canonical_geometry_corpus_sha256(normalized_corpus)
        if not hmac.compare_digest(actual_hash, corpus_hash):
            _fail("reference key corpus_sha256 does not match the canonical corpus")
        corpus_ids = {sample["sample_id"] for sample in normalized_corpus["samples"]}
        if corpus_ids != record_ids:
            _fail("reference key sample-ID set does not match the corpus")
    return {
        "schema_version": REFERENCE_SCHEMA,
        "corpus_sha256": corpus_hash,
        "records": records,
    }


def canonical_geometry_reference_bytes(value: Any) -> bytes:
    return _canonical_json_bytes(validate_geometry_reference_key(value))


def _validated_opaque_id_key(value: Any) -> bytes:
    if not isinstance(value, bytes) or not 32 <= len(value) <= 4_096:
        _fail("opaque_id_key must contain 32 to 4096 private bytes")
    return value


def _opaque_id(prefix: str, key: bytes, *parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return prefix + hmac.new(key, payload, hashlib.sha256).hexdigest()


def issuer_group_id_for_ticker(ticker: str, *, opaque_id_key: bytes) -> str:
    """Create a private stable issuer group from the normalized ticker.

    The backend has no issuer master identifier.  By default the normalized
    ticker is therefore the frozen cluster key.  The HMAC prevents recovery by
    hashing the finite public ticker universe.  Cross-listings remain separate
    until the backend has a verified issuer master.
    """

    key = _validated_opaque_id_key(opaque_id_key)
    symbol = str(ticker or "").strip().upper()
    if not _TICKER_RE.fullmatch(symbol):
        _fail("ticker contains unsupported characters")
    return _opaque_id(
        "iss_", key, "coilingview-geometry-lab:issuer:v1", symbol
    )


def issuer_partition_bucket(
    issuer_group_id: str,
    *,
    seed: str = DEFAULT_SPLIT_SEED,
) -> int:
    """Return the stable 0..9999 bucket used by the frozen 60/20/20 rule."""

    normalized = _normalized_id(
        issuer_group_id, pattern=_ISSUER_ID_RE, field="issuer_group_id"
    )
    digest = hashlib.sha256(f"{seed}\0{normalized}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % _SPLIT_BUCKETS


def partition_for_issuer_group(
    issuer_group_id: str,
    *,
    seed: str = DEFAULT_SPLIT_SEED,
) -> PartitionName:
    """Assign an issuer cluster to a membership-stable 60/20/20 partition."""

    bucket = issuer_partition_bucket(issuer_group_id, seed=seed)
    if bucket < _DEVELOPMENT_END:
        return PARTITION_DEVELOPMENT
    if bucket < _VALIDATION_END:
        return PARTITION_VALIDATION
    return PARTITION_SEALED_HOLDOUT


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    _fail(f"non-standard JSON number is forbidden: {value}")


def _load_json_path(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GeometryLabExportError(f"cannot read UTF-8 JSON from {path}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
        )
    except GeometryLabExportError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise GeometryLabExportError(f"invalid JSON in {path}: {exc}") from exc


def _expected_hash_for_source(
    source: ReviewExportSource,
    expected: str | Mapping[str, str] | None,
    *,
    source_count: int,
) -> str | None:
    if expected is None:
        return None
    if isinstance(expected, str):
        if source_count != 1:
            _fail("one expected export hash can only verify one review export")
        return expected
    if isinstance(source, Mapping):
        _fail(
            "in-memory review exports cannot be matched to a path-keyed "
            "expected hash"
        )
    path = Path(source)
    for key in (str(path), str(path.resolve()), path.name):
        if key in expected:
            return expected[key]
    _fail(f"no expected export hash was provided for {path}")


def load_finalized_review_export(
    source: ReviewExportSource,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load a trusted schema-v4 export or its finalize-response hash envelope.

    A bare export is accepted only with the independently retained SHA-256 from
    finalization.  The finalize-response envelope carries that same identity.
    """

    raw: Any
    source_label: str
    if isinstance(source, Mapping):
        raw = deepcopy(dict(source))
        source_label = "review export"
    else:
        path = Path(source)
        raw = _load_json_path(path)
        source_label = str(path)
    raw_object = _require_object(raw, field=source_label)

    envelope_hash: str | None = None
    envelope_finalized_at: Any = None
    if "export" in raw_object:
        _strict_keys(
            raw_object,
            required={"export", "sha256", "finalized_at"},
            field=f"{source_label} finalize envelope",
        )
        export = _require_object(raw_object["export"], field=f"{source_label}.export")
        envelope_hash_value = raw_object.get("sha256")
        if not isinstance(envelope_hash_value, str) or not _HEX_64_RE.fullmatch(
            envelope_hash_value
        ):
            _fail(f"{source_label} finalize envelope requires a SHA-256")
        envelope_hash = envelope_hash_value
        envelope_finalized_at = raw_object.get("finalized_at")
    else:
        export = raw_object

    if envelope_hash is None and expected_sha256 is None:
        _fail(
            f"{source_label} is a bare review export; its expected finalization "
            "SHA-256 is required"
        )

    actual_hash = backend_review_export_sha256(export)
    for label, claimed in (
        ("finalize envelope", envelope_hash),
        ("caller", expected_sha256),
    ):
        if claimed is not None:
            if not isinstance(claimed, str) or not _HEX_64_RE.fullmatch(claimed):
                _fail(f"{label} expected export hash must be lowercase SHA-256")
            if not hmac.compare_digest(claimed, actual_hash):
                _fail(f"{source_label} content does not match its {label} hash")

    session = _require_object(export.get("session"), field="session")
    finalized_at = session.get("finalized_at")
    if envelope_finalized_at is not None and _parse_timestamp(
        envelope_finalized_at, field="envelope.finalized_at"
    ) != _parse_timestamp(finalized_at, field="session.finalized_at"):
        _fail("finalize envelope timestamp does not match the export")
    return export, actual_hash


def _same(label: str, *values: Any) -> None:
    if not values:
        return
    first = values[0]
    if any(value != first for value in values[1:]):
        _fail(f"{label} does not match across frozen identities")


def _same_hash(label: str, *values: Any) -> None:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not _HEX_64_RE.fullmatch(value):
            _fail(f"{label} requires lowercase SHA-256 identities")
        normalized.append(value)
    first = normalized[0]
    if any(not hmac.compare_digest(first, value) for value in normalized[1:]):
        _fail(f"{label} hash drift detected")


def _manifest_roster(
    manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    raw_items = _require_list(manifest.get("items"), field="manifest.items")
    item_map: dict[str, dict[str, Any]] = {}
    item_order: list[str] = []
    for position, raw_item in enumerate(raw_items, start=1):
        item = _require_object(raw_item, field=f"manifest.items[{position - 1}]")
        ticker = str(item.get("ticker", "")).strip().upper()
        if not _TICKER_RE.fullmatch(ticker):
            _fail("manifest item ticker is invalid")
        if ticker in item_map:
            _fail(f"manifest contains duplicate ticker {ticker}")
        missing_hashes = [
            key for key in _REQUIRED_MANIFEST_ITEM_HASHES if key not in item
        ]
        if missing_hashes:
            _fail(
                f"manifest item {ticker} lacks frozen identities: "
                + ", ".join(missing_hashes)
        )
        for key in _REQUIRED_MANIFEST_ITEM_HASHES:
            if not isinstance(item[key], str) or not _HEX_64_RE.fullmatch(item[key]):
                _fail(f"manifest item {ticker} {key} is not SHA-256")
        for position_field in ("position", "universe_position"):
            declared_position = item.get(position_field)
            if declared_position is not None and _exact_int(
                declared_position,
                field=f"manifest item {ticker} {position_field}",
                minimum=1,
            ) != position:
                _fail(f"manifest item {ticker} position drift detected")
        item_order.append(ticker)
        item_map[ticker] = item

    ordered_universe = manifest.get("ordered_universe")
    if ordered_universe is not None:
        declared_order = [str(value).strip().upper() for value in _require_list(
            ordered_universe, field="manifest.ordered_universe"
        )]
        if declared_order != item_order:
            _fail("manifest ordered_universe does not match manifest items")
    item_count = manifest.get("item_count")
    if item_count is not None and _exact_int(
        item_count, field="manifest.item_count", minimum=0
    ) != len(item_order):
        _fail("manifest item_count does not match its roster")
    return item_order, item_map


def _validate_manifest_binding(
    export: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    source: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    session = _require_object(export.get("session"), field="session")
    session_snapshot = _require_object(
        session.get("snapshot"), field="session.snapshot"
    )
    frozen_manifest = _require_object(
        session_snapshot.get("frozen_manifest"),
        field="session.snapshot.frozen_manifest",
    )
    manifest_hash = manifest.get("_manifest_sha256")
    _same_hash("frozen manifest", frozen_manifest.get("sha256"), manifest_hash)
    _same(
        "manifest schema",
        frozen_manifest.get("schema_version"),
        manifest.get("schema_version"),
    )
    _same("manifest kind", frozen_manifest.get("kind"), manifest.get("kind"))
    _same("frozen source", session_snapshot.get("frozen_source"), source)

    manifest_order, manifest_items = _manifest_roster(manifest)
    export_items = [
        _require_object(value, field="session item")
        for value in _require_list(session.get("items"), field="session.items")
    ]
    export_order = [
        str(item.get("ticker", "")).strip().upper() for item in export_items
    ]
    if export_order != manifest_order:
        _fail("review export must contain the complete frozen manifest roster in order")
    if _exact_int(
        session_snapshot.get("frozen_item_count"),
        field="session.snapshot.frozen_item_count",
        minimum=0,
    ) != len(export_items):
        _fail("session frozen_item_count does not match its roster")
    if _exact_int(
        frozen_manifest.get("item_count"),
        field="session.snapshot.frozen_manifest.item_count",
        minimum=0,
    ) != len(export_items):
        _fail("session frozen manifest item_count does not match its roster")

    manifest_policy = (
        manifest.get("review_policy")
        if isinstance(manifest.get("review_policy"), dict)
        else {}
    )
    detector_hidden = bool(manifest_policy.get("detector_outputs_hidden"))
    if detector_hidden and not (
        manifest_policy.get("model_reveal_allowed") is False
        and manifest_policy.get("production_effect") == "none"
        and manifest_policy.get("candidate_rules_visible") is False
        and manifest_policy.get("coordinator_key_visible") is False
    ):
        _fail("frozen manifest detector-blind policy is not fail-closed")
    expected_review_policy = {
        "detector_outputs_hidden": detector_hidden,
        "model_reveal_allowed": not detector_hidden,
        "production_effect": "none" if detector_hidden else None,
        "candidate_rules_visible": False if detector_hidden else None,
        "coordinator_key_visible": False if detector_hidden else None,
    }
    if session_snapshot.get("review_policy") != expected_review_policy:
        _fail("session review policy does not match the live frozen manifest")

    top_frozen_run = export.get("frozen_run")
    snapshot_frozen_run = session_snapshot.get("frozen_run")
    if top_frozen_run != snapshot_frozen_run:
        _fail("frozen_run drift detected inside the export")
    expected_frozen_run = manifest.get("run")
    if expected_frozen_run is None:
        source_run = manifest.get("source_run")
        expected_frozen_run = {
            "corpus_id": manifest.get("corpus_id"),
            "source_run": source_run,
            "algorithm_version": (
                source_run.get("algorithm_version")
                if isinstance(source_run, dict)
                else export.get("algorithm_version")
            ),
            "canonicalization": manifest.get("canonicalization"),
            "generator": manifest.get("generator"),
        }
    expected_frozen_run = _require_object(
        expected_frozen_run, field="manifest frozen run"
    )
    if top_frozen_run != expected_frozen_run:
        _fail("frozen_run does not match the live frozen manifest")
    if not detector_hidden:
        _same(
            "review algorithm version",
            export.get("algorithm_version"),
            expected_frozen_run.get("algorithm_version"),
        )
    if frozen_manifest.get("source_run") is not None and (
        frozen_manifest.get("source_run") != manifest.get("source_run")
    ):
        _fail("frozen manifest source_run drift detected")
    return export_items, manifest_items


def _validate_finalized_session(
    export: Mapping[str, Any], *, now: datetime
) -> tuple[dict[str, Any], datetime]:
    if export.get("schema_version") != 4:
        _fail("geometry exporter accepts only review export schema_version 4")
    if export.get("kind") != "coilingview.fresh-review-session-feedback":
        _fail("geometry exporter accepts only finalized fresh-review exports")
    session = _require_object(export.get("session"), field="session")
    finalized_raw = session.get("finalized_at")
    if finalized_raw is None:
        _fail("review session is pending; finalized_at is required")
    finalized_at = _parse_timestamp(finalized_raw, field="session.finalized_at")
    exported_at = _parse_timestamp(export.get("exported_at"), field="exported_at")
    if exported_at != finalized_at:
        _fail("exported_at must equal session.finalized_at")
    if finalized_at > now + _FUTURE_CLOCK_TOLERANCE:
        _fail("review export finalization timestamp is in the future")
    created_at = _parse_timestamp(session.get("created_at"), field="session.created_at")
    if created_at > finalized_at:
        _fail("review session creation cannot follow finalization")

    items = _require_list(session.get("items"), field="session.items")
    statuses = Counter(
        str(_require_object(item, field="session item").get("status"))
        for item in items
    )
    if statuses.get("pending"):
        _fail("review session still contains pending items")
    if statuses.get("skipped"):
        _fail("skipped review items are unresolved and cannot be exported")
    if set(statuses) != {"reviewed"}:
        _fail("every geometry export item must be reviewed")
    expected_counts = {
        "pending": 0,
        "reviewed": len(items),
        "skipped": 0,
        "total": len(items),
    }
    raw_counts = _require_object(session.get("counts"), field="session.counts")
    if set(raw_counts) != set(expected_counts) or any(
        _exact_int(raw_counts.get(key), field=f"session.counts.{key}", minimum=0)
        != expected
        for key, expected in expected_counts.items()
    ):
        _fail("session counts do not match the finalized reviewed roster")
    if session.get("next_pending_ticker") is not None:
        _fail("finalized review export cannot retain a next pending ticker")
    if not items:
        _fail("review export has no reviewed items")
    return session, finalized_at


def _record_map(export: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    event_ids: set[str] = set()
    for raw_wrapper in _require_list(export.get("records"), field="records"):
        wrapper = _require_object(raw_wrapper, field="record wrapper")
        record_id = wrapper.get("id")
        if (
            isinstance(record_id, bool)
            or not isinstance(record_id, int)
            or record_id < 1
        ):
            _fail("record wrapper id must be a positive integer")
        if record_id in records:
            _fail(f"duplicate linked review id {record_id}")
        event_id = str(wrapper.get("event_id") or "")
        if not event_id or event_id in event_ids:
            _fail("record event_id values must be nonblank and unique")
        event_ids.add(event_id)
        records[record_id] = wrapper
    return records


def _validated_snapshot_context(
    *,
    source: str,
    ticker: str,
    item: Mapping[str, Any],
    manifest_item: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        identity = review_snapshot_identity(source, ticker)
        verify_manifest_identity(identity, dict(manifest_item))
        blind = load_blind_review_context(source, ticker)
    except (ReviewSnapshotError, KeyError, TypeError, ValueError) as exc:
        raise GeometryLabExportError(
            f"{ticker} frozen snapshot/manifest verification failed: {exc}"
        ) from exc

    _same_hash(
        f"{ticker} bars identity",
        identity.get("bars_hash"),
        blind.get("bars_hash"),
        item.get("bars_hash"),
        manifest_item.get("backend_bars_identity_sha256"),
    )
    _same_hash(
        f"{ticker} snapshot identity",
        identity.get("snapshot_sha256"),
        blind.get("snapshot_sha256"),
        manifest_item.get("snapshot_sha256"),
    )
    _same_hash(
        f"{ticker} screen snapshot identity",
        identity.get("screen_snapshot_sha256"),
        manifest_item.get("screen_snapshot_sha256"),
    )
    _same_hash(
        f"{ticker} sample identity",
        identity.get("sample_id"),
        blind.get("sample_id"),
        item.get("sample_id"),
    )
    if not bool(identity.get("reviewable")) or not bool(blind.get("reviewable")):
        _fail(f"{ticker} frozen snapshot is not reviewable")
    if item.get("reviewable") is not True:
        _fail(f"{ticker} session item is not reviewable")

    item_snapshot = _require_object(
        item.get("snapshot"), field=f"{ticker} item snapshot"
    )
    frozen = _require_object(
        item_snapshot.get("frozen"), field=f"{ticker} item frozen identity"
    )
    _same(f"{ticker} frozen source", frozen.get("source"), source, blind.get("source"))
    _same_hash(
        f"{ticker} frozen sample",
        frozen.get("sample_id"),
        blind.get("sample_id"),
    )
    _same_hash(f"{ticker} frozen bars", frozen.get("bars_hash"), blind.get("bars_hash"))
    _same_hash(
        f"{ticker} frozen snapshot",
        frozen.get("snapshot_sha256"),
        blind.get("snapshot_sha256"),
    )
    bars = _require_list(blind.get("monthly_bars"), field=f"{ticker} monthly bars")
    if not bars:
        _fail(f"{ticker} frozen snapshot has no monthly bars")
    _same(f"{ticker} frozen data date", frozen.get("data_date"), bars[-1].get("date"))

    if manifest_item.get("bar_count") is not None and _exact_int(
        manifest_item["bar_count"],
        field=f"{ticker} manifest bar_count",
        minimum=0,
    ) != len(bars):
        _fail(f"{ticker} manifest bar_count drift detected")
    if manifest_item.get("snapshot_file") is not None and (
        manifest_item.get("snapshot_file") != f"{ticker}.json"
    ):
        _fail(f"{ticker} manifest snapshot filename drift detected")
    if manifest_item.get("first_data_date") is not None:
        _same(
            f"{ticker} first data date",
            manifest_item.get("first_data_date"),
            bars[0].get("date"),
        )
    if manifest_item.get("last_data_date") is not None:
        _same(
            f"{ticker} last data date",
            manifest_item.get("last_data_date"),
            bars[-1].get("date"),
        )
    if manifest_item.get("canonical_monthly_bars_sha256") is not None:
        _same_hash(
            f"{ticker} canonical monthly bars",
            manifest_item.get("canonical_monthly_bars_sha256"),
            backend_sha256_json(bars),
        )
    return identity, blind


def _validated_gold(
    *,
    export: Mapping[str, Any],
    session: Mapping[str, Any],
    finalized_at: datetime,
    item: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    source: str,
    ticker: str,
    blind: Mapping[str, Any],
) -> dict[str, Any]:
    record = _require_object(wrapper.get("record"), field=f"{ticker} record")
    session_id = _exact_int(session.get("id"), field="session.id", minimum=1)
    linked_review_id = item.get("linked_review_id")
    _same(f"{ticker} linked review id", linked_review_id, wrapper.get("id"))
    linked_event_id = item.get("linked_event_id")
    _same(
        f"{ticker} linked event id",
        linked_event_id,
        wrapper.get("event_id"),
        record.get("eventId"),
    )
    _same(
        f"{ticker} record ticker",
        ticker,
        str(wrapper.get("ticker", "")).strip().upper(),
        str(record.get("ticker", "")).strip().upper(),
    )
    _same(
        f"{ticker} record interval",
        wrapper.get("interval"),
        record.get("interval"),
        "3M",
    )
    _same(f"{ticker} record session", record.get("sessionId"), session_id)

    sample_id = str(blind.get("sample_id"))
    bars_hash = str(blind.get("bars_hash"))
    _same_hash(f"{ticker} record sample", record.get("sampleId"), sample_id)
    frozen_context = _require_object(
        record.get("frozenContext"), field=f"{ticker} frozenContext"
    )
    _same(f"{ticker} record frozen source", frozen_context.get("source"), source)
    _same_hash(
        f"{ticker} record frozen sample",
        frozen_context.get("sampleId"),
        sample_id,
    )
    _same_hash(
        f"{ticker} record frozen bars", frozen_context.get("barsHash"), bars_hash
    )

    if record.get("schemaVersion") != 6:
        _fail(f"{ticker} requires finalized capture schemaVersion 6")
    if record.get("labelPolicyVersion") != 2 or record.get("captureOnly") is not True:
        _fail(f"{ticker} is not a capture-only schema-v6 review")
    provenance = _require_object(record.get("provenance"), field=f"{ticker} provenance")
    if (
        provenance.get("frozen") is not True
        or provenance.get("reviewOverrideApplied") is not False
    ):
        _fail(f"{ticker} provenance must remain frozen and override-free")
    _same(f"{ticker} provenance source", provenance.get("source"), source)
    _same_hash(f"{ticker} provenance sample", provenance.get("sampleId"), sample_id)
    _same_hash(f"{ticker} provenance bars", provenance.get("barsHash"), bars_hash)
    monthly_bars = _require_list(
        blind.get("monthly_bars"), field=f"{ticker} monthly bars"
    )
    _same(
        f"{ticker} provenance data date",
        provenance.get("dataDate"),
        monthly_bars[-1].get("date"),
    )

    review_policy = _require_object(
        _require_object(
            session.get("snapshot"), field="session.snapshot"
        ).get("review_policy"),
        field="session.snapshot.review_policy",
    )
    detector_hidden = bool(review_policy.get("detector_outputs_hidden"))
    if provenance.get("detectorOutputsHidden") is not detector_hidden:
        _fail(f"{ticker} detector visibility drift detected")
    if detector_hidden and record.get("algorithm") is not None:
        _fail(f"{ticker} detector-blind record unexpectedly contains algorithm output")
    expected_algorithm_version = None if detector_hidden else export.get(
        "algorithm_version"
    )
    _same(
        f"{ticker} algorithm version",
        record.get("algorithmVersion"),
        provenance.get("algorithmVersion"),
        expected_algorithm_version,
    )

    raw_capture = _require_object(
        record.get("wholePattern"), field=f"{ticker} wholePattern"
    )
    raw_gold = _require_object(
        record.get("wholePatternGold"), field=f"{ticker} wholePatternGold"
    )
    if canonical_gold_json(raw_gold.get("capture")) != canonical_gold_json(raw_capture):
        _fail(f"{ticker} wholePatternGold capture drift detected")
    try:
        capture = WholePatternGoldCapture.model_validate(raw_capture)
        validate_api_whole_pattern_capture(
            capture, session_id=session_id, sample_id=sample_id
        )
        rematerialized = validate_materialized_gold_label_against_bars(
            raw_gold, monthly_bars
        )
        independent = materialize_gold_label(raw_capture, monthly_bars)
    except (TypeError, ValueError) as exc:
        raise GeometryLabExportError(
            f"{ticker} whole-pattern gold rematerialization failed: {exc}"
        ) from exc
    if canonical_gold_json(rematerialized) != canonical_gold_json(independent):
        _fail(f"{ticker} whole-pattern gold rematerialization drift detected")

    if rematerialized.get("top_review_complete") is not True:
        _fail(f"{ticker} top review is unresolved")
    judgments = _require_object(
        rematerialized.get("judgments"), field=f"{ticker} judgments"
    )
    if "uncertain" in judgments.values() or judgments.get("action") == "abstain":
        _fail(f"{ticker} contains unresolved whole-pattern judgments")
    if rematerialized.get("outcome_visible_during_label") is not False:
        _fail(
            f"{ticker} outcome-visible labels are not eligible for the "
            "geometry corpus"
        )

    cutoff = _parse_iso_date(
        rematerialized.get("cutoff_date"), field=f"{ticker} cutoff_date"
    )
    if cutoff > finalized_at.date():
        _fail(f"{ticker} reviewed cutoff is later than session finalization")
    _same(f"{ticker} reviewed cutoff", record.get("asOf"), cutoff.isoformat())
    labeled_at = _parse_timestamp(
        rematerialized.get("labeled_at"), field=f"{ticker} labeled_at"
    )
    if labeled_at > finalized_at:
        _fail(f"{ticker} label timestamp follows session finalization")
    wrapper_created = _parse_timestamp(
        wrapper.get("created_at"), field=f"{ticker} record created_at"
    )
    if wrapper_created > finalized_at:
        _fail(f"{ticker} record timestamp follows session finalization")
    server_created = _parse_timestamp(
        record.get("serverCreatedAt"), field=f"{ticker} serverCreatedAt"
    )
    if server_created != wrapper_created:
        _fail(f"{ticker} server/wrapper record timestamp drift detected")
    completed_at = _parse_timestamp(
        item.get("completed_at"), field=f"{ticker} completed_at"
    )
    if completed_at != wrapper_created:
        _fail(f"{ticker} item completion timestamp drift detected")

    locked_at = _parse_timestamp(
        item.get("base_classification_locked_at"),
        field=f"{ticker} base_classification_locked_at",
    )
    if locked_at > completed_at:
        _fail(f"{ticker} base classification was locked after review completion")
    raw_base = _require_object(
        item.get("base_classification"), field=f"{ticker} base_classification"
    )
    raw_learning = _require_object(
        record.get("learningCapture"), field=f"{ticker} learningCapture"
    )
    try:
        base = BaseClassification.model_validate(raw_base)
        learning = ReviewLearningCaptureV5.model_validate(raw_learning)
    except ValueError as exc:
        raise GeometryLabExportError(
            f"{ticker} blind learning/base capture is invalid: {exc}"
        ) from exc
    _same(f"{ticker} locked base path", base.base_path, learning.base_path)
    _same(
        f"{ticker} locked failed base rules",
        base.failed_base_rules,
        learning.failed_base_rules,
    )
    _same(
        f"{ticker} locked base rationale",
        base.rationale,
        learning.base_rationale,
    )
    _same(
        f"{ticker} finalized shape",
        record.get("coilLabel"),
        judgments.get("shape"),
        learning.derived_coil_label(),
    )
    if record.get("decision") not in {"approved", "corrected"}:
        _fail(f"{ticker} finalized decision is unsupported")
    if judgments.get("shape") == "coil":
        if record.get("humanGrade") not in {"A", "B", "C"}:
            _fail(f"{ticker} coil label requires a human grade")
    elif record.get("humanGrade") is not None:
        _fail(f"{ticker} non-coil label cannot carry a human grade")
    reviewer = _require_object(export.get("reviewer"), field="reviewer")
    _same(
        f"{ticker} reviewer identity",
        learning.reviewer_name,
        rematerialized.get("labeler"),
        reviewer.get("name"),
    )
    return rematerialized


def _model_visible_sample(
    *,
    blind: Mapping[str, Any],
    sample_id: str,
    issuer_group_id: str,
    setup_group_id: str,
    cutoff_date: str,
) -> dict[str, Any]:
    """Construct the only object that can cross into the model-visible lane."""

    cutoff = _parse_iso_date(cutoff_date, field="cutoff_date")
    visible: list[dict[str, Any]] = []
    for raw_bar in _require_list(blind.get("monthly_bars"), field="monthly bars"):
        bar = _require_object(raw_bar, field="monthly bar")
        bar_date = _parse_iso_date(bar.get("date"), field="monthly bar date")
        if bar_date > cutoff:
            continue
        visible.append(
            {
                "date": bar_date.isoformat(),
                "open": _finite_positive_float(bar.get("open"), field="bar open"),
                "high": _finite_positive_float(bar.get("high"), field="bar high"),
                "low": _finite_positive_float(bar.get("low"), field="bar low"),
                "close": _finite_positive_float(bar.get("close"), field="bar close"),
            }
        )
    if len(visible) < 2 or visible[-1]["date"] != cutoff.isoformat():
        _fail("physical cutoff truncation must end on the reviewed frozen bar")
    return {
        "sample_id": sample_id,
        "issuer_group_id": issuer_group_id,
        "setup_group_id": setup_group_id,
        "cutoff_date": cutoff.isoformat(),
        # Frozen v1 calls this daily_bars.  These values are exact monthly bars.
        "daily_bars": visible,
    }


def _reference_record(
    *,
    ticker: str,
    identity: Mapping[str, Any],
    sample_id: str,
    partition: PartitionName,
    gold: Mapping[str, Any],
    source_export_sha256: str,
    frozen_manifest_sha256: str,
    opaque_id_key_sha256: str,
) -> dict[str, Any]:
    setup_label = str(gold.get("setup_id") or "").strip()
    if not 1 <= len(setup_label) <= 160:
        _fail(f"{ticker} gold setup_id cannot fit reference-key v1")
    screen_snapshot = identity.get("screen_snapshot")
    display_name = ticker
    if isinstance(screen_snapshot, dict):
        candidate = screen_snapshot.get("company_name")
        if isinstance(candidate, str) and 1 <= len(candidate.strip()) <= 160:
            display_name = candidate.strip()
    notes_payload = {
        "bars_sha256": identity.get("bars_hash"),
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "source_evaluation_role": gold.get("evaluation_role"),
        "source_export_sha256": source_export_sha256,
        "gold_capture_sha256": (gold.get("provenance") or {}).get("capture_sha256"),
        "gold_label_sha256": gold.get("label_sha256"),
        "opaque_id_key_sha256": opaque_id_key_sha256,
        "judgments": gold.get("judgments"),
        "partition": partition,
        "source_adjustment_provenance": "unavailable",
        "source_bar_frequency": "monthly",
        "snapshot_sha256": identity.get("snapshot_sha256"),
        "usage_constraint": "monthly_ood_only",
    }
    notes = _backend_canonical_json(notes_payload)
    if len(notes) > 2000:
        _fail(f"{ticker} compact reference summary exceeds v1 notes limit")
    return {
        "sample_id": sample_id,
        "ticker": ticker,
        "display_name": display_name,
        "setup_label": setup_label,
        "notes": notes,
    }


def _candidate_from_item(
    *,
    export: Mapping[str, Any],
    export_sha256: str,
    session: Mapping[str, Any],
    finalized_at: datetime,
    item: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    manifest_item: Mapping[str, Any],
    source: str,
    opaque_id_key: bytes,
    split_seed: str,
) -> _Candidate:
    ticker = str(item.get("ticker", "")).strip().upper()
    if not _TICKER_RE.fullmatch(ticker):
        _fail("session item ticker is invalid")
    identity, blind = _validated_snapshot_context(
        source=source,
        ticker=ticker,
        item=item,
        manifest_item=manifest_item,
    )
    gold = _validated_gold(
        export=export,
        session=session,
        finalized_at=finalized_at,
        item=item,
        wrapper=wrapper,
        source=source,
        ticker=ticker,
        blind=blind,
    )

    issuer_group_id = issuer_group_id_for_ticker(
        ticker, opaque_id_key=opaque_id_key
    )
    partition = partition_for_issuer_group(issuer_group_id, seed=split_seed)
    setup_group_id = _opaque_id(
        "set_",
        opaque_id_key,
        "coilingview-geometry-lab:setup:v1",
        issuer_group_id,
        str(gold["setup_id"]),
    )
    sample_id = _opaque_id(
        "smp_",
        opaque_id_key,
        "coilingview-geometry-lab:sample:v1",
        str(blind["sample_id"]),
        str(gold["episode_id"]),
        str(gold["setup_id"]),
        str(gold["cutoff_date"]),
    )
    sample = _model_visible_sample(
        blind=blind,
        sample_id=sample_id,
        issuer_group_id=issuer_group_id,
        setup_group_id=setup_group_id,
        cutoff_date=str(gold["cutoff_date"]),
    )
    reference = _reference_record(
        ticker=ticker,
        identity=identity,
        sample_id=sample_id,
        partition=partition,
        gold=gold,
        source_export_sha256=export_sha256,
        frozen_manifest_sha256=str(
            _require_object(
                _require_object(session.get("snapshot"), field="session.snapshot").get(
                    "frozen_manifest"
                ),
                field="session.snapshot.frozen_manifest",
            ).get("sha256")
        ),
        opaque_id_key_sha256=hashlib.sha256(opaque_id_key).hexdigest(),
    )
    return _Candidate(
        partition=partition,
        sample=sample,
        reference=reference,
        created_at=(
            finalized_at.isoformat(timespec="microseconds").replace("+00:00", "Z")
        ),
        source_export_sha256=export_sha256,
    )


def _normalize_sources(
    review_exports: ReviewExportSource | Iterable[ReviewExportSource],
) -> list[ReviewExportSource]:
    if isinstance(review_exports, (str, os.PathLike, Mapping)):
        return [review_exports]
    sources = list(review_exports)
    if not sources:
        _fail("at least one finalized review export is required")
    return sources


def build_geometry_lab_exports(
    review_exports: ReviewExportSource | Iterable[ReviewExportSource],
    *,
    opaque_id_key: bytes,
    split_seed: str = DEFAULT_SPLIT_SEED,
    expected_export_sha256: str | Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[PartitionName, GeometryPartitionArtifacts]:
    """Build strict v1 artifact pairs from finalized schema-v4 exports.

    Candidate nominations are intentionally ignored.  They are unreviewed and
    have no manifest-bound frozen snapshot in the review export, so they remain
    outside both the labeled corpus and its evaluation-only key.  The opaque ID
    key must be a stable private secret reused for every corpus generation.
    """

    sources = _normalize_sources(review_exports)
    id_key = _validated_opaque_id_key(opaque_id_key)
    if split_seed != DEFAULT_SPLIT_SEED:
        _fail("the frozen issuer split seed cannot be changed")
    if now is not None and (now.tzinfo is None or now.utcoffset() is None):
        _fail("now must be timezone-aware")
    clock = (
        now.astimezone(timezone.utc)
        if now is not None
        else datetime.now(timezone.utc)
    )

    candidates: list[_Candidate] = []
    for source_value in sources:
        expected = _expected_hash_for_source(
            source_value, expected_export_sha256, source_count=len(sources)
        )
        export, export_hash = load_finalized_review_export(
            source_value, expected_sha256=expected
        )
        session, finalized_at = _validate_finalized_session(export, now=clock)
        source = str(session.get("source") or "")
        try:
            manifest = load_review_manifest(source)
        except (ReviewSnapshotError, TypeError, ValueError) as exc:
            raise GeometryLabExportError(
                f"cannot load frozen review manifest for {source}: {exc}"
            ) from exc
        export_items, manifest_items = _validate_manifest_binding(
            export, manifest, source=source
        )
        records = _record_map(export)
        if len(records) != len(export_items):
            _fail("review export must contain exactly one linked record per item")
        used_record_ids: set[int] = set()
        for position, item in enumerate(export_items):
            if _exact_int(
                item.get("position"),
                field="review export item position",
                minimum=0,
            ) != position:
                _fail("review export item positions must be complete and ordered")
            linked_id = item.get("linked_review_id")
            if isinstance(linked_id, bool) or not isinstance(linked_id, int):
                _fail("reviewed item requires a linked review id")
            wrapper = records.get(linked_id)
            if wrapper is None or linked_id in used_record_ids:
                _fail("reviewed item link is missing or duplicated")
            used_record_ids.add(linked_id)
            ticker = str(item.get("ticker", "")).strip().upper()
            candidates.append(
                _candidate_from_item(
                    export=export,
                    export_sha256=export_hash,
                    session=session,
                    finalized_at=finalized_at,
                    item=item,
                    wrapper=wrapper,
                    manifest_item=manifest_items[ticker],
                    source=source,
                    opaque_id_key=id_key,
                    split_seed=split_seed,
                )
            )
        if used_record_ids != set(records):
            _fail("review export contains an unlinked record")

    unique: dict[str, _Candidate] = {}
    for candidate in candidates:
        sample_id = str(candidate.sample["sample_id"])
        existing = unique.get(sample_id)
        if existing is None:
            unique[sample_id] = candidate
            continue
        if (
            existing.partition != candidate.partition
            or existing.sample != candidate.sample
            or existing.reference != candidate.reference
        ):
            _fail(f"conflicting duplicate geometry sample {sample_id}")

    grouped: dict[PartitionName, list[_Candidate]] = {
        partition: [] for partition in PARTITIONS
    }
    for candidate in unique.values():
        grouped[candidate.partition].append(candidate)

    artifacts: dict[PartitionName, GeometryPartitionArtifacts] = {}
    for partition in PARTITIONS:
        members = sorted(
            grouped[partition], key=lambda value: value.sample["sample_id"]
        )
        if not members:
            continue
        samples = [member.sample for member in members]
        identity_hash = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "partition": partition,
                    "sample_ids": [sample["sample_id"] for sample in samples],
                }
            )
        ).hexdigest()[:24]
        corpus = validate_geometry_input_corpus(
            {
                "schema_version": CORPUS_SCHEMA,
                "corpus_id": (
                    "coilingview-geometry-"
                    f"{_PARTITION_DIRECTORIES[partition]}-{identity_hash}"
                ),
                "created_at": max(member.created_at for member in members),
                "adjustment": ADJUSTMENT,
                "samples": samples,
            }
        )
        corpus_hash = canonical_geometry_corpus_sha256(corpus)
        reference = validate_geometry_reference_key(
            {
                "schema_version": REFERENCE_SCHEMA,
                "corpus_sha256": corpus_hash,
                "records": [member.reference for member in members],
            },
            corpus=corpus,
        )
        artifacts[partition] = GeometryPartitionArtifacts(
            partition=partition,
            corpus=corpus,
            reference_key=reference,
            corpus_sha256=corpus_hash,
            source_export_sha256s=tuple(
                sorted({member.source_export_sha256 for member in members})
            ),
        )
    if not artifacts:
        _fail("no eligible reviewed geometry samples were produced")
    return artifacts


def _write_immutable(path: Path, payload: bytes, *, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 0o600 if private else 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError:
            try:
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                existing_descriptor = os.open(path, flags)
                try:
                    existing_mode = os.fstat(existing_descriptor).st_mode & 0o777
                    with os.fdopen(existing_descriptor, "rb") as existing_handle:
                        existing_descriptor = -1
                        existing = existing_handle.read()
                finally:
                    if existing_descriptor >= 0:
                        os.close(existing_descriptor)
            except OSError as exc:
                raise GeometryLabExportError(
                    f"cannot verify existing artifact {path}"
                ) from exc
            if existing != payload:
                _fail(f"refusing to overwrite differing immutable artifact {path}")
            if private and existing_mode != mode:
                _fail(f"existing private artifact has unsafe permissions: {path}")
        except OSError as exc:
            raise GeometryLabExportError(
                f"cannot verify existing artifact {path}"
            ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def write_geometry_lab_exports(
    artifacts: Mapping[PartitionName, GeometryPartitionArtifacts],
    output_dir: PathLike,
) -> dict[PartitionName, WrittenGeometryPartition]:
    """Write canonical immutable pairs; reference keys receive mode ``0600``."""

    unexpected_partitions = set(artifacts) - set(PARTITIONS)
    if unexpected_partitions:
        _fail(
            "unsupported artifact partitions: "
            + ", ".join(sorted(unexpected_partitions))
        )
    destination = Path(output_dir)
    written: dict[PartitionName, WrittenGeometryPartition] = {}
    for partition in PARTITIONS:
        pair = artifacts.get(partition)
        if pair is None:
            continue
        if pair.partition != partition:
            _fail("artifact mapping key does not match its partition")
        partition_directory = _PARTITION_DIRECTORIES[partition]
        corpus_path = (
            destination
            / "model-visible"
            / partition_directory
            / "geometry-input-corpus.v1.json"
        )
        reference_path = (
            destination
            / "evaluation-only"
            / partition_directory
            / "geometry-reference-key.v1.json"
        )
        corpus_bytes = canonical_geometry_corpus_bytes(pair.corpus)
        corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()
        if not hmac.compare_digest(corpus_hash, pair.corpus_sha256):
            _fail(f"{partition} corpus hash drift before write")
        validate_geometry_reference_key(pair.reference_key, corpus=pair.corpus)
        reference_bytes = canonical_geometry_reference_bytes(pair.reference_key)
        _write_immutable(
            corpus_path,
            corpus_bytes,
            private=partition == PARTITION_SEALED_HOLDOUT,
        )
        _write_immutable(reference_path, reference_bytes, private=True)
        written[partition] = WrittenGeometryPartition(
            partition=partition,
            corpus_path=corpus_path.resolve(),
            reference_key_path=reference_path.resolve(),
            corpus_sha256=corpus_hash,
            reference_key_sha256=hashlib.sha256(reference_bytes).hexdigest(),
            sample_count=len(pair.corpus["samples"]),
        )
    return written


def export_geometry_lab(
    review_exports: ReviewExportSource | Iterable[ReviewExportSource],
    output_dir: PathLike,
    *,
    opaque_id_key: bytes,
    **build_options: Any,
) -> dict[PartitionName, WrittenGeometryPartition]:
    """Validate, build, and immutably write all populated partition pairs."""

    return write_geometry_lab_exports(
        build_geometry_lab_exports(
            review_exports,
            opaque_id_key=opaque_id_key,
            **build_options,
        ),
        output_dir,
    )


def verify_frozen_lab_contract(root: PathLike) -> dict[str, Any]:
    """Verify the exact schemas and canonical fixtures frozen at commit 82d27ba."""

    lab_root = Path(root)

    def git_bytes(*arguments: str) -> bytes:
        try:
            return subprocess.run(
                ["git", "-C", str(lab_root), *arguments],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise GeometryLabExportError(
                f"cannot read frozen geometry-lab Git object at {lab_root}"
            ) from exc

    git_bytes("cat-file", "-e", f"{LAB_CONTRACT_COMMIT}^{{commit}}")
    try:
        schema_listing = git_bytes(
            "ls-tree", "--name-only", f"{LAB_CONTRACT_COMMIT}:schemas"
        ).decode("utf-8")
    except UnicodeError as exc:
        raise GeometryLabExportError(
            "frozen geometry-lab schema tree is not UTF-8"
        ) from exc
    actual_schema_names = {
        name
        for name in schema_listing.splitlines()
        if name.endswith(".schema.json")
    }
    if actual_schema_names != set(_LAB_SCHEMA_SHA256):
        _fail("geometry-lab schema filename set has drifted from frozen v1")
    for filename, expected in _LAB_SCHEMA_SHA256.items():
        raw_bytes = git_bytes(
            "show", f"{LAB_CONTRACT_COMMIT}:schemas/{filename}"
        )
        actual = hashlib.sha256(raw_bytes).hexdigest()
        if not hmac.compare_digest(actual, expected):
            _fail(f"frozen geometry-lab schema hash drift: {filename}")

    loaded: dict[str, Any] = {}
    for filename, expected in _LAB_FIXTURE_FILE_SHA256.items():
        try:
            raw_bytes = git_bytes(
                "show",
                f"{LAB_CONTRACT_COMMIT}:fixtures/synthetic/{filename}",
            )
            text = raw_bytes.decode("utf-8")
            loaded[filename] = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonstandard_number,
            )
        except GeometryLabExportError:
            raise
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise GeometryLabExportError(
                f"invalid frozen geometry-lab fixture: {filename}"
            ) from exc
        actual = hashlib.sha256(raw_bytes).hexdigest()
        if not hmac.compare_digest(actual, expected):
            _fail(f"frozen geometry-lab fixture file hash drift: {filename}")
    corpus = validate_geometry_input_corpus(
        loaded["geometry-input-corpus.v1.json"]
    )
    corpus_hash = canonical_geometry_corpus_sha256(corpus)
    if not hmac.compare_digest(corpus_hash, _LAB_FIXTURE_CORPUS_SHA256):
        _fail("frozen geometry-lab fixture canonical hash drift")
    validate_geometry_reference_key(
        loaded["geometry-reference-key.v1.json"], corpus=corpus
    )
    return {
        "commit": LAB_CONTRACT_COMMIT,
        "schema_files": tuple(sorted(_LAB_SCHEMA_SHA256)),
        "fixture_corpus_sha256": corpus_hash,
    }


def _parse_assignments(values: Sequence[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key.strip() or not item.strip():
            _fail(f"{label} values must use KEY=VALUE")
        if key in parsed:
            _fail(f"duplicate {label} key: {key}")
        parsed[key] = item
    return parsed


def load_opaque_id_key_file(path: PathLike) -> bytes:
    """Read a private regular file containing the stable opaque-ID HMAC key."""

    key_path = Path(path)
    descriptor = -1
    try:
        descriptor = os.open(
            key_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _fail("opaque ID key path must be a regular file")
        if metadata.st_mode & 0o077:
            _fail("opaque ID key file must not be accessible by group or others")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            key = handle.read(4_097)
    except GeometryLabExportError:
        raise
    except OSError as exc:
        raise GeometryLabExportError(
            f"cannot read opaque ID key file {key_path}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _validated_opaque_id_key(key)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export finalized CoilingView geometry into frozen lab v1 artifacts."
        )
    )
    parser.add_argument("review_exports", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--opaque-id-key-file",
        required=True,
        type=Path,
        help="private stable HMAC key file (mode 0600, at least 32 bytes)",
    )
    parser.add_argument(
        "--expected-export-sha256",
        action="append",
        default=[],
        metavar="PATH=SHA256",
    )
    parser.add_argument(
        "--lab-contract-root",
        type=Path,
        help="optionally verify the exact frozen lab schemas and fixtures first",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.lab_contract_root is not None:
            verify_frozen_lab_contract(args.lab_contract_root)
        opaque_id_key = load_opaque_id_key_file(args.opaque_id_key_file)
        expected = _parse_assignments(
            args.expected_export_sha256, label="expected export hash"
        )
        written = export_geometry_lab(
            args.review_exports,
            args.output_dir,
            opaque_id_key=opaque_id_key,
            expected_export_sha256=expected or None,
        )
    except (GeometryLabExportError, OSError) as exc:
        parser.error(str(exc))
    summary = {
        partition: {
            "corpus_path": str(result.corpus_path),
            "reference_key_path": str(result.reference_key_path),
            "corpus_sha256": result.corpus_sha256,
            "reference_key_sha256": result.reference_key_sha256,
            "sample_count": result.sample_count,
        }
        for partition, result in written.items()
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
