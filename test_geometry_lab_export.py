"""Fail-closed tests for the independent geometry-lab exporter."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from calendar import monthrange
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app as app_module
import review_snapshots
import reviews as reviews_module
from geometry_lab_export import (
    ADJUSTMENT,
    CORPUS_SCHEMA,
    LAB_CONTRACT_COMMIT,
    PARTITION_DEVELOPMENT,
    PARTITION_SEALED_HOLDOUT,
    PARTITION_VALIDATION,
    REFERENCE_SCHEMA,
    GeometryLabExportError,
    backend_review_export_sha256,
    build_geometry_lab_exports,
    canonical_geometry_corpus_sha256,
    export_geometry_lab,
    issuer_group_id_for_ticker,
    issuer_partition_bucket,
    load_opaque_id_key_file,
    partition_for_issuer_group,
    validate_geometry_input_corpus,
    validate_geometry_reference_key,
    verify_frozen_lab_contract,
)
from gold_labels import materialize_gold_label, sha256_json
from reviews import ReviewStore
from test_fresh_review_sessions import (
    TOKEN,
    _base_learning_capture,
    _create,
    _headers,
    _lock_base,
    _make_corpus,
)
from test_whole_pattern_capture_integration import _schema_v6_payload


LAB_ROOT = Path(
    os.environ.get(
        "COILINGVIEW_GEOMETRY_LAB_ROOT",
        "/Users/ankitmishra/Documents/Codex/2026-08-26/coilingview-geometry-lab",
    )
)
TEST_OPAQUE_ID_KEY = b"test-only-coilingview-geometry-opaque-id-key-v1"


@dataclass
class FinalizedCase:
    envelope: dict[str, Any]
    snapshot_root: Path
    source: str
    context: dict[str, Any]


def _embedded_frozen_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    first_bars = [
        ("2025-01-06", 100.0, 101.0, 99.0, 100.5),
        ("2025-01-07", 100.5, 102.0, 100.0, 101.5),
        ("2025-01-08", 101.5, 103.0, 101.0, 102.5),
        ("2025-01-09", 102.5, 104.0, 102.0, 103.5),
        ("2025-01-10", 103.5, 105.0, 103.0, 104.5),
    ]
    second_bars = [
        ("2025-01-06", 50.0, 50.5, 49.5, 50.25),
        ("2025-01-07", 50.25, 51.0, 50.0, 50.75),
        ("2025-01-08", 50.75, 51.5, 50.5, 51.25),
        ("2025-01-09", 51.25, 52.0, 51.0, 51.75),
        ("2025-01-10", 51.75, 52.5, 51.5, 52.25),
    ]

    def bars(
        rows: list[tuple[str, float, float, float, float]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "date": bar_date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }
            for bar_date, open_price, high, low, close in rows
        ]

    corpus = {
        "adjustment": ADJUSTMENT,
        "corpus_id": "synthetic-exporter-contract-v1",
        "created_at": "2026-01-02T00:00:00Z",
        "samples": [
            {
                "cutoff_date": "2025-01-10",
                "daily_bars": bars(first_bars),
                "issuer_group_id": "iss_1111111111111111",
                "sample_id": "smp_1111111111111111",
                "setup_group_id": "set_1111111111111111",
            },
            {
                "cutoff_date": "2025-01-10",
                "daily_bars": bars(second_bars),
                "issuer_group_id": "iss_2222222222222222",
                "sample_id": "smp_2222222222222222",
                "setup_group_id": "set_2222222222222222",
            },
        ],
        "schema_version": CORPUS_SCHEMA,
    }
    reference = {
        "corpus_sha256": (
            "8b9fde4cba88733319be0639a1bf07dc99d1ab7bb26039eeadfdfd0071c7e5b3"
        ),
        "records": [
            {
                "display_name": "Synthetic Issuer A",
                "notes": "Synthetic contract fixture; not market data.",
                "sample_id": "smp_1111111111111111",
                "setup_label": "Synthetic coil A",
                "ticker": "SYN-A",
            },
            {
                "display_name": "Synthetic Issuer B",
                "notes": "Synthetic contract fixture; not market data.",
                "sample_id": "smp_2222222222222222",
                "setup_label": "Synthetic coil B",
                "ticker": "SYN-B",
            },
        ],
        "schema_version": REFERENCE_SCHEMA,
    }
    return corpus, reference


def _strengthen_manifest(root: Path, source: str, tickers: list[str]) -> None:
    """Upgrade the generic API-test corpus to the production hash manifest."""

    path = root / Path(source).stem / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for position, (ticker, item) in enumerate(
        zip(tickers, manifest["items"], strict=True), start=1
    ):
        identity = review_snapshots.review_snapshot_identity(source, ticker)
        context = review_snapshots.load_blind_review_context(source, ticker)
        bars = context["monthly_bars"]
        item.update(
            {
                "position": position,
                "snapshot_file": f"{ticker}.json",
                "backend_bars_identity_sha256": identity["bars_hash"],
                "snapshot_sha256": identity["snapshot_sha256"],
                "screen_snapshot_sha256": identity[
                    "screen_snapshot_sha256"
                ],
                "canonical_monthly_bars_sha256": sha256_json(bars),
                "bar_count": len(bars),
                "first_data_date": bars[0]["date"],
                "last_data_date": bars[-1]["date"],
            }
        )
    manifest["item_count"] = len(tickers)
    path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


@pytest.fixture
def finalized_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FinalizedCase:
    source = "geometry-export.csv"
    snapshot_root = tmp_path / "review_snapshots"
    store = ReviewStore(sqlite_path=tmp_path / "reviews.db")
    monkeypatch.setattr(reviews_module, "_store", store)
    monkeypatch.setattr(
        review_snapshots, "REVIEW_SNAPSHOT_ROOT", snapshot_root
    )
    _make_corpus(snapshot_root, source, ["AAA"])
    _strengthen_manifest(snapshot_root, source, ["AAA"])

    client = TestClient(app_module.app)
    session = _create(client, source, ["AAA"], token=TOKEN)["session"]
    learning = _base_learning_capture()
    item, context = _lock_base(
        client,
        session,
        "AAA",
        learning_capture=learning,
        token=TOKEN,
    )
    payload = _schema_v6_payload(
        session,
        item,
        context,
        learning,
        key="geometry-export-schema-v6",
    )
    finalized_item = client.post(
        f"/api/review-sessions/{session['id']}/items/AAA/finalize",
        headers=_headers(TOKEN),
        json=payload,
    )
    assert finalized_item.status_code == 200, finalized_item.text
    finalized = client.post(
        f"/api/review-sessions/{session['id']}/finalize",
        headers=_headers(TOKEN),
    )
    assert finalized.status_code == 200, finalized.text
    return FinalizedCase(
        envelope=finalized.json(),
        snapshot_root=snapshot_root,
        source=source,
        context=context,
    )


def _rehash(export: dict[str, Any]) -> dict[str, Any]:
    frozen = deepcopy(export)
    return {
        "export": frozen,
        "sha256": backend_review_export_sha256(frozen),
        "finalized_at": frozen["session"]["finalized_at"],
    }


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _nested_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _nested_keys(child)}
    return set()


def _build(source: Any, **options: Any):
    return build_geometry_lab_exports(
        source,
        opaque_id_key=TEST_OPAQUE_ID_KEY,
        **options,
    )


def _only_pair(envelope: dict[str, Any]):
    artifacts = _build(envelope)
    assert len(artifacts) == 1
    return next(iter(artifacts.values()))


def test_happy_path_writes_immutable_strict_v1_pair(
    finalized_case: FinalizedCase, tmp_path: Path
) -> None:
    pair = _only_pair(finalized_case.envelope)
    assert pair.corpus["schema_version"] == CORPUS_SCHEMA
    assert pair.reference_key["schema_version"] == REFERENCE_SCHEMA
    assert pair.corpus["adjustment"] == ADJUSTMENT
    assert pair.reference_key["corpus_sha256"] == pair.corpus_sha256
    assert pair.corpus_sha256 == canonical_geometry_corpus_sha256(pair.corpus)
    provenance = json.loads(pair.reference_key["records"][0]["notes"])
    assert provenance["source_bar_frequency"] == "monthly"
    assert provenance["source_adjustment_provenance"] == "unavailable"
    assert provenance["usage_constraint"] == "monthly_ood_only"
    assert {record["sample_id"] for record in pair.reference_key["records"]} == {
        sample["sample_id"] for sample in pair.corpus["samples"]
    }

    written = export_geometry_lab(
        finalized_case.envelope,
        tmp_path / "out",
        opaque_id_key=TEST_OPAQUE_ID_KEY,
    )
    result = written[pair.partition]
    assert result.corpus_path.name == "geometry-input-corpus.v1.json"
    assert result.reference_key_path.name == "geometry-reference-key.v1.json"
    assert "model-visible" in result.corpus_path.parts
    assert "evaluation-only" in result.reference_key_path.parts
    assert result.corpus_sha256 == pair.corpus_sha256
    assert result.reference_key_path.stat().st_mode & 0o777 == 0o600
    assert result.corpus_path.read_bytes().endswith(b"\n")
    assert result.reference_key_path.read_bytes().endswith(b"\n")

    # Idempotence is byte-for-byte; a repeat does not rewrite or drift hashes.
    repeated = export_geometry_lab(
        finalized_case.envelope,
        tmp_path / "out",
        opaque_id_key=TEST_OPAQUE_ID_KEY,
    )
    assert repeated == written
    result.corpus_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(GeometryLabExportError, match="refusing to overwrite"):
        export_geometry_lab(
            finalized_case.envelope,
            tmp_path / "out",
            opaque_id_key=TEST_OPAQUE_ID_KEY,
        )


def test_physical_cutoff_truncation_uses_exact_monthly_ohlc_only(
    finalized_case: FinalizedCase,
) -> None:
    export = deepcopy(finalized_case.envelope["export"])
    record = export["records"][0]["record"]
    capture = deepcopy(record["wholePattern"])
    cutoff = finalized_case.context["quarterly_bars"][-2]["date"]
    assert cutoff < finalized_case.context["monthly_bars"][-1]["date"]
    cutoff_date = date.fromisoformat(cutoff)
    capture["cutoffDate"] = cutoff
    capture["decisionAsOf"] = cutoff_date.replace(
        day=monthrange(cutoff_date.year, cutoff_date.month)[1]
    ).isoformat()
    record["asOf"] = cutoff
    record["wholePattern"] = capture
    record["wholePatternGold"] = materialize_gold_label(
        capture, finalized_case.context["monthly_bars"]
    )

    pair = _only_pair(_rehash(export))
    sample = pair.corpus["samples"][0]
    visible = sample["daily_bars"]
    assert sample["cutoff_date"] == cutoff
    assert visible[-1]["date"] == cutoff
    assert len(visible) < len(finalized_case.context["monthly_bars"])
    assert all(bar["date"] <= cutoff for bar in visible)
    assert all(set(bar) == {"date", "open", "high", "low", "close"} for bar in visible)
    expected = [
        {key: bar[key] for key in ("date", "open", "high", "low", "close")}
        for bar in finalized_case.context["monthly_bars"]
        if bar["date"] <= cutoff
    ]
    assert visible == expected


def test_model_visible_and_reference_key_are_mechanically_separate(
    finalized_case: FinalizedCase,
) -> None:
    export = deepcopy(finalized_case.envelope["export"])
    export["candidate_nominations"] = [
        {
            "ticker": "LEAK.NS",
            "universe": "boundary",
            "rationale": "PENDING-BOUNDARY-LABEL-MUST-STAY-SEALED",
            "history_as_of": "2026-06-01",
            "bars_hash": "f" * 64,
            "revision": 1,
            "created_at": export["exported_at"],
            "updated_at": export["exported_at"],
        }
    ]
    pair = _only_pair(_rehash(export))
    corpus_text = json.dumps(pair.corpus, sort_keys=True)
    reference_text = json.dumps(pair.reference_key, sort_keys=True)
    assert "AAA" not in corpus_text
    assert "AAA-long-base" not in corpus_text
    assert TEST_OPAQUE_ID_KEY.decode("ascii") not in corpus_text
    assert "PENDING-BOUNDARY" not in corpus_text
    assert "PENDING-BOUNDARY" not in reference_text
    assert "AAA" in reference_text
    assert "AAA-long-base" in reference_text
    forbidden = {
        "ticker",
        "display_name",
        "setup_label",
        "notes",
        "wholePattern",
        "wholePatternGold",
        "judgments",
        "label_sha256",
        "algorithm",
        "analysis",
        "screen_snapshot",
        "source_features",
        "outcome",
        "volume",
    }
    assert forbidden.isdisjoint(_nested_keys(pair.corpus))


def test_issuer_grouping_and_split_are_deterministic_and_issuer_disjoint() -> None:
    with pytest.raises(GeometryLabExportError, match="opaque_id_key"):
        issuer_group_id_for_ticker("AAA", opaque_id_key=b"too-short")
    aaa = issuer_group_id_for_ticker(
        "AAA", opaque_id_key=TEST_OPAQUE_ID_KEY
    )
    assert aaa == (
        "iss_a362700c114154b918657b6155129f0f137a345ad515e1df436e33bd3d40d8e0"
    )
    assert issuer_partition_bucket(aaa) == 2_970
    assert partition_for_issuer_group(aaa) == PARTITION_DEVELOPMENT
    assert aaa == issuer_group_id_for_ticker(
        "aaa", opaque_id_key=TEST_OPAQUE_ID_KEY
    )
    assert aaa != issuer_group_id_for_ticker(
        "BBB", opaque_id_key=TEST_OPAQUE_ID_KEY
    )
    assert aaa != issuer_group_id_for_ticker(
        "AAA", opaque_id_key=b"different-private-opaque-id-key-for-tests"
    )
    assert partition_for_issuer_group(aaa) == partition_for_issuer_group(aaa)

    counts = {
        PARTITION_DEVELOPMENT: 0,
        PARTITION_VALIDATION: 0,
        PARTITION_SEALED_HOLDOUT: 0,
    }
    for index in range(2_000):
        issuer_id = issuer_group_id_for_ticker(
            f"T{index:04d}", opaque_id_key=TEST_OPAQUE_ID_KEY
        )
        bucket = issuer_partition_bucket(issuer_id)
        partition = partition_for_issuer_group(issuer_id)
        counts[partition] += 1
        if partition == PARTITION_DEVELOPMENT:
            assert bucket < 6_000
        elif partition == PARTITION_VALIDATION:
            assert 6_000 <= bucket < 8_000
        else:
            assert 8_000 <= bucket < 10_000
    assert 0.55 < counts[PARTITION_DEVELOPMENT] / 2_000 < 0.65
    assert 0.16 < counts[PARTITION_VALIDATION] / 2_000 < 0.24
    assert 0.16 < counts[PARTITION_SEALED_HOLDOUT] / 2_000 < 0.24


def test_bare_review_export_requires_its_independent_finalization_hash(
    finalized_case: FinalizedCase,
) -> None:
    bare = finalized_case.envelope["export"]
    with pytest.raises(GeometryLabExportError, match="expected finalization SHA-256"):
        _build(bare)

    artifacts = _build(
        bare,
        expected_export_sha256=finalized_case.envelope["sha256"],
    )
    assert sum(len(pair.corpus["samples"]) for pair in artifacts.values()) == 1
    with pytest.raises(GeometryLabExportError, match="split seed cannot be changed"):
        _build(finalized_case.envelope, split_seed="reshuffle")


def test_opaque_id_key_file_is_private_regular_and_bounded(tmp_path: Path) -> None:
    key_path = tmp_path / "opaque-id.key"
    key_path.write_bytes(TEST_OPAQUE_ID_KEY)
    key_path.chmod(0o600)
    assert load_opaque_id_key_file(key_path) == TEST_OPAQUE_ID_KEY

    key_path.chmod(0o640)
    with pytest.raises(GeometryLabExportError, match="group or others"):
        load_opaque_id_key_file(key_path)

    key_path.chmod(0o600)
    link_path = tmp_path / "opaque-id-link.key"
    link_path.symlink_to(key_path)
    with pytest.raises(GeometryLabExportError, match="cannot read"):
        load_opaque_id_key_file(link_path)

    oversized = tmp_path / "oversized.key"
    oversized.write_bytes(b"x" * 4_097)
    oversized.chmod(0o600)
    with pytest.raises(GeometryLabExportError, match="32 to 4096"):
        load_opaque_id_key_file(oversized)


def test_rehashed_session_policy_drift_from_manifest_is_rejected(
    finalized_case: FinalizedCase,
) -> None:
    export = deepcopy(finalized_case.envelope["export"])
    export["session"]["snapshot"]["review_policy"][
        "model_reveal_allowed"
    ] = False
    with pytest.raises(GeometryLabExportError, match="review policy"):
        _build(_rehash(export))


def test_pending_and_unresolved_reviews_are_rejected(
    finalized_case: FinalizedCase,
) -> None:
    pending = deepcopy(finalized_case.envelope["export"])
    pending["session"]["items"][0]["status"] = "pending"
    pending["session"]["counts"] = {
        "pending": 1,
        "reviewed": 0,
        "skipped": 0,
        "total": 1,
    }
    pending["session"]["next_pending_ticker"] = "AAA"
    with pytest.raises(GeometryLabExportError, match="pending"):
        _build(_rehash(pending))

    unresolved = deepcopy(finalized_case.envelope["export"])
    unresolved_record = unresolved["records"][0]["record"]
    unresolved_capture = deepcopy(unresolved_record["wholePattern"])
    unresolved_capture["topReviewComplete"] = False
    unresolved_record["wholePattern"] = unresolved_capture
    unresolved_record["wholePatternGold"] = materialize_gold_label(
        unresolved_capture, finalized_case.context["monthly_bars"]
    )
    with pytest.raises(GeometryLabExportError, match="unresolved"):
        _build(_rehash(unresolved))


def test_reviewed_cutoff_after_finalization_is_rejected(
    finalized_case: FinalizedCase,
) -> None:
    export = deepcopy(finalized_case.envelope["export"])
    export["session"]["created_at"] = "2009-01-01T00:00:00Z"
    export["session"]["finalized_at"] = "2019-11-01T00:00:00Z"
    export["exported_at"] = "2019-11-01T00:00:00Z"
    with pytest.raises(GeometryLabExportError, match="cutoff.*later than"):
        _build(_rehash(export))


@pytest.mark.parametrize("target", ["snapshot", "manifest"])
def test_tampered_snapshot_or_manifest_is_rejected(
    finalized_case: FinalizedCase, target: str
) -> None:
    folder = finalized_case.snapshot_root / Path(finalized_case.source).stem
    if target == "snapshot":
        path = folder / "AAA.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["monthly_bars"][0]["close"] += 0.01
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path = folder / "manifest.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(GeometryLabExportError, match="snapshot|manifest|hash drift"):
        _build(finalized_case.envelope)


def test_outer_and_gold_hash_drift_are_rejected(
    finalized_case: FinalizedCase,
) -> None:
    outer = deepcopy(finalized_case.envelope)
    outer["export"]["reviewer"]["name"] = "tampered"
    with pytest.raises(GeometryLabExportError, match="envelope hash"):
        _build(outer)

    inner = deepcopy(finalized_case.envelope["export"])
    inner["records"][0]["record"]["wholePatternGold"]["label_sha256"] = "0" * 64
    with pytest.raises(GeometryLabExportError, match="rematerialization failed"):
        _build(_rehash(inner))

    derived = deepcopy(finalized_case.envelope["export"])
    gold = derived["records"][0]["record"]["wholePatternGold"]
    gold["structures"][0]["line"]["value_at_cutoff"] += 1.0
    unhashed = dict(gold)
    unhashed.pop("label_sha256")
    gold["label_sha256"] = sha256_json(unhashed)
    with pytest.raises(GeometryLabExportError, match="rematerialization failed"):
        _build(_rehash(derived))


def test_frozen_lab_fixture_schema_and_generated_artifacts_are_compatible(
    finalized_case: FinalizedCase, tmp_path: Path
) -> None:
    fixture_corpus, fixture_reference = _embedded_frozen_fixture()
    normalized_fixture = validate_geometry_input_corpus(fixture_corpus)
    assert canonical_geometry_corpus_sha256(normalized_fixture) == (
        "8b9fde4cba88733319be0639a1bf07dc99d1ab7bb26039eeadfdfd0071c7e5b3"
    )
    validate_geometry_reference_key(fixture_reference, corpus=normalized_fixture)
    wrong_json_type = deepcopy(fixture_corpus)
    wrong_json_type["samples"][0]["daily_bars"][0]["open"] = "100.0"
    with pytest.raises(GeometryLabExportError, match="finite positive number"):
        validate_geometry_input_corpus(wrong_json_type)
    extreme_number = deepcopy(fixture_corpus)
    extreme_number["samples"][0]["daily_bars"][0]["open"] = 10**400
    with pytest.raises(GeometryLabExportError, match="finite positive number"):
        validate_geometry_input_corpus(extreme_number)

    pair = _only_pair(finalized_case.envelope)
    validate_geometry_input_corpus(pair.corpus)
    validate_geometry_reference_key(pair.reference_key, corpus=pair.corpus)
    if not LAB_ROOT.is_dir():
        return

    verified = verify_frozen_lab_contract(LAB_ROOT)
    assert verified == {
        "commit": LAB_CONTRACT_COMMIT,
        "schema_files": (
            "geometry-embed-request.v1.schema.json",
            "geometry-index-manifest.v1.schema.json",
            "geometry-input-corpus.v1.schema.json",
            "geometry-neighbors-request.v1.schema.json",
            "geometry-reference-key.v1.schema.json",
        ),
        "fixture_corpus_sha256": (
            "8b9fde4cba88733319be0639a1bf07dc99d1ab7bb26039eeadfdfd0071c7e5b3"
        ),
    }

    archive = subprocess.run(
        [
            "git",
            "-C",
            str(LAB_ROOT),
            "archive",
            LAB_CONTRACT_COMMIT,
            "src",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frozen_checkout = tmp_path / "frozen-lab"
    frozen_checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(frozen_checkout, filter="data")

    frozen_src = str(frozen_checkout / "src")
    prior_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "coilingview_geometry_lab"
        or name.startswith("coilingview_geometry_lab.")
    }
    for name in prior_modules:
        sys.modules.pop(name)
    sys.path.insert(0, frozen_src)
    try:
        from coilingview_geometry_lab.canonical import canonical_json_sha256
        from coilingview_geometry_lab.contracts import (
            GeometryInputCorpusV1,
            GeometryReferenceKeyV1,
        )

        lab_corpus = GeometryInputCorpusV1.parse_obj(pair.corpus)
        lab_reference = GeometryReferenceKeyV1.parse_obj(pair.reference_key)
        assert canonical_json_sha256(lab_corpus) == pair.corpus_sha256
        assert lab_reference.corpus_sha256 == pair.corpus_sha256
        assert {record.sample_id for record in lab_reference.records} == {
            sample.sample_id for sample in lab_corpus.samples
        }
    finally:
        sys.path.remove(frozen_src)
        for name in tuple(sys.modules):
            if name == "coilingview_geometry_lab" or name.startswith(
                "coilingview_geometry_lab."
            ):
                sys.modules.pop(name)
        sys.modules.update(prior_modules)
