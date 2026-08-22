from __future__ import annotations

import copy

import pytest

import automatic_exemplar_evaluator as evaluator
from eodhd_ingestion import (
    EodhdClient,
    EodhdIngestionError,
    build_frozen_snapshot,
    ingest_symbol,
    load_frozen_snapshot,
    normalize_monthly_rows,
    validate_frozen_snapshot,
    write_frozen_snapshot,
)


CODE_SHA = "a" * 40
FETCHED_AT = "2026-08-22T03:30:00Z"


def _provider_rows() -> list[dict]:
    return [
        {
            "date": "2024-01-31",
            "open": 100,
            "high": 112,
            "low": 98,
            "close": 110,
            "adjusted_close": 108.5,
            "volume": 1_500_000,
        },
        {
            "date": "2024-02-29",
            "open": 110,
            "high": 116,
            "low": 104,
            "close": 114,
            "adjusted_close": 112.4,
            "volume": 1_750_000,
        },
        {
            "date": "2024-03-28",
            "open": 114,
            "high": 120,
            "low": 109,
            "close": 118,
            "adjusted_close": 116.1,
            "volume": 1_900_000,
        },
    ]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return copy.deepcopy(self._payload)


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls: list[dict] = []

    def get(self, url, *, params, timeout):
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        return _FakeResponse(self.payload)


def _snapshot() -> dict:
    return build_frozen_snapshot(
        symbol="AAPL.US",
        date_from="2024-01-01",
        date_to="2024-03-31",
        provider_rows=_provider_rows(),
        fetched_at=FETCHED_AT,
        code_sha=CODE_SHA,
    )


def test_client_uses_documented_monthly_request_and_snapshot_hides_token():
    session = _FakeSession(_provider_rows())
    client = EodhdClient(
        "top-secret",
        session=session,
        timeout_seconds=12.5,
    )

    snapshot = ingest_symbol(
        client,
        symbol="aapl.us",
        date_from="2024-01-01",
        date_to="2024-03-31",
        fetched_at=FETCHED_AT,
        code_sha=CODE_SHA,
    )

    assert session.calls == [
        {
            "url": "https://eodhd.com/api/eod/AAPL.US",
            "params": {
                "api_token": "top-secret",
                "fmt": "json",
                "period": "m",
                "order": "a",
                "from": "2024-01-01",
                "to": "2024-03-31",
            },
            "timeout": 12.5,
        }
    ]
    assert snapshot["symbol"] == "AAPL.US"
    assert snapshot["request"]["params"] == {
        "fmt": "json",
        "period": "m",
        "order": "a",
        "from": "2024-01-01",
        "to": "2024-03-31",
    }
    assert "top-secret" not in str(snapshot)
    assert snapshot["provider_semantics"] == {
        "ohlc_adjustment": "raw_unadjusted",
        "adjusted_close_used": False,
        "volume_adjustment": "provider_split_adjusted",
    }
    assert snapshot["monthly_bars"][0] == {
        "date": "2024-01-31",
        "open": 100.0,
        "high": 112.0,
        "low": 98.0,
        "close": 110.0,
        "volume": 1_500_000.0,
    }
    validate_frozen_snapshot(snapshot)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda rows: rows[0].update(high=99),
            "OHLC containment",
        ),
        (
            lambda rows: rows[1].update(date="2024-01-15"),
            "strictly ascending|at most one bar per month",
        ),
        (
            lambda rows: rows[2].update(date="2024-04-01"),
            "outside the requested range",
        ),
        (
            lambda rows: rows[0].update(volume=-1),
            "volume must be nonnegative",
        ),
    ],
)
def test_monthly_normalization_fails_closed(mutate, message):
    rows = _provider_rows()
    mutate(rows)
    with pytest.raises(EodhdIngestionError, match=message):
        normalize_monthly_rows(
            rows,
            date_from="2024-01-01",
            date_to="2024-03-31",
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda snapshot: snapshot["provider_rows"][0].update(close=109),
        lambda snapshot: snapshot["monthly_bars"][0].update(close=109),
        lambda snapshot: snapshot.update(code_sha="b" * 40),
        lambda snapshot: snapshot["request"]["params"].update(api_token="leak"),
    ],
)
def test_snapshot_validation_detects_tampering(mutate):
    snapshot = _snapshot()
    mutate(snapshot)
    with pytest.raises(EodhdIngestionError):
        validate_frozen_snapshot(snapshot)


def test_snapshot_write_load_and_no_silent_overwrite(tmp_path):
    snapshot = _snapshot()
    path = tmp_path / "AAPL.US.json"

    assert write_frozen_snapshot(snapshot, path) == path
    assert load_frozen_snapshot(path) == snapshot
    with pytest.raises(EodhdIngestionError, match="already exists"):
        write_frozen_snapshot(snapshot, path)


def test_evaluator_accepts_inline_and_corpus_relative_eodhd_snapshot(tmp_path):
    snapshot = _snapshot()
    path = tmp_path / "snapshots" / "AAPL.US.json"
    write_frozen_snapshot(snapshot, path)

    inline_bars, inline_source = evaluator._load_episode_input(
        {"ticker": "AAPL.US", "eodhd_snapshot": snapshot}
    )
    file_bars, file_source = evaluator._load_episode_input(
        {
            "ticker": "AAPL.US",
            "eodhd_snapshot": "snapshots/AAPL.US.json",
        },
        source_root=tmp_path,
    )

    assert inline_bars == file_bars == snapshot["monthly_bars"]
    assert inline_source == file_source
    assert inline_source["snapshot_sha256"] == snapshot["snapshot_sha256"]
    assert inline_source["code_sha"] == CODE_SHA


def test_evaluator_rejects_ticker_snapshot_mismatch():
    with pytest.raises(evaluator.BenchmarkCorpusError, match="does not match"):
        evaluator._load_episode_input(
            {"ticker": "MSFT.US", "eodhd_snapshot": _snapshot()}
        )


def _eodhd_corpus(*, code_sha=None) -> dict:
    corpus = {
        "schema_version": evaluator.CORPUS_SCHEMA_VERSION,
        "kind": evaluator.CORPUS_KIND,
        "corpus_id": "eodhd-sha-boundary",
        "episodes": [
            {
                "split": "development",
                "ticker": "AAPL.US",
                "eodhd_snapshot": _snapshot(),
            }
        ],
    }
    if code_sha is not None:
        corpus["code_sha"] = code_sha
    return corpus


def test_eodhd_corpus_requires_exact_code_sha_before_loading_gold():
    with pytest.raises(evaluator.BenchmarkCorpusError, match="requires.*code_sha"):
        evaluator.evaluate_corpus(_eodhd_corpus())


def test_eodhd_snapshot_sha_must_match_corpus_before_loading_gold():
    with pytest.raises(evaluator.BenchmarkCorpusError, match="snapshot code_sha"):
        evaluator.evaluate_corpus(_eodhd_corpus(code_sha="b" * 40))


def test_runtime_checkout_sha_must_match_corpus_before_loading_snapshot():
    with pytest.raises(evaluator.BenchmarkCorpusError, match="running checkout"):
        evaluator.evaluate_corpus(
            _eodhd_corpus(code_sha=CODE_SHA),
            expected_code_sha="c" * 40,
        )
