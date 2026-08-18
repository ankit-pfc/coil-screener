from __future__ import annotations

import gzip
import json

import pytest

from scripts.pull_eodhd_benchmark_history import (
    PullError,
    _resume_item_valid,
    pull_one,
    resolve_provider_symbol,
    summarize_daily_rows,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("AAPL", "AAPL.US"),
        ("ALV.DE", "ALV.XETRA"),
        ("ULVR.L", "ULVR.LSE"),
        ("BHP.AX", "BHP.AU"),
        ("2330.TW", "2330.TW"),
    ],
)
def test_reviewed_provider_symbol_mappings(ticker, expected):
    assert resolve_provider_symbol(ticker) == (expected, None)


@pytest.mark.parametrize("ticker", ["RELIANCE.NS", "7203.T"])
def test_unavailable_exchange_mappings_fail_closed(ticker):
    provider_symbol, reason = resolve_provider_symbol(ticker)
    assert provider_symbol is None
    assert "does not expose" in str(reason)


def test_daily_summary_rejects_duplicate_dates():
    rows = [
        {"date": "2020-01-02", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 3},
        {"date": "2020-01-02", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 3},
    ]
    with pytest.raises(PullError, match="duplicate"):
        summarize_daily_rows(rows)


def test_pull_one_archives_raw_payloads_without_token(tmp_path):
    daily = [
        {"date": "1999-01-04", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 3},
        {"date": "2026-08-17", "open": 2, "high": 3, "low": 2, "close": 3, "volume": 4},
    ]
    splits = [{"date": "2000-01-03", "split": "2/1"}]
    calls = []

    def request_get(url, **kwargs):
        calls.append((url, kwargs))
        if "/splits/" in url:
            return FakeResponse(splits)
        return FakeResponse(daily)

    item = pull_one(
        "EXM",
        "EXM.US",
        tmp_path,
        token="private-test-token",
        request_get=request_get,
    )

    assert item["provider_history_start"] == "1999-01-04"
    assert item["provider_history_end"] == "2026-08-17"
    assert item["daily_bar_count"] == 2
    assert item["split_event_count"] == 1
    assert len(calls) == 2
    assert all(call[1]["params"]["api_token"] == "private-test-token" for call in calls)
    with gzip.open(tmp_path / item["raw_daily_file"], "rt", encoding="utf-8") as handle:
        assert json.load(handle) == daily
    assert "private-test-token" not in json.dumps(item)
    assert _resume_item_valid(item, tmp_path)

    (tmp_path / item["raw_daily_file"]).write_bytes(b"tampered")
    assert not _resume_item_valid(item, tmp_path)
