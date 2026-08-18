from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

import screen_monthly
from history_providers import (
    EODHD_TOKEN_ENV,
    FROZEN_HISTORY_DIR_ENV,
    HistoryProviderError,
    fetch_eodhd_daily_history,
    load_frozen_daily_history,
    resolve_eodhd_symbol,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def eodhd_payloads():
    bars = [
        {
            "date": "2020-08-28",
            "open": 500,
            "high": 504,
            "low": 496,
            "close": 500,
            "volume": 100,
        },
        {
            "date": "2020-08-31",
            "open": 125,
            "high": 130,
            "low": 124,
            "close": 129,
            "volume": 400,
        },
    ]
    splits = [{"date": "2020-08-31", "split": "4.000000/1.000000"}]
    return bars, splits


def test_eodhd_returns_raw_ohlc_with_explicit_split_events():
    bars, splits = eodhd_payloads()
    calls = []

    def request_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(splits if "/splits/" in url else bars)

    frame = fetch_eodhd_daily_history(
        "AAPL",
        api_token="private-test-token",
        request_get=request_get,
    )

    assert frame.attrs["source"] == "eodhd"
    assert frame.attrs["provider_symbol"] == "AAPL.US"
    assert frame.loc[pd.Timestamp("2020-08-31"), "Stock Splits"] == 4.0
    assert "private-test-token" not in json.dumps(frame.attrs)
    assert len(calls) == 2
    assert all(call[1]["params"]["from"] == "1900-01-01" for call in calls)


def test_international_eodhd_suffixes_are_never_guessed():
    with pytest.raises(HistoryProviderError, match="explicit EODHD symbol mapping"):
        resolve_eodhd_symbol("INFY.NS")
    assert resolve_eodhd_symbol(
        "INFY.NS", symbol_map={"INFY.NS": "INFY.NSE"}
    ) == "INFY.NSE"


def test_frozen_csv_requires_raw_ohlc_and_split_provenance(tmp_path, monkeypatch):
    source = tmp_path / "AAA.csv"
    source.write_text(
        "Date,Open,High,Low,Close,Volume,Stock Splits\n"
        "1990-01-02,10,11,9,10.5,100,0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(FROZEN_HISTORY_DIR_ENV, str(tmp_path))

    frame = load_frozen_daily_history("AAA")

    assert frame.attrs["source"] == "frozen_csv"
    assert frame.attrs["provider_history_start"] == "1990-01-02"


def test_screen_monthly_can_use_eodhd_without_changing_the_default(monkeypatch):
    bars, splits = eodhd_payloads()

    def request_get(url, **_kwargs):
        return FakeResponse(splits if "/splits/" in url else bars)

    daily = fetch_eodhd_daily_history(
        "AAPL",
        api_token="private-test-token",
        request_get=request_get,
    )
    monkeypatch.setenv("COILINGVIEW_HISTORY_PROVIDER", "eodhd")
    monkeypatch.setenv(EODHD_TOKEN_ENV, "private-test-token")
    monkeypatch.setattr(
        screen_monthly,
        "fetch_eodhd_daily_history",
        lambda _ticker: daily,
    )

    monthly = screen_monthly.fetch_monthly_history("AAPL")

    assert monthly is not None
    assert monthly.attrs["source"] == "eodhd"
    assert monthly.attrs["adjustment_mode"] == "split_adjusted"
    assert monthly.attrs["adjustment_source"] == "eodhd_splits_api"
    assert monthly.iloc[0]["High"] == 130


def test_hsbc_1999_three_for_one_event_keeps_pre_post_price_basis_continuous():
    daily = pd.DataFrame(
        {
            "Open": [90.0, 30.0],
            "High": [93.0, 31.0],
            "Low": [87.0, 29.0],
            "Close": [90.0, 30.0],
            "Volume": [100.0, 300.0],
            "Stock Splits": [0.0, 3.0],
        },
        index=pd.to_datetime(["1999-07-02", "1999-07-05"]),
    )
    daily.attrs.update(
        {
            "source": "frozen_csv",
            "adjustment_source": "hkex_1999_three_for_one",
        }
    )

    monthly = screen_monthly._split_adjust_and_aggregate_monthly(daily)

    assert monthly.iloc[0]["Open"] == pytest.approx(30.0)
    assert monthly.iloc[0]["Close"] == pytest.approx(30.0)
    assert monthly.iloc[0]["Volume"] == pytest.approx(600.0)


def test_long_history_pilot_freezes_completed_quarters_without_labels(tmp_path):
    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    (frozen_root / "AAA.csv").write_text(
        "Date,Open,High,Low,Close,Volume,Stock Splits\n"
        "1980-01-02,10,11,9,10.5,100,0\n"
        "1980-02-01,10.5,12,10,11.5,120,0\n"
        "1980-03-31,11.5,13,11,12.5,140,0\n"
        "1980-04-01,12.5,14,12,13.5,160,2\n",
        encoding="utf-8",
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "symbols": [
                    {
                        "ticker": "AAA",
                        "company_name": "AAA Holdings plc",
                        "listed_since": "1980-01-01",
                        "listing_date_source": "https://example.test/aaa-listing",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pilot"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_long_history_pilot.py",
            "--provider",
            "frozen_csv",
            "--frozen-root",
            str(frozen_root),
            "--spec",
            str(spec),
            "--output",
            str(output),
            "--as-of",
            "1980-03-31",
        ],
        check=True,
        cwd=__file__.rsplit("/", 1)[0],
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    snapshot = json.loads((output / "AAA.json").read_text(encoding="utf-8"))
    assert result.stdout.strip().endswith("manifest.json")
    assert manifest["holdout_labels_included"] is False
    assert manifest["counts"]["verified_listing_quarter_to_date"] == 1
    assert manifest["items"][0]["company_name"] == "AAA Holdings plc"
    assert snapshot["company_name"] == "AAA Holdings plc"
    assert snapshot["coverage"]["first_bar_date"] == "1980-01-01"
    assert snapshot["coverage"]["status"] == "verified_listing_quarter_to_date"
    assert snapshot["coverage"]["listing_date_source"].startswith("https://")
    assert snapshot["coverage"]["admitted_daily_end"] == "1980-03-31"
    assert snapshot["security_identity_sha256"]
    assert snapshot["coverage_sha256"]
    assert snapshot["monthly_bars"][-1]["date"] == "1980-03-01"
    assert snapshot["monthly_bars"][-1]["close"] == 12.5
    assert snapshot["provider"]["adjustment_mode"] == "split_adjusted"
    assert snapshot["provider"]["provider_history_end"] == "1980-04-01"
    assert snapshot["provider"]["admitted_daily_end"] == "1980-03-31"


def test_long_history_pilot_requires_full_security_identity(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"symbols": [{"ticker": "AAA"}]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_long_history_pilot.py",
            "--provider",
            "frozen_csv",
            "--spec",
            str(spec),
            "--output",
            str(tmp_path / "pilot"),
        ],
        cwd=__file__.rsplit("/", 1)[0],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "full company_name" in result.stderr


def test_long_history_pilot_fails_closed_without_writing_truncated_corpus(tmp_path):
    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    (frozen_root / "AAA.csv").write_text(
        "Date,Open,High,Low,Close,Volume,Stock Splits\n"
        "1980-02-01,10,11,9,10.5,100,0\n"
        "1980-03-31,11,12,10,11.5,120,0\n",
        encoding="utf-8",
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "symbols": [
                    {
                        "ticker": "AAA",
                        "company_name": "AAA Holdings plc",
                        "listed_since": "1980-01-01",
                        "listing_date_source": "https://example.test/aaa-listing",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pilot"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_long_history_pilot.py",
            "--provider",
            "frozen_csv",
            "--frozen-root",
            str(frozen_root),
            "--spec",
            str(spec),
            "--output",
            str(output),
            "--as-of",
            "1980-03-31",
        ],
        cwd=__file__.rsplit("/", 1)[0],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "no corpus was written" in result.stderr
    assert not output.exists()
