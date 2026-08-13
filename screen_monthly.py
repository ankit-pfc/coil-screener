from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Callable, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


DEFAULT_TICKERS = [
    "AER",
    "AVT",
    "BDC",
    "DD",
    "EWY",
    "LAZ",
    "PPC",
    "PTCT",
    "STLD",
    "TEX",
    "UTHR",
    "MSFT",
    "AAPL",
    "XOM",
    "CAT",
]


# A deliberately cross-market review set for detector calibration. These are
# liquid Yahoo Finance listings spanning multiple market structures, currencies,
# price scales, listing ages, and sector mixes. Keeping the symbols in-repo makes
# a review pass reproducible instead of depending on a changing web index.
INTERNATIONAL_REVIEW_TICKERS = [
    # Canada
    "SHOP.TO", "RY.TO", "TD.TO", "CNR.TO", "ENB.TO",
    # United Kingdom
    "AZN.L", "SHEL.L", "ULVR.L", "HSBA.L", "REL.L",
    # Germany
    "SAP.DE", "SIE.DE", "ALV.DE", "BAS.DE", "DTE.DE",
    # France
    "MC.PA", "OR.PA", "SAN.PA", "SU.PA", "AIR.PA",
    # Netherlands
    "ASML.AS", "ADYEN.AS", "PHIA.AS",
    # Switzerland
    "NESN.SW", "NOVN.SW", "GIVN.SW", "UBSG.SW",
    # Japan
    "7203.T", "6758.T", "9984.T", "8306.T", "8035.T",
    # Hong Kong
    "0700.HK", "9988.HK", "1299.HK", "0005.HK", "3690.HK",
    # India
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    # Australia
    "BHP.AX", "CBA.AX", "CSL.AX", "WES.AX", "MQG.AX",
    # Brazil
    "VALE3.SA", "PETR4.SA", "ITUB4.SA", "WEGE3.SA",
    # South Korea
    "005930.KS", "000660.KS", "035420.KS", "051910.KS",
    # Taiwan
    "2330.TW", "2317.TW", "2454.TW",
    # South Africa
    "NPN.JO", "SOL.JO", "SBK.JO",
]


@dataclass
class ScreenResult:
    ticker: str
    age_years: float
    last_close: float
    score_total: float
    score_long_coil: float
    score_tight_resistance: float
    score_ascending_compression: float
    pos_in_10y_range: float
    dist_to_10y_high_pct: float
    range_ratio_24_120: float
    range_ratio_24_60: float
    low_36m_above_10y_low_pct: float
    slope_high_60m: float
    slope_low_60m: float
    trend_r2_60m: float
    peak_age_months: float
    old_peak_similarity: float


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def normalize_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    std = float(series.std(ddof=0))
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - float(series.mean())) / std


def fit_slope(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) < 2:
        return np.nan
    x = np.arange(len(clean), dtype=float)
    y = clean.to_numpy(dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def fit_trend_r2(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) < 3:
        return np.nan
    x = np.arange(len(clean), dtype=float)
    y = clean.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def _split_adjust_and_aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Build split-only adjusted monthly OHLCV from raw daily provider bars.

    Yahoo's ``auto_adjust`` uses Adj Close and therefore includes distributions.
    Validation needs a narrower, auditable policy: raw daily OHLC is expressed
    in the latest share units using only reported ``Stock Splits`` events, then
    aggregated to calendar months. The split-day candle is already post-split,
    so only events strictly after a daily candle affect that candle.
    """
    required = ["Open", "High", "Low", "Close"]
    missing = [name for name in [*required, "Stock Splits"] if name not in daily]
    if missing:
        raise ValueError(
            "provider history cannot prove split adjustment; missing "
            + ", ".join(missing)
        )
    frame = daily.copy().sort_index()
    if frame.empty:
        return frame
    prices = frame[required].apply(pd.to_numeric, errors="coerce")
    price_values = prices.to_numpy(dtype=float)
    if not np.isfinite(price_values).all() or (price_values <= 0).any():
        raise ValueError("raw daily provider history contains invalid OHLC")

    splits = pd.to_numeric(frame["Stock Splits"], errors="coerce")
    invalid_split = (
        splits.isna()
        | ~np.isfinite(splits)
        | ((splits != 0.0) & (splits <= 0.0))
    )
    if bool(invalid_split.any()):
        raise ValueError("provider history contains an invalid split factor")
    split_events = splits.mask(splits == 0.0, 1.0)
    inclusive_factor = split_events.iloc[::-1].cumprod().iloc[::-1]
    future_factor = inclusive_factor / split_events

    adjusted = prices.div(future_factor, axis=0)
    if "Volume" in frame:
        volume = pd.to_numeric(frame["Volume"], errors="coerce")
        volume = volume.where(np.isfinite(volume) & (volume >= 0.0))
        adjusted["Volume"] = volume.mul(future_factor, axis=0)
    else:
        adjusted["Volume"] = np.nan

    index = pd.DatetimeIndex(adjusted.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    periods = index.to_period("M")
    prices_monthly = adjusted.groupby(periods).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    )
    volume_monthly = adjusted["Volume"].groupby(periods).agg(
        lambda values: values.sum() if values.notna().all() else np.nan
    )
    monthly = prices_monthly.assign(Volume=volume_monthly)
    monthly.index = monthly.index.to_timestamp()
    monthly.attrs.update(
        {
            "adjustment_mode": "split_adjusted",
            "adjustment_source": "yfinance_stock_splits",
            "source_interval": "1d",
            "adjustment_transform_version": "yfinance-stock-splits-v1",
        }
    )
    return monthly


def fetch_monthly_history(ticker: str) -> Optional[pd.DataFrame]:
    data = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False,
        multi_level_index=False,
    )
    if data is None or data.empty:
        return None
    return _split_adjust_and_aggregate_monthly(data)


def normalize_tickers(tickers: Iterable[str]) -> List[str]:
    return list(
        dict.fromkeys(
            ticker.strip().upper()
            for ticker in tickers
            if ticker and ticker.strip()
        )
    )


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_monthly_histories(tickers: List[str], batch_size: int = 10) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}

    for batch in chunked(tickers, batch_size):
        data = yf.download(
            batch,
            period="max",
            interval="1d",
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        if data is None or data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for ticker in batch:
                if ticker not in data.columns.get_level_values(0):
                    continue
                ticker_df = data[ticker]
                if not ticker_df.empty:
                    histories[ticker] = _split_adjust_and_aggregate_monthly(
                        ticker_df
                    )
        else:
            ticker_df = data
            if not ticker_df.empty and len(batch) == 1:
                histories[batch[0]] = _split_adjust_and_aggregate_monthly(
                    ticker_df
                )

    return histories


def compute_features(ticker: str, monthly: pd.DataFrame) -> Optional[ScreenResult]:
    monthly = monthly.copy()
    monthly = monthly.sort_index()

    if len(monthly) < 120:
        return None

    close = monthly["Close"].astype(float)
    high = monthly["High"].astype(float)
    low = monthly["Low"].astype(float)

    last_close = float(close.iloc[-1])
    age_years = len(monthly) / 12.0

    high_120 = float(high.iloc[-120:].max())
    low_120 = float(low.iloc[-120:].min())
    high_60 = float(high.iloc[-60:].max())
    low_60 = float(low.iloc[-60:].min())
    high_24 = float(high.iloc[-24:].max())
    low_24 = float(low.iloc[-24:].min())
    low_36 = float(low.iloc[-36:].min())

    range_120 = high_120 - low_120
    range_60 = high_60 - low_60
    range_24 = high_24 - low_24

    if range_120 <= 0 or range_60 <= 0:
        return None

    pos_in_10y_range = (last_close - low_120) / range_120
    dist_to_10y_high_pct = (high_120 - last_close) / high_120 if high_120 > 0 else np.nan
    range_ratio_24_120 = range_24 / range_120
    range_ratio_24_60 = range_24 / range_60
    low_36m_above_10y_low_pct = (low_36 - low_120) / low_120 if low_120 > 0 else np.nan

    rolling_high_12 = high.rolling(12).max().iloc[-60:]
    rolling_low_12 = low.rolling(12).min().iloc[-60:]
    slope_high_60m = fit_slope(normalize_series(rolling_high_12))
    slope_low_60m = fit_slope(normalize_series(rolling_low_12))
    trend_r2_60m = fit_trend_r2(normalize_series(close.iloc[-60:]))

    older_high_window = high.iloc[:-24]
    older_high = float(older_high_window.max()) if not older_high_window.empty else high_120
    old_peak_similarity = older_high / high_120 if high_120 > 0 else np.nan
    had_old_peak = old_peak_similarity >= 0.80 if not np.isnan(old_peak_similarity) else False
    if had_old_peak:
        prior_peak_idx = older_high_window.idxmax()
        peak_age_months = float((monthly.index[-1].to_period("M") - prior_peak_idx.to_period("M")).n)
    else:
        peak_age_months = 0.0

    compression_quality = np.mean(
        [
            clamp01((0.60 - range_ratio_24_120) / 0.35),
            clamp01((0.70 - range_ratio_24_60) / 0.35),
        ]
    )
    old_peak_score = np.mean(
        [
            1.0 if had_old_peak else 0.0,
            clamp01((peak_age_months - 60.0) / 120.0),
            clamp01((old_peak_similarity - 0.80) / 0.20) if not np.isnan(old_peak_similarity) else 0.0,
        ]
    )
    anti_trend_penalty = np.mean(
        [
            clamp01((trend_r2_60m - 0.65) / 0.25) if not np.isnan(trend_r2_60m) else 0.0,
            clamp01((slope_high_60m - 0.04) / 0.05) if not np.isnan(slope_high_60m) else 0.0,
            clamp01((range_ratio_24_120 - 0.65) / 0.25),
        ]
    )

    score_long_coil = np.mean(
        [
            clamp01((age_years - 10.0) / 15.0),
            clamp01((pos_in_10y_range - 0.70) / 0.25),
            compression_quality,
            clamp01(low_36m_above_10y_low_pct / 1.50) if not np.isnan(low_36m_above_10y_low_pct) else 0.0,
            old_peak_score,
        ]
    )

    score_tight_resistance = np.mean(
        [
            clamp01((0.15 - dist_to_10y_high_pct) / 0.15) if not np.isnan(dist_to_10y_high_pct) else 0.0,
            clamp01((pos_in_10y_range - 0.75) / 0.20),
            clamp01((0.50 - range_ratio_24_120) / 0.30),
            clamp01((0.70 - range_ratio_24_60) / 0.35),
        ]
    )

    score_ascending_compression = np.mean(
        [
            clamp01((slope_low_60m - 0.005) / 0.05) if not np.isnan(slope_low_60m) else 0.0,
            clamp01((0.03 - abs(slope_high_60m)) / 0.03) if not np.isnan(slope_high_60m) else 0.0,
            clamp01((pos_in_10y_range - 0.65) / 0.25),
            compression_quality,
        ]
    )

    raw_score_total = float(
        0.40 * score_long_coil
        + 0.35 * score_tight_resistance
        + 0.25 * score_ascending_compression
    )
    score_total = clamp01(raw_score_total - 0.35 * anti_trend_penalty)

    return ScreenResult(
        ticker=ticker,
        age_years=age_years,
        last_close=last_close,
        score_total=score_total,
        score_long_coil=float(score_long_coil),
        score_tight_resistance=float(score_tight_resistance),
        score_ascending_compression=float(score_ascending_compression),
        pos_in_10y_range=float(pos_in_10y_range),
        dist_to_10y_high_pct=float(dist_to_10y_high_pct),
        range_ratio_24_120=float(range_ratio_24_120),
        range_ratio_24_60=float(range_ratio_24_60),
        low_36m_above_10y_low_pct=float(low_36m_above_10y_low_pct),
        slope_high_60m=float(slope_high_60m),
        slope_low_60m=float(slope_low_60m),
        trend_r2_60m=float(trend_r2_60m),
        peak_age_months=float(peak_age_months),
        old_peak_similarity=float(old_peak_similarity),
    )


def run_legacy_numeric_screen(tickers: Iterable[str]) -> pd.DataFrame:
    """Original numeric-only first pass, retained as a diagnostic utility."""
    results: List[ScreenResult] = []
    ticker_list = list(tickers)
    histories = fetch_monthly_histories(ticker_list)

    for ticker in ticker_list:
        monthly = histories.get(ticker)
        if monthly is None:
            continue
        result = compute_features(ticker, monthly)
        if result is not None:
            results.append(result)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in results])
    return df.sort_values(["score_total", "score_long_coil"], ascending=False).reset_index(drop=True)


LIFECYCLE_ORDER = {
    "pre_breakout": 0,
    "forming": 1,
    "breaking_out": 2,
    "post_breakout": 3,
    "no_structure": 4,
}
GRADE_ORDER = {"A": 0, "B": 1, "C": 2}


def _screen_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    lifecycle = str(row.get("lifecycle") or "no_structure")
    grade_rank = GRADE_ORDER.get(row.get("grade"), 3)
    score = float(row.get("coil_score") or 0.0)
    # Only the actionable pre-breakout bucket is grade-first. Other lifecycle
    # groups remain contiguous and are ordered by structural score.
    if lifecycle == "pre_breakout":
        return (LIFECYCLE_ORDER[lifecycle], grade_rank, -score, row["ticker"])
    return (LIFECYCLE_ORDER.get(lifecycle, 99), -score, row["ticker"])


def _lifecycle_row(
    ticker: str, payload: dict[str, Any], analysis: dict[str, Any]
) -> dict[str, Any]:
    legacy = dict(payload.get("features") or {})
    bars = payload.get("bars") or []
    # Structural eligibility is intentionally broader than the old 120-month
    # numeric feature gate. Keep the basic quote usable when diagnostics are
    # absent so lifecycle-only rows still render correctly in API consumers.
    if "last_close" not in legacy:
        legacy["last_close"] = bars[-1].get("close") if bars else None
    if "age_years" not in legacy:
        legacy["age_years"] = round(len(bars) / 12.0, 4) if bars else None
    # Keep a stable legacy-compatible JSON shape even when the numeric model
    # cannot run. Those fields are diagnostics, so null is preferable to
    # dropping an otherwise valid structural result.
    for field_name in ScreenResult.__dataclass_fields__:
        legacy.setdefault(field_name, None)
    active_lid = analysis.get("active_lid") or {}
    resistance = analysis.get("resistance") or {}
    metrics = analysis.get("metrics") or {}
    review = analysis.get("review") or {}
    freshness = payload.get("freshness") or {}
    analysis_metadata = analysis.get("analysis_metadata") or {}
    data_quality = analysis_metadata.get("data_quality") or {}
    row = {
        **legacy,
        "ticker": ticker,
        "lifecycle": analysis.get("lifecycle", "no_structure"),
        "status": analysis.get("status", "no_structure"),
        "grade": analysis.get("grade"),
        "lid_grade": active_lid.get("grade") or resistance.get("lid_grade"),
        "coil_score": analysis.get("coil_score", 0.0),
        "lid_slope_pct_per_year": active_lid.get("slope_pct_per_year"),
        "proximity_pct": metrics.get("proximity_pct"),
        # Raw ratio and the v2.2 band enum both ship: the number drives sorting
        # and diagnostics, the enum is the classification consumers act on.
        "current_price_position": metrics.get("current_price_position"),
        "span_years": active_lid.get("span_years"),
        "touches": active_lid.get("touch_count"),
        # Alias retained for consumers that used the analysis object naming.
        "touch_count": active_lid.get("touch_count"),
        "reviewed": bool(review.get("reviewed")),
        "review_status": review.get("decision"),
        "review_id": review.get("review_id"),
        "review_as_of": review.get("review_as_of"),
        "review_algorithm_version": review.get("review_algorithm_version"),
        "review_stale": bool(review.get("stale")),
        "review_effective": review.get("effective", "algorithm"),
        "data_date": freshness.get("last_bar_date") or analysis.get("as_of"),
        "freshness": freshness.get("status", "unknown"),
        "fetched_at": freshness.get("fetched_at"),
        "data_source": freshness.get("source"),
        "adjustment_mode": data_quality.get("adjustment_mode")
        or freshness.get("adjustment_mode"),
        "data_quality_status": data_quality.get("status"),
        "analysis_variant": analysis_metadata.get("variant"),
        "analysis_mode": analysis_metadata.get("mode"),
    }
    return row


def run_lifecycle_screen(
    tickers: Iterable[str],
    *,
    fetch_monthly: Callable[[str], Optional[pd.DataFrame]] = fetch_monthly_history,
    review_override_for: Optional[Callable[[str], Optional[dict[str, Any]]]] = None,
    review_state_for: Optional[Callable[[str], Optional[dict[str, Any]]]] = None,
    force_refresh: bool = False,
    analysis_variant: str = "v2_3_1",
    analysis_mode: str = "effective",
) -> dict[str, Any]:
    """Structurally screen every ticker without a legacy-score gate.

    Numeric first-pass features remain in each row for diagnostics and legacy
    CSV consumers. Classification and ranking use the schema-v2 coil analysis.
    ``review_state_for`` supplies the latest human decision so rows can carry
    review status and staleness alongside the effective analysis.
    """
    from coil_analysis import (
        ALGORITHM_VERSION,
        ANALYSIS_MODE_ALGORITHM_ONLY,
        ANALYSIS_MODE_EFFECTIVE,
        ANALYSIS_VARIANT_V2_3_1,
        ANALYSIS_VARIANT_V2_4_VALIDATION,
        DEFAULT_CONFIG,
        analyze_coil,
    )
    from history_cache import get_history_payload
    from reviews import annotate_review

    symbols = normalize_tickers(tickers)
    if analysis_variant not in {
        ANALYSIS_VARIANT_V2_3_1,
        ANALYSIS_VARIANT_V2_4_VALIDATION,
    }:
        raise ValueError(f"unsupported analysis variant: {analysis_variant}")
    if analysis_mode not in {
        ANALYSIS_MODE_ALGORITHM_ONLY,
        ANALYSIS_MODE_EFFECTIVE,
    }:
        raise ValueError(f"unsupported analysis mode: {analysis_mode}")
    if (
        analysis_variant == ANALYSIS_VARIANT_V2_4_VALIDATION
        and analysis_mode != ANALYSIS_MODE_ALGORITHM_ONLY
    ):
        raise ValueError("v2_4_validation is available only in algorithm_only mode")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for ticker in symbols:
        try:
            payload = get_history_payload(
                ticker,
                fetch_monthly,
                compute_features,
                force_refresh=force_refresh,
            )
            if payload is None:
                failures.append(
                    {"ticker": ticker, "error": "No usable monthly history found."}
                )
                continue
            override = (
                review_override_for(ticker)
                if analysis_mode == ANALYSIS_MODE_EFFECTIVE and review_override_for
                else None
            )
            freshness = payload.get("freshness") or {}
            analysis = analyze_coil(
                payload["bars"],
                ticker=ticker,
                review_override=override,
                variant=analysis_variant,
                mode=analysis_mode,
                adjustment_mode=str(
                    freshness.get("adjustment_mode") or "unknown"
                ),
            )
            if analysis_mode == ANALYSIS_MODE_EFFECTIVE:
                state = review_state_for(ticker) if review_state_for else None
                annotate_review(analysis, state, algorithm_version=ALGORITHM_VERSION)
            quality = (analysis.get("analysis_metadata") or {}).get("data_quality") or {}
            if quality.get("blocked"):
                failures.append(
                    {
                        "ticker": ticker,
                        "error": "Strict OHLC integrity checks failed.",
                        "data_quality": quality,
                    }
                )
                continue
            bar_count = analysis.get("bar_count")
            if isinstance(bar_count, int) and bar_count < DEFAULT_CONFIG.min_bars:
                failures.append(
                    {
                        "ticker": ticker,
                        "error": (
                            f"Insufficient monthly history: {bar_count} bars; "
                            f"requires {DEFAULT_CONFIG.min_bars}."
                        ),
                    }
                )
                continue
            results.append(_lifecycle_row(ticker, payload, analysis))
        except Exception as exc:
            # One provider/analysis failure must not discard the rest of a run.
            failures.append({"ticker": ticker, "error": str(exc) or type(exc).__name__})

    results.sort(key=_screen_sort_key)
    bucket_counts = {name: 0 for name in LIFECYCLE_ORDER}
    for result in results:
        lifecycle = str(result.get("lifecycle") or "no_structure")
        bucket_counts[lifecycle] = bucket_counts.get(lifecycle, 0) + 1
    screened_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
    return {
        "results": results,
        "bucket_counts": bucket_counts,
        "failures": failures,
        "algorithm_version": (
            "2.4.0-validation"
            if analysis_variant == ANALYSIS_VARIANT_V2_4_VALIDATION
            else ALGORITHM_VERSION
        ),
        "analysis_variant": analysis_variant,
        "analysis_mode": analysis_mode,
        "screened_at": screened_at,
    }


def run_screen(
    tickers: Iterable[str],
    *,
    force_refresh: bool = False,
    analysis_variant: str = "v2_3_1",
    analysis_mode: str = "effective",
) -> pd.DataFrame:
    """Compatibility DataFrame wrapper over the lifecycle-aware screener.

    Existing CLI/CSV callers still receive a DataFrame, now enriched with v2
    structure. Envelope metadata is retained in ``DataFrame.attrs``.
    """
    review_override_for = None
    if analysis_mode == "effective":
        from reviews import get_review_store

        store = get_review_store()
        review_override_for = lambda ticker: store.get_override(ticker, "3M")
    run = run_lifecycle_screen(
        tickers,
        review_override_for=review_override_for,
        force_refresh=force_refresh,
        analysis_variant=analysis_variant,
        analysis_mode=analysis_mode,
    )
    df = pd.DataFrame(run["results"])
    df.attrs.update({key: value for key, value in run.items() if key != "results"})
    return df


def load_sp500_tickers() -> List[str]:
    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "coil-screening/0.1"},
        timeout=30,
    )
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]
    symbols = table["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return symbols.tolist()


def build_ticker_list(
    explicit_tickers: Optional[Iterable[str]] = None,
    universe: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[str]:
    normalized_explicit = normalize_tickers(explicit_tickers or [])

    if universe == "sp500":
        universe_tickers = load_sp500_tickers()
        if limit:
            universe_tickers = universe_tickers[:limit]
        return normalize_tickers(universe_tickers + normalized_explicit)

    if universe == "international":
        universe_tickers = INTERNATIONAL_REVIEW_TICKERS
        if limit:
            universe_tickers = universe_tickers[:limit]
        return normalize_tickers(universe_tickers + normalized_explicit)

    if normalized_explicit:
        return normalized_explicit

    return DEFAULT_TICKERS.copy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monthly coil-effect first-pass screener")
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Ticker list to screen. If omitted, a starter watchlist is used.",
    )
    parser.add_argument(
        "--csv",
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--universe",
        choices=["sp500", "international"],
        help="Load a predefined ticker universe.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on the number of universe tickers loaded before adding explicit ticker arguments.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refresh every symbol from yfinance before screening.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tickers = build_ticker_list(
        explicit_tickers=args.tickers,
        universe=args.universe,
        limit=args.limit,
    )

    df = run_screen(tickers, force_refresh=args.force_refresh)
    if df.empty:
        print("No results.")
        return

    columns = [
        "ticker",
        "lifecycle",
        "grade",
        "lid_grade",
        "coil_score",
        "lid_slope_pct_per_year",
        "touches",
        "span_years",
        "reviewed",
        "score_total",
        "score_long_coil",
        "age_years",
        "last_close",
    ]

    printable = df.reindex(columns=columns).copy()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(printable.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved full results to {args.csv}")


if __name__ == "__main__":
    main()
