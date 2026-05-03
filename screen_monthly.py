from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import StringIO
from typing import Iterable, List, Optional

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


def fetch_monthly_history(ticker: str) -> Optional[pd.DataFrame]:
    data = yf.download(
        ticker,
        period="max",
        interval="1mo",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if data is None or data.empty:
        return None
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    if data.empty:
        return None
    return data


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
            interval="1mo",
            auto_adjust=False,
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
                ticker_df = data[ticker].dropna(subset=["Open", "High", "Low", "Close"])
                if not ticker_df.empty:
                    histories[ticker] = ticker_df
        else:
            ticker_df = data.dropna(subset=["Open", "High", "Low", "Close"])
            if not ticker_df.empty and len(batch) == 1:
                histories[batch[0]] = ticker_df

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


def run_screen(tickers: Iterable[str]) -> pd.DataFrame:
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
        choices=["sp500"],
        help="Load a predefined ticker universe.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on the number of universe tickers loaded before adding explicit ticker arguments.",
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

    df = run_screen(tickers)
    if df.empty:
        print("No results.")
        return

    columns = [
        "ticker",
        "score_total",
        "score_long_coil",
        "score_tight_resistance",
        "score_ascending_compression",
        "age_years",
        "last_close",
        "pos_in_10y_range",
        "dist_to_10y_high_pct",
        "range_ratio_24_120",
        "low_36m_above_10y_low_pct",
    ]

    printable = df[columns].copy()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(printable.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved full results to {args.csv}")


if __name__ == "__main__":
    main()
