from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
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
class ScreenConfig:
    min_history_months: int = 120
    long_range_months: int = 120
    mid_range_months: int = 60
    recent_range_months: int = 24
    support_low_months: int = 36
    rolling_window_months: int = 12

    weight_long_coil: float = 0.40
    weight_tight_resistance: float = 0.35
    weight_ascending_compression: float = 0.25
    anti_trend_penalty_weight: float = 0.35

    compression_recent_long_target: float = 0.60
    compression_recent_mid_target: float = 0.70
    compression_tolerance: float = 0.35
    tight_resistance_distance_pct: float = 0.15
    old_peak_min_similarity: float = 0.80
    old_peak_exclusion_months: int = 24
    old_peak_min_age_months: float = 60.0
    old_peak_full_age_span_months: float = 120.0
    low_support_scale_pct: float = 1.50

    long_age_min_years: float = 10.0
    long_age_full_span_years: float = 15.0
    long_position_start: float = 0.70
    long_position_span: float = 0.25
    tight_position_start: float = 0.75
    tight_position_span: float = 0.20
    tight_range_long_target: float = 0.50
    tight_range_long_tolerance: float = 0.30
    ascending_position_start: float = 0.65
    ascending_position_span: float = 0.25

    ascending_low_slope_start: float = 0.005
    ascending_low_slope_span: float = 0.05
    ascending_high_slope_abs_max: float = 0.03
    trend_r2_penalty_start: float = 0.65
    trend_r2_penalty_span: float = 0.25
    high_slope_penalty_start: float = 0.04
    high_slope_penalty_span: float = 0.05
    wide_range_penalty_start: float = 0.65
    wide_range_penalty_span: float = 0.25


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
    range_ratio_recent_long: float
    range_ratio_recent_mid: float
    support_low_above_long_low_pct: float
    slope_high_mid: float
    slope_low_mid: float
    trend_r2_mid: float


def default_config_dict() -> dict[str, float | int]:
    return asdict(ScreenConfig())


def build_config(overrides: Optional[dict[str, float | int]] = None) -> ScreenConfig:
    config_values = default_config_dict()
    if overrides:
        allowed = {field.name for field in fields(ScreenConfig)}
        for key, value in overrides.items():
            if key not in allowed or value is None:
                continue
            if isinstance(config_values[key], int):
                config_values[key] = max(1, int(value))
            else:
                config_values[key] = float(value)

    positive_keys = [
        "compression_tolerance",
        "tight_resistance_distance_pct",
        "old_peak_full_age_span_months",
        "low_support_scale_pct",
        "long_age_full_span_years",
        "long_position_span",
        "tight_position_span",
        "tight_range_long_tolerance",
        "ascending_position_span",
        "ascending_low_slope_span",
        "ascending_high_slope_abs_max",
        "trend_r2_penalty_span",
        "high_slope_penalty_span",
        "wide_range_penalty_span",
    ]
    for key in positive_keys:
        config_values[key] = max(0.001, float(config_values[key]))

    for key in [
        "weight_long_coil",
        "weight_tight_resistance",
        "weight_ascending_compression",
        "anti_trend_penalty_weight",
    ]:
        config_values[key] = max(0.0, float(config_values[key]))

    return ScreenConfig(**config_values)


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


def weighted_mean(parts: list[tuple[float, float]]) -> float:
    total_weight = sum(max(0.0, float(weight)) for _, weight in parts)
    if total_weight == 0:
        return 0.0
    return float(
        sum(float(value) * max(0.0, float(weight)) for value, weight in parts)
        / total_weight
    )


def compute_features(
    ticker: str,
    monthly: pd.DataFrame,
    config: Optional[ScreenConfig] = None,
) -> Optional[ScreenResult]:
    config = config or ScreenConfig()
    monthly = monthly.copy()
    monthly = monthly.sort_index()

    required_months = max(
        config.min_history_months,
        config.long_range_months,
        config.mid_range_months,
        config.recent_range_months,
        config.support_low_months,
    )
    if len(monthly) < required_months:
        return None

    close = monthly["Close"].astype(float)
    high = monthly["High"].astype(float)
    low = monthly["Low"].astype(float)

    last_close = float(close.iloc[-1])
    age_years = len(monthly) / 12.0

    high_long = float(high.iloc[-config.long_range_months:].max())
    low_long = float(low.iloc[-config.long_range_months:].min())
    high_mid = float(high.iloc[-config.mid_range_months:].max())
    low_mid = float(low.iloc[-config.mid_range_months:].min())
    high_recent = float(high.iloc[-config.recent_range_months:].max())
    low_recent = float(low.iloc[-config.recent_range_months:].min())
    low_support = float(low.iloc[-config.support_low_months:].min())

    range_long = high_long - low_long
    range_mid = high_mid - low_mid
    range_recent = high_recent - low_recent

    if range_long <= 0 or range_mid <= 0:
        return None

    pos_in_long_range = (last_close - low_long) / range_long
    dist_to_long_high_pct = (high_long - last_close) / high_long if high_long > 0 else np.nan
    range_ratio_recent_long = range_recent / range_long
    range_ratio_recent_mid = range_recent / range_mid
    support_low_above_long_low_pct = (low_support - low_long) / low_long if low_long > 0 else np.nan

    rolling_high = high.rolling(config.rolling_window_months).max().iloc[-config.mid_range_months:]
    rolling_low = low.rolling(config.rolling_window_months).min().iloc[-config.mid_range_months:]
    slope_high_mid = fit_slope(normalize_series(rolling_high))
    slope_low_mid = fit_slope(normalize_series(rolling_low))
    trend_r2_mid = fit_trend_r2(normalize_series(close.iloc[-config.mid_range_months:]))

    older_high_window = high.iloc[:-config.old_peak_exclusion_months]
    older_high = float(older_high_window.max()) if not older_high_window.empty else high_long
    old_peak_similarity = older_high / high_long if high_long > 0 else np.nan
    had_old_peak = (
        old_peak_similarity >= config.old_peak_min_similarity
        if not np.isnan(old_peak_similarity)
        else False
    )
    if had_old_peak:
        prior_peak_idx = older_high_window.idxmax()
        peak_age_months = float((monthly.index[-1].to_period("M") - prior_peak_idx.to_period("M")).n)
    else:
        peak_age_months = 0.0

    compression_quality = np.mean(
        [
            clamp01(
                (config.compression_recent_long_target - range_ratio_recent_long)
                / config.compression_tolerance
            ),
            clamp01(
                (config.compression_recent_mid_target - range_ratio_recent_mid)
                / config.compression_tolerance
            ),
        ]
    )
    old_peak_score = np.mean(
        [
            1.0 if had_old_peak else 0.0,
            clamp01(
                (peak_age_months - config.old_peak_min_age_months)
                / config.old_peak_full_age_span_months
            ),
            clamp01(
                (old_peak_similarity - config.old_peak_min_similarity)
                / max(0.001, 1.0 - config.old_peak_min_similarity)
            )
            if not np.isnan(old_peak_similarity)
            else 0.0,
        ]
    )
    anti_trend_penalty = np.mean(
        [
            clamp01((trend_r2_mid - config.trend_r2_penalty_start) / config.trend_r2_penalty_span)
            if not np.isnan(trend_r2_mid)
            else 0.0,
            clamp01((slope_high_mid - config.high_slope_penalty_start) / config.high_slope_penalty_span)
            if not np.isnan(slope_high_mid)
            else 0.0,
            clamp01((range_ratio_recent_long - config.wide_range_penalty_start) / config.wide_range_penalty_span),
        ]
    )

    score_long_coil = np.mean(
        [
            clamp01((age_years - config.long_age_min_years) / config.long_age_full_span_years),
            clamp01((pos_in_long_range - config.long_position_start) / config.long_position_span),
            compression_quality,
            clamp01(support_low_above_long_low_pct / config.low_support_scale_pct)
            if not np.isnan(support_low_above_long_low_pct)
            else 0.0,
            old_peak_score,
        ]
    )

    score_tight_resistance = np.mean(
        [
            clamp01((config.tight_resistance_distance_pct - dist_to_long_high_pct) / config.tight_resistance_distance_pct)
            if not np.isnan(dist_to_long_high_pct)
            else 0.0,
            clamp01((pos_in_long_range - config.tight_position_start) / config.tight_position_span),
            clamp01((config.tight_range_long_target - range_ratio_recent_long) / config.tight_range_long_tolerance),
            clamp01(
                (config.compression_recent_mid_target - range_ratio_recent_mid)
                / config.compression_tolerance
            ),
        ]
    )

    score_ascending_compression = np.mean(
        [
            clamp01((slope_low_mid - config.ascending_low_slope_start) / config.ascending_low_slope_span)
            if not np.isnan(slope_low_mid)
            else 0.0,
            clamp01((config.ascending_high_slope_abs_max - abs(slope_high_mid)) / config.ascending_high_slope_abs_max)
            if not np.isnan(slope_high_mid)
            else 0.0,
            clamp01((pos_in_long_range - config.ascending_position_start) / config.ascending_position_span),
            compression_quality,
        ]
    )

    raw_score_total = weighted_mean(
        [
            (score_long_coil, config.weight_long_coil),
            (score_tight_resistance, config.weight_tight_resistance),
            (score_ascending_compression, config.weight_ascending_compression),
        ]
    )
    score_total = clamp01(raw_score_total - config.anti_trend_penalty_weight * anti_trend_penalty)

    return ScreenResult(
        ticker=ticker,
        age_years=age_years,
        last_close=last_close,
        score_total=score_total,
        score_long_coil=float(score_long_coil),
        score_tight_resistance=float(score_tight_resistance),
        score_ascending_compression=float(score_ascending_compression),
        pos_in_10y_range=float(pos_in_long_range),
        dist_to_10y_high_pct=float(dist_to_long_high_pct),
        range_ratio_24_120=float(range_ratio_recent_long),
        range_ratio_24_60=float(range_ratio_recent_mid),
        low_36m_above_10y_low_pct=float(support_low_above_long_low_pct),
        slope_high_60m=float(slope_high_mid),
        slope_low_60m=float(slope_low_mid),
        trend_r2_60m=float(trend_r2_mid),
        peak_age_months=float(peak_age_months),
        old_peak_similarity=float(old_peak_similarity),
        range_ratio_recent_long=float(range_ratio_recent_long),
        range_ratio_recent_mid=float(range_ratio_recent_mid),
        support_low_above_long_low_pct=float(support_low_above_long_low_pct),
        slope_high_mid=float(slope_high_mid),
        slope_low_mid=float(slope_low_mid),
        trend_r2_mid=float(trend_r2_mid),
    )


def run_screen(tickers: Iterable[str], config: Optional[ScreenConfig] = None) -> pd.DataFrame:
    config = config or ScreenConfig()
    results: List[ScreenResult] = []
    ticker_list = list(tickers)
    histories = fetch_monthly_histories(ticker_list)

    for ticker in ticker_list:
        monthly = histories.get(ticker)
        if monthly is None:
            continue
        result = compute_features(ticker, monthly, config=config)
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
    parser.add_argument(
        "--config-json",
        help="Optional JSON object of ScreenConfig overrides.",
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

    config = build_config(json.loads(args.config_json) if args.config_json else None)
    df = run_screen(tickers, config=config)
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
