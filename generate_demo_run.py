"""Generate the curated demo saved-run CSV.

The frontend auto-loads the newest ``*.csv`` (by mtime) on boot via
``/api/saved-runs``. For a credible demo we want the picks + chart patterns to
show genuinely *coiling* names: long monthly compression, >=2 comparable major
highs pressed under a flat resistance ceiling, NOT runaway/trending names
(which the screener's ``anti_trend_penalty`` is designed to push down).

CURATED_UNIVERSE below is hand-picked from the broader S&P-500 screen, biased
toward FLAT coils: small/near-zero ``slope_high_60m``, moderate (not extreme)
``trend_r2_60m``, tight ``range_ratio_24_120``, price near the 10y high. These
make the demo charts read as classic sideways coils rather than parabolic runs.

Run:  .venv/bin/python generate_demo_run.py
Output: demo_curated_coils_results.csv  (sorted by score_total desc)
"""
from __future__ import annotations

from pathlib import Path

from screen_monthly import normalize_tickers, run_screen

# Flat, not-yet-broken-out coilers (>=2 comparable major highs under a ceiling),
# selected for low high-slope / contained trend so charts look like real coils.
CURATED_UNIVERSE = [
    "UNP",   # flat rail, tight range, pressed under multi-year ceiling
    "REG",   # sideways REIT, low slope, repeated highs
    "CSX",   # flat rail, contained trend
    "NSC",   # flat rail, near-zero high slope
    "BG",    # multi-year sideways, very low trend_r2
    "LH",    # flat, contained, near 10y high
    "MSCI",  # broad consolidation under prior peak
    "SPG",   # range-bound REIT under resistance
    "VTR",   # sideways healthcare REIT
    "FCX",   # cyclical coil under multi-year resistance
    "NUE",   # steel name basing near highs
]

OUTPUT = Path(__file__).resolve().parent / "demo_curated_coils_results.csv"


def main() -> None:
    tickers = normalize_tickers(CURATED_UNIVERSE)
    df = run_screen(tickers)
    if df.empty:
        raise SystemExit("No results produced - is yfinance reachable?")
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT.name}")
    print(df[["ticker", "score_total", "pos_in_10y_range", "range_ratio_24_120"]].to_string(index=False))


if __name__ == "__main__":
    main()
