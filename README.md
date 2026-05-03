# Coil Screening

See also: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) and [AGENTS.md](AGENTS.md).

## Goal

Build a two-stage workflow for finding stocks that show long-duration "coil effect" structures on monthly charts.

## Current Understanding

The workflow has two major stages:

1. Numeric screening on monthly OHLCV data to reduce the universe to names that are likely to show long-term compression.
2. Visual analysis on standardized monthly chart images to decide whether the chart geometry actually looks like a coil.

The screen should optimize for recall, not perfection. It should produce a manageable candidate set that is richer in potential coil structures than the full market universe.

The visual layer should optimize for geometric understanding:

- multi-year compression
- repeated interaction with a long-term ceiling or boundary
- shrinking swing amplitude
- price tightening near the right edge
- breakout pressure or confirmed breakout behavior

## Immediate Plan

1. Set up a local Python environment for market data access and screening.
2. Install the minimum packages needed to fetch historical price data and work with monthly bars.
3. Implement a first-pass screener with a few candidate definitions:
   - long coil candidate
   - tight near resistance
   - ascending compression / pre-breakout
4. Test the screener on a small ticker list first.
5. Expand to a larger universe once the feature logic looks sane.
6. Add standardized chart rendering for the highest-ranked names.
7. Design the visual-labeling and model pipeline after the numeric filter is producing useful candidates.

## Initial Technical Direction

The practical first build is:

- Python virtual environment
- `pandas` for data handling
- `yfinance` for an initial low-friction data source
- optional `openbb` later if we want a broader provider layer or MCP integration

We are intentionally starting with a lightweight setup so we can get the monthly screener working before adding a heavier finance stack.

## Planned Outputs

- reproducible local environment
- monthly screener script
- candidate ranking output
- chart rendering pipeline for review
- follow-up design for visual pattern detection

## Current Project Files

- `screen_monthly.py`: first-pass monthly screener
- `first_pass_results.csv`: initial live output from the starter ticker list
- `app.py`: FastAPI backend for the screener and chart endpoints
- `static/`: plain HTML/CSS/JS review UI

## Configurable Screening

The scoring criteria now flow through `ScreenConfig` in `screen_monthly.py`.

The default config preserves the original first-pass behavior, while the API and UI can override key controls such as:

- score weights
- anti-trend penalty strength
- recent / mid / long range windows
- compression targets
- near-high distance
- old-peak similarity
- minimum history length

API endpoint:

`GET /api/default-config`

CLI override example:

`python screen_monthly.py AER AVT CAT --config-json '{"weight_long_coil": 0.7, "anti_trend_penalty_weight": 0.5}'`

## Persisted Runs

Live API/UI screens are saved under `storage/runs/<run_id>/` by default.

Each run writes:

- `metadata.json`: timestamp, request, resolved tickers, config, git commit, counts, failures
- `results.csv`: ranked result rows for that exact run

Runtime storage is ignored by Git. Configure it with environment variables:

- `COIL_STORAGE_DIR`: root for runtime storage; defaults to `./storage`
- `COIL_RUNS_DIR`: optional override for run artifacts
- `COIL_CACHE_DIR`: optional override for market-data cache artifacts

Examples:

```bash
COIL_STORAGE_DIR=/app/storage python -m uvicorn app:app --host 0.0.0.0 --port 8010
```

```bash
COIL_STORAGE_DIR=/var/lib/coil-screener python -m uvicorn app:app
```

Use:

- `GET /api/runs` to list persisted live runs
- `GET /api/runs/{run_id}` to load metadata and results
- `GET /api/storage` to inspect active storage paths

## Environment

Create a local virtual environment:

`python3 -m venv .venv`

Activate it:

`source .venv/bin/activate`

Install dependencies:

`pip install -r requirements.txt`

## First Run Notes

The first live run completed successfully and ranked a starter list that included the example tickers plus a few controls.

Initial observation:

- the screener is finding some of the example names (`AER`, `UTHR`, `AVT`, `STLD`)
- it is also over-scoring straightforward strong trend names (`CAT`, `AAPL`)

This means the first version is useful for narrowing the universe, but it still needs calibration to separate:

- true long-duration compression
- strong secular uptrends that happen to be near highs

## Benchmark Notes

We ran a mixed benchmark using:

- the first 50 S&P 500 names from the live universe source
- plus Amrut's example tickers

Output file:

- `sp50_plus_amrut_results.csv`

Amrut example ranks in that mixed benchmark:

- `AVT`: 7
- `TEX`: 13
- `UTHR`: 14
- `AER`: 17
- `STLD`: 19
- `EWY`: 31
- `DD`: 32
- `PTCT`: 35
- `PPC`: 50
- `BDC`: 51
- `LAZ`: 56

Interpretation:

- some example names are now surfacing in the upper part of the ranked set
- several others are still too low, so the screen is directionally useful but not yet aligned enough to act as a trustworthy first filter

## Next Calibration Step

The next version should penalize charts that are simply trending up without enough evidence of multi-year compression. Likely adjustments:

- stronger penalty for very wide recent ranges
- stronger requirement for an older major high / prior stall
- explicit measure of shrinking swing amplitude over multiple windows
- explicit distinction between "near highs" and "compressing under a boundary"

## How To Run

Activate the environment:

`source .venv/bin/activate`

Run the starter screen:

`python screen_monthly.py --csv first_pass_results.csv`

Run custom tickers:

`python screen_monthly.py AER EWY STLD PTCT BDC`

Run the review UI:

`source .venv/bin/activate`

`python -m uvicorn app:app --host 127.0.0.1 --port 8010`
