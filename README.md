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

## Long-history research pilot

The production default remains `yfinance`. Detector research can now use an
explicit provider boundary without changing API generations, review storage,
or the sealed v2.4 benchmark:

- `eodhd` fetches raw daily OHLCV plus the separate split-event history;
- `frozen_csv` imports a dated vendor export for offline/reproducible trials;
- CoilingView applies the same split-only transform, completed-quarter cutoff,
  and strict OHLC integrity checks to either input.

Every non-US EODHD symbol must be mapped explicitly; exchange suffixes are not
guessed. The API token is read from the environment and is never persisted.

Build the example development corpus from EODHD:

```bash
export COILINGVIEW_EODHD_API_TOKEN='replace-with-a-local-secret'
python3 scripts/build_long_history_pilot.py \
  --provider eodhd \
  --spec examples/long_history_pilot.example.json \
  --output /private/tmp/coilingview-long-history-pilot \
  --as-of 2026-06-30
```

For a provider-delivered export, create one `<TICKER>.csv` per symbol with the
columns `Date,Open,High,Low,Close,Volume,Stock Splits`, then run:

```bash
python3 scripts/build_long_history_pilot.py \
  --provider frozen_csv \
  --frozen-root /path/to/frozen-daily-files \
  --spec examples/long_history_pilot.example.json \
  --output /private/tmp/coilingview-long-history-pilot \
  --as-of 2026-06-30
```

Add a trustworthy `listed_since` date to every spec entry before using the
manifest as a coverage audit. A start within 366 days is recorded as plausible
inception coverage; a later start is explicitly `truncated`. The output is an
unlabeled development corpus and intentionally contains no benchmark holdout
labels.

## Planned Outputs

- reproducible local environment
- monthly screener script
- candidate ranking output
- chart rendering pipeline for review
- follow-up design for visual pattern detection

## Current Project Files

- `screen_monthly.py`: first-pass monthly screener
- `coil_analysis.py`: deterministic major-top / resistance-slope / coil grading
- `first_pass_results.csv`: initial live output from the starter ticker list
- `app.py`: FastAPI backend for the screener and chart endpoints
- `static/`: plain HTML/CSS/JS review UI

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

Build a fresh, cross-market feedback batch (Canada, Europe, Japan, Hong Kong,
India, Australia, Brazil, South Korea, Taiwan, and South Africa):

`python screen_monthly.py --universe international --force-refresh --csv international_review_results.csv`

The generated CSV appears under Saved runs in the frontend. Load it and choose
**Review results** to approve major tops or save corrected tops. A new
detector `ALGORITHM_VERSION` automatically returns older decisions to pending.
Use **Download feedback (.md)** at any point to export the complete session as
a human-readable summary plus a canonical versioned JSON corpus. The raw API
form is available at `GET /api/review-sessions/{session_id}/export`.

The schema-v4 review form records two independent judgments:

- the major-top verdict (`approved` or `corrected`); and
- the chart classification (`coil`, `not_coil`, or `uncertain`) with an
  independent human A/B/C grade, confidence, and rationale.

Keeping these labels separate means a reviewer can say that the major tops are
correct while the chart is not a coil. New events also carry a stable UUID,
label-policy version, and reviewed-bar content hash for cross-export
deduplication and reproducibility. Schema-v3 events remain readable.

Review events are append-only. They are intentionally accumulated across
sessions for batch analysis, calibration, and future training rather than
triggering a one-off model update per click.

Combine one or more `.md`/`.json` exports into a deduplicated candidate corpus:

`python -m review_corpus build feedback-1.md feedback-2.md -o candidate-corpus.json`

Print coverage, verdict, label, confidence, and grade-agreement counts:

`python -m review_corpus report feedback-1.md feedback-2.md`

The materializer preserves every revision and quarantines legacy,
low-confidence, uncertain, or non-reproducible samples. It never edits
`major_high_feedback.json` or promotes labels into detector goldens
automatically; curator review and a frozen evaluation split remain separate
offline steps.

Run the review UI:

`source .venv/bin/activate`

`python -m uvicorn app:app --host 127.0.0.1 --port 8010`

Locally, feedback is stored in `reviews.db`. In production, set `DATABASE_URL`
for PostgreSQL or attach a Railway volume; when `RAILWAY_VOLUME_MOUNT_PATH` is
present the SQLite database is stored at `<mount>/reviews.db`. An explicit
`REVIEW_DB_PATH` takes precedence.

## Deterministic Structure Analysis

`coil_analysis.py` is the precision layer between the numeric screen and human
review: it detects major tops on the monthly series, fits the resistance line
through them, measures its slope, and grades the coil. It addresses the
calibration gaps above (explicit shrinking-amplitude measure, explicit
"compressing under a boundary" vs "near highs" distinction).

What qualifies as a coil (the working definition):

1. Structure: >= 2 major tops (high-prominence swing highs) on one straight
   resistance line within tolerance.
2. Lid slope: flat-to-gently-rising, normalized to %/yr of the line value at
   the last bar. Bands: A < 2, B < 6, C < 12 (and gently falling to -3).
3. Sealed: no monthly close escaped above the line before the recent bars.
4. Wound: pullbacks below the lid shrink, or price is pressed at the lid
   (>= 90% of the line).
5. Loaded: last close >= 70% of the line, and the base is not a smooth
   exponential trend (log-close R^2 gate rejects mega-cap uptrend envelopes).

Output per ticker: `status` (`no_structure` / `basing` / `coiling` /
`breaking_out` / `broken_out`), `grade` (A/B/C once every gate passes),
`resistance.lid_grade` (slope band alone — how a chart gets hand-graded even
mid-base), the anchor/touch points as `{idx, date, price}`, the support line,
pullback depths, and human-readable notes. Anchors use the same shape as the
vision pipeline's mapped highs, so the frontend trendline primitive can render
either source.

Run from the CLI (cache-first, like the API):

`python coil_analysis.py KN UEC COF`

`python coil_analysis.py --saved-run demo_curated_coils_results.csv`

Replay a chart as it looked before its breakout (backtest/tuning mode):

`python coil_analysis.py KN --as-of 2025-09-30`

Or over HTTP: `GET /api/coil/{ticker}?as_of=YYYY-MM-DD`.

Validation against Amrut's graded reference charts (replayed pre-breakout):

| ticker | hand grade | as_of      | result                            |
|--------|-----------|------------|-----------------------------------|
| KN     | A         | 2025-09-30 | coiling A, lid +0.5%/yr, 3 touches |
| UEC    | A         | 2024-06-30 | coiling A, lid +0.8%/yr           |
| COF    | B         | 2024-06-30 | coiling B, lid +2.6%/yr           |
| EAT    | B         | 2024-09-30 | coiling B, lid +2.5%/yr, 3 touches |
| CF     | B         | 2026-02-28 | basing, lid_grade B (+4.3%/yr) — never pressed the lid, gapped out from depth |

Controls: MSFT/AAPL (secular uptrends) never receive a grade. Thresholds live
in `CoilConfig` (`coil_analysis.py`); tune only against flat, not-yet-broken
coiling charts — breakouts are filtered upstream by the screener.

## Vision Tagging

The vision pipeline captures the React chart in frontend capture mode, sends the
image to Roboflow hosted inference, maps detections back to chart `{idx, date,
price}` anchors, and writes reviewable suggestion artifacts to disk. Human
review remains the source of truth.

Required environment variables:

`ROBOFLOW_API_KEY`

`ROBOFLOW_MODEL_ID` such as `your-project/1`

Install the browser once:

`python -m playwright install chromium`

Run a single ticker while the frontend dev server is available:

`python -m vision.run --base-url http://127.0.0.1:5173 --ticker AAPL`

Seed Roboflow with chart captures from an existing screened stock list:

`python -m vision.seed_dataset --base-url http://127.0.0.1:5173 --saved-run demo_curated_coils_results.csv --project-id coiling-view --upload`

Check whether Roboflow has uploaded images, annotations, and a trained version:

`ROBOFLOW_WORKSPACE=your-workspace python -m vision.roboflow_status --model-id coiling-view/1`

Artifacts are written under `vision_runs/<run_id>/`:

- `images/` chart captures
- `raw/` Roboflow responses
- `mapped/` AI point and trendline suggestions
- `debug/` annotated captures
- `predictions/` API-facing prediction JSON
- `manifest.json` run-level artifact index

Dataset seed artifacts are written under `vision_dataset_uploads/<run_id>/`.
Those captures are the source images for Roboflow annotation/training; after a
Roboflow version is trained, set `ROBOFLOW_MODEL_ID` to that version and run the
vision pipeline again.
