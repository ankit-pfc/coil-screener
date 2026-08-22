# Detector-only candidate engine

This package evaluates the automatic coil detector against whole-pattern gold
labels without allowing human coordinates or review overrides into prediction.
It is research infrastructure: it produces reproducible benchmark evidence, not
trading advice or a production override.

## Trust boundary

Each benchmark episode contains two independent artifacts:

1. a human capture made from frozen candles; and
2. an automatic prediction made from OHLCV through the same historical cutoff.

`gold_labels.py` owns the capture and materialized-label contracts.
`automatic_exemplar_evaluator.py` owns the detector-only runner and report
contract. The runner strips every non-OHLCV field, physically truncates future
bars, and makes the analyzer call with `review_override=None`.

The runner fails closed if:

- the materialized label cannot be reproduced from its embedded capture and
  frozen bars;
- source, capture, bar, or label hashes do not match;
- the detector returns human-effective provenance or a `human_review` boundary;
- the detector result does not identify the exact cutoff and bar count; or
- repeated cutoffs from one setup cross evaluation splits.

## Point-in-time dates

`cutoffDate` and `decisionAsOf` are deliberately distinct:

- `cutoffDate` is the date key of the last frozen monthly bar. A provider may
  label a completed month at month-start, for example `2019-12-01`.
- `decisionAsOf` is the historical calendar date on which that candle could be
  used. It cannot precede the represented month end.

The runner truncates physical data at `cutoffDate` and passes `decisionAsOf` to
the detector for calendar-completeness checks. Both values are part of the
decision boundary.

## Gold-label contract

The capture stores candle clicks and semantic roles, never client-computed
prices, slopes, or recognition dates. Materialization snaps clicks to completed
quarterly candles and derives geometry deterministically.

The schema records independently:

- shape, maturity, lifecycle, readiness, action, and confidence;
- primary/alternate and parent/child structures;
- construction anchors, recognition confirmation, supporting touches, and
  deliberately excluded highs;
- major bottoms, undercuts, and outliers;
- base, congestion, compression, breakout, retest, and post-breakout phases;
- breakout, failed-breakout, retest, continuation, and invalidation events; and
- immutable source-evidence hashes and top-review completeness.

Materialization rejects invalid OHLC containment, duplicate months, incomplete
interior/cutoff quarters, impossible event chronology, missing references,
cyclic structure relationships, conflicting clicks, invalid resistance bands,
and any mismatch with a stored canonical hash.

Quarterly volume is available only when all three distinct monthly volumes are
present. A partial sum is never treated as quarterly volume.

## Corpus envelope

```json
{
  "schema_version": 1,
  "kind": "coilingview.detector-benchmark-corpus",
  "corpus_id": "example-development-wave-1",
  "code_sha": "FULL_PUSHED_GIT_SHA",
  "episodes": [
    {
      "split": "development",
      "sampling_stratum": "predicted_positive",
      "ticker": "EXAMPLE.US",
      "eodhd_snapshot": "snapshots/EXAMPLE.US.json",
      "gold_capture": {}
    }
  ]
}
```

An episode may use a previously materialized `gold_label` instead of
`gold_capture`. It must retain the normalized embedded capture so the evaluator
can reproduce the label. `monthly_bars` may still be embedded directly, and
frozen review snapshots may be referenced with the existing `source` and
`ticker` fields. Relative EODHD snapshot paths resolve from the corpus file.

All cutoffs from one `setup_id` must remain in one split.

## Frozen EODHD ingestion

`eodhd_ingestion.py` calls EODHD's historical EOD endpoint with `fmt=json`,
`period=m`, and `order=a`. The provider documents the returned OHLC as raw and
unadjusted, `adjusted_close` as split/dividend adjusted, and volume as
split-adjusted. The adapter preserves the full provider rows but passes only
raw OHLCV to the detector, matching the existing unadjusted detector input.

The token is read from `EODHD_API_TOKEN` and is never written to a snapshot.
Before any request, the CLI requires a clean Git worktree and verifies that
HEAD is contained by an `origin/*` tracking ref. The full SHA is sealed into
every snapshot.

Run ingestion from the exact pushed candidate-engine checkout and write output
outside the repository so the checkout stays clean:

```bash
export EODHD_API_TOKEN='...'
.venv/bin/python eodhd_ingestion.py \
  --symbol AAPL.US \
  --symbol MSFT.US \
  --from 1990-01-01 \
  --to 2026-06-30 \
  --output-dir /absolute/path/to/frozen-eodhd-snapshots
```

Each snapshot seals:

- the exact pushed Git SHA and fetch timestamp;
- token-free endpoint parameters and provider adjustment semantics;
- canonical provider rows and their SHA-256;
- normalized detector OHLCV and its SHA-256; and
- a SHA-256 over the complete snapshot envelope.

The adapter fails on malformed dates, unordered or duplicate months, bars
outside the requested range, invalid/non-finite OHLCV, OHLC containment errors,
negative volume, token-bearing provenance, or any derived/hash mismatch. It
also refuses to overwrite an existing snapshot.

An EODHD-backed corpus must include `code_sha`. The evaluator requires every
snapshot SHA boundary to equal that corpus SHA, and its CLI requires the corpus
SHA to equal the clean pushed checkout currently executing the detector.

## Metrics and release gates

The evaluator reports per-episode and aggregate geometry, classification,
lifecycle, action, abstention, false-action, and estimated intervention
metrics. General top detection is supported only when the detector emits an
explicit `candidate_tops` collection. Selected lid points are a membership
proxy and cannot be relabeled as general top detection.

Confidence intervals for F1 and macro-F1 use a setup-cluster percentile
bootstrap. Every historical checkpoint from a sampled setup stays together.
Wilson intervals remain limited to genuinely binomial proportions.

Estimated correction is detector-versus-gold disagreement. It is not observed
human effort. An observed correction rate requires a separate post-prediction
human review event.

## Running the evaluator

From the repository root:

```bash
.venv/bin/python automatic_exemplar_evaluator.py benchmark.json \
  --output benchmark-report.json \
  --candle-tolerance 1 \
  --line-tolerance-pct 5 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 0
```

The report records the algorithm version, default-config SHA-256, corpus
SHA-256, exact code SHA, every EODHD source/snapshot SHA-256, report SHA-256,
bootstrap method, seed, sample count, and setup count.

## Strict blind eligibility

A report is eligible to be described as the final blind benchmark only when:

- every episode is `blind_benchmark`, outcome-hidden, and top-review complete;
- at least 400 distinct real setup clusters are present;
- no setup, issuer episode, or alternate cutoff crosses splits;
- sampling strata and protocol metadata were frozen before reveal;
- every source and derived hash validates; and
- detector thresholds, rules, prompts, and selection policy remain untouched
  after the benchmark cohort is revealed.

The required setup-level composition is 30% frozen-model predicted positives,
30% near-boundary negatives, 20% hard traps, and 20% random population. The
runner recomputes this composition from episode strata.

The blind protocol also attests that identity, detector output, and reference
answers were hidden; sessions were finalized before reveal; future leakage was
audited; at least 25% received independent double review; and at least 10%
received repeat review after a washout period.

Development examples and outcome-visible captures cannot be presented as
holdout validation, regardless of their metric values.

## Authoritative-run SHA policy

An authoritative data run must start from a clean checkout of the exact pushed
candidate-engine commit. Record the full Git SHA beside the input snapshot and
the generated report. If detector code, default configuration, label
materialization, evaluator logic, gate rules, or ingestion normalization changes
afterward, push the change and rerun from the new SHA. Do not reuse a report
hash from an earlier algorithm tree.
