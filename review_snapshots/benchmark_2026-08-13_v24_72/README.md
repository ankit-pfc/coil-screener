# Lean v2.4 72-sample blind benchmark

This directory is the immutable canonical corpus for the fail-closed benchmark
evaluator. It deliberately uses a non-workbench manifest kind. Reviewer tasks
are materialized into the separate opaque batch A–D saved-run directories.

The 72 unique samples are balanced across India, the United States, and global
ex-U.S. listings, with 36 development, 18 validation, and 18 untouched holdout
samples. Twelve development samples are queued a second time, producing 84
review tasks. The reviewer-facing queue omits repeat relationships. Cohort names
are sampling intentions, not labels; blind review must confirm them.

All histories use raw Yahoo daily OHLC plus reported split events, adjusted into
latest share units and then aggregated to months. Every analysis cutoff is a
completed quarter. The manifest freezes the bars, source fingerprints, quality
disposition, ticker family, split, cohort, and `as_of` date.

Holdout labels are deliberately absent. `benchmark_v24.load_labels()` rejects a
holdout artifact until the configuration-freeze file matches the corpus,
protocol, registered algorithm-only detector versions, and v2.4 configuration.
An audit with missing blind labels must report `inconclusive`; it may never infer
truth from either detector or promote v2.4 on sampling strata alone.

The development sweep is limited to the 54 registered combinations in
`protocol.json`. `scripts/select_v24_configuration.py` refuses to run until all
36 development labels, all 12 repeats, and all 18 validation labels are
present; it applies the validation partition exactly once by creating its
selection report exclusively. The counterbalanced timing order is coordinator
metadata and is omitted from the reviewer queue. An exported review must record
the order actually used before the review-time gate can pass.

Workbench collection is partition-isolated. Development (36), repeat (12),
validation (18), and holdout (18) each have a separate saved-run manifest, so
duplicate tickers never share a session and a development/validation export can
never contain holdout records. The API rejects creation of the holdout session
until a valid `configuration-freeze.json` and deployed code commit are supplied.
The source names are opaque batches A–D and the protected UI does not display
them, so the reviewer cannot infer partitions or repeats. The repeat session
always locks its independent blind label first. After that lock, the UI performs
an explicit, automatically timed manual-versus-assisted exercise in the frozen
6/6 counterbalanced order, recorded in `reviewOrder` at finalization.

Build (network required):

```bash
python3 scripts/build_v24_benchmark.py
```

Pre-label integrity audit:

```bash
python3 scripts/evaluate_v24_benchmark.py --output review_snapshots/benchmark_2026-08-13_v24_72/decision.prelabel.json
```

The evaluator exits `2` for both `no_go` and `inconclusive`; only a fully passing
frozen holdout returns success.

Frozen decision command (after label collection, selection, and freeze):

```bash
python3 scripts/evaluate_v24_benchmark.py --freeze review_snapshots/benchmark_2026-08-13_v24_72/configuration-freeze.json --repeat-labels review_snapshots/benchmark_2026-08-13_v24_72/development-labels.json --holdout-labels review_snapshots/benchmark_2026-08-13_v24_72/holdout-labels.json --output review_snapshots/benchmark_2026-08-13_v24_72/decision.json
```
