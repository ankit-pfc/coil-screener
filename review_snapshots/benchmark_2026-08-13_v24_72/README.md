# Lean v2.4 72-sample blind benchmark

This directory is the immutable canonical corpus for the fail-closed benchmark
evaluator. It deliberately uses a non-workbench manifest kind. Reviewer tasks
are materialized into the separate opaque batch A–D saved-run directories.

The 72 unique samples are balanced across India, the United States, and global
ex-U.S. listings, with 36 development, 18 validation, and 18 untouched holdout
samples. Twelve development samples are queued a second time, producing 84
review tasks. The reviewer-facing queue omits repeat relationships. Cohort names
are sampling intentions, not labels; blind review must confirm them.

Every sample must use frozen, source-identified raw daily OHLC plus reported
split events, cut at that sample's `as_of` date before adjustment, and then
aggregated to months. Every analysis cutoff is a completed quarter. The
manifest freezes the full company name, sourced listing date, listing-quarter
coverage report, bars, source fingerprints, quality disposition, ticker
family, split, cohort, and `as_of` date. A provider's `max` response is not
accepted as proof of listing history.

Holdout labels are deliberately absent. `benchmark_v24.load_labels()` rejects a
holdout artifact until the configuration-freeze file matches the corpus,
protocol, registered algorithm-only detector versions, and v2.4 configuration.
An audit with missing blind labels must report `inconclusive`; it may never infer
truth from either detector or promote v2.4 on sampling strata alone.

The development sweep is limited to the 54 registered combinations in
`protocol.json`. `scripts/select_v24_configuration.py` refuses to run until all
36 development labels, all 12 repeats, and all 18 validation labels are
present; it applies the validation partition exactly once by creating its
selection report exclusively. Review timing is an optional observation, not a
detector-promotion gate: the deployed assisted-review experiment was retired
when blind labeling moved to the Codex-native chart flow.

Workbench collection is partition-isolated. Development (36), repeat (12),
validation (18), and holdout (18) each have a separate saved-run manifest, so
duplicate tickers never share a session and a development/validation export can
never contain holdout records. The API rejects creation of the holdout session
until a valid `configuration-freeze.json` and deployed code commit are supplied.
The source names are opaque batches A–D and the protected UI does not display
them, so the reviewer cannot infer partitions or repeats. The repeat session
always records an independent blind label without exposing detector evidence.

Build from a verified 72-ticker long-history corpus (provider acquisition may
require network access):

```bash
python3 scripts/build_v24_benchmark.py \
  --long-history-root /path/to/frozen-verified-long-history
```

## Blind review runbook

Run these commands from a clean tracked checkout. Run the API from this exact
commit on port 8000 and the CoilingView frontend on port 5173. Keep the review
database durable. If session creation is protected, set the same
`REVIEW_SESSION_CREATE_KEY` in the API and coordinator shell.

Create one stable capability token per batch. Keep these environment variables
private and reuse the same value if a session-creation request is retried:

```bash
export COILINGVIEW_V24_BATCH_A_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export COILINGVIEW_V24_BATCH_B_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export COILINGVIEW_V24_BATCH_C_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python3 scripts/create_v24_review_session.py --batch a
python3 scripts/create_v24_review_session.py --batch b
python3 scripts/create_v24_review_session.py --batch c
```

Each command prints a capability URL. Send only that URL to the assigned
reviewer. Complete and finalize A, then B, then C, retaining the three exact
schema-v5 JSON exports. Batch names and repeat relationships remain hidden in
the workbench UI. Do not create batch D yet.

Import A plus B as the development/repeat label artifact, then C as validation:

```bash
python3 scripts/import_v24_benchmark_labels.py batch-a.json batch-b.json --partition development --output review_snapshots/benchmark_2026-08-13_v24_72/development-labels.json
python3 scripts/import_v24_benchmark_labels.py batch-c.json --partition validation --output review_snapshots/benchmark_2026-08-13_v24_72/validation-labels.json
python3 scripts/select_v24_configuration.py --development-labels review_snapshots/benchmark_2026-08-13_v24_72/development-labels.json --validation-labels review_snapshots/benchmark_2026-08-13_v24_72/validation-labels.json
python3 scripts/freeze_v24_benchmark.py
```

Restart the API from the same Git commit with
`COILINGVIEW_V24_BENCHMARK_FREEZE` set to the absolute path of the generated
`configuration-freeze.json` and `COILINGVIEW_CODE_COMMIT` set to that 40-digit
commit. Only then create D with its own stable token:

```bash
export COILINGVIEW_V24_BATCH_D_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python3 scripts/create_v24_review_session.py --batch d
```

After D is finalized, import and register its export, then run the frozen
decision command below:

```bash
python3 scripts/import_v24_benchmark_labels.py batch-d.json --partition holdout --freeze review_snapshots/benchmark_2026-08-13_v24_72/configuration-freeze.json --output review_snapshots/benchmark_2026-08-13_v24_72/holdout-labels.json
python3 scripts/freeze_v24_benchmark.py --register-holdout-labels review_snapshots/benchmark_2026-08-13_v24_72/holdout-labels.json
python3 scripts/evaluate_v24_benchmark.py --freeze review_snapshots/benchmark_2026-08-13_v24_72/configuration-freeze.json --repeat-labels review_snapshots/benchmark_2026-08-13_v24_72/development-labels.json --holdout-labels review_snapshots/benchmark_2026-08-13_v24_72/holdout-labels.json --output review_snapshots/benchmark_2026-08-13_v24_72/decision.json
```

The coordinator must not inspect or distribute D labels before the freeze
exists. Preserve all artifacts and their hashes with the final decision.

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
