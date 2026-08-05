# July 27 v2.2.0 Amrut review evidence

This directory freezes the exact saved-run inputs for Amrut’s screened-stock
review. It is an immutable evidence package, not a corrected market-data set,
not a detector benchmark, and not validated training truth.

The corpus is tied to:

- saved run: `screen_2026-07-27_v2.2.0.csv`
- algorithm: `2.2.0`
- code commit: `65a04cf6d13d4e31a4e9b15cbab54fa150a935ba`
- source CSV SHA-256:
  `ae845f887498bbeab4ea082512cfb60f1d11ce5098213f7647b4fb045be1b45a`

## Contents

- `source_run.csv` is an exact byte copy of the saved run.
- `manifest.json` contains the ordered 79-name universe, source identities,
  cohort and prior-use flags, per-ticker hashes, date ranges, bar counts, and
  mechanical data-quality findings.
- `<TICKER>.json` contains the exact ordered monthly bars, exact CSV row,
  cache/run provenance, cohort labels, and detailed quality findings for one
  ticker. The required review-app shape is:
  `schema_version=1`,
  `kind=coilingview.saved-run-review-snapshot`, the exact source CSV filename,
  ticker, `run.algorithm_version`, `screen_snapshot.ticker`, and
  `monthly_bars`.
- `build_corpus.py` deterministically materializes the package from the exact
  local cache and saved CSV.
- `verify_corpus.py` proves the saved-run copy, snapshot bytes, canonical
  identities, cache equality (when the source cache is present), item order,
  labels, and quality findings.

There are 79 snapshots and 27,114 monthly bars. The stored snapshot files
occupy 5,418,461 bytes. The current `manifest.json` SHA-256 is:

`cee6fe3042c6765d5dd873e7f93418a3e9f7ee73ebb95aa11dabd6e9b0fd31e5`

## Review and leakage boundaries

The run is not a clean out-of-sample validation set:

- 18 names are explicit calibration/control references.
- 61 names come from the prospective international review universe.
- 23 names already occur in `validation/major_high_feedback.json`.
- 8 names are explicitly used in v2.2.0 source comments to pin or illustrate
  thresholds: `CF`, `CNR.TO`, `EAT`, `FCX`, `KN`, `LH`, `REG`, and `SPG`.

These labels are recorded per ticker so the app and downstream feedback
document can distinguish prospective review from prior exposure. They must not
be shown as model truth or used to imply independent validation. A genuinely
held-out evaluation split must be created separately.

Human annotations produced from this corpus remain proposed evidence. They
must pass curation, conflict review, reproducibility checks, leakage checks,
and an independently frozen evaluation before any rule is promoted into the
algorithm.

## Data-quality boundary

No source price was repaired, dropped, rescaled, or normalized. The exact
source cache file has a byte-level SHA-256 identity, while each exact bar array
also has a canonical JSON identity.

The deterministic checks found:

- 61 tickers clear by these limited checks;
- 6 tickers flagged for review by extreme-wick/discontinuity heuristics;
- 12 tickers quarantined because at least one price is nonpositive or violates
  OHLC containment;
- 15 OHLC-containment failures;
- 5 nonpositive price fields;
- 47 extreme-wick flags; and
- 17 extreme-discontinuity flags.

The 12 quarantined names are:

`PETR4.SA`, `INFY.NS`, `SHEL.L`, `VALE3.SA`, `SOL.JO`, `ENB.TO`,
`WEGE3.SA`, `SBK.JO`, `ITUB4.SA`, `NPN.JO`, `HDFCBANK.NS`, and `0700.HK`.

The 6 heuristic-only flagged names are:

`ULVR.L`, `0005.HK`, `RELIANCE.NS`, `CBA.AX`, `REL.L`, and `TCS.NS`.

“Clear” means only that these specific automated checks did not fire. It does
not certify the vendor data. Likewise, a heuristic flag may represent a real
market event, corporate action, or vendor adjustment rather than an error.

The exact checks are:

1. Dates must be unique, chronological `YYYY-MM-DD` values.
2. OHLC fields must be finite numbers, strictly positive, and satisfy
   `high >= open`, `high >= close`, `high >= low`, `low <= open`, and
   `low <= close`.
3. An extreme wick is flagged when the larger wick divided by the median
   absolute open/close is at least `0.50` and at least `6x` the trailing
   24-bar median full-range ratio, after at least 12 baseline bars.
4. An extreme discontinuity is flagged when the median absolute open/close
   differs from the previous close by a symmetric scale ratio of at least
   `4x`.

Quarantined bars remain present so reviewers and engineers can see the exact
failure. They must not be silently passed through the analyzer. Corrected
vendor data belongs in a new, separately identified corpus; it must never
overwrite this evidence.

## Canonical identities

Canonical JSON hashes use UTF-8 with sorted object keys, no whitespace,
unescaped Unicode, and no non-finite numbers. Stored JSON uses sorted keys,
two-space indentation, and a final LF. The manifest records:

- exact source-cache byte hash;
- canonical monthly-bars hash;
- the backend-compatible bars identity hash over schema version, `1M`
  interval, ticker, and bars;
- exact CSV-row hash; and
- complete snapshot-file byte hash.

The source cache also contains derived feature objects, at least one of which
uses non-standard `NaN`. Those derived objects are not rewritten into portable
snapshot JSON. Their exact original files are still identified by byte hash,
and the saved CSV row contains the run’s portable feature output.

## Rebuild and verify

From the `coil-screener` directory:

```bash
python3 review_snapshots/screen_2026-07-27_v2.2.0/build_corpus.py
python3 review_snapshots/screen_2026-07-27_v2.2.0/verify_corpus.py
```

Building requires the exact ignored `cache/` files and the original saved-run
CSV. Verification remains self-contained for snapshot identities and uses the
original cache, when available, to prove byte identity and exact bar equality.
