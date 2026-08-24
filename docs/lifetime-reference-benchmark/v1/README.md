# Lifetime reference benchmark — v1

This is an outcome-revealed development benchmark, not a blind accuracy estimate and not an investment signal. Both detectors receive only frozen OHLCV prefixes. Human reference geometry is introduced afterward for comparison.

## What ran

- 24 exact reviewed examples and 9 source-estimated portfolio examples (33 labelled development charts).
- 43 July shadow charts with no exact geometry joined.
- 24 otherwise-reviewable July charts withheld because their ticker identities overlap the sealed 24-chart blind queue.
- Production `analyze_coil()` and experimental lifetime geometry ran independently; production code was not changed.
- Hybrid lifetime-boundary lifecycle scoring did not run because there is no supported detector-only boundary-injection interface yet.

## Exact clicked geometry (23 line-positive, 1 explicit no-line)

| Measure | Current detector | Lifetime geometry |
|---|---:|---:|
| Line found on positive examples | 95.7% | 95.7% |
| Direction match when a line was emitted | 40.9% | 68.2% |
| Primary line within 10% RMS at clicked anchors | 17.4% | 34.8% |
| Strict 5%/slope/anchor geometry match | 0.0% | 13.0% |
| Line drawn on the explicit no-line example | 100.0% | 100.0% |

The 10% RMS line is a descriptive review aid. The stricter row additionally requires projected-value, slope, direction, and clicked-anchor timing agreement. Neither is population accuracy because these examples are outcome-visible development material.

## Source-estimated portfolio geometry

| Measure | Current detector | Lifetime geometry |
|---|---:|---:|
| Line found | 100.0% | 100.0% |
| Primary line within 10% RMS | 0.0% | 100.0% |

## Shadow comparison (no exact geometry joined)

| Relationship | Charts |
|---|---:|
| Both aligned | 1 |
| Both drew different geometry | 29 |
| Current only | 7 |
| Lifetime only | 2 |
| Neither | 4 |

These are disagreement buckets, not wins or losses. Source-screen context is not hidden: 22 cases had prior review context and 8 were tuning anchors. Only 20 cases are prospective, not previously reviewed, and not tuning anchors; 19 of those are detector disagreements and form the independent review queue.

## Point-in-time recognition replay

Geometry is expected only from Amrut's first-recognizable milestone onward. Earlier detector lines are counted as false-early presence; geometry is scored only where the contemporaneously available reference has at least two anchors.

| Measure | Current detector | Lifetime geometry |
|---|---:|---:|
| False-early line before recognition | 93.9% | 75.8% |
| Line present at first-recognizable checkpoint | 93.8% | 75.0% |
| Presence transitions | 1 | 7 |
| Anchor churn while continuously present | 33 | 13 |

`False-early` means the geometry existed before the whole setup's reference recognition milestone; it does not by itself make the resistance line invalid. Recognition lag is recorded separately in `summary.json`; replay checkpoints are sparse milestones, not a quarter-by-quarter latency study.

## Files

- [`labelled_comparison.csv`](labelled_comparison.csv): one row per outcome-revealed development chart.
- [`shadow_comparison.csv`](shadow_comparison.csv): one row per safe shadow chart, including prior-review, tuning-anchor, and cohort-role context.
- [`replay.csv`](replay.csv): point-in-time runs at available Amrut milestones, with future reference anchors withheld.
- [`summary.json`](summary.json): aggregate counts and error taxonomy.
- [`manifest.json`](manifest.json): source hashes, detector versions, protocol, and blind exclusions.
- [`chart_index.md`](chart_index.md): ticker-to-chart index for every labelled failure and safe shadow disagreement.
- `setups/`: compact per-observation JSON records; raw price bars are not duplicated.

## Interpretation boundary

A mathematically credible resistance line is not automatically a coil. ADM and short-history cases show why congestion, maturity, compression, and lifecycle must remain separate downstream decisions. The next engineering step is a supported algorithmic-boundary adapter with parity tests against the unchanged current detector.
