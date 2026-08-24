# Observed-lifetime reference-line experiment

This is a development comparison, not a production detector release. The experimental detector receives only frozen monthly OHLCV bars. Amrut's outcome-revealed geometry is added afterward as a dashed comparison overlay.

- Frozen source: `amrut_portfolio_exemplars_2026-08-21.csv`
- Cutoff: `full frozen history`
- Price definition: quarterly candle high
- Time definition: calendar-quarter ordinal (missing quarters do not compress slope)
- Production `analyze_coil()` remains unchanged

| Ticker | Detector anchors | Amrut anchors | Direction | Reference-anchor RMS error | Structures retained |
|---|---|---|---:|---:|---:|
| [1070.HK](1070.HK.png) | 2010-01-01, 2015-04-01 | 2010-01-01, 2015-04-01, 2020-10-01, 2024-06-01 | match | 8.71% | 3 |
| [0836.HK](0836.HK.png) | 2007-10-01, 2021-12-01 | 2007-10-01, 2021-12-01 | match | 0.00% | 3 |
| [GMDCLTD.NS](GMDCLTD.NS.png) | 2007-11-01, 2024-02-01 | 2007-11-01, 2024-02-01 | match | 0.00% | 2 |
| [0981.HK](0981.HK.png) | 2004-03-01, 2020-07-01 | 2004-03-01, 2020-07-01 | match | 0.00% | 1 |

## How to read the charts

The top pane shows why the highest observed price is not automatically the active resistance. Red X markers remain in the record but are demoted when they cannot form a credible price-and-time family. The lower pane fixes the earliest credible pair of anchors, projects it through calendar time, accepts later tops inside an asymmetric band, and retains a lower repeated-price family when one exists.

The numeric RMS comparison is descriptive only. These are teaching examples with estimated, outcome-revealed reference geometry, so it cannot establish population accuracy.

## Historical cutoff replay

This is the stricter check: what the detector could draw using only bars available at Amrut's dated milestones. `Aligned` means the direction matched and the detector line was within 10% RMS of the reference anchors; it is not a claim that the thresholds are final.

| Ticker | First recognisable | First watch | First actionable |
|---|---|---|---|
| 1070.HK | no line yet | aligned (2010-01-01, 2015-04-01) | aligned (2010-01-01, 2015-04-01) |
| 0836.HK | different line (2007-10-01, 2013-05-01) | aligned (2007-10-01, 2021-12-01) | — |
| GMDCLTD.NS | different line (2007-11-01, 2022-04-01) | aligned (2007-11-01, 2024-02-01) | aligned (2007-11-01, 2024-02-01) |
| 0981.HK | no line yet | aligned (2004-03-01, 2020-07-01) | aligned (2004-03-01, 2020-07-01) |

The gap between first recognition and first watch is material: this version waits for a later price rejection before confirming a quarterly high. That makes it intentionally slower and prevents a live, still-rising quarter from becoming an anchor, but it also means some structures appear later than Amrut says they become visually recognisable.
