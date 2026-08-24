# Benchmark chart index

Every chart below is a development or no-exact-geometry shadow comparison. Purple reference geometry, where present, is added only after both detector runs are locked.

## Labelled failures and strict disagreements

| Ticker | Reference quality | Lifetime assessment | Anchor RMS | Chart |
|---|---|---|---:|---|
| ACIW | exact_clicks | different_geometry | 23.22% | [open chart](charts/labelled-failures/27e0341867a3dae9e9f2.png) |
| ADM | exact_clicks | false_line | n/a | [open chart](charts/labelled-failures/13adfae210ff5810acb6.png) |
| AEO | exact_clicks | different_geometry | 39.65% | [open chart](charts/labelled-failures/d23b872f9da69e5afb9e.png) |
| AER | exact_clicks | different_geometry | 2.10% | [open chart](charts/labelled-failures/92a7cca51d000d580e9f.png) |
| AGRO | exact_clicks | different_geometry | 3.33% | [open chart](charts/labelled-failures/005755511dc9d3c39475.png) |
| AGYS | exact_clicks | different_geometry | 1.33% | [open chart](charts/labelled-failures/43dbd17227cadadc4bfd.png) |
| AMKR | exact_clicks | different_geometry | 24.55% | [open chart](charts/labelled-failures/8a6eaa2fdea4c7a2cf76.png) |
| ASB | exact_clicks | different_geometry | 17.45% | [open chart](charts/labelled-failures/8f5d9cd3e52e17c4f073.png) |
| AVT | exact_clicks | different_geometry | 2.98% | [open chart](charts/labelled-failures/6ccc66999103cd7a517a.png) |
| BCO | exact_clicks | different_geometry | 5.50% | [open chart](charts/labelled-failures/ee931bfaa2734962533a.png) |
| BDC | exact_clicks | different_geometry | 12.64% | [open chart](charts/labelled-failures/e5448cbf68b63f5781ed.png) |
| CBT | exact_clicks | different_geometry | 116.09% | [open chart](charts/labelled-failures/c377b70ad34cd70732da.png) |
| CECO | exact_clicks | missed_line | n/a | [open chart](charts/labelled-failures/6503d265994407fa7d5b.png) |
| CNA | exact_clicks | different_geometry | 15.66% | [open chart](charts/labelled-failures/718e0e8cadd7d7719163.png) |
| FOR | exact_clicks | different_geometry | 34.47% | [open chart](charts/labelled-failures/8be434a2e9858086b3df.png) |
| KAI | exact_clicks | different_geometry | 42.02% | [open chart](charts/labelled-failures/290d096d99a64efdae98.png) |
| KFY | exact_clicks | different_geometry | 7.13% | [open chart](charts/labelled-failures/42521d3ba42ba517ec6c.png) |
| LAZ | exact_clicks | different_geometry | 2.15% | [open chart](charts/labelled-failures/f7a376e47daa9c547100.png) |
| MMYT | exact_clicks | different_geometry | 2.83% | [open chart](charts/labelled-failures/593402cf66ad535f4184.png) |
| NATR | exact_clicks | different_geometry | 4.56% | [open chart](charts/labelled-failures/79d17947d82e9724ce6a.png) |
| VICR | exact_clicks | different_geometry | 58.60% | [open chart](charts/labelled-failures/c8bb28eed1eda02e29da.png) |
| 1070.HK | estimated_source | different_geometry | 8.71% | [open chart](charts/labelled-failures/f46d37ab13e9737f4af8.png) |
| 0148.HK | estimated_source | different_geometry | 4.87% | [open chart](charts/labelled-failures/afbd46c0f7e35824a527.png) |
| 3808.HK | estimated_source | different_geometry | 8.11% | [open chart](charts/labelled-failures/8f147fe72007cfa372ee.png) |

A `different_geometry` row can still be visually close: the strict assessment also checks projected value, annualized slope, and construction-anchor timing. Use `labelled_comparison.csv` for the separate 10% RMS flag.

## Independent shadow review queue

Prospective cases with no prior-review or tuning-anchor context are isolated here.

| Ticker | Flow relationship | Chart |
|---|---|---|
| 1299.HK | current_only | [open chart](charts/shadow-disagreements/07141c3a4461b30ffb68.png) |
| AZN.L | both_different | [open chart](charts/shadow-disagreements/9e7e1593817897dee07e.png) |
| PHIA.AS | both_different | [open chart](charts/shadow-disagreements/fcd996d70b3329fff1bb.png) |
| BAS.DE | both_different | [open chart](charts/shadow-disagreements/c786e2a2cbe9da12cd28.png) |
| SAN.PA | both_different | [open chart](charts/shadow-disagreements/e218cadccb0654a07df9.png) |
| 2317.TW | both_different | [open chart](charts/shadow-disagreements/cf05a5b4023cd4fbe4ef.png) |
| REL.L | both_different | [open chart](charts/shadow-disagreements/d607696da3265c6a6908.png) |
| HSBA.L | both_different | [open chart](charts/shadow-disagreements/c73d2250dff2e7099b33.png) |
| BHP.AX | both_different | [open chart](charts/shadow-disagreements/d7d62ccc6ebabe2842c9.png) |
| OR.PA | current_only | [open chart](charts/shadow-disagreements/c9db50cde1393893b0d6.png) |
| SU.PA | both_different | [open chart](charts/shadow-disagreements/73f5d41ec18cdf6fa32d.png) |
| 6758.T | both_different | [open chart](charts/shadow-disagreements/e89fbac6c8f80cfee129.png) |
| 2454.TW | both_different | [open chart](charts/shadow-disagreements/2d11e22e29a65ca1909b.png) |
| SAP.DE | both_different | [open chart](charts/shadow-disagreements/b62125d4726aca012af6.png) |
| WES.AX | both_different | [open chart](charts/shadow-disagreements/e770517291d710da857e.png) |
| ASML.AS | lifetime_only | [open chart](charts/shadow-disagreements/6a1a628c0743bba9775d.png) |
| CSL.AX | lifetime_only | [open chart](charts/shadow-disagreements/cdd8ab2b20f2b70656cf.png) |
| MQG.AX | both_different | [open chart](charts/shadow-disagreements/3b5708042ce377ae27d0.png) |
| TD.TO | current_only | [open chart](charts/shadow-disagreements/5c6962ff95af3c2924b9.png) |

## Contextual shadow disagreements

These rows carried prior-review, tuning-anchor, or control-reference context in the source screen and are kept separate from independent review.

| Ticker | Source context | Flow relationship | Chart |
|---|---|---|---|
| REG | prior review, tuning anchor, control reference | both_different | [open chart](charts/shadow-disagreements/b541aab4a79f0ba4a9c4.png) |
| SPG | prior review, tuning anchor, control reference | both_different | [open chart](charts/shadow-disagreements/f5dc2c7b989aea5a8cd1.png) |
| ULVR.L | prior review | both_different | [open chart](charts/shadow-disagreements/c82d9031eca543156e41.png) |
| 0005.HK | prior review | current_only | [open chart](charts/shadow-disagreements/b31316f270ea3bd203cc.png) |
| VTR | prior review, control reference | both_different | [open chart](charts/shadow-disagreements/595febb8421ace340379.png) |
| UNP | prior review, control reference | both_different | [open chart](charts/shadow-disagreements/57381ac4f551f9c7cd99.png) |
| LH | prior review, tuning anchor, control reference | both_different | [open chart](charts/shadow-disagreements/24997ffc950a4da82a19.png) |
| FCX | prior review, tuning anchor, control reference | both_different | [open chart](charts/shadow-disagreements/a3e0d1f1e45e26cec95f.png) |
| NOVN.SW | prior review | both_different | [open chart](charts/shadow-disagreements/13a3a19411db63b1387b.png) |
| NUE | prior review, control reference | current_only | [open chart](charts/shadow-disagreements/b965c1eee479189f6181.png) |
| CF | prior review, tuning anchor, control reference | both_different | [open chart](charts/shadow-disagreements/b3351bdb85c8b14de18b.png) |
| KN | prior review, tuning anchor, control reference | both_different | [open chart](charts/shadow-disagreements/84df235fd41ecb62cdf0.png) |
| ALV.DE | prior review | both_different | [open chart](charts/shadow-disagreements/f9c408d4f038450b70b6.png) |
| EAT | prior review, tuning anchor, control reference | both_different | [open chart](charts/shadow-disagreements/1118ad37dcd66bd35718.png) |
| MSFT | prior review, control reference | current_only | [open chart](charts/shadow-disagreements/0f673bebd45967ac780e.png) |
| COF | prior review, control reference | both_different | [open chart](charts/shadow-disagreements/5689b004694cd008d946.png) |
| CSX | prior review, control reference | current_only | [open chart](charts/shadow-disagreements/990a6fb279de9f5d74ad.png) |
| NSC | prior review, control reference | both_different | [open chart](charts/shadow-disagreements/2d108295a27f784aed29.png) |
| UEC | prior review, control reference | both_different | [open chart](charts/shadow-disagreements/7b0fd33f825a8a5dbe3d.png) |

The sealed 24-chart blind queue and duplicate ticker identities are absent from this index.
