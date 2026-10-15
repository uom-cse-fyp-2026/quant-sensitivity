# Pre-registered predictions

Written at Gate 0 (2026-10-15), before any pilot, development or held-out run. Plan §5.4.
Each row is a check on whether the code and the reasoning agree. A miss is a finding to
explain, not a failure. Gate 4 checks every row in `T2_predictions_check.md`.

Static scales come from absmax over 512 × 1,024 D0 values, which is about 5, so the static
step is about 5/127 ≈ 0.039 for A8 and 5/7 ≈ 0.71 for A4.

| ID | Case | Prediction | Reason |
| --- | --- | --- | --- |
| P1 | D1, static, any Q | Large error from clipping | D0 absmax is about 5, so ±20 outliers saturate. |
| P2 | D1, dynamic, Q1 | Most normal values become 0 (zero fraction ≳ 0.8) | Scale ≈ 20/7 ≈ 2.9, so \|x\| < 1.4 rounds to zero. |
| P3 | D1, dynamic, Q2 vs Q1 | Q2 much better | Only about 27% of groups contain an outlier; the rest keep fine scales. |
| P4 | D1, static, Q2 vs Q1 | Little difference | Static group scales from iid D0 are nearly identical. |
| P5 | D2, static Q0/Q1/Q2 and dynamic Q1 | All codes equal; layer_norm output collapses to 0 (collapsed flag set) | The 0.001 noise is far below half a step: static A8 maps every value to code 26, static A4 to 1, dynamic A4 to 7. Static Q2 group scales are nearly identical, so x̂ is (almost) constant. |
| P5b | D2, dynamic Q2 | All codes equal 7, but layer_norm does **not** collapse: its output is a 32-step staircase uncorrelated with the true noise (SQNR ≤ 0 dB, collapsed flag not set) | Each group's x̂ equals that group's absmax, so x̂ varies between groups by about 0.001. layer_norm amplifies those between-group steps instead of the within-group noise. |
| P6 | D2, dynamic Q0 | Codes split between 126 and 127 (about 2:1); layer_norm keeps only a 1-bit trace of the noise, so its error is still large (SQNR well under 20 dB) | Dynamic A8 step ≈ 1.003/127 ≈ 0.0079. The noise moves x/s by about 0.13 codes around 126.6, so values straddle the 126.5 rounding boundary. |
| P7 | `gelu`, all regimes | Amplification κ near 1 | Elementwise error stays local. |

## Revision before freezing

The plan's original D2 row said every symmetric quantizer under both policies would map D2
to a single code, "even for dynamic A8". A pure-Python check at Gate 0 (1,024 samples of
`1 + 0.001·N(0, 1)`) gave 674 values at code 127 and 350 at code 126 under dynamic A8, and a
single code in the static A8 and dynamic A4 cases. A smoke check with the Gate 0 code then
showed that dynamic Q2 also gives a single code but not a collapsed output (SQNR −1.7 dB,
collapsed flag 0 on one pilot tensor), because group scales differ. The row was split into
P5, P5b and P6 before any experiment ran. The conclusion that layer_norm on D2 is badly
broken under every symmetric scheme still stands; the mechanism differs by scheme.

## Note on the D2 reference

D2's variance (about 1e-6) is below the layer_norm eps (1e-5), so the FP32 reference itself
is eps-dominated: its output std is about 0.30, not 1. Quantized and reference outputs use
the same eps, so the comparison is fair, but D2 layer_norm results describe this eps
setting, not an eps-free normalisation.
