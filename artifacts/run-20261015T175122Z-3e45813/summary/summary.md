# Quantization sensitivity results: run-20261015T175122Z-3e45813

Commit `3e45813`, Intel(R) Xeon(R) CPU E7-4820 v3 @ 1.90GHz. 300 samples per distribution, 21,600 rows, 0 non-finite. Medians over samples; brackets are 95% bootstrap CIs.

Distributions: D0 normal, D1 random outliers, D2 near-constant, D3 cancellation, D4 heavy-tailed, D5 fixed outlier channels.

Schemes: Q0 A8 per-tensor, Q1 A4 per-tensor, Q2 A4 per-group-32.

## 1. Loss of precision against the FP32 baseline

**sum**: |ΔY| / ‖A‖₁

| scheme | D0 | D1 | D2 | D3 | D4 | D5 |
| --- | --- | --- | --- | --- | --- | --- |
| Q0 | 1.98e-04 [1.76e-04, 2.21e-04] | 9.25e-04 [8.24e-04, 1.05e-03] | 1.44e-03 [1.37e-03, 1.49e-03] | 1.90e-04 [1.66e-04, 2.20e-04] | 7.62e-04 [6.51e-04, 8.68e-04] | 1.04e-03 [9.14e-04, 1.17e-03] |
| Q1 | 3.96e-03 [3.42e-03, 4.39e-03] | 1.53e-02 [1.41e-02, 1.70e-02] | 3.20e-03 [3.15e-03, 3.23e-03] | 1.27e-03 [1.15e-03, 1.44e-03] | 1.18e-02 [1.04e-02, 1.33e-02] | 1.64e-02 [1.49e-02, 1.89e-02] |
| Q2 | 2.76e-03 [2.35e-03, 3.15e-03] | 7.56e-03 [6.40e-03, 9.00e-03] | 2.06e-03 [2.05e-03, 2.07e-03] | 2.40e-03 [2.12e-03, 2.78e-03] | 4.65e-03 [4.26e-03, 5.04e-03] | 7.71e-03 [6.56e-03, 8.87e-03] |

**softmax**: KL(Y ‖ Y_q), nats

| scheme | D0 | D1 | D2 | D3 | D4 | D5 |
| --- | --- | --- | --- | --- | --- | --- |
| Q0 | 2.98e-05 [2.92e-05, 3.10e-05] | 7.17e-10 [7.10e-10, 7.26e-10] | 3.54e-06 [3.42e-06, 3.65e-06] | 2.56e-05 [2.49e-05, 2.65e-05] | 8.14e-05 [7.32e-05, 9.00e-05] | 5.10e-04 [4.44e-04, 5.63e-04] |
| Q1 | 9.79e-03 [9.48e-03, 1.00e-02] | 2.71e-07 [2.69e-07, 2.75e-07] | 4.98e-07 [4.95e-07, 5.01e-07] | 8.37e-03 [8.19e-03, 8.61e-03] | 2.67e-02 [2.52e-02, 3.23e-02] | 1.44e-01 [1.28e-01, 1.61e-01] |
| Q2 | 4.38e-03 [4.34e-03, 4.42e-03] | 7.44e-08 [7.29e-08, 7.67e-08] | 5.85e-07 [5.80e-07, 5.89e-07] | 4.29e-03 [4.25e-03, 4.33e-03] | 4.62e-03 [4.25e-03, 5.21e-03] | 1.55e-05 [7.59e-08, 1.31e-03] |

**layer_norm**: ‖ΔY‖₂ / ‖Y‖₂

| scheme | D0 | D1 | D2 | D3 | D4 | D5 |
| --- | --- | --- | --- | --- | --- | --- |
| Q0 | 7.72e-03 [7.63e-03, 7.87e-03] | 2.04e-02 [2.04e-02, 2.05e-02] | 1.80e+00 [1.79e+00, 1.81e+00] | 7.20e-03 [7.13e-03, 7.29e-03] | 2.07e-02 [1.99e-02, 2.14e-02] | 2.39e-02 [2.38e-02, 2.40e-02] |
| Q1 | 1.40e-01 [1.38e-01, 1.41e-01] | 3.45e-01 [3.44e-01, 3.45e-01] | 1.00e+00 [1.00e+00, 1.00e+00] | 1.30e-01 [1.29e-01, 1.32e-01] | 3.54e-01 [3.44e-01, 3.63e-01] | 3.96e-01 [3.95e-01, 3.98e-01] |
| Q2 | 9.64e-02 [9.61e-02, 9.68e-02] | 1.84e-01 [1.83e-01, 1.86e-01] | 1.09e+00 [1.08e+00, 1.09e+00] | 9.62e-02 [9.56e-02, 9.67e-02] | 1.43e-01 [1.41e-01, 1.44e-01] | 1.69e-01 [1.68e-01, 1.70e-01] |

**gelu**: ‖ΔY‖₂ / ‖Y‖₂

| scheme | D0 | D1 | D2 | D3 | D4 | D5 |
| --- | --- | --- | --- | --- | --- | --- |
| Q0 | 8.04e-03 [7.94e-03, 8.17e-03] | 1.98e-02 [1.98e-02, 1.99e-02] | 3.90e-03 [3.86e-03, 3.92e-03] | 7.46e-03 [7.38e-03, 7.53e-03] | 1.95e-02 [1.90e-02, 2.04e-02] | 2.32e-02 [2.31e-02, 2.33e-02] |
| Q1 | 1.46e-01 [1.44e-01, 1.47e-01] | 3.27e-01 [3.26e-01, 3.28e-01] | 4.31e-03 [4.25e-03, 4.35e-03] | 1.36e-01 [1.34e-01, 1.37e-01] | 3.50e-01 [3.37e-01, 3.67e-01] | 3.75e-01 [3.74e-01, 3.77e-01] |
| Q2 | 9.96e-02 [9.92e-02, 1.00e-01] | 1.73e-01 [1.71e-01, 1.75e-01] | 3.00e-03 [2.99e-03, 3.02e-03] | 9.97e-02 [9.92e-02, 1.00e-01] | 1.36e-01 [1.35e-01, 1.38e-01] | 1.59e-01 [1.57e-01, 1.60e-01] |

## 2. Are reductions more sensitive than elementwise ops?

Median amplification κ = (output relative error) / (input relative error). κ ≈ 1 means the operator passes the input's quantization error through unchanged; κ ≫ 1 means it magnifies it. The last column is the spread of the median across distributions (max / min).

**Q0 A8 per-tensor**

| operator | D0 | D1 | D2 | D3 | D4 | D5 | spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sum | 0.895 | 2.27 | 0.474 | 144 | 1 | 1.78 | 304× |
| softmax | 0.967 | — | 0.879 | 0.961 | 0.477 | 0.999 | 2.09× |
| layer_norm | 1 | 1 | 599 | 0.999 | 1 | 1 | 600× |
| gelu | 1.03 | 0.972 | 1.29 | 1.04 | 0.96 | 0.971 | 1.34× |

**Q1 A4 per-tensor**

| operator | D0 | D1 | D2 | D3 | D4 | D5 | spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sum | 1.04 | 2.23 | 0.954 | 50.2 | 0.973 | 1.65 | 52.6× |
| softmax | 0.96 | — | 0.299 | 0.944 | 0.48 | 1.01 | 3.37× |
| layer_norm | 0.992 | 0.985 | 299 | 0.993 | 0.972 | 0.987 | 307× |
| gelu | 1.03 | 0.935 | 1.29 | 1.03 | 0.964 | 0.934 | 1.38× |

**Q2 A4 per-group-32**

| operator | D0 | D1 | D2 | D3 | D4 | D5 | spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sum | 0.941 | 2.09 | 0.886 | 135 | 1.02 | 1.86 | 153× |
| softmax | 0.836 | — | 0.464 | 0.833 | 0.145 | 0.905 | 6.25× |
| layer_norm | 0.996 | 0.995 | 467 | 0.996 | 0.993 | 0.994 | 470× |
| gelu | 1.03 | 0.935 | 1.29 | 1.03 | 0.958 | 0.932 | 1.38× |

κ values at FP32 noise level (output error within 1000× of the baseline's own FP32 rounding) are excluded: sum D0 Q0 (1), sum D1 Q0 (1), sum D2 Q0 (1), sum D3 Q0 (3), sum D3 Q1 (1), sum D3 Q2 (1), sum D5 Q0 (1), softmax D1 Q0 (300), softmax D1 Q1 (300), softmax D1 Q2 (300), softmax D4 Q0 (35), softmax D4 Q1 (14), softmax D4 Q2 (33), softmax D5 Q2 (149).

## 3. Does the best scheme depend on the input?

Schemes ordered by median error, best first.

| operator | D0 | D1 | D2 | D3 | D4 | D5 | distinct orders |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sum | Q0 < Q2 < Q1 | Q0 < Q2 < Q1 | Q0 < Q2 < Q1 | Q0 < Q1 < Q2 | Q0 < Q2 < Q1 | Q0 < Q2 < Q1 | 2 |
| softmax | Q0 < Q2 < Q1 | Q0 < Q2 < Q1 | Q1 < Q2 < Q0 | Q0 < Q2 < Q1 | Q0 < Q2 < Q1 | Q2 < Q0 < Q1 | 3 |
| layer_norm | Q0 < Q2 < Q1 | Q0 < Q2 < Q1 | Q1 < Q2 < Q0 † | Q0 < Q2 < Q1 | Q0 < Q2 < Q1 | Q0 < Q2 < Q1 | 2 |
| gelu | Q0 < Q2 < Q1 | Q0 < Q2 < Q1 | Q2 < Q0 < Q1 | Q0 < Q2 < Q1 | Q0 < Q2 < Q1 | Q0 < Q2 < Q1 | 2 |

† every scheme has median normalised L2 ≥ 0.99: the output is essentially lost whichever scheme is used.

## 4. Softmax top-1 changes

Share of samples whose top-1 entry changed.

| scheme | D0 | D1 | D2 | D3 | D4 | D5 |
| --- | --- | --- | --- | --- | --- | --- |
| Q0 | 0.03 | 0.00 | 1.00 | 0.03 | 0.02 | 0.06 |
| Q1 | 0.52 | 0.00 | 1.00 | 0.52 | 0.26 | 0.80 |
| Q2 | 0.05 | 0.00 | 1.00 | 0.04 | 0.01 | 0.07 |
