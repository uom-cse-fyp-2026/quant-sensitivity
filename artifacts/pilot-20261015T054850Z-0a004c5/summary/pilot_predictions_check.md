# Pilot check against reports/predictions.md (run pilot-20261015T054850Z-0a004c5)

Pilot numbers are for debugging only and are never reported. Medians over pilot tensors.

## P1: D1 static, large error from clipping

| operator | Q | static clip fraction | static primary | dynamic primary |
| --- | --- | --- | --- | --- |
| sum | Q0 | 0.0098 | 0.254 | 0.754 |
| sum | Q1 | 0.0098 | 3.98 | 18.4 |
| sum | Q2 | 0.0098 | 4.2 | 8.76 |
| softmax | Q0 | 0.0098 | 1.17 | 7.02e-10 |
| softmax | Q1 | 0.0098 | 1.18 | 2.66e-07 |
| softmax | Q2 | 0.0098 | 1.77 | 7.27e-08 |
| layer_norm | Q0 | 0.0098 | 0.632 | 0.0204 |
| layer_norm | Q1 | 0.0098 | 0.653 | 0.344 |
| layer_norm | Q2 | 0.0098 | 0.705 | 0.184 |
| gelu | Q0 | 0.0098 | 0.68 | 0.0198 |
| gelu | Q1 | 0.0098 | 0.686 | 0.325 |
| gelu | Q2 | 0.0098 | 0.72 | 0.172 |

## P2: D1 dynamic Q1, most normal values become 0

Median zero fraction: 0.842 (predicted about 0.8 or more).

## P3 / P4: D1, Q2 vs Q1 (primary error ratio Q2 / Q1)

| operator | dynamic (P3: well below 1) | static (P4: near 1) |
| --- | --- | --- |
| sum | 0.475 | 1.05 |
| softmax | 0.273 | 1.49 |
| layer_norm | 0.536 | 1.08 |
| gelu | 0.528 | 1.05 |

## P5 / P5b / P6: D2 layer_norm

| Q | policy | collapsed rate | median SQNR (dB) | prediction |
| --- | --- | --- | --- | --- |
| Q0 | static | 1.00 | 0.00 | P5 collapse |
| Q1 | static | 1.00 | 0.00 | P5 collapse |
| Q2 | static | 0.00 | -10.69 | P5 collapse |
| Q1 | dynamic | 1.00 | 0.00 | P5 collapse |
| Q2 | dynamic | 0.00 | -0.72 | P5b no collapse, SQNR <= 0 |
| Q0 | dynamic | 0.00 | -5.12 | P6 no collapse, SQNR well under 20 |

## P7: gelu amplification near 1

| regime | Q0 dyn | Q1 dyn | Q2 dyn | Q0 static | Q1 static | Q2 static |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | 1.04 | 1.04 | 1.04 | 1.03 | 1.04 | 1.04 |
| D1 | 0.968 | 0.932 | 0.929 | 1.02 | 1.01 | 1.01 |
| D2 | 1.29 | 1.29 | 1.29 | 1.29 | 1.23 | 1.28 |
| D3 | 1.04 | 1.03 | 1.03 | 1.04 | 1.03 | 1.03 |
| D4 | 0.954 | 0.966 | 0.948 | 1.1 | 1 | 1.01 |
| D5 | 0.973 | 0.937 | 0.927 | 1.02 | 1.02 | 1.01 |
