# Pilot check against reports/predictions.md (run pilot-20261015T061356Z-2cb75d2)

Pilot numbers are for debugging only and are never reported. Medians over pilot tensors.

## P1: D1 static, large error from clipping

| operator | Q | static clip fraction | static primary | dynamic primary |
| --- | --- | --- | --- | --- |
| sum | Q0 | 0.0098 | 0.244 | 0.997 |
| sum | Q1 | 0.0098 | 4.64 | 14.8 |
| sum | Q2 | 0.0098 | 3.52 | 7.29 |
| softmax | Q0 | 0.0098 | 1.19 | 7.16e-10 |
| softmax | Q1 | 0.0098 | 1.2 | 2.67e-07 |
| softmax | Q2 | 0.0098 | 1.87 | 7.56e-08 |
| layer_norm | Q0 | 0.0098 | 0.633 | 0.0205 |
| layer_norm | Q1 | 0.0098 | 0.654 | 0.344 |
| layer_norm | Q2 | 0.0098 | 0.713 | 0.186 |
| gelu | Q0 | 0.0098 | 0.681 | 0.0198 |
| gelu | Q1 | 0.0098 | 0.686 | 0.325 |
| gelu | Q2 | 0.0098 | 0.723 | 0.174 |

## P2: D1 dynamic Q1, most normal values become 0

Median zero fraction: 0.841 (predicted about 0.8 or more).

## P3 / P4: D1, Q2 vs Q1 (primary error ratio Q2 / Q1)

| operator | dynamic (P3: well below 1) | static (P4: near 1) |
| --- | --- | --- |
| sum | 0.492 | 0.758 |
| softmax | 0.283 | 1.55 |
| layer_norm | 0.54 | 1.09 |
| gelu | 0.537 | 1.05 |

## P5 / P5b / P6: D2 layer_norm

| Q | policy | collapsed rate | median SQNR (dB) | prediction |
| --- | --- | --- | --- | --- |
| Q0 | static | 0.00 | -8.92 | P5 collapse |
| Q1 | static | 1.00 | 0.00 | P5 collapse |
| Q2 | static | 0.00 | -10.80 | P5 collapse |
| Q1 | dynamic | 1.00 | 0.00 | P5 collapse |
| Q2 | dynamic | 0.00 | -0.74 | P5b no collapse, SQNR <= 0 |
| Q0 | dynamic | 0.00 | -5.14 | P6 no collapse, SQNR well under 20 |

## P7: gelu amplification near 1

| regime | Q0 dyn | Q1 dyn | Q2 dyn | Q0 static | Q1 static | Q2 static |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | 1.04 | 1.03 | 1.03 | 1.04 | 1.04 | 1.04 |
| D1 | 0.97 | 0.931 | 0.934 | 1.02 | 1.01 | 1.01 |
| D2 | 1.29 | 1.29 | 1.29 | 1.29 | 1.23 | 1.31 |
| D3 | 1.04 | 1.04 | 1.03 | 1.04 | 1.04 | 1.04 |
| D4 | 0.954 | 0.966 | 0.948 | 1.1 | 1 | 1.03 |
| D5 | 0.972 | 0.934 | 0.939 | 1.02 | 1.02 | 1.03 |
