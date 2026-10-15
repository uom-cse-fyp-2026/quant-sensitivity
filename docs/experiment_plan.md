# NeuroMinds: Quantization Sensitivity Experiment

Oct 15, 2026 · @Deepthi Damruwan

## Objective

Show empirically how a computation's loss of precision depends on both its input and the
quantization scheme, which is the case for adaptive quantization. The experiment follows the
supervisor's description exactly:

1. Take 2–3 operators that are sensitive to numerical perturbation (reduction ops and
   non-linear ops with reductions), and **verify** the assertion that they are more
   sensitive than elementwise ops.
2. Define a few input distributions with contrasting properties and draw FP32 samples
   A0, A1, … from them. Compute the baseline `Y = Op(A, NoQuant)` in FP32.
3. Define 2–3 quantization schemes Q0, Q1, ….
4. Compute `Y0 = Op(A, Q0)`, `Y1 = Op(A, Q1)`, ….
5. Quantify `Y0 − Y`, `Y1 − Y`, … and compare the loss of precision across schemes.

This is fake quantization: the input is quantized and dequantized, then the operator runs in
FP32. It measures numerical error only, not speed or energy.

## 1. Operators

| ID | PyTorch call | Type | Why it is sensitive |
| --- | --- | --- | --- |
| `sum` | `x.sum(dim=-1)` | Reduction | Errors accumulate over n terms; a near-zero sum (cancellation) magnifies them. |
| `softmax` | `torch.softmax(x, dim=-1)` | Non-linear + reduction | `exp` magnifies logit error; the normalising sum couples every element. |
| `layer_norm` | `F.layer_norm(x, (n,), weight=None, bias=None, eps=1e-5)` | Normalisation reduction | Mean and variance couple every element; a small variance amplifies error. |
| `gelu` | `F.gelu(x, approximate='none')` | Elementwise | **Control** for the assertion: error should stay local to each element. |

Only the operator's input is quantized. The operator is identical for the baseline and for
every scheme.

## 2. Input distributions

Each sample A is a 1-D FP32 tensor of length n = 1,024, generated in FP32 first and only then
quantized. 300 samples per distribution.

| ID | Name | Generation | Contrasting property |
| --- | --- | --- | --- |
| D0 | Normal | `N(0, 1)` | Typical, well-behaved baseline |
| D1 | Random outliers | D0 with exactly 1% of positions (random per sample) set to ±20, balanced signs | A few huge values stretch the range |
| D2 | Near-constant | `1.0 + 0.001 · N(0, 1)` | Tiny variation around an offset |
| D3 | Cancellation | 512 values a ~ N(0, 1) paired with −a + δ, δ ~ N(0, 0.01²), shuffled | Sum close to zero relative to its terms |
| D4 | Heavy-tailed | Student-t(3) / √3 (unit variance) | Many moderate outliers |
| D5 | Fixed outlier channels | D0 plus ±20 at 8 fixed positions (the same in every sample) | Systematic outlier channels, as in LLM activations |

**Reproducibility.** Each sample has its own `torch.Generator`, seeded with
`int(sha256(f"{regime}|{sample_id}").hexdigest()[:16], 16)`. Normals are drawn in FP64 and
rounded to FP32, because PyTorch's FP32 `randn` differs in the last bit between Linux and
Windows; this way the input tensors are bit-identical on every machine, and a test pins their
hashes. D5's positions and signs come from one fixed seed.

## 3. Baseline and quantization schemes

**Baseline (`NoQuant`):** `Y = Op(A)` in FP32, computed once per sample and operator.

All schemes are signed, symmetric, uniform, round-half-to-even and saturating, with zero point
0. The scale comes from the input's own absmax, `s = absmax / qmax` (floor 1e-12):

```math
q = \mathrm{clamp}(\mathrm{round}(x / s),\ -q_{max},\ q_{max}), \qquad \hat{A} = s \cdot q
```

| ID | Bits | Grid | Scale granularity | Controlled change |
| --- | --- | --- | --- | --- |
| Q0 | 8 | [−127, 127] | One scale per tensor | Higher-precision scheme |
| Q1 | 4 | [−7, 7] | One scale per tensor | Bit width only (vs Q0) |
| Q2 | 4 | [−7, 7] | One scale per contiguous group of 32 | Granularity only (vs Q1) |

## 4. Comparing outputs

Every quantized output Y_q is compared with the baseline Y of the same sample, in FP64.

| Operator | Error measures |
| --- | --- |
| `sum` | Absolute error \|Y_q − Y\|, and \|Y_q − Y\| / ‖A‖₁ (scale-free and safe under cancellation) |
| `softmax` | KL(Y ‖ Y_q) in nats, normalised L2, top-1 changed |
| `layer_norm`, `gelu` | Normalised L2 = ‖Y_q − Y‖₂ / ‖Y‖₂, MAE, max absolute error |

For every vector output, SQNR in dB = −20 · log₁₀(normalised L2). The headline error per
operator is |ΔY| / ‖A‖₁ for `sum`, KL for `softmax`, and normalised L2 for `layer_norm` and
`gelu`.

**Verifying the sensitivity assertion.** The amplification factor

```math
\kappa = \frac{\lVert Y_q - Y \rVert_2 / \lVert Y \rVert_2}{\lVert \hat{A} - A \rVert_2 / \lVert A \rVert_2}
```

is the output's relative error divided by the input's relative quantization error: an
empirical condition number. If reductions are more sensitive than elementwise ops, κ for
`sum`, `softmax` and `layer_norm` should be larger and vary more across distributions than κ
for `gelu`. κ is left empty when it is undefined (zero input error, or ‖Y‖ < 1e-12).

**FP32 noise.** The FP32 baseline has its own rounding error, measured against an FP64
evaluation of the same operator: `fp32_noise = ‖Y − Y₆₄‖₂ / ‖Y₆₄‖₂`. When a scheme's output
error is within 1,000× of this, the error is FP32 rounding rather than quantization, and the
row's `at_fp32_noise` flag is set. Flagged κ values are excluded from κ summaries and counted.
(A comparison between two machines showed κ varying by up to 73% at 10–30× the noise level
and by 0.4% above 1,000×.)

**Statistics.** The unit of analysis is the sample: 300 per distribution, not 300 × 1,024
elements. For every operator × distribution × scheme we report the median, IQR, p95 and a 95%
bootstrap confidence interval of the median (2,000 resamples, fixed seed). Errors span orders
of magnitude, so figures use log scales and compare medians. Non-finite values are never
dropped; they are counted and the run fails.

## 5. Outputs

`just run` writes `artifacts/<run_id>/`:

- `raw/results.csv`: one row per (distribution, sample, operator, scheme), 6 × 300 × 4 × 3 = 21,600 rows
- `manifest.json`: git commit, dirty flag, machine, library versions, config checksum, row and non-finite counts, runtime
- `run.log`
- `summary/summary.csv`: median, IQR, p95 and bootstrap CI per operator × distribution × scheme
- `summary/summary.md`: the tables for the report, including the sensitivity check and the scheme ranking
- `plots/F1_inputs.png`: histograms of the six distributions
- `plots/F2_error_heatmap.png`: median error per distribution × scheme, one panel per operator
- `plots/F3_amplification.png`: κ by operator and distribution, the test of the assertion

## 6. Running

All runs execute on the VM. CPU only, one thread, deterministic algorithms: a rerun on the same
machine gives bit-identical numbers. Inputs are identical on every machine; operator outputs can
differ in the last bits between CPUs (different `exp`/`erf` kernels), so reported numbers come
from the VM. Commands are in the README.
