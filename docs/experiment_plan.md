# NeuroMinds: Quantization Sensitivity Experiment Plan

Oct 15, 2026 · @Deepthi Damruwan

> Source of truth for the Stage 1 experiment. Any change after Gate 0 must also be logged in the
> "Deviations" section of `reports/findings.md`, with the date and reason.

## Scope of this phase

The supervisor's experiment description, and where this plan covers each step:

| Supervisor's step | Where |
| --- | --- |
| Take 2–3 operators sensitive to numerical perturbation (reductions, non-linear ops with reductions), and verify that they beat elementwise ops | §3 (`sum`, `softmax`, `layer_norm`, with `gelu` as the elementwise control); verified by the amplification factor κ, §6.3 and RQ2 |
| Define a few input distributions with interesting, contrasting properties; draw FP32 samples A0, A1, …; compute the baseline Y = Op(A, NoQuant) | §4.1 (D0–D5), §4.2 sweeps; FP32 baseline §6.1 |
| Define 2–3 quantization schemes Q0, Q1, … | §5 (Q0 A8 per-tensor, Q1 A4 per-tensor, Q2 A4 per-group, each under static and dynamic calibration) |
| Observe Y0 = Op(A, Q0), Y1 = Op(A, Q1), … | §8 Gates 2–3 |
| Quantify Y0 − Y, Y1 − Y, or compare loss of precision another reasonable way | §6 metrics and statistics, §9 analysis |

**Deferred:** everything that involves LLMs or goes beyond the supervisor's description for
now: the real-activation replay from Llama-3.2-1B (§4.3), the Tier 2 quantizers and
`matvec` (§5.5, §3), and Stage 2 (§10). They stay in this plan for later but are not part of
the current runs. All experiments run on the VM; results come back with `scp` for the report.

## 1. Purpose and final design

The experiment tests one claim: **no single static quantization scheme is best across inputs, and a cheap input statistic predicts which scheme an input needs.** It shows more than "quantization adds error". It shows that the *ranking* and the *feasibility* of schemes change with the input. That is the evidence an adaptive, inspector-driven framework needs.

This v2 keeps the v1 protocol (controlled quantizers, disjoint splits, held-out evaluation, inspector feasibility) and makes six changes:

| # | Change from v1 | Why |
| --- | --- | --- |
| 1 | Add **GELU as an elementwise control** to the core operators | The supervisor asked us to *verify* that reductions are more sensitive than elementwise ops. That needs an elementwise op in the same harness. |
| 2 | Add a **dynamic (per-input) calibration policy** beside v1's static D0-calibrated scales | Static scales mix two effects: the input's structure and its mismatch with the calibration data. Dynamic scales isolate the input's own structure. They are also where per-group scaling can actually matter. |
| 3 | Add an **FP64 reference** beside the FP32 baseline | It measures the FP32 floor, so quantization error is never confused with FP32 rounding (this matters for the cancellation regime). |
| 4 | Add an **amplification factor** and **SQNR** to the metrics | Amplification (output relative error ÷ input relative error) is the empirical condition number. It is the direct test of the sensitivity assertion. |
| 5 | Add **parameter sweeps** (outlier size, outlier rate, vector length) | Curves where the best scheme switches at a threshold argue for adaptivity much more strongly than four fixed points. |
| 6 | Add **two LLM-motivated regimes** (heavy-tailed, fixed outlier channels) and a **real-activation replay** tier | They tie the synthetic stress tests to what LLM activations actually look like. |

**Core matrix (Tier 1, the result we present first):**

| Factor | Levels |
| --- | --- |
| Operators | `sum`, `softmax`, `layer_norm`, plus `gelu` as the elementwise control |
| Input regimes | D0 normal, D1 random outliers, D2 near-constant, D3 cancellation, D4 heavy-tailed, D5 fixed outlier channels |
| Quantizers | Q0 A8 per-tensor, Q1 A4 per-tensor, Q2 A4 per-group (g = 32) |
| Calibration | Static (frozen from D0), Dynamic (per input) |
| References | FP32 (primary, the supervisor's `NoQuant`), FP64 (numerical floor) |
| Samples | 300 held-out tensors per regime, length 1,024 |

Tier 1 has 4 × 6 × 3 × 2 × 300 = **43,200 quantized evaluations** plus 7,200 reference pairs. That is minutes of CPU time, so compute does not limit the design; rigor does.

**Tiers:**

1. **Tier 1, core matrix:** the matrix above. Required.
2. **Tier 1b, sweeps:** the continuous sweeps in §4. Required, because they carry the crossover argument.
3. **Tier 2, extensions (deferred):** FP8 E4M3 and A8 asymmetric quantizers, a `matvec` operator, and real activations replayed from Llama-3.2-1B. Not run in this phase.
4. **Stage 2 (deferred):** the W4A8 vs W4A16 linear-projection prototype (§10). This is a separate effort and not part of this experiment.

**Boundary:** this is fake quantization (quantize, dequantize, compute in FP32). It measures numerical sensitivity only. It says nothing about speed, energy or real INT4 kernels.

## 2. Research questions and hypotheses

Four questions, each answered by a named analysis in §9. Freeze them in `configs/experiment.yaml` before the final run.

| ID | Question | Hypothesis | Analysis |
| --- | --- | --- | --- |
| RQ1 | Does error under a fixed quantizer change across input regimes? | **H1:** for at least one operator and quantizer, error distributions differ between regimes beyond bootstrap uncertainty. | Per-regime median, p95, 95% CIs |
| RQ2 | Are reductions more sensitive than elementwise ops? (the supervisor's assertion) | **H2:** `sum`, `softmax` and `layer_norm` show higher, more input-dependent amplification than `gelu` in at least one regime. | Amplification factor by operator × regime |
| RQ3 | Does the ranking or feasibility of quantizers depend on the input? | **H3:** the best scheme, or the cheapest scheme within an error budget, differs across regimes or sweep values. | Ranking table, crossover curves |
| RQ4 | Can a cheap input descriptor predict budget violations on unseen inputs? | **H4:** some descriptor beats a constant predictor on held-out data, with an acceptable false-safe rate. | Inspector feasibility (Gate 5) |

H2 is stated as a test, not an assumption. `exp` inside softmax is elementwise and ill-conditioned for large inputs, and GELU may be sensitive near its curved region. Report whatever the data shows.

**Null results are valid results.** If regimes do not differ, Q2 never helps, or no descriptor predicts risk, report that plainly. Do not tune the design after seeing held-out data to manufacture a success.

**This experiment cannot show:**

- that an INT4 or INT8 kernel is faster or more energy-efficient;
- that an operator-level error budget guarantees model perplexity or task accuracy;
- that A4 kernels exist on the target x86-64 or AArch64 devices;
- that inspection is worth its scan, dispatch and switching cost.

Those belong to Stage 2 and the end-to-end evaluation.

## 3. Operators

Three reduction-type operators carry the study, and one elementwise operator is the control that tests the supervisor's assertion. Every operator runs in FP32 on a 1-D float32 tensor of length 1,024 (other lengths only in the §4 sweep).

| ID | Definition | Type | Why it is sensitive |
| --- | --- | --- | --- |
| `sum` | `x.sum(dim=-1)` | Reduction | Error accumulates over n terms. Cancellation: the condition number is Σ\|xᵢ\| / \|Σxᵢ\|, which is huge when the true sum is near zero. |
| `softmax` | `torch.softmax(x, dim=-1)` | Nonlinear + reduction | Max and normalising sum couple every element. `exp` magnifies logit error. Attention depends on it. |
| `layer_norm` | `F.layer_norm(x, (n,), weight=None, bias=None, eps=1e-5)` | Normalisation reduction | Mean and variance couple every element. Division by a small variance amplifies error; outliers dominate the variance. |
| `gelu` (control) | `F.gelu(x)` (exact erf form) | Elementwise | Error stays local to each element. Expected low, roughly input-independent amplification. |
| `matvec` (Tier 2, deferred) | `W @ x`, W fixed FP32 `[1024, 1024]` from `N(0, 1/1024)`, seed-fixed | Reduction (dot products) | The operation that LLM activation quantization actually feeds. Bridges to the Stage 2 W4A8 work. |

**Rules that hold for every operator:**

- Quantize the operator's **input only**. For softmax that means the logits, not the probabilities. For layer_norm that means its input, not its output or affine parameters.
- The operator function, shape, axis, epsilon and FP32 arithmetic are identical across Q0, Q1 and Q2. Only the input representation changes.
- No weight quantization and no output quantization in Tier 1. In `matvec`, W stays FP32.
- Record the exact PyTorch call, including `approximate='none'` for GELU and `weight=None, bias=None` for layer_norm.

**Why `sum` and not only `matvec`:** `sum` is a dot product with an all-ones vector. It isolates accumulation and cancellation with no weight effects, which makes the mechanism easy to read. `matvec` then shows whether the same effect survives with realistic weights.

## 4. Input regimes

Six fixed regimes form the core matrix and five sweeps trace where behaviour changes. A deferred replay tier will later check the findings against real LLM activations. Always generate the FP32 tensor first and quantize afterwards. Never generate quantized values directly.

### 4.1 Fixed regimes (Tier 1)

| ID | Name | Generation (length n = 1,024, FP32) | What it probes |
| --- | --- | --- | --- |
| D0 | Typical normal | `torch.randn(n)` | Baseline, and the static calibration source |
| D1 | Random outliers | D0, then exactly round(0.01 · n) positions (random each sample) set to ±20 with balanced random signs | Clipping and range under unpredictable outliers |
| D2 | Near-constant | `1.0 + 0.001 * torch.randn(n)` | Resolution near a common offset; normalisation of tiny variance |
| D3 | Cancellation | 512 values aᵢ ~ N(0, 1), paired with −aᵢ + δᵢ, δᵢ ~ N(0, 0.01²), then shuffled | Sum close to zero relative to the size of its terms |
| D4 | Heavy-tailed | Student-t with ν = 3, divided by √3 so variance is 1 | Many moderate outliers instead of a few huge ones |
| D5 | Fixed outlier channels | D0, plus 8 channel positions chosen once with a fixed seed. Each channel has a fixed sign and a value of ±20 + N(0, 1), the same positions in every sample. | Systematic outlier channels, the pattern reported in LLM activations |

D1 and D5 differ only in whether outlier positions are random or fixed. That contrast matters: fixed channels are what per-channel or per-group scaling is designed for, while random outliers are not.

These regimes are mechanistic stress tests, not models of real activations. Run the **full matrix** for every operator, even when a regime mainly targets one operator (D3 targets `sum`, D2 targets `layer_norm`). Do not drop combinations after seeing results.

### 4.2 Sweeps (Tier 1b)

Each sweep varies one property and holds the rest at D0 defaults. Use 100 held-out samples per sweep point, all three quantizers, and both calibration policies.

| Sweep | Parameter | Values | Operators |
| --- | --- | --- | --- |
| S1 | Outlier magnitude (D1 form, 1% rate) | 2, 5, 10, 20, 50, 100 | all |
| S2 | Outlier rate (D1 form, magnitude 20) | 0.1%, 0.5%, 1%, 2%, 5% | all |
| S3 | Vector length n (D0 and D3) | 256, 1,024, 4,096, 16,384 | all |
| S4 | Logit scale, x = s · N(0, 1) | 0.5, 1, 2, 4, 8 | `softmax` |
| S5 | Cancellation residual σ_δ (D3 form) | 10⁻³, 10⁻², 10⁻¹, 1 | `sum` (`matvec` deferred) |

For S3, recalibrate static scales on D0 at each length n. That keeps the comparison about length, not about a mismatched calibration. The key output of every sweep is the **crossover point**: the parameter value where the best or cheapest feasible quantizer changes.

### 4.3 Real-activation replay (Tier 2, deferred)

Not run in this phase. Kept here for later.

Capture real tensors from Llama-3.2-1B (Qwen2.5-0.5B if model access is a problem) with forward hooks, then run them through the same harness unchanged.

- **Tensors to capture:** inputs to RMSNorm, pre-softmax attention scores, inputs to the MLP `down_proj`, and residual hidden states.
- **Layers:** one early, one middle and one late layer.
- **Prompts:** WikiText-2, split into disjoint calibration, development and held-out prompt sets.
- **Storage:** FP32 `.pt` files with a manifest recording prompt IDs, layer, tensor name and token positions.
- **Operators for replay:** add `rms_norm`, since Llama uses RMSNorm rather than LayerNorm. Llama-3.2-1B's hidden size of 2,048 divides evenly by the group size of 32.

The replay answers one question: where do real tensors fall on the synthetic descriptor axes, and do the synthetic conclusions hold for them?

## 5. Quantization schemes and calibration

Three controlled quantizers, each run under two calibration policies. Q0 → Q1 changes only bit width, and Q1 → Q2 changes only granularity, so each comparison has one cause.

### 5.1 Common rules

All quantizers are signed, symmetric, uniform, round-to-nearest-even and saturating, with zero point 0:

```math
q = \mathrm{clamp}(\mathrm{round}(x / s),\ -q_{max},\ q_{max}), \qquad \hat{x} = s \cdot q
```

- A8 grid: [−127, 127]. A4 grid: [−7, 7]. The −128 and −8 codes are deliberately unused.
- `x̂` goes into the ordinary FP32 operator. This simulates numerics only; it is not 4-bit packing or integer arithmetic.
- Scale floor: `s = max(raw_s, 1e-12)`. Scales are kept in FP32 for every quantizer.
- Verify with a unit test that `torch.round` rounds halfway values to even on the PyTorch version in use.

### 5.2 Quantizers

| ID | Bits | Granularity | Effective bits per element | Controlled change |
| --- | --- | --- | --- | --- |
| Q0 A8-PT | 8 | One scale per tensor | 8 | Higher-bit baseline |
| Q1 A4-PT | 4 | One scale per tensor | 4 | Bit width only (vs Q0) |
| Q2 A4-PG32 | 4 | One scale per contiguous group of 32 | 4.5 (one FP16 scale per 32 values, cost accounting only) | Granularity only (vs Q1) |

Effective bits per element is the cost axis for "cheapest scheme within budget" in §9.

### 5.3 Calibration policies

| Policy | How scales are set | Role |
| --- | --- | --- |
| **Static** | Absmax over 512 D0-only calibration tensors, frozen before any evaluation: `s = absmax / qmax` per tensor (Q0, Q1) or per group position (Q2). Saved with a checksum. | Deployment-like fixed scales. Tests distribution shift away from "typical" inputs. |
| **Dynamic** | Absmax of the incoming tensor itself, per tensor or per group, recomputed for every input. | Isolates each input's own structure. Upper bound for scale adaptation; its scan cost is a Stage 2 question. |

Run both policies on the same held-out tensors. The paired difference between them separates "the input is hard" from "the input is unlike the calibration data".

### 5.4 Pre-registered predictions

Committed in `reports/predictions.md` before running. Each one is a check on whether the code and the reasoning agree. A miss is a finding to explain, not a failure. The D2 rows were revised at Gate 0 after a numerical check; `reports/predictions.md` records why.

### 5.5 Tier 2 quantizers (deferred)

- **Q3 FP8 E4M3, per tensor:** `s = absmax / 448`; clamp `x / s` to ±448 *before* casting with `.to(torch.float8_e4m3fn)`, because out-of-range casts can produce NaN. A format contrast: non-uniform steps tolerate outliers differently from integers.
- **Q4 A8 asymmetric, per tensor, dynamic:** min–max range with a zero point, grid [0, 255]. Expected to recover D2.

### 5.6 Clipping diagnostics

For every sample and quantizer, record `clip_fraction` (share of |x| above `qmax · s`), `zero_fraction` (share of codes equal to 0) and `input_mse`. These explain errors; they are not error measures themselves.

### 5.7 Implementation

See `src/quantization.py`.

## 6. References, metrics and statistics

Every quantized output is compared with the FP32 reference (the supervisor's baseline). An FP64 reference measures the FP32 floor, so a quantization error smaller than FP32's own rounding is never reported as meaningful.

### 6.1 References

- `y_ref32 = Op(x)` in FP32, computed **once** per input and reused for all quantizers.
- `y_ref64 = Op(x.double())`, also once per input.
- `fp32_floor = metric(y_ref64, y_ref32)`, saved per sample. If a quantization error is within 10× of this floor, flag it as "at numerical floor".
- `rel_l2_floor = ‖y_ref32 − y_ref64‖₂ / ‖y_ref64‖₂`, the noise level of κ's numerator. `kappa_at_floor` is 1 when the output's relative error is within 1,000× of it; such κ values are FP32 noise and are never reported as findings. (Added at Gate 0: comparing VM and laptop pilots, κ differed between machines by up to 73% at 10–30× the floor, 22% at 30–100×, 1.8% at 100–1,000× and 0.4% above 1,000×.)
- Compute all metrics in FP64 from the stored FP32 outputs.

### 6.2 Metrics by operator

| Operator | Primary | Secondary | Diagnostic |
| --- | --- | --- | --- |
| `sum` | Absolute error \|Δy\| | Input-L1-normalised error \|Δy\| / ‖x‖₁ | Cancellation ratio \|Σx\| / ‖x‖₁ |
| `softmax` | KL(p_ref ‖ p_q), natural log, probabilities clamped at 10⁻¹² and renormalised | Normalised L2, max absolute error | Top-1 changed (yes/no), top-2 logit gap |
| `layer_norm` | Normalised L2 | MAE, max absolute error | Input variance; "collapsed" flag when output std < 10⁻³ |
| `gelu` | Normalised L2 | MAE, max absolute error | — |

Normalised L2 is ‖y_q − y_ref‖₂ / (‖y_ref‖₂ + 10⁻¹²). For every vector output, also report **SQNR in dB** = −20 · log₁₀(normalised L2). SQNR is the one metric that compares across vector operators on a single axis. Never use |Δy| / |y| as the primary `sum` metric: it explodes near cancellation.

### 6.3 Amplification factor (tests H2)

```math
\kappa = \frac{\lVert y_q - y_{ref} \rVert_2 \,/\, \lVert y_{ref} \rVert_2}{\lVert \hat{x} - x \rVert_2 \,/\, \lVert x \rVert_2}
```

κ is the empirical condition number: how much the operator magnifies the input's relative quantization error. Expect κ near 1 for `gelu`, and large, regime-dependent κ for `sum` under D3 and for `layer_norm` under D2. Large κ under cancellation is the *intended* signal here, not an artefact.

- Plot κ on a log₁₀ scale; report median and IQR, excluding rows with `kappa_at_floor = 1` (count them).
- Set κ to null (not 0 or infinity) when ‖x̂ − x‖ = 0 or ‖y_ref‖ < 10⁻¹², and count those cases.

### 6.4 Statistics

The unit of analysis is the **input tensor**, not the element. 300 tensors per stratum are 300 observations, not 307,200.

- Per operator × regime × quantizer × policy: median, IQR, mean, p95 and p99.
- Paired differences on the same tensors: Q1 − Q0 (bit width), Q2 − Q1 (granularity), dynamic − static (calibration).
- 95% bootstrap confidence intervals: 2,000 resamples of tensors within each stratum, with a fixed seed.
- Errors spanning orders of magnitude: analyse and plot on log₁₀ scale; compare medians, not means.
- Optional p-values: Wilcoxon signed-rank on paired differences, with Holm correction across all comparisons. Effect sizes and CIs come first; p-values are secondary.
- Never drop, truncate or replace NaN, infinite or extreme values. Flag them and report counts.

## 7. Data splits and reproducibility

Four disjoint splits keep calibration, tuning and reporting apart. Only the held-out split produces numbers that go in the report.

| Split | Size | Regimes | Allowed use |
| --- | --- | --- | --- |
| Pilot | 100 per regime | D0–D5 | Debugging, sanity plots, timing. Never reported. Static scales come from a separate throwaway pilot-calibration set. |
| Calibration | 512 tensors | D0 only | Static scales, computed once and frozen |
| Development | 200 per regime; 50 per sweep point | D0–D5, S1–S5 | Choosing error budgets; fitting and selecting inspector rules |
| Held-out | 300 per regime; 100 per sweep point | D0–D5, S1–S5 | All reported results, evaluated once |

The same held-out FP32 tensors go through every quantizer and both calibration policies, so all comparisons are paired.

### 7.1 Seeds

- Derive each sample's seed from its identity, not from a global counter: `seed = int(sha256(f"{split}|{stratum_key}|{sample_id}").hexdigest()[:16], 16)` (64 bits, so no collisions across the ~10,000 tensors). The key format is defined in `configs/experiment.yaml`.
- Generate each tensor with its own `torch.Generator().manual_seed(seed)`. Never rely on the global RNG state.
- The D5 outlier channel positions and signs come from one fixed seed recorded in the config.
- Adding a regime or sample then never changes any existing tensor.

### 7.2 Determinism and environment

- Run on CPU. Set `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True)`, and keep autocast off. FP32 reductions can change with thread count, which would leak into the FP32 floor.
- Normals are drawn in FP64 and rounded to FP32, so input tensors are bit-identical across machines (a pinned-hash test checks this). Operator outputs (`exp`, `erf`, vectorised reductions) still differ in the last bits between CPUs and operating systems, so runs are bit-identical only on the same machine; primary metrics agree across machines to about 1e-4 relative. Reported numbers come from the VM only. The manifest records PyTorch's CPU kernel capability.
- Dependencies are pinned in `uv.lock` (Python 3.12, CPU-only PyTorch).
- A determinism test regenerates 10 tensors per regime and checks they are bit-identical.

### 7.3 Manifest

Each run writes `artifacts/<run_id>/manifest.json` with:

- UTC timestamp, git commit and a dirty-tree flag (refuse to run final splits on a dirty tree);
- VM CPU model, ISA, core count, OS and kernel;
- Python, PyTorch and NumPy versions, and the thread setting;
- config checksum, static-scale file checksum and the rounding-mode test result;
- split sizes, seed rule, and counts of completed, failed and non-finite rows.

## 8. Execution procedure

The work runs through gates. Each gate ends with a committed output, and no gate starts until the previous one has passed. All runs execute on the VM.

1. **Gate 0: set up and pre-register.**
   - Write `configs/experiment.yaml` with every factor, level, metric, split size and seed rule from §3–§7.
   - Commit the predictions to `reports/predictions.md`.
   - Implement the generators, quantizers, operators and metrics; pass every unit test in §8.1.
   - Run the pilot (100 per regime), check sanity plots, and fix bugs. Pilot numbers are never reported.
   - Done when: config, predictions and passing tests are committed, and the pilot shows no missing or non-finite rows.
2. **Gate 1: freeze static scales.** Generate the 512 D0 calibration tensors, compute Q0, Q1 and Q2 scales, and save them to `artifacts/calibration/scales.pt` with a checksum. Confirm that no development or held-out tensor touched them.
3. **Gate 2: core run (Tier 1).** For each split (development, then held-out), regime, sample, operator, quantizer and policy: generate x, compute both references once, extract descriptors from x alone, quantize, run the operator, compute metrics, and write one row. Assert that the expected row count matches exactly.
4. **Gate 3: sweeps (Tier 1b).** Run S1–S5 through the same loop with `sweep_id` and `sweep_value` filled in. Recalibrate static scales per length for S3 only.
5. **Gate 4: analysis.** Produce the §9 tables and figures from the held-out rows with one script, `src/analysis.py`, that reads only the raw CSVs. Compare results against the pre-registered predictions.
6. **Gate 5: inspector feasibility.** Fix the error budgets on development data, fit the rules, freeze them, then evaluate once on held-out data (§9.3).
7. **Gate 6: Tier 2 extensions (deferred).** Real-activation replay, Q3/Q4 and `matvec`. Not run in this phase.

### 8.1 Unit tests (must pass before Gate 1)

The scale-freeze and completeness tests need Gate 1–2 outputs; they are written now and skip until those outputs exist.

- [x] Determinism: the same seed and regime give bit-identical tensors.
- [x] Generators: shape, dtype FP32, and empirical statistics within tolerance (D0 mean ≈ 0 and variance ≈ 1; D3 cancellation ratio < 0.01; D5 channels fixed across samples).
- [x] Quantizer ranges: Q0 codes in [−127, 127], Q1 and Q2 codes in [−7, 7]; `x̂ == codes · s` within FP32 tolerance.
- [x] Rounding: halfway values round to even.
- [x] Grouping: Q2 uses exactly 32 contiguous groups of 32, with no off-by-one.
- [x] Clipping: a constructed input with known out-of-range values gives the expected clip fraction.
- [x] Zero input: no NaN, no division by zero.
- [x] Identity: bypassing quantization gives zero error on every metric.
- [x] Softmax output is finite, non-negative and sums to 1 within 10⁻⁶.
- [ ] Scale freeze: the static-scale checksum is identical in development and held-out runs (skips until Gate 2).
- [x] Leakage: no held-out tensor appears in calibration or development.
- [ ] Completeness: every expected (split, regime/sweep point, sample, operator, quantizer, policy) row appears exactly once (skips until Gate 2).

### 8.2 Raw data schema

One long-format row per (split, regime or sweep point, sample, operator, quantizer, policy), written to `artifacts/<run_id>/raw/{development,heldout}.csv`. Fields that do not apply are empty, not zero.

```csv
run_id,git_commit,split,regime,sweep_id,sweep_value,sample_id,seed,operator,quantizer,calib_policy,bits,granularity,group_size,effective_bits,n,input_mean,input_var,input_absmax,input_abs_p99,max_over_p99,kurtosis,cancellation_ratio,top2_gap,clip_fraction,zero_fraction,input_mse,input_rel_l2,abs_error,l1_norm_error,norm_l2,sqnr_db,mae,max_abs_error,softmax_kl,top1_changed,collapsed,kappa,fp32_floor,at_floor,rel_l2_floor,kappa_at_floor,ref_finite,quant_finite
```

### 8.3 Repository layout

```
quant-sensitivity/
├── CLAUDE.md                # context for Claude Code
├── README.md
├── pyproject.toml, uv.lock  # pinned environment
├── configs/experiment.yaml
├── docs/experiment_plan.md  # this plan
├── src/
│   ├── config.py            # config loading, determinism, design enumeration
│   ├── distributions.py     # D0–D5, sweeps, seed rule
│   ├── quantization.py      # Q0–Q2, static/dynamic calibration
│   ├── operators.py
│   ├── metrics.py           # incl. SQNR, kappa, FP64 floor, CSV schema
│   ├── features.py          # inspector descriptors
│   ├── run_experiment.py    # gates 0–3 (gate 0 = pilot)
│   ├── sanity.py            # gate 0 pilot plots and checks
│   ├── analysis.py          # gate 4: tables + figures from raw CSV only
│   └── inspector.py         # gate 5
├── tests/
├── artifacts/<run_id>/{raw,summary,plots,manifest.json}
└── reports/{predictions.md,findings.md}
```

## 9. Analysis, figures and decision criteria

The headline result is the **cheapest-feasible-scheme map**: for each operator and regime, the lowest-cost quantizer that stays within budget. If that map is not constant, adaptivity is justified. Every other figure explains why the map looks the way it does.

### 9.1 Error budgets

The project PDFs set a whole-model budget (ΔPPL ≤ 0.5) but no operator-level one, so budgets must be agreed with the supervisor and frozen on **development data** before the held-out analysis. Candidates to propose:

| Operator | Candidate budget τ (to confirm) |
| --- | --- |
| `layer_norm`, `gelu` | SQNR ≥ 30 dB (strict) and ≥ 20 dB (loose) |
| `softmax` | KL ≤ 10⁻³ nats and top-1 unchanged |
| `sum` | Input-L1-normalised error ≤ 10⁻³ |

Run every budget-dependent analysis at both the strict and loose budget. A conclusion that flips between them must be reported as budget-dependent.

### 9.2 Feasibility and ranking

- A scheme is **feasible** for an operator × regime × policy cell when at least 95% of held-out tensors meet τ. Report that fraction with a bootstrap CI.
- The **cheapest feasible** scheme has the fewest effective bits per element (§5.2). Cells where nothing is feasible are labelled "none".
- Count **ranking flips**: pairs of regimes, under the same operator and policy, whose median-error ordering of Q0–Q2 differs.
- For each sweep, locate the **crossover**: the parameter value where the cheapest feasible scheme changes. Report it with a bootstrap interval.

### 9.3 Inspector feasibility (Gate 5)

1. Label each (tensor, operator, quantizer) as **unsafe** when its error exceeds τ.
2. Use only descriptors computable from x before any decision: absmax, abs p99, max/p99, mean, variance, kurtosis, top-2 gap, and clip and zero fractions under the fixed Q1 scales. Descriptors that need the reference output are forbidden.
3. Fit candidate rules on the training half of the development split: single-feature thresholds, then a depth-2 decision tree. Select on the other half.
4. Freeze the rule, then evaluate once on held-out data.
5. Compare against four baselines:
   - always aggressive (Q1);
   - always conservative (Q0);
   - the best **static per-operator** choice (the strong baseline);
   - an offline oracle that picks the cheapest safe scheme per tensor (an upper bound, not deployable).
6. Report the false-safe rate (aggressive chosen but unsafe), false-fallback rate (conservative chosen but aggressive was safe), AUROC, confusion matrix, and average effective bits versus the oracle.
7. Record each descriptor's cost: one pass over x, a sort, or a full quantization. Percentiles and kurtosis are not free at runtime.

### 9.4 Figures and tables

| File | Shows | Answers |
| --- | --- | --- |
| `F1_error_heatmap.png` | Median primary metric, operator × regime × quantizer, one panel per policy | RQ1 |
| `F2_amplification.png` | κ (log scale) by operator × regime, `gelu` as control | RQ2 |
| `F3_paired_effects.png` | Q1 − Q0, Q2 − Q1 and dynamic − static differences with CIs | RQ1, RQ3 |
| `F4_error_tails.png` | ECDFs per regime, p95 marked | RQ1 |
| `F5_sweep_crossovers.png` | Median error vs sweep value per quantizer, CI bands, crossover marked | RQ3 |
| `F6_cheapest_feasible_map.png` | Operator × regime grid coloured by cheapest feasible scheme, strict and loose budgets | RQ3, headline |
| `F7_error_vs_descriptors.png` | Error against the best descriptors | RQ4 |
| `F8_inspector.png` | Confusion matrix and ROC against the baselines | RQ4 |
| `T1_summary.csv` | Median, IQR, p95, p99, CI per stratum | All |
| `T2_predictions_check.md` | Each prediction: confirmed or not, with the number | Sanity |

Every figure states the metric and its units, the split, the number of tensors, the calibration policy and the run ID. Never put two different metrics on one unlabelled axis.

### 9.5 Decision criteria

**Evidence for adaptive quantization** requires H1 plus at least one of:

- the cheapest-feasible map differs across regimes or sweep values under both budgets; or
- an inspector rule beats the best static per-operator choice on held-out data at a false-safe rate the supervisor accepts.

**Stop or rethink** if any of these holds:

- descriptors predict no better than a constant rule;
- the aggressive path is unsafe for nearly all inputs, or safe for nearly all;
- an apparent gain disappears without leakage or without retuning on held-out data.

A clean negative result is still a valid finding for the project.

## 10. Stage 2 bridge: W4A8 vs W4A16 (deferred)

Not part of this phase; kept for later. Stage 2 is a separate effort, started only if §9.5 supports it. It moves from simulated numerics to a real kernel on one CPU.

- **Target:** one real Llama-3.2-1B linear or MLP projection on one identified x86-64 or AArch64 device.
- **Paths:** fixed W4 weights with A8 activations (aggressive) versus the same W4 weights with FP16/BF16 activations (conservative). An FP16/BF16 end-to-end model is the quality anchor only.
- **Controls:** identical packed weights, scales, shape, layout, bias and accumulation where the kernels allow; record any difference.
- **Inputs:** the activations captured in Tier 2, with the same prompt splits.
- **Measure:** operator error, model perplexity on WikiText-2, kernel wall-clock time, quantize/pack cost, inspector cost, dispatch cost, memory and code size.
- **Break-even rule** from the supervisor review: adaptation pays only when the reuse count H, the per-execution saving ΔT, the inspection cost I and the switch cost S satisfy:

```math
H \cdot \Delta T > I + S
```

How Stage 1 feeds Stage 2:

- the best descriptor from Gate 5 becomes the first inspector feature;
- its measured cost is the first estimate of I;
- the Tier 2 replay shows how often real activations sit in risky regions, which bounds the possible gain.

If fixed-shape kernels run at the same latency regardless of input values, do not claim input-dependent latency. The input then changes numerical risk only, and the gain must come from choosing a different valid path.

The proposal's targets (controller overhead < 5%, efficiency ≥ 1.2×, ΔPPL ≤ 0.5) apply at the system and model level in Stage 2. They are not operator-level error budgets.

## 11. Workflow: VM, laptop and the report

Code lives in this GitHub repo. Every run executes on the VM; the report is written on the laptop from results copied back with `scp`.

| Where | Holds | Used for |
| --- | --- | --- |
| GitHub repo | Code, config, tests, `CLAUDE.md`, summaries, figures, manifests | Version control; the single source of truth for code |
| VM | Python environment, raw runs under `artifacts/` | Executing the experiment |
| Laptop | Results pulled from the VM, report, slides | Interpreting results, writing |

### 11.1 Steps

1. **Set up the VM:** clone the repo, run `uv sync` (pinned CPU PyTorch), then `uv run pytest`.
2. **Work gate by gate.** Review each gate's diff and test output, then commit with a message naming the gate.
3. **Run long jobs in `tmux`**, logging to `artifacts/<run_id>/run.log`, so an SSH drop does not kill a run.
4. **Pull results:** `scp -r <user>@<vm-host>:~/quant-sensitivity/artifacts/<run_id> ./artifacts/`. Raw CSVs and logs stay out of git; summary tables, figures and manifests can be committed.
5. **Log deviations.** Any change to the plan after Gate 0 goes in a "Deviations" section of `reports/findings.md`, with the date and reason.

## 12. Schedule, roles and open questions

### 12.1 Schedule

| Day | Work | Done when |
| --- | --- | --- |
| 1 | Freeze config and predictions; create repo, environment, `CLAUDE.md` | Config and predictions committed |
| 2 | Generators D0–D5 and sweeps, with tests | Statistics tests pass |
| 3 | Quantizers, both calibration policies, operators, metrics, with tests | All §8.1 tests pass |
| 4 | Pilot run on the VM and debugging (Gate 0) | No missing or non-finite rows |
| 5 | Freeze static scales (Gate 1); core run (Gate 2) | Row count exact; manifest complete |
| 6 | Sweeps (Gate 3) | Row count exact |
| 7–8 | Analysis and figures (Gate 4); predictions check | F1–F6, T1, T2 generated by one script |
| 9 | Agree budgets on development data; fit inspector rules | Rules frozen and committed |
| 10 | Held-out inspector evaluation (Gate 5); draft `findings.md` | F7, F8 and findings drafted |
| 11 | Report and supervisor presentation | Both reviewed by the team |

### 12.2 Roles

| Role | Owns |
| --- | --- |
| Inputs and sensitivity | Generators, sweeps, descriptors |
| Quantization | Q0–Q2, calibration, quantizer tests |
| Inspector | Budgets, rules, baselines, descriptor cost |
| Evaluation | Metrics, statistics, figures, manifests, report |

### 12.3 Open choices (plan defaults until the supervisor says otherwise)

- Static and dynamic calibration are both primary.
- Q2 group size is 32.
- Error budgets are the §9.1 candidates.

### 12.4 Definition of done (Tier 1 and 1b)

- [ ] Config and predictions frozen
- [ ] All §8.1 unit tests pass
- [ ] Calibration, development and held-out splits reproducible from seeds
- [ ] Complete held-out raw data with descriptors and diagnostics, zero unexplained non-finite rows
- [ ] F1–F8 and T1–T2 generated by `analysis.py` from raw CSVs only
- [ ] Every prediction checked
- [ ] RQ1–RQ4 answered in `findings.md`, including null results
- [ ] Another team member reruns one gate from a clean clone and gets identical numbers
- [ ] Supervisor has reviewed the findings
