# quant-sensitivity

NeuroMinds: how much precision does an operator lose under different quantization schemes,
and does that depend on the input? The experiment follows the supervisor's description:

1. Operators `sum`, `softmax` and `layer_norm` (reductions), with `gelu` (elementwise) to
   check that reductions are more sensitive.
2. Six contrasting FP32 input distributions, 300 samples each; baseline `Y = Op(A, NoQuant)`.
3. Three schemes: Q0 A8 per-tensor, Q1 A4 per-tensor, Q2 A4 per-group-32.
4. `Y_q = Op(A, Q)` for every sample and scheme.
5. Quantify `Y_q − Y`: error tables, an amplification check, and how the best scheme changes
   with the input.

Full description: [docs/experiment_plan.md](docs/experiment_plan.md).

This is fake (simulated) quantization: each input is quantized to integer codes and
immediately dequantized, then the same FP32 operator runs on it. It isolates the numerical
error caused by the quantized input; it does not measure speed or energy.

## Results

From the run on the VM ([run-20261015T175122Z-3e45813](artifacts/run-20261015T175122Z-3e45813/summary/summary.md):
21,600 comparisons, none missing or non-finite; medians over 300 samples).

- **4 bits costs 18–330× more error than 8 bits** on normal inputs. On those inputs,
  `layer_norm` keeps 42 dB (0.77% error) at A8 but 17 dB (14%) at A4. Per-group scales
  (Q2) cut the 4-bit error by 30–55%.
- **The loss depends on the input.** With the same 4-bit scheme, `layer_norm` error
  ranges from 13% to 100% across inputs, and `softmax` KL spans six orders of magnitude.
- **The sensitivity claim holds for `sum` and `layer_norm`, not `softmax`.** Median
  amplification κ reaches 144 for `sum` on cancelling inputs and 599 for `layer_norm` on
  near-constant inputs, while `gelu` (elementwise) stays between 0.93 and 1.29. `softmax`
  does not amplify error, but at 4 bits its top-1 entry changes in up to 80% of samples.
- **No single scheme is best.** On fixed outlier channels, 4-bit per-group beats 8-bit
  per-tensor for `softmax` (33× lower KL). Under cancellation, per-tensor beats per-group
  for `sum`. On near-constant inputs, no scheme preserves the `layer_norm` output.

| Figure | Shows |
| --- | --- |
| [F1_inputs.png](artifacts/run-20261015T175122Z-3e45813/plots/F1_inputs.png) | The six input distributions |
| [F2_error_heatmap.png](artifacts/run-20261015T175122Z-3e45813/plots/F2_error_heatmap.png) | Median error per distribution and scheme, per operator |
| [F3_amplification.png](artifacts/run-20261015T175122Z-3e45813/plots/F3_amplification.png) | κ by operator and distribution |

Caveats: the inputs are synthetic stress tests, not real model activations. κ values at FP32
rounding level are excluded (all `softmax` rows on D1, some on D4 and D5).

## Code

| File | What it does |
| --- | --- |
| `configs/experiment.yaml` | Operators, distributions, schemes, sample count, seeds |
| `src/distributions.py` | Input distributions D0–D5 |
| `src/quantization.py` | Schemes Q0–Q2 |
| `src/operators.py` | The four operators and the FP32 baseline |
| `src/metrics.py` | Error measures and the amplification factor κ |
| `src/run_experiment.py` | Runs everything and writes the raw CSV |
| `src/analysis.py` | Tables and figures from the raw CSV |

## Running on the VM

The VM is shared, so everything stays inside your home directory: no sudo, no system
packages, no global config. uv and just install to `~/.local/bin`, and the project uses uv's
own standalone Python 3.12.

One-time setup:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv tool install rust-just
```

```bash
git clone git@github.com:uom-cse-fyp-2026/quant-sensitivity.git && cd quant-sensitivity && just pull
```

Then, from `~/quant-sensitivity`:

| Command | What it does |
| --- | --- |
| `just pull` | Get the latest code and update the environment |
| `just test` | Run the unit tests |
| `just run` | Run the experiment and its analysis inside tmux (about 30 s) |
| `just pull-run` | Pull, test, then run |
| `just attach` (or `just fg`) | Bring the run back to the foreground |
| `just status` | Is a run going? Shows the end of the newest run log |
| `just stop` | Stop a running job |
| `just push` | Commit and push the newest finished run |
| `just analyze` | Redo the tables and figures for the newest run |
| `just runs` | List the runs on this machine |

In tmux, press **Ctrl+B then D** to detach and close SSH whenever you like; `just attach`
brings it back. When the run finishes, the window shows the exit status and waits for Enter.

## What a run writes

`artifacts/run-<UTC time>-<commit>/`:

- `raw/results.csv`: one row per (distribution, sample, operator, scheme), 21,600 rows
- `manifest.json`: commit, machine, versions, row and non-finite counts
- `run.log`
- `summary/summary.md`: the report tables (loss of precision, sensitivity check, scheme ranking)
- `summary/summary.csv`: medians, IQR, p95 and bootstrap CIs
- `plots/F1_inputs.png`, `plots/F2_error_heatmap.png`, `plots/F3_amplification.png`

The run exits non-zero if any row is missing or non-finite. `just push` gzips the raw CSV,
commits the run folder, rebases onto any newer code and pushes.
