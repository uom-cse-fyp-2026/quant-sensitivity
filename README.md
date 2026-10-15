# quant-sensitivity

NeuroMinds Stage 1: an activation-only, fake-quantization study of how operator error
(`sum`, `softmax`, `layer_norm`, with `gelu` as an elementwise control) depends on the
input distribution, and whether a cheap input descriptor predicts which quantizer an
input needs.

- Protocol: [docs/experiment_plan.md](docs/experiment_plan.md)
- Context for Claude Code: [CLAUDE.md](CLAUDE.md)
- Work is tracked gate by gate in GitHub issues.

## Running on the VM

```bash
git clone git@github.com:uom-cse-fyp-2026/quant-sensitivity.git
cd quant-sensitivity
uv sync
uv run pytest
```

Gate 0 pilot (100 tensors per regime, about a minute including plots). Use tmux so an SSH
drop does not kill the run:

```bash
tmux new -s pilot
uv run python -m src.run_experiment --config configs/experiment.yaml --gate 0
```

The run writes `artifacts/pilot-<UTC time>-<commit>/` with:

- `raw/pilot.csv`: one row per (regime, sample, operator, quantizer, policy), 14,400 rows
- `manifest.json`: commit, machine, versions, checksums, row and non-finite counts
- `run.log`
- `plots/pilot_histograms.png`, `summary/regime_stats.csv`, `summary/pilot_medians.csv`,
  `summary/pilot_predictions_check.md`

The command exits non-zero if any row is missing or non-finite. Pilot numbers are for
debugging only and are never reported.

## Pulling results from the VM

Copy a finished run back to the laptop with:

```bash
scp -r <user>@<vm-host>:~/quant-sensitivity/artifacts/<run_id> ./artifacts/
```

Raw CSVs and logs stay out of git.
