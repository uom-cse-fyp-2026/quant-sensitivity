# quant-sensitivity

NeuroMinds Stage 1: an activation-only, fake-quantization study of how operator error
(`sum`, `softmax`, `layer_norm`, with `gelu` as an elementwise control) depends on the
input distribution, and whether a cheap input descriptor predicts which quantizer an
input needs.

- Protocol: [docs/experiment_plan.md](docs/experiment_plan.md)
- Context for Claude Code: [CLAUDE.md](CLAUDE.md)
- Work is tracked gate by gate in GitHub issues.

## Running on the VM

The VM is shared, so everything stays inside your home directory: no sudo, no system
packages, no global config. uv installs to `~/.local/bin`, and the project uses uv's own
standalone Python 3.12 (under `~/.local/share/uv/python`), never the system interpreter.

One-time uv install:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone git@github.com:uom-cse-fyp-2026/quant-sensitivity.git
cd quant-sensitivity
uv sync --locked
uv run pytest
```

### Running in the background

Long runs go through `scripts/run_bg.sh`, which detaches the command from the SSH session
(`nohup` + `setsid`), so you can close the connection while it runs. Output goes to
`logs/<name>-<UTC time>.log`, ending with the exit status.

Gate 0 pilot (100 tensors per regime, about a minute including plots):

```bash
scripts/run_bg.sh pilot uv run python -m src.run_experiment --config configs/experiment.yaml --gate 0
```

Check on it later (after reconnecting, from `~/quant-sensitivity`):

```bash
scripts/status.sh pilot
```

Stop it early with `kill $(cat logs/pilot.pid)`.

The run writes `artifacts/pilot-<UTC time>-<commit>/` with:

- `raw/pilot.csv`: one row per (regime, sample, operator, quantizer, policy), 14,400 rows
- `manifest.json`: commit, machine, versions, checksums, row and non-finite counts
- `run.log`
- `plots/pilot_histograms.png`, `summary/regime_stats.csv`, `summary/pilot_medians.csv`,
  `summary/pilot_predictions_check.md`

The run exits non-zero if any row is missing or non-finite. Pilot numbers are for
debugging only and are never reported.

## Sending results back

When `scripts/status.sh` shows `exit status 0`, commit and push the newest run folder (raw
CSVs are gzipped first; run outputs are gitignored, so the script adds them with `-f`):

```bash
scripts/push_results.sh
```

Or copy a run to the laptop without git:

```bash
scp -r <user>@<vm-host>:~/quant-sensitivity/artifacts/<run_id> ./artifacts/
```
