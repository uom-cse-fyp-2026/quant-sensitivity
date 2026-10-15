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
packages, no global config. uv and just install to `~/.local/bin`, and the project uses
uv's own standalone Python 3.12, never the system interpreter.

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

### Commands

Run `just` to list them. From `~/quant-sensitivity`:

| Command | What it does |
| --- | --- |
| `just pull` | Get the latest code and update the environment |
| `just test` | Run the unit tests |
| `just run` | Run the pilot (gate 0) inside tmux; `just run 1` for gate 1, and so on |
| `just pull-run` | Pull, update, test, then run |
| `just attach` (or `just fg`) | Bring the running job back to the foreground |
| `just status` | Is a run going? Shows the end of the newest run log |
| `just log` | Follow the newest run log live |
| `just stop` | Stop a running job |
| `just push` | Commit and push the newest finished run |
| `just freeze` | Gate 1: freeze the static calibration scales |
| `just push-scales` | Commit and push the frozen scales |
| `just runs` | List the runs on this machine |

Runs happen in a tmux session, so press **Ctrl+B then D** to detach and close SSH whenever you
like. Reconnect and type `just attach` to see it again. When the run finishes, the window
shows the exit status and waits for Enter.

### What a pilot run writes

`artifacts/pilot-<UTC time>-<commit>/` with:

- `raw/pilot.csv`: one row per (regime, sample, operator, quantizer, policy), 14,400 rows
- `manifest.json`: commit, machine, versions, checksums, row and non-finite counts
- `run.log`
- `plots/pilot_histograms.png`, `summary/regime_stats.csv`, `summary/pilot_medians.csv`,
  `summary/pilot_predictions_check.md`

The run exits non-zero if any row is missing or non-finite. Pilot numbers are for
debugging only and are never reported.

### Gate 1: freezing the static scales

`just freeze` computes the static absmax scales from the 512 D0 calibration tensors at each
length (1,024 for the core runs; 256, 4,096 and 16,384 for the S3 sweep) and saves them once
to `artifacts/calibration/scales.pt`. It takes a few seconds, refuses to run on a dirty git
tree, and refuses to overwrite scales that already exist. The manifest records the file's
SHA-256 (which every later run must match) and a content checksum that is reproducible on
any machine. Then `just push-scales`.

### Pushing results

`just push` gzips the raw CSV, commits the run folder (run outputs are gitignored, so it adds
them with `-f`), rebases onto any newer code, and pushes. To copy a run without git instead:

```bash
scp -r <user>@<vm-host>:~/quant-sensitivity/artifacts/<run_id> ./artifacts/
```
