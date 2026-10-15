# quant-sensitivity

NeuroMinds Stage 1: an activation-only, fake-quantization study of how operator error
(`sum`, `softmax`, `layer_norm`, with `gelu` as an elementwise control) depends on the
input distribution, and whether a cheap input descriptor predicts which quantizer an
input needs.

- Protocol: [docs/experiment_plan.md](docs/experiment_plan.md)
- Context for Claude Code: [CLAUDE.md](CLAUDE.md)
- Work is tracked gate by gate in GitHub issues.

## Pulling results from the VM

Runs execute on the VM under `artifacts/<run_id>/`. Copy a finished run back with:

```bash
scp -r <user>@<vm-host>:~/quant-sensitivity/artifacts/<run_id> ./artifacts/
```
