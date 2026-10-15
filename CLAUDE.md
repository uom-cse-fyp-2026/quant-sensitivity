# NeuroMinds quantization sensitivity experiment

FYP: Compiler-Directed Adaptive Quantization Framework for Edge LLMs (Univ. of Moratuwa).
This repo is one experiment: fake quantization of operator inputs, measuring how much
precision each operator loses and how that depends on the input. Description:
docs/experiment_plan.md.

## Scope: the supervisor's five steps, nothing more
1. 2-3 sensitive operators (sum, softmax, layer_norm) plus gelu, to verify that reductions
   are more sensitive than elementwise ops.
2. Contrasting FP32 input distributions D0-D5; baseline Y = Op(A, NoQuant) in FP32.
3. Schemes Q0 A8 per-tensor, Q1 A4 per-tensor, Q2 A4 per-group-32 (symmetric, round-half-even,
   saturating, scale = the input's own absmax).
4. Y_q = Op(A_hat) for every sample and scheme.
5. Quantify Y_q - Y (src/metrics.py) and summarise (src/analysis.py).
Do not add calibration splits, sweeps, inspectors, LLM work or other extras unless asked.

## Rules
- Generate FP32 inputs first (normals drawn in FP64, rounded to FP32), then quantize.
  Per-sample torch.Generator seeded from sha256(regime|sample_id). Pinned-hash tests guard this.
- CPU only, torch.set_num_threads(1), deterministic algorithms.
- One CSV row per (distribution, sample, operator, scheme). Inapplicable fields empty, never 0.
- Never drop or replace NaN/inf; the run fails and reports them.
- The unit of analysis is the sample. Bootstrap CIs over samples.
- kappa rows flagged at_fp32_noise are excluded from kappa summaries.
- Tables and figures come only from src/analysis.py reading the raw CSV.

## Where things run
- Reported runs execute on the VM, which is shared: user-local commands only (no sudo, no global config).
- VM commands go through the justfile: just pull, just test, just run, just attach, just status, just push.
  just run starts the job in tmux session "qs" (Ctrl+B then D to detach).

## Commands
- Tests: uv run pytest
- Run: uv run python -m src.run_experiment   (add --samples 3 for a quick check)
- Analysis only: uv run python -m src.analysis --run artifacts/<run_id>
