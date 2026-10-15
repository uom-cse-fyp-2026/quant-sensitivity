# NeuroMinds quantization sensitivity experiment

FYP: Compiler-Directed Adaptive Quantization Framework for Edge LLMs (Univ. of Moratuwa).
This repo is Stage 1: an activation-only, fake-quantization study of how operator error
depends on input distribution. Full protocol: docs/experiment_plan.md (source of truth).

## Claim under test
No single static quantizer is best across inputs; a cheap input descriptor predicts
which quantizer an input needs. Null results are valid. Never tune on held-out data.

## Design (do not change without updating docs/experiment_plan.md and reports/findings.md "Deviations")
- Operators: sum, softmax, layer_norm (weight=None, bias=None, eps=1e-5), gelu (control).
- Deferred for now: everything LLM-related and Tier 2 (Llama activation replay, matvec, rms_norm, FP8/asymmetric quantizers) and Stage 2. Do not implement unless asked.
- Regimes: D0 normal, D1 random outliers, D2 near-constant, D3 cancellation, D4 Student-t(3)/sqrt(3), D5 fixed outlier channels. Sweeps S1-S5.
- Quantizers: Q0 A8 per-tensor, Q1 A4 per-tensor, Q2 A4 per-group g=32. Symmetric, round-half-even, saturating.
- Calibration: static (absmax over 512 D0 tensors, frozen, checksummed) and dynamic (per input).
- References: FP32 primary, FP64 floor. Metrics computed in FP64.
- Splits: pilot / calibration (D0 only) / development / held-out. Held-out evaluated once.

## Rules
- Generate FP32 first, then quantize. Per-sample torch.Generator seeded from sha256(split|regime|sample_id).
- CPU only, torch.set_num_threads(1), deterministic algorithms, no autocast.
- One long-format CSV row per (split, regime/sweep point, sample, operator, quantizer, policy). Inapplicable fields empty, never 0.
- Never drop or replace NaN/inf. Stop and report.
- Unit of analysis is the input tensor. Bootstrap CIs over tensors.
- Figures come only from src/analysis.py reading raw CSVs.
- Work gate by gate (plan section 8). Stop at the end of each gate for human review before continuing.

## Where things run
- Code is written and unit-tested locally or on the VM; all reported runs (pilot, Gates 1-6) execute on the VM.
- Long runs go in tmux and log to artifacts/<run_id>/run.log.
- Raw CSVs stay out of git (artifacts/*/raw/ is gitignored). Summaries, figures and manifests are committed.
- Results come back to the laptop with scp (see README "Pulling results from the VM").

## Commands
- Environment: uv sync (Python 3.12, CPU-only torch, pinned in uv.lock)
- Tests: uv run pytest -q
- Run: uv run python -m src.run_experiment --config configs/experiment.yaml --gate <n>
- Analysis: uv run python -m src.analysis --run <run_id>

## Code map
- src/config.py: config loading, deterministic torch setup, design enumeration (strata, seeds, expected rows)
- src/distributions.py: D0-D5 generators; random-draw order is part of the definition, freeze it after Gate 1
- src/quantization.py, src/operators.py, src/metrics.py (incl. CSV schema), src/features.py
