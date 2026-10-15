"""Run the experiment (plan §5). Runs on the VM.

    uv run python -m src.run_experiment

For every distribution D0-D5 and sample A: compute the FP32 baseline Y = Op(A) once per
operator, quantize A with each scheme Q0-Q2, compute Y_q = Op(A_hat), and write one row of
errors per (distribution, sample, operator, scheme). Then run the analysis.

Writes artifacts/<run_id>/: raw/results.csv, manifest.json, run.log, summary/, plots/.
Exits non-zero if any row is missing or non-finite.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.config import DEFAULT_CONFIG, ROOT, configure_torch, expected_row_keys, file_checksum, load_config, planned_samples
from src.distributions import generate
from src.metrics import CSV_FIELDS, evaluate, make_row
from src.operators import apply, baselines
from src.quantization import quantize, quantizers_from_config

log = logging.getLogger("run_experiment")

# Float-valued metric columns checked for NaN before a row is accepted.
FLOAT_METRICS = ("abs_error", "l1_norm_error", "norm_l2", "sqnr_db", "mae", "max_abs_error",
                 "softmax_kl", "kappa", "input_rel_l2", "fp32_noise")


def git_state() -> tuple[str, bool]:
    """Current commit hash and whether tracked files have uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "HEAD") or "unknown", bool(git("status", "--porcelain", "--untracked-files=no"))


def machine_info() -> dict:
    """Hardware and library versions for the manifest, so results can be traced to a machine."""
    cpu = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((l.split(":", 1)[1].strip() for l in cpuinfo.read_text().splitlines()
                    if l.startswith("model name")), cpu)
    return {
        "cpu_model": cpu, "isa": platform.machine(), "cpu_count": os.cpu_count(),
        "os": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "numpy": np.__version__, "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def setup_logging(out_dir: Path) -> None:
    """Log to both the console and the run folder's run.log."""
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(out_dir / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def sample_rows(ids: dict, regime: str, sample_id: int, seed: int, x: torch.Tensor,
                operators: list[str], qzs: dict) -> list[dict]:
    """Every row for one input A: the baseline once per operator, quantization once per scheme."""
    # Each scheme quantizes the input once; the result is reused for every operator.
    quant = {qid: quantize(x, qz) for qid, qz in qzs.items()}
    rows = []
    for op in operators:
        y32, y64 = baselines(op, x)
        for qid, r in quant.items():
            m = evaluate(op, x, r.x_hat, y32, y64, apply(op, r.x_hat))
            rows.append(make_row(**ids, regime=regime, sample_id=sample_id, seed=seed, operator=op,
                                 quantizer=qid, bits=qzs[qid].bits, group_size=qzs[qid].group_size,
                                 n=x.numel(), zero_fraction=r.zero_fraction, **m))
    return rows


def is_non_finite(row: dict) -> bool:
    """Non-finite outputs, or a NaN metric. An infinite SQNR (exactly zero error) is valid."""
    if row["ref_finite"] == 0 or row["quant_finite"] == 0:
        return True
    return any(isinstance(row[f], float) and math.isnan(row[f]) for f in FLOAT_METRICS)


def run(cfg: dict, config_path: Path, out_root: Path, samples: int | None, analysis: bool) -> int:
    """Run the full experiment into a new artifacts/<run_id>/ folder.

    Streams one CSV row per (distribution, sample, operator, scheme), checks the written rows
    against the expected set, writes the manifest, and optionally runs the analysis.
    Returns the process exit code: 0 when every row is present and finite, 1 otherwise.
    """
    commit, dirty = git_state()
    run_id = f"run-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{commit[:7]}"
    out_dir = out_root / run_id
    (out_dir / "raw").mkdir(parents=True)
    setup_logging(out_dir)
    log.info("run %s (commit %s%s)", run_id, commit[:7], ", uncommitted changes" if dirty else "")

    qzs = quantizers_from_config(cfg)
    operators = list(cfg["operators"])
    n = cfg["vector_length"]
    ids = {"run_id": run_id, "git_commit": commit}
    counts = {"rows": 0, "tensors": 0, "non_finite_rows": 0}
    seen = set()
    t0 = time.perf_counter()
    with open(out_dir / "raw" / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for regime, params, i, seed in planned_samples(cfg, samples):
            x = generate(regime, n, params, seed)
            for row in sample_rows(ids, regime, i, seed, x, operators, qzs):
                # Guard against writing the same result twice.
                key = (regime, i, row["operator"], row["quantizer"])
                if key in seen:
                    raise RuntimeError(f"duplicate row {key}")
                seen.add(key)
                counts["non_finite_rows"] += is_non_finite(row)
                writer.writerow(row)
                counts["rows"] += 1
            counts["tensors"] += 1
            if counts["tensors"] % 300 == 0:
                log.info("%d tensors, %d rows, %.1fs", counts["tensors"], counts["rows"], time.perf_counter() - t0)

    # Completeness check: every planned row written, and nothing unplanned.
    expected = expected_row_keys(cfg, samples)
    counts.update(expected_rows=len(expected), missing_rows=len(expected - seen),
                  unexpected_rows=len(seen - expected), seconds=round(time.perf_counter() - t0, 2))
    ok = counts["missing_rows"] == counts["unexpected_rows"] == counts["non_finite_rows"] == 0
    log.info("rows %d / expected %d (missing %d, unexpected %d), non-finite %d, %.1fs",
             counts["rows"], counts["expected_rows"], counts["missing_rows"], counts["unexpected_rows"],
             counts["non_finite_rows"], counts["seconds"])

    (out_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_id,
        "utc_timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty,
        "machine": machine_info(),
        "config_path": str(config_path), "config_checksum": file_checksum(config_path),
        "samples_per_regime": samples or cfg["samples_per_regime"],
        "seed_rule": "int(sha256(f'{regime}|{sample_id}').hexdigest()[:16], 16)",
        "counts": counts,
        "status": "ok" if ok else "missing or non-finite rows; see run.log",
    }, indent=2), encoding="utf-8")

    if analysis:
        from src.analysis import analyze
        for p in analyze(out_dir, cfg):
            log.info("wrote %s", p.relative_to(out_dir).as_posix())
    log.info("done: %s -> %s", "OK" if ok else "PROBLEMS FOUND", out_dir)
    return 0 if ok else 1


def main(argv=None) -> int:
    """Command-line entry point: parse arguments, configure PyTorch, run."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-root", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--samples", type=int, default=None, help="samples per distribution (default: config, 300)")
    ap.add_argument("--no-analysis", action="store_true", help="skip tables and figures")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    configure_torch(cfg)
    return run(cfg, args.config, args.out_root, args.samples, not args.no_analysis)


if __name__ == "__main__":
    sys.exit(main())
