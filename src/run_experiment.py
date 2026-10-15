"""Experiment runner (plan §8). Runs on the VM.

    uv run python -m src.run_experiment --config configs/experiment.yaml --gate 0

Gate 0 is the pilot: 100 tensors per regime through every operator, quantizer and
calibration policy. Its static policy uses throwaway scales from a separate
pilot-calibration set that are never saved, so the frozen Gate 1 scales stay untouched.
Pilot numbers are for debugging only and are never reported. Each pilot run writes
artifacts/<run_id>/ with raw/pilot.csv, manifest.json, run.log, sanity plots and summaries.

Gate 1 freezes the static scales: absmax over the 512 D0 calibration tensors at each
length, saved once to artifacts/calibration/scales.pt with a checksum. It refuses to run
on a dirty git tree and refuses to overwrite existing scales.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
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

from src.config import (DEFAULT_CONFIG, ROOT, configure_torch, expected_row_keys, file_checksum,
                        load_config, planned_samples, stratum_key)
from src.distributions import generate_stratum
from src.features import describe
from src.metrics import CSV_FIELDS, evaluate, make_row, quantizer_fields
from src.operators import apply, references
from src.quantization import quantize, quantizers_from_config, static_scale

log = logging.getLogger("run_experiment")

FLOAT_METRICS = ("abs_error", "l1_norm_error", "norm_l2", "sqnr_db", "mae", "max_abs_error",
                 "softmax_kl", "kappa", "input_rel_l2", "input_mse", "fp32_floor", "rel_l2_floor")


# ---------------------------------------------------------------- run metadata

def git_state() -> tuple[str, bool]:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD") or "unknown"
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return commit, dirty


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def machine_info() -> dict:
    return {
        "cpu_model": cpu_model(),
        "isa": platform.machine(),
        "cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "numpy": np.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def rounding_is_half_even() -> bool:
    got = torch.round(torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -2.5])).tolist()
    return got == [0.0, 2.0, 2.0, -0.0, -2.0, -2.0]


def scales_checksum(scales: dict) -> str:
    h = hashlib.sha256()
    for (n, qid) in sorted(scales):
        h.update(f"{n}|{qid}|".encode())
        h.update(scales[(n, qid)].numpy().tobytes())
    return h.hexdigest()


def make_run_id(prefix: str, commit: str) -> str:
    return f"{prefix}-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{commit[:7]}"


def setup_logging(out_dir: Path) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(out_dir / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


# ---------------------------------------------------------------- core loop

def compute_static_scales(cfg: dict, split: str, qzs: dict) -> dict:
    """Static absmax scales per (length, quantizer) from a calibration split's D0 tensors."""
    batches: dict[int, list[torch.Tensor]] = {}
    for st, _, seed in planned_samples(cfg, split):
        batches.setdefault(st.n, []).append(generate_stratum(st, seed))
    return {(n, qid): static_scale(torch.stack(xs), qz)
            for n, xs in batches.items() for qid, qz in qzs.items()}


def evaluate_sample(ids: dict, st, sample_id: int, seed: int, x: torch.Tensor,
                    qzs: dict, policies: list[str], static_scales: dict) -> list[dict]:
    """Every row for one input tensor: references once per operator, quantization once per scheme."""
    desc = describe(x)
    quant = {}
    for qid, qz in qzs.items():
        for policy in policies:
            scale = static_scales[(st.n, qid)] if policy == "static" else None
            quant[(qid, policy)] = quantize(x, qz, scale)
    rows = []
    for op in st.operators:
        y32, y64 = references(op, x)
        for (qid, policy), r in quant.items():
            m = evaluate(op, x, r.x_hat, y32, y64, apply(op, r.x_hat))
            rows.append(make_row(
                **ids, regime=st.regime, sweep_id=st.sweep_id, sweep_value=st.sweep_value,
                sample_id=sample_id, seed=seed, operator=op, **quantizer_fields(qzs[qid]),
                calib_policy=policy, n=st.n, **desc, clip_fraction=r.clip_fraction,
                zero_fraction=r.zero_fraction, input_mse=r.input_mse, **m))
    return rows


def row_is_non_finite(row: dict) -> bool:
    """Non-finite outputs, or a NaN metric. An infinite SQNR (exactly zero error) is valid."""
    if row["ref_finite"] == 0 or row["quant_finite"] == 0:
        return True
    return any(isinstance(row[f], float) and math.isnan(row[f]) for f in FLOAT_METRICS)


def run_split(cfg: dict, split: str, out_dir: Path, ids: dict, static_scales: dict,
              samples_per_stratum: int | None = None) -> dict:
    qzs = quantizers_from_config(cfg)
    policies = cfg["calibration"]["policies"]
    path = out_dir / "raw" / f"{split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"rows": 0, "non_finite_rows": 0, "infinite_sqnr_rows": 0, "tensors": 0}
    seen = set()
    t0 = time.perf_counter()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for st, i, seed in planned_samples(cfg, split):
            if samples_per_stratum is not None and i >= samples_per_stratum:
                continue
            x = generate_stratum(st, seed)
            for row in evaluate_sample({**ids, "split": split}, st, i, seed, x, qzs, policies, static_scales):
                key = (split, stratum_key(st.regime, st.sweep_id, st.sweep_value), i,
                       row["operator"], row["quantizer"], row["calib_policy"])
                if key in seen:
                    raise RuntimeError(f"duplicate row {key}")
                seen.add(key)
                stats["non_finite_rows"] += row_is_non_finite(row)
                stats["infinite_sqnr_rows"] += row["sqnr_db"] == math.inf
                writer.writerow(row)
                stats["rows"] += 1
            stats["tensors"] += 1
            if stats["tensors"] % 100 == 0:
                log.info("%s: %d tensors, %d rows, %.1fs", split, stats["tensors"], stats["rows"],
                         time.perf_counter() - t0)
    expected = {k for k in expected_row_keys(cfg, split)
                if samples_per_stratum is None or k[2] < samples_per_stratum}
    stats["expected_rows"] = len(expected)
    stats["missing_rows"] = len(expected - seen)
    stats["unexpected_rows"] = len(seen - expected)
    stats["seconds"] = round(time.perf_counter() - t0, 2)
    return stats


def scales_summary(scales: dict) -> dict:
    return {f"n={n} {qid}": ({"value": s.item()} if s.numel() == 1 else
                             {"groups": s.numel(), "min": s.min().item(), "max": s.max().item()})
            for (n, qid), s in sorted(scales.items())}


def save_static_scales(path: Path, scales: dict, meta: dict) -> str:
    """Save frozen scales (keys "<n>|<quantizer>") and return the file's SHA-256."""
    torch.save({"scales": {f"{n}|{qid}": s.detach().clone() for (n, qid), s in sorted(scales.items())},
                "meta": meta}, path)
    return file_checksum(path)


def load_static_scales(path: Path) -> tuple[dict, dict]:
    """Load frozen scales as {(n, quantizer): tensor} plus their metadata."""
    obj = torch.load(path, weights_only=True)
    scales = {}
    for key, s in obj["scales"].items():
        n, qid = key.split("|")
        scales[(int(n), qid)] = s
    return scales, obj["meta"]


def calibration_leakage(cfg: dict) -> dict:
    """Seeds of calibration tensors must not appear in any other split."""
    cal = {seed for _, _, seed in planned_samples(cfg, "calibration")}
    other = {seed for split in ("pilot", "pilot_calibration", "development", "heldout")
             for _, _, seed in planned_samples(cfg, split)}
    return {"calibration_tensors": len(cal), "other_split_tensors": len(other),
            "overlapping_seeds": len(cal & other)}


def write_manifest(out_dir: Path, **fields) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(fields, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------- gates

def run_pilot(cfg: dict, config_path: Path, out_root: Path, samples: int | None, plots: bool) -> int:
    commit, dirty = git_state()
    run_id = make_run_id("pilot", commit)
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    setup_logging(out_dir)
    log.info("run %s (commit %s%s)", run_id, commit[:7], ", dirty tree" if dirty else "")

    qzs = quantizers_from_config(cfg)
    static_scales = compute_static_scales(cfg, "pilot_calibration", qzs)
    for (n, qid), s in sorted(static_scales.items()):
        log.info("pilot static scale n=%d %s: %s", n, qid,
                 f"{s.item():.6g}" if s.numel() == 1 else f"{s.numel()} groups, {s.min():.4g}..{s.max():.4g}")

    t0 = time.perf_counter()
    stats = run_split(cfg, "pilot", out_dir, {"run_id": run_id, "git_commit": commit}, static_scales, samples)
    log.info("pilot rows %d / expected %d (missing %d, unexpected %d), non-finite %d, infinite SQNR %d, %.1fs",
             stats["rows"], stats["expected_rows"], stats["missing_rows"], stats["unexpected_rows"],
             stats["non_finite_rows"], stats["infinite_sqnr_rows"], stats["seconds"])

    if plots:
        from src.sanity import pilot_sanity
        for p in pilot_sanity(cfg, out_dir, "pilot", samples):
            log.info("wrote %s", p.relative_to(out_dir))

    ok = stats["missing_rows"] == stats["unexpected_rows"] == stats["non_finite_rows"] == 0
    write_manifest(
        out_dir,
        run_id=run_id, gate=0, splits=["pilot"],
        utc_timestamp=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        git_commit=commit, git_dirty=dirty,
        machine=machine_info(),
        config_path=str(config_path), config_checksum=file_checksum(config_path),
        static_scales="pilot-only throwaway scales from the pilot_calibration split (not saved)",
        static_scale_checksum=None, pilot_scale_checksum=scales_checksum(static_scales),
        rounding_half_even=rounding_is_half_even(),
        seed_rule="int(sha256(f'{split}|{stratum_key}|{sample_id}').hexdigest()[:16], 16)",
        split_sizes={"pilot_per_regime": samples or cfg["splits"]["pilot"]["per_regime"],
                     "pilot_calibration": cfg["splits"]["pilot_calibration"]["n_tensors"]},
        samples_override=samples,
        counts={"pilot": stats},
        runtime_seconds=round(time.perf_counter() - t0, 2),
        status="ok" if ok else "incomplete or non-finite rows; see run.log",
    )
    log.info("done: %s -> %s", "OK" if ok else "PROBLEMS FOUND", out_dir)
    return 0 if ok else 1


def run_freeze(cfg: dict, config_path: Path, out_root: Path, allow_dirty: bool) -> int:
    commit, dirty = git_state()
    if dirty and not allow_dirty:
        print("refusing to freeze scales on a dirty git tree; commit or stash first", file=sys.stderr)
        return 2
    calib_dir = out_root / "calibration"
    scale_path = calib_dir / "scales.pt"
    if scale_path.exists():
        print(f"{scale_path} already exists: static scales are frozen and are never recomputed. "
              "If they truly must change, delete the folder by hand and log a deviation in "
              "reports/findings.md.", file=sys.stderr)
        return 2
    calib_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(calib_dir)
    run_id = make_run_id("calibration", commit)
    log.info("run %s (commit %s%s)", run_id, commit[:7], ", dirty tree" if dirty else "")
    t0 = time.perf_counter()

    leakage = calibration_leakage(cfg)
    log.info("leakage check: %s", leakage)
    if leakage["overlapping_seeds"]:
        log.error("calibration seeds overlap other splits; not freezing")
        return 1
    if {st.regime for st, _, _ in planned_samples(cfg, "calibration")} != {cfg["calibration"]["static"]["source_regime"]}:
        log.error("calibration split contains a regime other than the configured source regime")
        return 1

    qzs = quantizers_from_config(cfg)
    scales = compute_static_scales(cfg, "calibration", qzs)
    per_length = {}
    for st, _, _ in planned_samples(cfg, "calibration"):
        per_length[st.n] = per_length.get(st.n, 0) + 1
    for (n, qid), sc in sorted(scales.items()):
        log.info("static scale n=%d %s: %s", n, qid,
                 f"{sc.item():.6g}" if sc.numel() == 1 else f"{sc.numel()} groups, {sc.min():.4g}..{sc.max():.4g}")
    if not all(torch.isfinite(sc).all() and (sc > 0).all() for sc in scales.values()):
        log.error("non-finite or non-positive scale; not freezing")
        return 1

    meta = {"run_id": run_id, "git_commit": commit, "config_checksum": file_checksum(config_path),
            "tensors_per_length": {str(n): k for n, k in sorted(per_length.items())}}
    checksum = save_static_scales(scale_path, scales, meta)
    log.info("saved %s, sha256 %s", scale_path, checksum)

    write_manifest(
        calib_dir,
        run_id=run_id, gate=1, splits=["calibration"],
        utc_timestamp=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        git_commit=commit, git_dirty=dirty, final=not dirty,
        machine=machine_info(),
        config_path=str(config_path), config_checksum=meta["config_checksum"],
        static_scale_file="calibration/scales.pt", static_scale_checksum=checksum,
        static_scale_content_checksum=scales_checksum(scales),
        scales=scales_summary(scales),
        tensors_per_length=per_length,
        leakage=leakage,
        rounding_half_even=rounding_is_half_even(),
        seed_rule="int(sha256(f'{split}|{stratum_key}|{sample_id}').hexdigest()[:16], 16)",
        runtime_seconds=round(time.perf_counter() - t0, 2),
        status="ok",
    )
    log.info("done: OK -> %s", calib_dir)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--gate", type=int, required=True,
                    help="0 = pilot, 1 = freeze static scales (gates 2-3 not implemented yet)")
    ap.add_argument("--out-root", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--samples", type=int, default=None,
                    help="pilot only: limit tensors per regime (for a quick smoke test)")
    ap.add_argument("--no-plots", action="store_true", help="skip sanity plots")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="gate 1: allow a dirty git tree (for testing only; the manifest marks it non-final)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    configure_torch(cfg)
    if args.gate == 0:
        return run_pilot(cfg, args.config, args.out_root, args.samples, not args.no_plots)
    if args.gate == 1:
        return run_freeze(cfg, args.config, args.out_root, args.allow_dirty)
    ap.error(f"gate {args.gate} is not implemented yet (see issue #7)")


if __name__ == "__main__":
    sys.exit(main())
