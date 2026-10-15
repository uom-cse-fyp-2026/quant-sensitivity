"""Experiment configuration, deterministic PyTorch setup and sample enumeration.

The runner and the tests share the enumeration defined here, so both agree on exactly which
input tensors and result rows a complete run contains.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import torch
import yaml

# Repository root, resolved from this file so commands work from any working directory.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiment.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Read the YAML experiment configuration into a plain dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_checksum(path: str | Path) -> str:
    """SHA-256 of a file's bytes, recorded in each run's manifest for traceability."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_torch(cfg: dict) -> None:
    """Make PyTorch deterministic: CPU only, a fixed thread count, deterministic kernels.

    FP32 reductions can change in the last bits with the number of threads, so the thread
    count is pinned to keep reruns on the same machine bit-identical.
    """
    env = cfg["environment"]
    if env["device"] != "cpu":
        raise ValueError("This experiment runs on CPU only")
    torch.set_num_threads(env["num_threads"])
    torch.use_deterministic_algorithms(env["deterministic_algorithms"])


def regime_params(cfg: dict, regime: str) -> dict:
    """Generator keyword arguments for one distribution (its config entry minus the name)."""
    return {k: v for k, v in cfg["regimes"][regime].items() if k != "name"}


def planned_samples(cfg: dict, samples_per_regime: int | None = None):
    """Yield (regime, params, sample_id, seed) for every input tensor A in the experiment.

    Args:
        cfg: The experiment configuration.
        samples_per_regime: Override for the number of samples per distribution
            (used for quick checks); defaults to the configured value.
    """
    # Imported here to avoid a circular import: distributions does not depend on config.
    from src.distributions import sample_seed
    k = cfg["samples_per_regime"] if samples_per_regime is None else samples_per_regime
    digits = cfg["seeds"]["sample_seed_hex_digits"]
    for regime in cfg["regimes"]:
        params = regime_params(cfg, regime)
        for i in range(k):
            yield regime, params, i, sample_seed(regime, i, digits)


def expected_row_keys(cfg: dict, samples_per_regime: int | None = None) -> set[tuple]:
    """The (regime, sample_id, operator, quantizer) key of every row a complete run writes.

    The runner compares this set with the rows it actually wrote to detect missing or
    unexpected results.
    """
    return {(regime, i, op, q)
            for (regime, _, i, _), op, q in itertools.product(
                planned_samples(cfg, samples_per_regime), cfg["operators"], cfg["quantizers"])}
