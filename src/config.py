"""Load the experiment config, set up deterministic torch, and enumerate the samples.

The runner and the tests share this enumeration, so they agree on what a complete run contains.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiment.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_checksum(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_torch(cfg: dict) -> None:
    env = cfg["environment"]
    if env["device"] != "cpu":
        raise ValueError("This experiment runs on CPU only")
    torch.set_num_threads(env["num_threads"])
    torch.use_deterministic_algorithms(env["deterministic_algorithms"])


def regime_params(cfg: dict, regime: str) -> dict:
    return {k: v for k, v in cfg["regimes"][regime].items() if k != "name"}


def planned_samples(cfg: dict, samples_per_regime: int | None = None):
    """Yield (regime, params, sample_id, seed) for every input tensor A."""
    from src.distributions import sample_seed
    k = cfg["samples_per_regime"] if samples_per_regime is None else samples_per_regime
    digits = cfg["seeds"]["sample_seed_hex_digits"]
    for regime in cfg["regimes"]:
        params = regime_params(cfg, regime)
        for i in range(k):
            yield regime, params, i, sample_seed(regime, i, digits)


def expected_row_keys(cfg: dict, samples_per_regime: int | None = None) -> set[tuple]:
    """(regime, sample_id, operator, quantizer) for every row a complete run writes."""
    return {(regime, i, op, q)
            for (regime, _, i, _), op, q in itertools.product(
                planned_samples(cfg, samples_per_regime), cfg["operators"], cfg["quantizers"])}
