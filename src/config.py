"""Load the experiment config, set up deterministic torch, and enumerate the design.

The design enumeration (strata, samples, expected rows) lives here so that the runner,
the completeness test and the analysis all agree on what a complete run contains.
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiment.yaml"

REPORTED_SPLITS = ("development", "heldout")


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


def fmt_value(v) -> str:
    return format(v, "g")


def stratum_key(regime: str, sweep_id: str | None = None, value=None) -> str:
    if sweep_id is None:
        return regime
    return f"{sweep_id}:{regime}={fmt_value(value)}"


@dataclass(frozen=True)
class Stratum:
    """One regime or sweep point: everything needed to generate its tensors."""
    key: str
    regime: str
    n: int
    params: dict = field(hash=False, compare=False)
    operators: tuple[str, ...] = ()
    sweep_id: str | None = None
    sweep_value: float | int | None = None


def regime_params(cfg: dict, regime: str) -> dict:
    return {k: v for k, v in cfg["regimes"][regime].items() if k != "name"}


def regime_strata(cfg: dict) -> list[Stratum]:
    n = cfg["vector_length"]
    ops = tuple(cfg["operators"]["core"])
    return [Stratum(r, r, n, regime_params(cfg, r), ops) for r in cfg["regimes"]]


def sweep_strata(cfg: dict) -> list[Stratum]:
    out = []
    for sid, sw in cfg["sweeps"].items():
        for base, value in itertools.product(sw["base"], sw["values"]):
            params = regime_params(cfg, base)
            n = cfg["vector_length"]
            if sw["param"] == "n":
                n = int(value)
            else:
                params[sw["param"]] = value
            out.append(Stratum(stratum_key(base, sid, value), base, n, params,
                               tuple(sw["operators"]), sid, value))
    return out


def calibration_strata(cfg: dict) -> list[Stratum]:
    """D0 at the core length, plus D0 at each S3 length that needs its own static scales."""
    n0 = cfg["vector_length"]
    src = cfg["calibration"]["static"]["source_regime"]
    lengths = {n0}
    for sid in cfg["calibration"]["static"]["recalibrate_per_length"]:
        lengths |= {int(v) for v in cfg["sweeps"][sid]["values"]}
    params = regime_params(cfg, src)
    return [Stratum(src if n == n0 else f"{src}:n={n}", src, n, params) for n in sorted(lengths)]


def split_strata(cfg: dict, split: str) -> list[Stratum]:
    if split in ("calibration", "pilot_calibration"):
        strata = calibration_strata(cfg)
        return strata if split == "calibration" else [s for s in strata if s.n == cfg["vector_length"]]
    strata = regime_strata(cfg)
    if cfg["splits"][split].get("sweeps", True):
        strata += sweep_strata(cfg)
    return strata


def samples_per_stratum(cfg: dict, split: str, stratum: Stratum) -> int:
    s = cfg["splits"][split]
    if "n_tensors" in s:
        return s["n_tensors"]
    return s["per_sweep_point"] if stratum.sweep_id else s["per_regime"]


def planned_samples(cfg: dict, split: str):
    """Yield (stratum, sample_id, seed) for every tensor in a split."""
    from src.distributions import sample_seed
    digits = cfg["seeds"]["sample_seed_hex_digits"]
    for st in split_strata(cfg, split):
        for i in range(samples_per_stratum(cfg, split, st)):
            yield st, i, sample_seed(split, st.key, i, digits)


def expected_row_keys(cfg: dict, split: str) -> list[tuple]:
    """(split, stratum_key, sample_id, operator, quantizer, policy) for every row a run must write."""
    quantizers = [q for q in cfg["quantizers"] if q.startswith("Q")]
    policies = cfg["calibration"]["policies"]
    keys = []
    for st, i, _ in planned_samples(cfg, split):
        for op, q, p in itertools.product(st.operators, quantizers, policies):
            keys.append((split, st.key, i, op, q, p))
    return keys
