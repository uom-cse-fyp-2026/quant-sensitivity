import hashlib
import math

import pytest
import torch

from src.config import planned_samples, regime_params
from src.distributions import d5_channels, generate, sample_seed

N = 1024


def sample(cfg, regime, i):
    return generate(regime, N, regime_params(cfg, regime), sample_seed(regime, i))


def pooled(cfg, regime, k=50):
    return torch.stack([sample(cfg, regime, i) for i in range(k)])


# sha256 (first 16 hex digits) of the first 3 samples per distribution. Inputs are drawn in
# FP64 and rounded to FP32, so these must match on every machine (laptop and VM alike).
PINNED_INPUT_HASHES = {
    "D0": "e8832ba7832504d6",
    "D1": "f9c84d040439c43c",
    "D2": "c1432e0e2caa6463",
    "D3": "243ad1f56dd68ace",
    "D4": "2ec5e4d0b81f21c7",
    "D5": "4932a0c35a7f5821",
}


@pytest.mark.parametrize("regime", list(PINNED_INPUT_HASHES))
def test_inputs_match_pinned_hashes(cfg, regime):
    h = hashlib.sha256()
    for i in range(3):
        h.update(sample(cfg, regime, i).numpy().tobytes())
    assert h.hexdigest()[:16] == PINNED_INPUT_HASHES[regime], "inputs differ from the pinned machine-independent values"


def test_shape_dtype_and_determinism(cfg):
    for regime in cfg["regimes"]:
        x = sample(cfg, regime, 0)
        assert x.shape == (N,) and x.dtype == torch.float32 and torch.isfinite(x).all()
        assert torch.equal(x, sample(cfg, regime, 0))


def test_global_rng_untouched(cfg):
    torch.manual_seed(0)
    before = torch.get_rng_state()
    for regime in cfg["regimes"]:
        sample(cfg, regime, 0)
    assert torch.equal(before, torch.get_rng_state())


def test_no_seed_collisions(cfg):
    seeds = [seed for *_, seed in planned_samples(cfg)]
    assert len(seeds) == len(set(seeds)) == 6 * cfg["samples_per_regime"]


def test_d0_statistics(cfg):
    x = pooled(cfg, "D0")
    assert abs(x.mean().item()) < 0.02 and abs(x.var().item() - 1) < 0.03


def test_d1_outliers(cfg):
    k = round(cfg["regimes"]["D1"]["outlier_rate"] * N)
    positions = set()
    for i in range(5):
        x = sample(cfg, "D1", i)
        idx = (x.abs() == 20).nonzero().flatten()
        assert len(idx) == k == 10
        assert (x[idx] > 0).sum() == (x[idx] < 0).sum() == k // 2
        positions.add(frozenset(idx.tolist()))
    assert len(positions) == 5, "D1 outlier positions must change from sample to sample"


def test_d2_statistics(cfg):
    x = pooled(cfg, "D2")
    assert abs(x.mean().item() - 1) < 1e-4 and abs(x.std().item() / 0.001 - 1) < 0.05


def test_d3_cancellation(cfg):
    for i in range(20):
        x = sample(cfg, "D3", i).double()
        assert x.sum().abs() / x.abs().sum() < 0.01


def test_d4_median_abs_matches_student_t(cfg):
    # median |t_3| = 0.76489, scaled by 1/sqrt(3)
    assert abs(pooled(cfg, "D4").abs().median().item() - 0.76489 / math.sqrt(3)) < 0.01


def test_d5_channels_fixed(cfg):
    p = cfg["regimes"]["D5"]
    pos, signs = d5_channels(N, p["n_channels"], p["channel_seed"])
    assert len(set(pos.tolist())) == 8 and signs.sum() == 0
    for i in range(5):
        x = sample(cfg, "D5", i)
        assert torch.equal(torch.sign(x[pos]), signs)
        assert ((x[pos].abs() - 20).abs() < 6).all()
        rest = torch.ones(N, dtype=torch.bool)
        rest[pos] = False
        assert x[rest].abs().max() < 7
