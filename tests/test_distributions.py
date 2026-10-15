import math

import pytest
import torch

from src.config import planned_samples, regime_strata, split_strata, sweep_strata
from src.distributions import d5_channels, generate, generate_stratum, sample_seed

SPLITS = ("pilot", "pilot_calibration", "calibration", "development", "heldout")


def strata_by_key(cfg):
    return {s.key: s for s in regime_strata(cfg) + sweep_strata(cfg)}


def pooled(cfg, key, k=50, split="development"):
    st = strata_by_key(cfg)[key]
    return torch.stack([generate_stratum(st, sample_seed(split, key, i)) for i in range(k)])


def test_determinism_bit_identical(cfg):
    for st in regime_strata(cfg):
        for i in range(10):
            seed = sample_seed("pilot", st.key, i)
            assert torch.equal(generate_stratum(st, seed), generate_stratum(st, seed)), st.key


def test_global_rng_untouched(cfg):
    torch.manual_seed(0)
    before = torch.get_rng_state()
    for st in regime_strata(cfg):
        generate_stratum(st, 123)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("split", ["pilot", "development", "heldout"])
def test_shape_and_dtype(cfg, split):
    for st in split_strata(cfg, split):
        x = generate_stratum(st, sample_seed(split, st.key, 0))
        assert x.shape == (st.n,) and x.dtype == torch.float32 and torch.isfinite(x).all(), st.key


def test_d0_statistics(cfg):
    x = pooled(cfg, "D0")
    assert abs(x.mean().item()) < 0.02
    assert abs(x.var().item() - 1) < 0.03


def test_d1_outliers(cfg):
    p = cfg["regimes"]["D1"]
    k = round(p["outlier_rate"] * cfg["vector_length"])
    st = strata_by_key(cfg)["D1"]
    positions = []
    for i in range(5):
        x = generate_stratum(st, sample_seed("development", "D1", i))
        idx = (x.abs() == p["outlier_magnitude"]).nonzero().flatten()
        assert len(idx) == k and k % 2 == 0
        assert (x[idx] > 0).sum() == (x[idx] < 0).sum() == k // 2
        positions.append(set(idx.tolist()))
    assert len({frozenset(s) for s in positions}) == 5, "D1 outlier positions must vary per sample"


def test_d1_odd_outlier_count_uses_one_random_sign(cfg):
    st = strata_by_key(cfg)["S2:D1=0.001"]   # round(0.001 * 1024) = 1 outlier
    x = generate_stratum(st, sample_seed("development", st.key, 0))
    assert (x.abs() == 20).sum() == 1


def test_d2_statistics(cfg):
    x = pooled(cfg, "D2")
    assert abs(x.mean().item() - 1) < 1e-4
    assert abs(x.std().item() / 0.001 - 1) < 0.05


def test_d3_cancellation_ratio(cfg):
    st = strata_by_key(cfg)["D3"]
    for i in range(20):
        x = generate_stratum(st, sample_seed("development", "D3", i)).double()
        assert x.sum().abs() / x.abs().sum() < 0.01


def test_d4_median_abs_matches_student_t(cfg):
    # median |t_3| = 0.76489; scaled by 1/sqrt(3)
    x = pooled(cfg, "D4")
    assert abs(x.abs().median().item() - 0.76489 / math.sqrt(3)) < 0.01


def test_d5_channels_fixed_across_samples_and_splits(cfg):
    p = cfg["regimes"]["D5"]
    pos, signs = d5_channels(cfg["vector_length"], p["n_channels"], p["channel_seed"])
    assert len(set(pos.tolist())) == p["n_channels"]
    assert signs.sum() == 0
    st = strata_by_key(cfg)["D5"]
    for split in ("pilot", "development", "heldout"):
        for i in range(5):
            x = generate_stratum(st, sample_seed(split, "D5", i))
            assert torch.equal(torch.sign(x[pos]), signs)
            assert ((x[pos].abs() - p["channel_magnitude"]).abs() < 6).all()
            others = torch.ones_like(x, dtype=torch.bool)
            others[pos] = False
            assert x[others].abs().max() < 7


def test_sweep_parameters_take_effect(cfg):
    s = strata_by_key(cfg)
    assert generate_stratum(s["S3:D3=16384"], 1).shape == (16384,)
    big = generate_stratum(s["S1:D1=100"], 1)
    assert (big.abs() == 100).sum() == 10
    wide = torch.stack([generate_stratum(s["S4:D0=8"], i) for i in range(20)])
    assert abs(wide.std().item() - 8) < 0.2


def test_no_seed_collisions_across_splits(cfg):
    seeds = {}
    for split in SPLITS:
        for st, i, seed in planned_samples(cfg, split):
            assert seed not in seeds, f"{(split, st.key, i)} collides with {seeds[seed]}"
            seeds[seed] = (split, st.key, i)


def test_heldout_tensors_not_in_other_splits(cfg):
    """Leakage: no held-out tensor equals a calibration or development tensor."""
    def digests(split):
        out = set()
        for st, i, seed in planned_samples(cfg, split):
            if st.key not in ("D0", "D1", "D5"):
                continue
            out.add(generate(st.regime, st.n, st.params, seed).numpy().tobytes())
        return out
    held = digests("heldout")
    assert held.isdisjoint(digests("development"))
    assert held.isdisjoint(digests("calibration"))


# sha256 (first 16 hex digits) of the first 3 pilot tensors per stratum. Inputs are drawn in
# FP64 and rounded to FP32, so these must match on every machine (laptop and VM alike).
PINNED_INPUT_HASHES = {
    "D0": "95c60dec66e422ec",
    "D1": "9eee615675357434",
    "D2": "a7d6a9f8e12b3dfc",
    "D3": "9b92bc94603d6f84",
    "D4": "67310815430f29c7",
    "D5": "d2e89d228ab970f2",
    "S3:D3=16384": "2562817d6ec89a81",
    "S4:D0=8": "42bad4845808ba88",
}


@pytest.mark.parametrize("key", list(PINNED_INPUT_HASHES))
def test_inputs_match_pinned_hashes(cfg, key):
    import hashlib
    st = strata_by_key(cfg)[key]
    h = hashlib.sha256()
    for i in range(3):
        h.update(generate_stratum(st, sample_seed("pilot", key, i)).numpy().tobytes())
    assert h.hexdigest()[:16] == PINNED_INPUT_HASHES[key], "input tensors differ from the pinned machine-independent values"
