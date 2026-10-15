"""The design enumeration matches the plan's counts (§1, §4.2, §7)."""
from src.config import expected_row_keys, planned_samples, split_strata


def count_samples(cfg, split, sweeps):
    return sum(1 for st, _, _ in planned_samples(cfg, split) if (st.sweep_id is not None) == sweeps)


def test_tier1_heldout_counts(cfg):
    rows = [k for k in expected_row_keys(cfg, "heldout") if ":" not in k[1]]
    assert len(rows) == 4 * 6 * 3 * 2 * 300 == 43_200
    assert count_samples(cfg, "heldout", sweeps=False) * 4 == 7_200   # reference pairs


def test_sweep_heldout_counts(cfg):
    strata = [s for s in split_strata(cfg, "heldout") if s.sweep_id]
    assert len(strata) == 6 + 5 + 8 + 5 + 4
    rows = [k for k in expected_row_keys(cfg, "heldout") if ":" in k[1]]
    assert len(rows) == (19 * 4 + 9 * 1) * 3 * 2 * 100


def test_split_sizes(cfg):
    assert count_samples(cfg, "pilot", sweeps=True) == 0
    assert count_samples(cfg, "pilot", sweeps=False) == 600
    assert count_samples(cfg, "development", sweeps=False) == 1200
    assert count_samples(cfg, "development", sweeps=True) == 28 * 50
    cal = [st for st, _, _ in planned_samples(cfg, "calibration")]
    assert {st.n for st in cal} == {256, 1024, 4096, 16384}
    assert all(st.regime == "D0" for st in cal)
    assert sum(st.n == 1024 for st in cal) == 512


def test_row_keys_unique(cfg):
    for split in ("pilot", "development", "heldout"):
        keys = expected_row_keys(cfg, split)
        assert len(keys) == len(set(keys))
