"""Gate 1: freezing the static calibration scales."""
import json

import pytest
import torch

from src import run_experiment
from src.config import file_checksum, planned_samples
from src.distributions import generate_stratum
from src.quantization import quantizers_from_config, static_scale
from src.run_experiment import load_static_scales, main, scales_checksum

LENGTHS = {256, 1024, 4096, 16384}


def freeze(out_root):
    return main(["--gate", "1", "--out-root", str(out_root), "--allow-dirty"])


@pytest.fixture(scope="module")
def calib_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("gate1")
    assert freeze(root) == 0
    return root / "calibration"


def test_scales_saved_with_matching_checksum(calib_dir):
    m = json.loads((calib_dir / "manifest.json").read_text())
    assert m["status"] == "ok" and m["gate"] == 1 and m["splits"] == ["calibration"]
    assert m["static_scale_checksum"] == file_checksum(calib_dir / "scales.pt")
    assert m["leakage"]["overlapping_seeds"] == 0
    assert m["tensors_per_length"] == {str(n): 512 for n in sorted(LENGTHS)}


def test_every_length_and_quantizer_present(cfg, calib_dir):
    scales, meta = load_static_scales(calib_dir / "scales.pt")
    assert set(scales) == {(n, q) for n in LENGTHS for q in ("Q0", "Q1", "Q2")}
    for n in LENGTHS:
        assert scales[(n, "Q0")].numel() == scales[(n, "Q1")].numel() == 1
        assert scales[(n, "Q2")].shape == (n // 32,)
        # Q0 and Q1 share the same absmax: s0 * 127 == s1 * 7
        torch.testing.assert_close(scales[(n, "Q0")] * 127, scales[(n, "Q1")] * 7)
    assert meta["tensors_per_length"]["1024"] == 512


def test_scales_equal_absmax_over_calibration_tensors(cfg, calib_dir):
    scales, _ = load_static_scales(calib_dir / "scales.pt")
    xs = torch.stack([generate_stratum(st, seed) for st, _, seed in planned_samples(cfg, "calibration")
                      if st.n == 1024])
    assert xs.shape == (512, 1024)
    qzs = quantizers_from_config(cfg)
    for q in ("Q0", "Q1", "Q2"):
        torch.testing.assert_close(scales[(1024, q)], static_scale(xs, qzs[q]), rtol=0, atol=0)
    assert 4.0 < (scales[(1024, "Q0")] * 127).item() < 6.0   # absmax of ~524k standard normals


def test_freeze_is_deterministic(calib_dir, tmp_path):
    assert freeze(tmp_path) == 0
    a, _ = load_static_scales(calib_dir / "scales.pt")
    b, _ = load_static_scales(tmp_path / "calibration" / "scales.pt")
    assert scales_checksum(a) == scales_checksum(b)


def test_refuses_to_overwrite(calib_dir):
    before = file_checksum(calib_dir / "scales.pt")
    assert freeze(calib_dir.parent) == 2
    assert file_checksum(calib_dir / "scales.pt") == before


def test_refuses_dirty_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(run_experiment, "git_state", lambda: ("0" * 40, True))
    assert main(["--gate", "1", "--out-root", str(tmp_path)]) == 2
    assert not (tmp_path / "calibration" / "scales.pt").exists()
