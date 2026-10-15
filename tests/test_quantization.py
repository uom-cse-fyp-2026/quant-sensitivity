import pytest
import torch

from src.quantization import (absmax_scale, fake_quant, quantize, quantizers_from_config,
                              static_scale)


@pytest.fixture(scope="module")
def qzs(cfg):
    return quantizers_from_config(cfg)


def randn(n=1024, seed=0, scale=1.0):
    return scale * torch.randn(n, generator=torch.Generator().manual_seed(seed))


def test_quantizer_definitions(qzs):
    assert (qzs["Q0"].bits, qzs["Q0"].qmax, qzs["Q0"].group_size) == (8, 127, None)
    assert (qzs["Q1"].bits, qzs["Q1"].qmax, qzs["Q1"].group_size) == (4, 7, None)
    assert (qzs["Q2"].bits, qzs["Q2"].qmax, qzs["Q2"].group_size) == (4, 7, 32)
    assert [qzs[q].effective_bits for q in ("Q0", "Q1", "Q2")] == [8.0, 4.0, 4.5]


@pytest.mark.parametrize("qid", ["Q0", "Q1", "Q2"])
@pytest.mark.parametrize("dynamic", [True, False])
def test_code_range_and_reconstruction(qzs, qid, dynamic):
    qz = qzs[qid]
    x = randn(scale=30.0)
    # a deliberately small static scale forces heavy saturation
    scale = None if dynamic else static_scale(randn(1024, 1).reshape(1, -1), qz)
    r = quantize(x, qz, scale)
    assert r.codes.min() >= -qz.qmax and r.codes.max() <= qz.qmax
    s = r.scale if qz.group_size is None else r.scale.repeat_interleave(qz.group_size)
    torch.testing.assert_close(r.x_hat, r.codes.float() * s, rtol=1e-6, atol=0)
    assert r.x_hat.dtype == torch.float32


def test_round_half_to_even():
    s = torch.tensor(0.5)
    x = torch.tensor([0.25, 0.75, 1.25, -0.25, -0.75, -1.25])   # x / s = ±0.5, ±1.5, ±2.5
    r = fake_quant(x, 8, s)
    assert r.codes.tolist() == [0, 2, 2, 0, -2, -2]


def test_grouping_is_32_contiguous_groups(qzs):
    qz = qzs["Q2"]
    x = randn(1024, 3, scale=0.1)
    x[5 * 32 + 7] = 50.0           # one outlier inside group 5
    scale = absmax_scale(x, qz.bits, qz.group_size)
    assert scale.shape == (32,)
    expected = x.reshape(32, 32).abs().amax(dim=1) / 7
    torch.testing.assert_close(scale, expected)
    assert scale[5] == pytest.approx(50.0 / 7)
    assert (scale[torch.arange(32) != 5] < 1).all()
    # group 5 is [160, 192) and uses the coarse scale; its neighbour group 4 [128, 160) keeps a fine one
    r = quantize(x, qz)
    assert (r.x_hat[160:192] - x[160:192]).abs().max() <= 50.0 / 14 + 1e-6
    assert (r.x_hat[128:160] - x[128:160]).abs().max() <= scale[4] / 2 + 1e-6


def test_static_group_scales_reduce_over_batch(qzs):
    qz = qzs["Q2"]
    batch = torch.stack([randn(1024, i) for i in range(8)])
    batch[3, 40] = 9.0
    s = static_scale(batch, qz)
    assert s.shape == (32,)
    assert s[1] == pytest.approx(9.0 / 7)
    torch.testing.assert_close(s, batch.reshape(8, 32, 32).abs().amax(dim=(0, 2)) / 7)


def test_clip_fraction_known_input():
    s = torch.tensor(0.01)
    x = torch.zeros(1024)
    x[:10] = 5.0        # 5 / 0.01 = 500 > 127
    x[10:20] = -1.27    # exactly at qmax * s: not clipped
    r = fake_quant(x, 8, s)
    assert r.clip_fraction == pytest.approx(10 / 1024)
    assert (r.codes[:10] == 127).all()


@pytest.mark.parametrize("qid", ["Q0", "Q1", "Q2"])
def test_dynamic_scale_never_clips(qzs, qid):
    for seed in range(20):
        assert quantize(randn(seed=seed, scale=3.7), qzs[qid]).clip_fraction == 0.0


@pytest.mark.parametrize("qid", ["Q0", "Q1", "Q2"])
def test_zero_input(qzs, qid):
    r = quantize(torch.zeros(1024), qzs[qid])
    assert torch.isfinite(r.x_hat).all() and (r.x_hat == 0).all()
    assert r.zero_fraction == 1.0 and r.clip_fraction == 0.0 and r.input_mse == 0.0


def test_group_size_must_divide_length(qzs):
    with pytest.raises(ValueError):
        quantize(torch.ones(1000), qzs["Q2"])
