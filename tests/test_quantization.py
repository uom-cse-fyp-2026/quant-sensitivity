import pytest
import torch

from src.quantization import absmax_scale, fake_quant, quantize, quantizers_from_config


@pytest.fixture(scope="module")
def qzs(cfg):
    return quantizers_from_config(cfg)


def randn(n=1024, seed=0, scale=1.0):
    return scale * torch.randn(n, generator=torch.Generator().manual_seed(seed))


def test_scheme_definitions(qzs):
    assert [(qzs[q].bits, qzs[q].qmax, qzs[q].group_size) for q in ("Q0", "Q1", "Q2")] == \
        [(8, 127, None), (4, 7, None), (4, 7, 32)]


@pytest.mark.parametrize("qid", ["Q0", "Q1", "Q2"])
def test_code_range_and_reconstruction(qzs, qid):
    qz = qzs[qid]
    r = quantize(randn(scale=30.0), qz)
    assert r.codes.abs().max() == qz.qmax          # absmax maps exactly to the top code
    s = r.scale if qz.group_size is None else r.scale.repeat_interleave(qz.group_size)
    torch.testing.assert_close(r.x_hat, r.codes.float() * s, rtol=1e-6, atol=0)
    assert r.x_hat.dtype == torch.float32


def test_round_half_to_even():
    x = torch.tensor([0.25, 0.75, 1.25, -0.25, -0.75, -1.25])   # x / 0.5 = ±0.5, ±1.5, ±2.5
    assert fake_quant(x, 8, torch.tensor(0.5)).codes.tolist() == [0, 2, 2, 0, -2, -2]


def test_saturation():
    x = torch.tensor([5.0, -5.0, 0.5])
    r = fake_quant(x, 4, torch.tensor(0.1))      # 5 / 0.1 = 50 > 7
    assert r.codes.tolist() == [7, -7, 5]


def test_grouping_is_32_contiguous_groups(qzs):
    x = randn(1024, 3, scale=0.1)
    x[5 * 32 + 7] = 50.0           # one outlier inside group 5, i.e. [160, 192)
    scale = absmax_scale(x, 4, 32)
    assert scale.shape == (32,)
    torch.testing.assert_close(scale, x.reshape(32, 32).abs().amax(dim=1) / 7)
    r = quantize(x, qzs["Q2"])
    # the neighbouring group [128, 160) keeps its own fine scale
    assert (r.x_hat[128:160] - x[128:160]).abs().max() <= scale[4] / 2 + 1e-6
    assert (r.x_hat[160:192] - x[160:192]).abs().max() <= scale[5] / 2 + 1e-6


def test_zero_fraction_with_one_outlier(qzs):
    x = randn(1024, 4)
    x[0] = 20.0                    # per-tensor A4 step = 20 / 7 ≈ 2.9: most normals round to 0
    assert quantize(x, qzs["Q1"]).zero_fraction > 0.8
    assert quantize(x, qzs["Q2"]).zero_fraction < 0.3


@pytest.mark.parametrize("qid", ["Q0", "Q1", "Q2"])
def test_zero_input(qzs, qid):
    r = quantize(torch.zeros(1024), qzs[qid])
    assert torch.isfinite(r.x_hat).all() and (r.x_hat == 0).all() and r.zero_fraction == 1.0


def test_group_size_must_divide_length(qzs):
    with pytest.raises(ValueError):
        quantize(torch.ones(1000), qzs["Q2"])
