import math

import pytest
import torch

from src import metrics
from src.config import regime_params
from src.distributions import generate, sample_seed
from src.metrics import CSV_FIELDS, HEADLINE, amplification, evaluate, make_row
from src.operators import OPERATORS, apply, baselines
from src.quantization import quantize, quantizers_from_config

OPS = list(OPERATORS)


def sample(cfg, regime, i=0):
    return generate(regime, 1024, regime_params(cfg, regime), sample_seed(regime, i))


def test_constants_match_config(cfg):
    m = cfg["metrics"]
    assert metrics.PROB_CLAMP == m["probability_clamp"]
    assert metrics.NORM_EPS == m["norm_l2_eps"]
    assert metrics.KAPPA_NULL_REF_NORM == m["kappa_null_ref_norm"]
    assert metrics.FP32_NOISE_FACTOR == m["fp32_noise_factor"]
    assert HEADLINE == m["headline"]
    assert set(OPS) == set(cfg["operators"])


@pytest.mark.parametrize("op", OPS)
def test_no_quantization_gives_zero_error(cfg, op):
    for regime in cfg["regimes"]:
        x = sample(cfg, regime)
        y32, y64 = baselines(op, x)
        m = evaluate(op, x, x.clone(), y32, y64, y32.clone())
        for f in ("abs_error", "l1_norm_error", "norm_l2", "mae", "max_abs_error", "softmax_kl", "input_rel_l2"):
            assert m.get(f) in (None, 0), (regime, f)
        assert m.get("sqnr_db", math.inf) == math.inf
        assert m.get("top1_changed", 0) == 0
        assert m["kappa"] is None and m["ref_finite"] == m["quant_finite"] == 1


@pytest.mark.parametrize("op", OPS)
def test_inapplicable_fields_absent(op):
    x = torch.randn(1024, generator=torch.Generator().manual_seed(1))
    y32, y64 = baselines(op, x)
    m = evaluate(op, x, x * 1.01, y32, y64, apply(op, x * 1.01))
    vector_only = {"norm_l2", "sqnr_db", "mae", "max_abs_error"}
    if op == "sum":
        assert vector_only.isdisjoint(m) and {"abs_error", "l1_norm_error"} <= set(m)
    else:
        assert vector_only <= set(m) and "abs_error" not in m
    assert ("softmax_kl" in m) == ("top1_changed" in m) == (op == "softmax")


def test_softmax_valid_for_every_input_and_scheme(cfg):
    qzs = quantizers_from_config(cfg)
    for regime in cfg["regimes"]:
        x = sample(cfg, regime)
        for x_in in [x] + [quantize(x, qz).x_hat for qz in qzs.values()]:
            p = apply("softmax", x_in)
            assert torch.isfinite(p).all() and (p >= 0).all()
            assert abs(p.double().sum().item() - 1) < 1e-6


def test_kappa_undefined_cases():
    x = torch.randn(64, generator=torch.Generator().manual_seed(2))
    y = apply("gelu", x)
    assert amplification(x, x.clone(), y, y) is None                       # no input error
    pair = torch.tensor([1.5, -1.5])
    assert amplification(pair, pair * 1.1, apply("sum", pair), apply("sum", pair * 1.1)) is None  # Y = 0


def test_kappa_gelu_near_one_and_sum_amplifies_cancellation(cfg):
    qz = quantizers_from_config(cfg)["Q0"]

    def median_kappa(op, regime):
        ks = []
        for i in range(30):
            x = sample(cfg, regime, i)
            r = quantize(x, qz)
            ks.append(amplification(x, r.x_hat, apply(op, x), apply(op, r.x_hat)))
        return sorted(ks)[15]

    assert 0.5 < median_kappa("gelu", "D0") < 2
    assert median_kappa("sum", "D3") > 20 * median_kappa("sum", "D0")   # about 100 vs about 1


def test_fp32_noise_flag(cfg):
    """Softmax on D1: the ±20 outliers hold all the mass and quantize exactly, so the output
    error sits at FP32 rounding level and must be flagged; gelu on D0 must not be."""
    q1 = quantizers_from_config(cfg)["Q1"]
    x1 = sample(cfg, "D1")
    y32, y64 = baselines("softmax", x1)
    xh = quantize(x1, q1).x_hat
    m = evaluate("softmax", x1, xh, y32, y64, apply("softmax", xh))
    assert m["fp32_noise"] > 0 and m["at_fp32_noise"] == 1
    x0 = sample(cfg, "D0")
    y32, y64 = baselines("gelu", x0)
    xh = quantize(x0, q1).x_hat
    assert evaluate("gelu", x0, xh, y32, y64, apply("gelu", xh))["at_fp32_noise"] == 0


def test_kl_not_negative_beyond_fp64_rounding(cfg):
    for regime in cfg["regimes"]:
        x = sample(cfg, regime)
        y32, y64 = baselines("softmax", x)
        assert metrics.softmax_kl(y64, y32) > -1e-15


def test_top1_changed_counts_ties():
    p = apply("softmax", torch.tensor([0.1, 0.9, 0.85, 0.2]))
    tied = apply("softmax", torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert metrics.error_metrics("softmax", torch.zeros(4), p, tied)["top1_changed"] == 1


def test_make_row():
    row = make_row(regime="D0", operator="sum", abs_error=0.1)
    assert list(row) == CSV_FIELDS and row["softmax_kl"] is None
    with pytest.raises(KeyError):
        make_row(not_a_field=1)
