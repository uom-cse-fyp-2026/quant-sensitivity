import math
import re

import pytest
import torch

from src import metrics
from src.config import ROOT, regime_strata, sweep_strata
from src.distributions import generate_stratum, sample_seed
from src.features import describe
from src.metrics import (CSV_FIELDS, PRIMARY, amplification, evaluate, make_row, quantizer_fields)
from src.operators import OPERATORS, apply, references
from src.quantization import quantize, quantizers_from_config

OPS = list(OPERATORS)


def regime_inputs(cfg, k=3):
    for st in regime_strata(cfg):
        for i in range(k):
            yield st.key, generate_stratum(st, sample_seed("pilot", st.key, i))


def test_metric_constants_match_config(cfg):
    m = cfg["metrics"]
    assert metrics.PROB_CLAMP == m["probability_clamp"]
    assert metrics.NORM_EPS == m["norm_l2_eps"]
    assert metrics.KAPPA_NULL_REF_NORM == m["kappa_null_ref_norm"]
    assert metrics.COLLAPSED_STD == m["collapsed_output_std"]
    assert metrics.AT_FLOOR_FACTOR == cfg["references"]["at_floor_factor"]
    assert metrics.KAPPA_FLOOR_FACTOR == m["kappa_floor_factor"]
    assert PRIMARY == m["primary"]
    assert set(OPS) == set(cfg["operators"]["core"])


def test_csv_schema_matches_plan():
    plan = (ROOT / "docs" / "experiment_plan.md").read_text(encoding="utf-8")
    line = re.search(r"^run_id,.*$", plan, re.M).group(0)
    assert CSV_FIELDS == line.split(",")


def test_references_dtypes():
    x = torch.randn(1024, generator=torch.Generator().manual_seed(0))
    for op in OPS:
        y32, y64 = references(op, x)
        assert y32.dtype == torch.float32 and y64.dtype == torch.float64
        assert y32.shape == y64.shape


@pytest.mark.parametrize("op", OPS)
def test_identity_gives_zero_error(cfg, op):
    for key, x in regime_inputs(cfg):
        y32, y64 = references(op, x)
        m = evaluate(op, x, x.clone(), y32, y64, y32.clone())
        for f in ("abs_error", "l1_norm_error", "norm_l2", "mae", "max_abs_error", "softmax_kl", "input_rel_l2"):
            if m.get(f) is not None:
                assert m[f] == 0, (key, f)
        if "sqnr_db" in m:
            assert m["sqnr_db"] == math.inf
        if op == "softmax":
            assert m["top1_changed"] == 0
        assert m["kappa"] is None and m["at_floor"] == 1
        assert m["ref_finite"] == m["quant_finite"] == 1


@pytest.mark.parametrize("op", OPS)
def test_inapplicable_fields_absent(op):
    x = torch.randn(1024, generator=torch.Generator().manual_seed(1))
    y32, y64 = references(op, x)
    m = evaluate(op, x, x * 1.01, y32, y64, apply(op, x * 1.01))
    vector_only = {"norm_l2", "sqnr_db", "mae", "max_abs_error"}
    if op == "sum":
        assert vector_only.isdisjoint(m) and "abs_error" in m and "l1_norm_error" in m
    else:
        assert vector_only <= set(m) and "abs_error" not in m
    assert ("softmax_kl" in m) == ("top1_changed" in m) == (op == "softmax")
    assert ("collapsed" in m) == (op == "layer_norm")


def test_softmax_valid_on_every_regime_and_quantizer(cfg):
    qzs = quantizers_from_config(cfg)
    strata = regime_strata(cfg) + [s for s in sweep_strata(cfg) if s.sweep_id == "S4"]
    for st in strata:
        x = generate_stratum(st, sample_seed("pilot", st.key, 0))
        for x_in in [x] + [quantize(x, qz).x_hat for qz in qzs.values()]:
            p = apply("softmax", x_in)
            assert torch.isfinite(p).all() and (p >= 0).all(), st.key
            assert abs(p.double().sum().item() - 1) < 1e-6, st.key


def test_kappa_null_cases():
    x = torch.randn(64, generator=torch.Generator().manual_seed(2))
    y = apply("gelu", x)
    assert amplification(x, x.clone(), y, y) is None                       # no input error
    pair = torch.tensor([1.5, -1.5])
    assert amplification(pair, pair * 1.1, apply("sum", pair), apply("sum", pair * 1.1)) is None  # ref ~ 0


def test_kappa_gelu_near_one_and_sum_large_under_cancellation(cfg):
    qz = quantizers_from_config(cfg)["Q0"]
    s = {st.key: st for st in regime_strata(cfg)}
    x0 = generate_stratum(s["D0"], 11)
    r = quantize(x0, qz)
    k_gelu = amplification(x0, r.x_hat, apply("gelu", x0), apply("gelu", r.x_hat))
    assert 0.2 < k_gelu < 5
    def median_sum_kappa(regime):
        ks = []
        for seed in range(30):
            x = generate_stratum(s[regime], seed)
            rq = quantize(x, qz)
            ks.append(amplification(x, rq.x_hat, apply("sum", x), apply("sum", rq.x_hat)))
        return sorted(ks)[len(ks) // 2]
    assert median_sum_kappa("D3") > 20 * median_sum_kappa("D0")   # about 100 vs about 1


def test_fp32_floor_is_small_on_d0(cfg):
    x = generate_stratum(regime_strata(cfg)[0], 5)
    for op in OPS:
        y32, y64 = references(op, x)
        floor = metrics.fp32_floor(op, x, y32, y64)
        assert 0 <= floor < 1e-5, op


def test_collapsed_flag_on_constant_codes():
    x = 1 + 0.001 * torch.randn(1024, generator=torch.Generator().manual_seed(3))
    y32, y64 = references("layer_norm", x)
    flat = torch.full_like(x, 1.0)
    m = evaluate("layer_norm", x, flat, y32, y64, apply("layer_norm", flat))
    assert m["collapsed"] == 1


def test_top1_changed_counts_ties():
    x = torch.tensor([0.1, 0.9, 0.85, 0.2])
    p = apply("softmax", x)
    tied = apply("softmax", torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert metrics.error_metrics("softmax", x, p, tied)["top1_changed"] == 1


def test_layer_norm_follows_length():
    x = torch.randn(256, generator=torch.Generator().manual_seed(4))
    y = apply("layer_norm", x)
    assert y.shape == (256,) and abs(y.mean().item()) < 1e-5


def test_descriptors():
    x = torch.randn(1024, generator=torch.Generator().manual_seed(6))
    d = describe(x)
    assert abs(d["kurtosis"]) < 0.6 and d["max_over_p99"] >= 1
    assert d["top2_gap"] >= 0
    c = describe(torch.ones(1024))
    assert c["kurtosis"] is None and c["max_over_p99"] == 1 and c["cancellation_ratio"] == 1
    z = describe(torch.zeros(1024))
    assert z["cancellation_ratio"] is None and z["max_over_p99"] is None


def test_make_row(cfg):
    qz = quantizers_from_config(cfg)["Q2"]
    row = make_row(split="pilot", operator="sum", **quantizer_fields(qz), abs_error=0.1)
    assert list(row) == CSV_FIELDS
    assert row["group_size"] == 32 and row["granularity"] == "per_group" and row["softmax_kl"] is None
    with pytest.raises(KeyError):
        make_row(not_a_field=1)


def test_kappa_at_floor_flags_fp32_noise(cfg):
    """Softmax on D1 under dynamic scales: outliers keep all the mass and quantize exactly,
    so the output error sits at FP32 resolution and kappa must be flagged."""
    qz = quantizers_from_config(cfg)["Q1"]
    s = {st.key: st for st in regime_strata(cfg)}
    x1 = generate_stratum(s["D1"], sample_seed("pilot", "D1", 0))
    r = quantize(x1, qz)
    y32, y64 = references("softmax", x1)
    m = evaluate("softmax", x1, r.x_hat, y32, y64, apply("softmax", r.x_hat))
    assert m["rel_l2_floor"] > 0 and m["kappa_at_floor"] == 1

    x0 = generate_stratum(s["D0"], sample_seed("pilot", "D0", 0))
    r0 = quantize(x0, qz)
    y32, y64 = references("gelu", x0)
    m0 = evaluate("gelu", x0, r0.x_hat, y32, y64, apply("gelu", r0.x_hat))
    assert m0["kappa_at_floor"] == 0
