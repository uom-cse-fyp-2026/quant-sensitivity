"""Comparing quantized outputs Y_q with the FP32 baseline Y (plan §4), and the CSV row schema.

Every metric is computed in FP64. A metric that does not apply to an operator is left out of
the dict, so it becomes an empty CSV field, never 0.
"""
from __future__ import annotations

import math

import torch

PROB_CLAMP = 1e-12
NORM_EPS = 1e-12
KAPPA_NULL_REF_NORM = 1e-12
# An output error within 1000x of the baseline's own FP32 rounding is FP32 noise, not
# quantization: across two machines kappa varied by up to 73% at 10-30x and 0.4% above 1000x.
FP32_NOISE_FACTOR = 1000

HEADLINE = {"sum": "l1_norm_error", "softmax": "softmax_kl", "layer_norm": "norm_l2", "gelu": "norm_l2"}

CSV_FIELDS = [
    "run_id", "git_commit", "regime", "sample_id", "seed", "operator", "quantizer", "bits",
    "group_size", "n", "input_rel_l2", "zero_fraction",
    "abs_error", "l1_norm_error", "norm_l2", "sqnr_db", "mae", "max_abs_error", "softmax_kl",
    "top1_changed", "kappa", "fp32_noise", "at_fp32_noise", "ref_finite", "quant_finite",
]


def _norm(t: torch.Tensor) -> float:
    return torch.linalg.vector_norm(t).item()


def _argmax_set(y: torch.Tensor) -> set[int]:
    return set((y == y.max()).nonzero().flatten().tolist())


def norm_l2(y_ref: torch.Tensor, y: torch.Tensor) -> float:
    return _norm(y - y_ref) / (_norm(y_ref) + NORM_EPS)


def sqnr_db(nl2: float) -> float:
    return math.inf if nl2 == 0 else -20 * math.log10(nl2)


def softmax_kl(p_ref: torch.Tensor, p: torch.Tensor) -> float:
    """KL(p_ref || p) in nats on clamped, renormalised FP64 distributions.

    Renormalising keeps it >= 0 up to FP64 rounding (about 1e-16) when p is an FP32 softmax,
    whose probabilities sum to 1 only within about 1e-7.
    """
    a, b = p_ref.double().clamp_min(PROB_CLAMP), p.double().clamp_min(PROB_CLAMP)
    a, b = a / a.sum(), b / b.sum()
    return (a * (a.log() - b.log())).sum().item()


def relative_error(y_ref: torch.Tensor, y: torch.Tensor) -> float | None:
    ref = _norm(y_ref.double())
    return _norm(y.double() - y_ref.double()) / ref if ref >= KAPPA_NULL_REF_NORM else None


def error_metrics(op: str, x: torch.Tensor, y_ref: torch.Tensor, y: torch.Tensor) -> dict:
    """Errors of y against y_ref for one operator. x is the original (unquantized) input."""
    x, y_ref, y = x.double(), y_ref.double(), y.double()
    if op == "sum":
        abs_err = (y - y_ref).abs().item()
        l1 = x.abs().sum().item()
        return {"abs_error": abs_err, "l1_norm_error": abs_err / l1 if l1 > 0 else None}
    diff = (y - y_ref).abs()
    nl2 = norm_l2(y_ref, y)
    out = {"norm_l2": nl2, "sqnr_db": sqnr_db(nl2), "mae": diff.mean().item(), "max_abs_error": diff.max().item()}
    if op == "softmax":
        out["softmax_kl"] = softmax_kl(y_ref, y)
        # changed unless the baseline's top-1 is the unique maximum (ties from coarse codes count)
        out["top1_changed"] = int(_argmax_set(y) != _argmax_set(y_ref))
    return out


def amplification(x: torch.Tensor, x_hat: torch.Tensor, y_ref: torch.Tensor, y: torch.Tensor) -> float | None:
    """kappa = output relative error / input relative error; None when undefined."""
    dx = _norm(x_hat.double() - x.double())
    out_rel = relative_error(y_ref, y)
    if dx == 0 or out_rel is None:
        return None
    return out_rel / (dx / _norm(x.double()))


def evaluate(op: str, x: torch.Tensor, x_hat: torch.Tensor,
             y32: torch.Tensor, y64: torch.Tensor, y_q: torch.Tensor) -> dict:
    """All metric fields for one (input, operator, scheme). y32 is the baseline Y."""
    m = error_metrics(op, x, y32, y_q)
    x_norm = _norm(x.double())
    noise = relative_error(y64, y32)
    out_rel = relative_error(y32, y_q)
    m.update(
        input_rel_l2=_norm(x_hat.double() - x.double()) / x_norm if x_norm > 0 else None,
        kappa=amplification(x, x_hat, y32, y_q),
        fp32_noise=noise,
        at_fp32_noise=None if noise is None or out_rel is None else int(out_rel <= FP32_NOISE_FACTOR * noise),
        ref_finite=int(torch.isfinite(y32).all().item()),
        quant_finite=int(torch.isfinite(y_q).all().item() and torch.isfinite(x_hat).all().item()),
    )
    return m


def make_row(**fields) -> dict:
    """One CSV row in schema order; missing fields are None (written as empty)."""
    unknown = set(fields) - set(CSV_FIELDS)
    if unknown:
        raise KeyError(f"fields not in the CSV schema: {sorted(unknown)}")
    return {k: fields.get(k) for k in CSV_FIELDS}
