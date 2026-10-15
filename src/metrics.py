"""Error metrics, amplification factor, FP32 floor and the raw CSV row schema (plan §6, §8.2).

Every metric is computed in FP64 from the stored outputs. A metric that does not apply to
an operator is left out of the dict, so it becomes an empty CSV field, never 0.
"""
from __future__ import annotations

import math

import torch

PROB_CLAMP = 1e-12
NORM_EPS = 1e-12
KAPPA_NULL_REF_NORM = 1e-12
COLLAPSED_STD = 1e-3
AT_FLOOR_FACTOR = 10
# kappa is a ratio of relative errors, so it needs a wider margin: at 1000x the floor, kappa
# agreed across machines to 0.4%; at 10-30x it differed by up to 73% (Gate 0 pilot).
KAPPA_FLOOR_FACTOR = 1000

PRIMARY = {"sum": "abs_error", "softmax": "softmax_kl", "layer_norm": "norm_l2", "gelu": "norm_l2"}

CSV_FIELDS = [
    "run_id", "git_commit", "split", "regime", "sweep_id", "sweep_value", "sample_id", "seed",
    "operator", "quantizer", "calib_policy", "bits", "granularity", "group_size", "effective_bits", "n",
    "input_mean", "input_var", "input_absmax", "input_abs_p99", "max_over_p99", "kurtosis",
    "cancellation_ratio", "top2_gap",
    "clip_fraction", "zero_fraction", "input_mse", "input_rel_l2",
    "abs_error", "l1_norm_error", "norm_l2", "sqnr_db", "mae", "max_abs_error", "softmax_kl",
    "top1_changed", "collapsed", "kappa", "fp32_floor", "at_floor", "rel_l2_floor", "kappa_at_floor",
    "ref_finite", "quant_finite",
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

    Renormalising matters: FP32 probabilities sum to 1 only within ~1e-7, and without it
    KL(p64 || p32) can come out negative.
    """
    a, b = p_ref.double().clamp_min(PROB_CLAMP), p.double().clamp_min(PROB_CLAMP)
    a, b = a / a.sum(), b / b.sum()
    return (a * (a.log() - b.log())).sum().item()


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
        out["top1_changed"] = int(_argmax_set(y) != _argmax_set(y_ref))
    if op == "layer_norm":
        out["collapsed"] = int(y.std(correction=0).item() < COLLAPSED_STD)
    return out


def amplification(x: torch.Tensor, x_hat: torch.Tensor, y_ref: torch.Tensor, y: torch.Tensor) -> float | None:
    """kappa = output relative L2 error / input relative L2 error; None when undefined (plan §6.3)."""
    x, x_hat, y_ref, y = x.double(), x_hat.double(), y_ref.double(), y.double()
    dx, ref = _norm(x_hat - x), _norm(y_ref)
    if dx == 0 or ref < KAPPA_NULL_REF_NORM:
        return None
    return (_norm(y - y_ref) / ref) / (dx / _norm(x))


def fp32_floor(op: str, x: torch.Tensor, y_ref32: torch.Tensor, y_ref64: torch.Tensor) -> float:
    """The primary metric of the FP32 reference measured against the FP64 reference."""
    return error_metrics(op, x, y_ref64, y_ref32)[PRIMARY[op]]


def rel_l2_floor(y_ref32: torch.Tensor, y_ref64: torch.Tensor) -> float | None:
    """Relative L2 error of the FP32 reference against FP64: the noise level of kappa's numerator."""
    ref = _norm(y_ref64.double())
    return _norm(y_ref32.double() - y_ref64.double()) / ref if ref >= KAPPA_NULL_REF_NORM else None


def evaluate(op: str, x: torch.Tensor, x_hat: torch.Tensor,
             y_ref32: torch.Tensor, y_ref64: torch.Tensor, y_q: torch.Tensor) -> dict:
    """All per-row metric fields for one (input, operator, quantizer, policy)."""
    m = error_metrics(op, x, y_ref32, y_q)
    floor = fp32_floor(op, x, y_ref32, y_ref64)
    x_norm = _norm(x.double())
    l2_floor = rel_l2_floor(y_ref32, y_ref64)
    ref_norm = _norm(y_ref32.double())
    out_rel = _norm(y_q.double() - y_ref32.double()) / ref_norm if ref_norm >= KAPPA_NULL_REF_NORM else None
    m.update(
        input_rel_l2=_norm(x_hat.double() - x.double()) / x_norm if x_norm > 0 else None,
        kappa=amplification(x, x_hat, y_ref32, y_q),
        fp32_floor=floor,
        at_floor=int(m[PRIMARY[op]] <= AT_FLOOR_FACTOR * abs(floor)),
        rel_l2_floor=l2_floor,
        # kappa's output error is within 1000x of FP32's own rounding: kappa is noise, not signal
        kappa_at_floor=(None if out_rel is None or l2_floor is None
                        else int(out_rel <= KAPPA_FLOOR_FACTOR * l2_floor)),
        ref_finite=int(torch.isfinite(y_ref32).all().item()),
        quant_finite=int(torch.isfinite(y_q).all().item() and torch.isfinite(x_hat).all().item()),
    )
    return m


def quantizer_fields(qz) -> dict:
    return {"quantizer": qz.id, "bits": qz.bits, "granularity": qz.granularity,
            "group_size": qz.group_size, "effective_bits": qz.effective_bits}


def make_row(**fields) -> dict:
    """One CSV row in schema order; missing fields are None (written as empty)."""
    unknown = set(fields) - set(CSV_FIELDS)
    if unknown:
        raise KeyError(f"fields not in the raw schema: {sorted(unknown)}")
    return {k: fields.get(k) for k in CSV_FIELDS}
