"""Inspector descriptors computed from the FP32 input x alone (plan §9.3 step 2).

Nothing here may look at a reference or quantized output. Each descriptor's runtime cost
class is recorded in DESCRIPTOR_COST for the Gate 5 cost comparison.
"""
from __future__ import annotations

import torch

from src.quantization import fake_quant

ABS_PERCENTILE = 0.99

DESCRIPTOR_COST = {
    "input_mean": "one pass",
    "input_var": "one pass",
    "input_absmax": "one pass",
    "input_abs_p99": "sort / selection",
    "max_over_p99": "sort / selection",
    "kurtosis": "one pass",
    "cancellation_ratio": "one pass",
    "top2_gap": "one pass",
    "q1_static_clip_fraction": "full quantization",
    "q1_static_zero_fraction": "full quantization",
}


def describe(x: torch.Tensor) -> dict:
    """Descriptors in the raw CSV schema. Undefined values (e.g. kurtosis of a constant) are None."""
    x = x.double()
    a = x.abs()
    mean = x.mean().item()
    var = x.var(correction=0).item()
    absmax = a.max().item()
    p99 = torch.quantile(a, ABS_PERCENTILE).item()   # linear interpolation
    l1 = a.sum().item()
    top2 = torch.topk(x, 2).values
    m4 = (x - mean).pow(4).mean().item()
    return {
        "input_mean": mean,
        "input_var": var,
        "input_absmax": absmax,
        "input_abs_p99": p99,
        "max_over_p99": absmax / p99 if p99 > 0 else None,
        "kurtosis": m4 / var**2 - 3 if var > 0 else None,   # excess kurtosis
        "cancellation_ratio": abs(x.sum().item()) / l1 if l1 > 0 else None,
        "top2_gap": (top2[0] - top2[1]).item(),
    }


def q1_static_descriptors(x: torch.Tensor, q1_static_scale: torch.Tensor, bits: int = 4) -> dict:
    """Clip and zero fractions of x under the frozen static Q1 scale (needs Gate 1 scales)."""
    r = fake_quant(x, bits, q1_static_scale)
    return {"q1_static_clip_fraction": r.clip_fraction, "q1_static_zero_fraction": r.zero_fraction}
