"""Quantization schemes Q0-Q2 (plan §3).

All schemes are signed, symmetric, uniform, round-half-even and saturating with zero point 0.
The scale is the input's own absmax / qmax, per tensor or per contiguous group:
A_hat = s * clamp(round(A / s), -qmax, qmax), computed in FP32 and fed to the FP32 operator.

This is fake (simulated) quantization: values are rounded to the integer grid and immediately
dequantized, so the operator itself still runs in FP32 and only the input carries the error.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Lower bound on the scale, so an all-zero input does not cause a division by zero.
SCALE_FLOOR = 1e-12


@dataclass(frozen=True)
class Quantizer:
    """One quantization scheme.

    Attributes:
        id: Short identifier, e.g. "Q1".
        label: Human-readable name, e.g. "A4 per-tensor".
        bits: Bit width of the signed integer codes.
        group_size: Values per scale for per-group quantization, or None for one scale per tensor.
    """
    id: str
    label: str
    bits: int
    group_size: int | None

    @property
    def qmax(self) -> int:
        return qmax(self.bits)


@dataclass
class QuantResult:
    """The output of quantizing one tensor, with diagnostics."""
    x_hat: torch.Tensor        # dequantized values, FP32, same shape as x
    codes: torch.Tensor        # integer codes, stored as int16
    scale: torch.Tensor        # 0-dim (per tensor) or [n / group] (per group)
    zero_fraction: float       # share of codes equal to 0 (values lost to rounding)


def quantizers_from_config(cfg: dict) -> dict[str, Quantizer]:
    """Build the schemes listed under `quantizers` in the configuration."""
    return {qid: Quantizer(qid, q["label"], q["bits"], q["group_size"])
            for qid, q in cfg["quantizers"].items()}


def qmax(bits: int) -> int:
    """Largest code of a symmetric signed grid: 127 for 8 bits, 7 for 4 bits.

    The most negative code (-128 or -8) is deliberately unused so the grid is symmetric.
    """
    return (1 << (bits - 1)) - 1


def absmax_scale(x: torch.Tensor, bits: int, group: int | None = None) -> torch.Tensor:
    """absmax / qmax of a 1-D tensor: one scale, or one per contiguous group.

    With this scale the largest magnitude maps exactly to the top code, so nothing is clipped.
    """
    if group is None:
        return x.abs().amax().div(qmax(bits)).clamp_min(SCALE_FLOOR)
    if x.shape[-1] % group:
        raise ValueError(f"length {x.shape[-1]} is not divisible by group size {group}")
    # Rows of the reshaped tensor are the contiguous groups.
    return x.reshape(-1, group).abs().amax(dim=1).div(qmax(bits)).clamp_min(SCALE_FLOOR)


def fake_quant(x: torch.Tensor, bits: int, scale: torch.Tensor, group: int | None = None) -> QuantResult:
    """Quantize and dequantize x with a given scale (one value, or one per group).

    torch.round rounds halfway values to the nearest even integer; clamp saturates values
    beyond the grid instead of letting them wrap around.
    """
    if x.dim() != 1:
        raise ValueError("fake_quant expects a 1-D tensor")
    if group is not None and scale.numel() * group != x.numel():
        raise ValueError(f"{scale.numel()} group scales do not cover length {x.numel()}")
    # Expand per-group scales to one scale per element.
    s = scale if group is None else scale.repeat_interleave(group)
    m = qmax(bits)
    q = torch.round(x / s).clamp(-m, m)
    return QuantResult(x_hat=(q * s).to(torch.float32), codes=q.to(torch.int16), scale=scale,
                       zero_fraction=(q == 0).double().mean().item())


def quantize(x: torch.Tensor, qz: Quantizer) -> QuantResult:
    """Apply a scheme to x, taking the scale from x's own absmax (dynamic scaling)."""
    return fake_quant(x, qz.bits, absmax_scale(x, qz.bits, qz.group_size), qz.group_size)
