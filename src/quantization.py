"""Quantization schemes Q0-Q2 (plan §3).

All schemes are signed, symmetric, uniform, round-half-even and saturating with zero point 0.
The scale is the input's own absmax / qmax, per tensor or per contiguous group:
A_hat = s * clamp(round(A / s), -qmax, qmax), computed in FP32 and fed to the FP32 operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

SCALE_FLOOR = 1e-12


@dataclass(frozen=True)
class Quantizer:
    id: str
    label: str
    bits: int
    group_size: int | None

    @property
    def qmax(self) -> int:
        return qmax(self.bits)


@dataclass
class QuantResult:
    x_hat: torch.Tensor        # FP32, same shape as x
    codes: torch.Tensor        # int16 integer codes
    scale: torch.Tensor        # 0-dim (per tensor) or [n / group] (per group)
    zero_fraction: float       # share of codes equal to 0


def quantizers_from_config(cfg: dict) -> dict[str, Quantizer]:
    return {qid: Quantizer(qid, q["label"], q["bits"], q["group_size"])
            for qid, q in cfg["quantizers"].items()}


def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def absmax_scale(x: torch.Tensor, bits: int, group: int | None = None) -> torch.Tensor:
    """absmax / qmax of a 1-D tensor: one scale, or one per contiguous group."""
    if group is None:
        return x.abs().amax().div(qmax(bits)).clamp_min(SCALE_FLOOR)
    if x.shape[-1] % group:
        raise ValueError(f"length {x.shape[-1]} is not divisible by group size {group}")
    return x.reshape(-1, group).abs().amax(dim=1).div(qmax(bits)).clamp_min(SCALE_FLOOR)


def fake_quant(x: torch.Tensor, bits: int, scale: torch.Tensor, group: int | None = None) -> QuantResult:
    if x.dim() != 1:
        raise ValueError("fake_quant expects a 1-D tensor")
    if group is not None and scale.numel() * group != x.numel():
        raise ValueError(f"{scale.numel()} group scales do not cover length {x.numel()}")
    s = scale if group is None else scale.repeat_interleave(group)
    m = qmax(bits)
    q = torch.round(x / s).clamp(-m, m)
    return QuantResult(x_hat=(q * s).to(torch.float32), codes=q.to(torch.int16), scale=scale,
                       zero_fraction=(q == 0).double().mean().item())


def quantize(x: torch.Tensor, qz: Quantizer) -> QuantResult:
    return fake_quant(x, qz.bits, absmax_scale(x, qz.bits, qz.group_size), qz.group_size)
