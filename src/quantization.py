"""Fake quantizers Q0-Q2 and the static / dynamic calibration policies (plan §5).

All quantizers are signed, symmetric, uniform, round-half-even and saturating with zero
point 0. x_hat = s * clamp(round(x / s), -qmax, qmax) is computed in FP32 and fed to the
ordinary FP32 operator. Scales stay in FP32 for every quantizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

SCALE_FLOOR = 1e-12
CLIP_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Quantizer:
    id: str
    label: str
    bits: int
    group_size: int | None
    effective_bits: float

    @property
    def granularity(self) -> str:
        return "per_tensor" if self.group_size is None else "per_group"

    @property
    def qmax(self) -> int:
        return qmax(self.bits)


@dataclass
class QuantResult:
    x_hat: torch.Tensor        # FP32, same shape as x
    codes: torch.Tensor        # int16 integer codes
    scale: torch.Tensor        # 0-dim (per tensor) or [n / group] (per group)
    clip_fraction: float
    zero_fraction: float
    input_mse: float


def quantizers_from_config(cfg: dict) -> dict[str, Quantizer]:
    return {qid: Quantizer(qid, q["label"], q["bits"], q["group_size"], q["effective_bits"])
            for qid, q in cfg["quantizers"].items() if qid.startswith("Q")}


def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def absmax_scale(x: torch.Tensor, bits: int, group: int | None = None) -> torch.Tensor:
    """absmax / qmax over every leading dim of x: one scale per tensor, or one per group position.

    x is [n] for a dynamic scale or [N, n] for a static scale over a calibration batch.
    """
    if group is None:
        return x.abs().amax().div(qmax(bits)).clamp_min(SCALE_FLOOR)
    if x.shape[-1] % group:
        raise ValueError(f"length {x.shape[-1]} is not divisible by group size {group}")
    g = x.reshape(*x.shape[:-1], -1, group)
    dims = tuple(range(g.dim() - 2)) + (g.dim() - 1,)
    return g.abs().amax(dim=dims).div(qmax(bits)).clamp_min(SCALE_FLOOR)


def fake_quant(x: torch.Tensor, bits: int, scale: torch.Tensor, group: int | None = None) -> QuantResult:
    if x.dim() != 1:
        raise ValueError("fake_quant expects a 1-D tensor")
    if group is not None and scale.numel() * group != x.numel():
        raise ValueError(f"{scale.numel()} group scales do not cover length {x.numel()}")
    s = scale if group is None else scale.repeat_interleave(group)
    m = qmax(bits)
    q = torch.round(x / s).clamp(-m, m)
    x_hat = (q * s).to(torch.float32)
    clipped = x.abs() > m * s * (1 + CLIP_TOLERANCE)
    err = x_hat.double() - x.double()
    return QuantResult(
        x_hat=x_hat,
        codes=q.to(torch.int16),
        scale=scale,
        clip_fraction=clipped.double().mean().item(),
        zero_fraction=(q == 0).double().mean().item(),
        input_mse=err.square().mean().item(),
    )


def dynamic_scale(x: torch.Tensor, qz: Quantizer) -> torch.Tensor:
    return absmax_scale(x, qz.bits, qz.group_size)


def static_scale(x_cal: torch.Tensor, qz: Quantizer) -> torch.Tensor:
    """Scale from a [N, n] calibration batch; computed once and frozen (Gate 1)."""
    if x_cal.dim() != 2:
        raise ValueError("static calibration expects a [N, n] batch")
    return absmax_scale(x_cal, qz.bits, qz.group_size)


def quantize(x: torch.Tensor, qz: Quantizer, scale: torch.Tensor | None = None) -> QuantResult:
    """Quantize with a frozen static scale, or with a dynamic scale when scale is None."""
    if scale is None:
        scale = dynamic_scale(x, qz)
    return fake_quant(x, qz.bits, scale, qz.group_size)
