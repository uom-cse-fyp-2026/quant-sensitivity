"""Operators under test (plan §1). The same call runs on FP32 and FP64 inputs.

Only the operator's input is quantized; shape, axis, eps and arithmetic are identical for the
baseline and every scheme.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

LAYER_NORM_EPS = 1e-5


@dataclass(frozen=True)
class Operator:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    kind: str      # "reduction" or "elementwise"
    output: str    # "scalar" or "vector"


def _sum(x):
    return x.sum(dim=-1)


def _softmax(x):
    return torch.softmax(x, dim=-1)


def _layer_norm(x):
    return F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=LAYER_NORM_EPS)


def _gelu(x):
    return F.gelu(x, approximate="none")


OPERATORS = {
    "sum": Operator("sum", _sum, "reduction", "scalar"),
    "softmax": Operator("softmax", _softmax, "reduction", "vector"),
    "layer_norm": Operator("layer_norm", _layer_norm, "reduction", "vector"),
    "gelu": Operator("gelu", _gelu, "elementwise", "vector"),
}


def apply(name: str, x: torch.Tensor) -> torch.Tensor:
    return OPERATORS[name].fn(x)


def baselines(name: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Y = Op(A, NoQuant) in FP32 (the baseline), and Op(A) in FP64 (to measure FP32 noise)."""
    if x.dtype != torch.float32:
        raise ValueError("baselines expect the FP32 input tensor")
    return apply(name, x), apply(name, x.double())
