"""Operators under test (plan §3). The same call runs on FP32 and FP64 inputs.

Only the operator's input is quantized; shape, axis, eps and arithmetic are identical
across quantizers. layer_norm takes its normalised shape from the input length so the
S3 length sweep uses the same definition.
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
    call: str      # exact PyTorch call, recorded for the report


def _sum(x):
    return x.sum(dim=-1)


def _softmax(x):
    return torch.softmax(x, dim=-1)


def _layer_norm(x):
    return F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=LAYER_NORM_EPS)


def _gelu(x):
    return F.gelu(x, approximate="none")


OPERATORS = {
    "sum": Operator("sum", _sum, "reduction", "scalar", "x.sum(dim=-1)"),
    "softmax": Operator("softmax", _softmax, "reduction", "vector", "torch.softmax(x, dim=-1)"),
    "layer_norm": Operator("layer_norm", _layer_norm, "reduction", "vector",
                           "F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=1e-5)"),
    "gelu": Operator("gelu", _gelu, "elementwise", "vector", "F.gelu(x, approximate='none')"),
}


def apply(name: str, x: torch.Tensor) -> torch.Tensor:
    return OPERATORS[name].fn(x)


def references(name: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 reference (primary) and FP64 reference (numerical floor), computed once per input."""
    if x.dtype != torch.float32:
        raise ValueError("references expect the FP32 input tensor")
    return apply(name, x), apply(name, x.double())
