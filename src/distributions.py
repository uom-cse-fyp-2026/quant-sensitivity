"""Input regimes D0-D5 and their sweep variants (plan §4), with the per-sample seed rule (§7.1).

Every tensor is generated in FP32 from its own torch.Generator; the global RNG is never used.
The order of random draws inside each generator is part of the definition: changing it
changes every tensor, so treat it as frozen once Gate 1 has run.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import torch

DTYPE = torch.float32


def sample_seed(split: str, stratum_key: str, sample_id: int, hex_digits: int = 16) -> int:
    digest = hashlib.sha256(f"{split}|{stratum_key}|{sample_id}".encode()).hexdigest()
    return int(digest[:hex_digits], 16)


def make_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def randn(n: int, g: torch.Generator) -> torch.Tensor:
    """Standard normals drawn in FP64, then rounded to FP32.

    torch's FP32 randn uses platform float math and differed in the last bit between the
    Linux VM and a Windows laptop; the FP64 draw and the FP64 -> FP32 cast were bit-identical.
    """
    return torch.randn(n, generator=g, dtype=torch.float64).to(DTYPE)


def balanced_signs(k: int, g: torch.Generator) -> torch.Tensor:
    """k // 2 positive and k // 2 negative signs in random order, plus one random sign if k is odd."""
    signs = [1.0] * (k // 2) + [-1.0] * (k // 2)
    if k % 2:
        signs.append(1.0 if torch.randint(0, 2, (1,), generator=g).item() else -1.0)
    signs = torch.tensor(signs, dtype=DTYPE)
    return signs[torch.randperm(k, generator=g)]


def d0(n: int, g: torch.Generator, scale: float = 1.0) -> torch.Tensor:
    x = randn(n, g)
    return x if scale == 1.0 else x * scale


def d1(n: int, g: torch.Generator, outlier_rate: float, outlier_magnitude: float, scale: float = 1.0) -> torch.Tensor:
    x = d0(n, g, scale)
    k = round(outlier_rate * n)
    if k == 0:
        return x
    pos = torch.randperm(n, generator=g)[:k]
    x[pos] = balanced_signs(k, g) * outlier_magnitude
    return x


def d2(n: int, g: torch.Generator, offset: float, noise_std: float) -> torch.Tensor:
    return offset + noise_std * randn(n, g)


def d3(n: int, g: torch.Generator, sigma_delta: float) -> torch.Tensor:
    if n % 2:
        raise ValueError("D3 needs an even length")
    a = randn(n // 2, g)
    delta = sigma_delta * randn(n // 2, g)
    x = torch.cat([a, -a + delta])
    return x[torch.randperm(n, generator=g)]


def d4(n: int, g: torch.Generator, df: int) -> torch.Tensor:
    """Student-t(df) scaled to unit variance, built from normals so it uses the generator."""
    if int(df) != df or df <= 2:
        raise ValueError("D4 needs an integer df > 2")
    df = int(df)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    chi2 = torch.randn(df, n, generator=g, dtype=torch.float64).square().sum(dim=0)
    t = z / torch.sqrt(chi2 / df)
    return (t / (df / (df - 2)) ** 0.5).to(DTYPE)


@lru_cache(maxsize=None)
def d5_channels(n: int, n_channels: int, channel_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed outlier channel positions and signs, identical for every sample and split."""
    g = make_generator(channel_seed)
    pos = torch.randperm(n, generator=g)[:n_channels]
    return pos, balanced_signs(n_channels, g)


def d5(n: int, g: torch.Generator, n_channels: int, channel_magnitude: float, channel_seed: int) -> torch.Tensor:
    x = d0(n, g)
    pos, signs = d5_channels(n, n_channels, channel_seed)
    x[pos] += signs * channel_magnitude
    return x


GENERATORS = {"D0": d0, "D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5}


def generate(regime: str, n: int, params: dict, seed: int) -> torch.Tensor:
    x = GENERATORS[regime](n, make_generator(seed), **params)
    assert x.dtype == DTYPE and x.shape == (n,)
    return x


def generate_stratum(stratum, seed: int) -> torch.Tensor:
    return generate(stratum.regime, stratum.n, stratum.params, seed)
