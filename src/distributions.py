"""Input distributions D0-D5 (plan §2) and the per-sample seed rule.

Every tensor is generated in FP32 from its own torch.Generator; the global RNG is never used.
The order of random draws inside each generator is part of the definition: changing it
changes every tensor (the pinned-hash test will fail).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import torch

DTYPE = torch.float32


def sample_seed(regime: str, sample_id: int, hex_digits: int = 16) -> int:
    """Seed derived from the sample's identity rather than from a running counter.

    Hashing "<regime>|<sample_id>" means adding or reordering distributions never changes an
    existing tensor. Sixteen hex digits give a 64-bit seed, so collisions are negligible.
    """
    digest = hashlib.sha256(f"{regime}|{sample_id}".encode()).hexdigest()
    return int(digest[:hex_digits], 16)


def make_generator(seed: int) -> torch.Generator:
    """A private CPU generator, so sampling never touches the global RNG state."""
    return torch.Generator(device="cpu").manual_seed(seed)


def randn(n: int, g: torch.Generator) -> torch.Tensor:
    """Standard normals drawn in FP64, then rounded to FP32.

    torch's FP32 randn uses platform float math and differed in the last bit between the
    Linux VM and a Windows laptop; the FP64 draw and the FP64 -> FP32 cast were bit-identical.
    """
    return torch.randn(n, generator=g, dtype=torch.float64).to(DTYPE)


def balanced_signs(k: int, g: torch.Generator) -> torch.Tensor:
    """k // 2 positive and k // 2 negative signs in random order, plus one random sign if k is odd.

    Balanced signs keep the outliers from shifting the mean, so their effect on the
    operators comes from their magnitude rather than from a bias.
    """
    signs = [1.0] * (k // 2) + [-1.0] * (k // 2)
    if k % 2:
        signs.append(1.0 if torch.randint(0, 2, (1,), generator=g).item() else -1.0)
    signs = torch.tensor(signs, dtype=DTYPE)
    return signs[torch.randperm(k, generator=g)]


def d0(n: int, g: torch.Generator) -> torch.Tensor:
    """D0, normal: x ~ N(0, 1). The well-behaved baseline input."""
    return randn(n, g)


def d1(n: int, g: torch.Generator, outlier_rate: float, outlier_magnitude: float) -> torch.Tensor:
    """D1, random outliers: D0 with round(rate * n) positions overwritten by ±magnitude.

    The positions change from sample to sample, unlike the fixed channels of D5.
    """
    x = randn(n, g)
    k = round(outlier_rate * n)
    if k == 0:
        return x
    pos = torch.randperm(n, generator=g)[:k]
    x[pos] = balanced_signs(k, g) * outlier_magnitude
    return x


def d2(n: int, g: torch.Generator, offset: float, noise_std: float) -> torch.Tensor:
    """D2, near-constant: offset + noise_std * N(0, 1), i.e. tiny variation around an offset."""
    return offset + noise_std * randn(n, g)


def d3(n: int, g: torch.Generator, sigma_delta: float) -> torch.Tensor:
    """D3, cancellation: n/2 values a ~ N(0, 1) paired with -a + delta, then shuffled.

    The exact sum is the sum of the small residuals delta ~ N(0, sigma_delta^2), so it is
    close to zero relative to the size of its terms.
    """
    if n % 2:
        raise ValueError("D3 needs an even length")
    a = randn(n // 2, g)
    delta = sigma_delta * randn(n // 2, g)
    x = torch.cat([a, -a + delta])
    return x[torch.randperm(n, generator=g)]


def d4(n: int, g: torch.Generator, df: int) -> torch.Tensor:
    """D4, heavy-tailed: Student-t(df) scaled to unit variance.

    Built as Z / sqrt(chi2(df) / df) from normals, because torch's StudentT distribution does
    not accept a generator. Computed in FP64 and rounded to FP32 once at the end.
    """
    if int(df) != df or df <= 2:
        raise ValueError("D4 needs an integer df > 2")
    df = int(df)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    chi2 = torch.randn(df, n, generator=g, dtype=torch.float64).square().sum(dim=0)
    t = z / torch.sqrt(chi2 / df)
    # Var(t) = df / (df - 2); dividing by its square root gives unit variance.
    return (t / (df / (df - 2)) ** 0.5).to(DTYPE)


@lru_cache(maxsize=None)
def d5_channels(n: int, n_channels: int, channel_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed outlier channel positions and signs, identical for every sample.

    Drawn once from their own seed and cached, so every D5 sample shares the same channels.
    """
    g = make_generator(channel_seed)
    pos = torch.randperm(n, generator=g)[:n_channels]
    return pos, balanced_signs(n_channels, g)


def d5(n: int, g: torch.Generator, n_channels: int, channel_magnitude: float, channel_seed: int) -> torch.Tensor:
    """D5, fixed outlier channels: D0 plus ±magnitude added at the same positions every sample.

    This mimics the systematic outlier channels reported in LLM activations.
    """
    x = randn(n, g)
    pos, signs = d5_channels(n, n_channels, channel_seed)
    x[pos] += signs * channel_magnitude
    return x


GENERATORS = {"D0": d0, "D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5}


def generate(regime: str, n: int, params: dict, seed: int) -> torch.Tensor:
    """Generate one FP32 input tensor of length n for a distribution, from its seed."""
    x = GENERATORS[regime](n, make_generator(seed), **params)
    assert x.dtype == DTYPE and x.shape == (n,)
    return x
