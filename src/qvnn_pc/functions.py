"""Standard 4D benchmark functions for function approximation experiments."""

from __future__ import annotations

import math

import torch


def _validate_4d_input(x: torch.Tensor) -> None:
    if x.shape[-1] != 4:
        raise ValueError(f"Expected last dimension of size 4, got {x.shape}.")


def ackley_4d(x: torch.Tensor) -> torch.Tensor:
    _validate_4d_input(x)
    n = x.shape[-1]
    squared_mean = x.square().mean(dim=-1)
    cosine_mean = torch.cos(2.0 * math.pi * x).mean(dim=-1)
    term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(squared_mean))
    term2 = -torch.exp(cosine_mean)
    return term1 + term2 + 20.0 + math.e


def griewank_4d(x: torch.Tensor) -> torch.Tensor:
    _validate_4d_input(x)
    idx = torch.arange(1, x.shape[-1] + 1, dtype=x.dtype, device=x.device)
    quadratic = x.square().sum(dim=-1) / 4000.0
    oscillatory = torch.cos(x / torch.sqrt(idx)).prod(dim=-1)
    return quadratic - oscillatory + 1.0


def schwefel_4d(x: torch.Tensor) -> torch.Tensor:
    _validate_4d_input(x)
    n = x.shape[-1]
    return 418.9829 * n - (x * torch.sin(torch.sqrt(torch.abs(x)))).sum(dim=-1)


def rastrigin_4d(x: torch.Tensor) -> torch.Tensor:
    _validate_4d_input(x)
    n = x.shape[-1]
    return 10.0 * n + (x.square() - 10.0 * torch.cos(2.0 * math.pi * x)).sum(dim=-1)
