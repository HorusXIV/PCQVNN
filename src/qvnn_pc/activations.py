"""Split quaternion activations and their componentwise derivatives."""

from __future__ import annotations

import torch


def split_tanh(q: torch.Tensor) -> torch.Tensor:
    return torch.tanh(q)


def split_tanh_derivative(q: torch.Tensor) -> torch.Tensor:
    tanh_q = torch.tanh(q)
    return 1.0 - tanh_q.square()


def split_sigmoid(q: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(q)


def split_sigmoid_derivative(q: torch.Tensor) -> torch.Tensor:
    sigma = torch.sigmoid(q)
    return sigma * (1.0 - sigma)


def split_swish(q: torch.Tensor) -> torch.Tensor:
    return q * torch.sigmoid(q)


def split_swish_derivative(q: torch.Tensor) -> torch.Tensor:
    sigma = torch.sigmoid(q)
    return sigma + q * sigma * (1.0 - sigma)


def split_leaky_relu(q: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    return torch.where(q >= 0.0, q, negative_slope * q)


def split_leaky_relu_derivative(q: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    ones = torch.ones_like(q)
    slope = torch.full_like(q, negative_slope)
    return torch.where(q >= 0.0, ones, slope)
