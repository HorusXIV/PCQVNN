"""Shared utilities for quaternion-valued models and training loops."""

from __future__ import annotations

import math
from typing import Literal

import torch

from qvnn_pc.quaternion import q_norm_sq

OptimizerName = Literal["Adadelta", "Adagrad", "SGD"]
TrainingState = dict[str, list[torch.Tensor]]


def validate_quaternion_batch(name: str, q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[-1] != 4:
        raise ValueError(f"Expected {name} to have shape (B, n, 4), got {tuple(q.shape)}.")


def scalar_regression_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    eps: float = 1e-9,
) -> dict[str, float]:
    validate_quaternion_batch("y_pred", y_pred)
    validate_quaternion_batch("y_true", y_true)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Expected matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")

    pred = y_pred[..., 0]
    target = y_true[..., 0]
    error = target - pred
    mse = error.square().mean()
    mae = error.abs().mean()
    mape = (error.abs() / target.abs().clamp_min(eps)).mean()

    residual_ss = error.square().sum()
    centered_ss = (target - target.mean()).square().sum()
    if centered_ss <= eps:
        r2 = torch.zeros((), dtype=target.dtype, device=target.device)
    else:
        r2 = 1.0 - residual_ss / centered_ss

    return {
        "mse": float(mse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "mape": float(mape.detach().cpu()),
        "r2": float(r2.detach().cpu()),
    }


def quaternion_mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    validate_quaternion_batch("y_pred", y_pred)
    validate_quaternion_batch("y_true", y_true)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Expected matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")
    return torch.linalg.vector_norm(y_pred - y_true, dim=-1).mean()


def source_style_weight_init(
    fan_out: int,
    fan_in: int,
    *,
    generator: torch.Generator | None,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    sigma = 1.0 / math.sqrt(2.0 * (fan_in + fan_out))
    theta = torch.empty(fan_out, fan_in, device=device, dtype=dtype).uniform_(
        -math.pi,
        math.pi,
        generator=generator,
    )
    phi = torch.empty(fan_out, fan_in, device=device, dtype=dtype).uniform_(
        -sigma,
        sigma,
        generator=generator,
    )
    imag_unit = torch.randn(fan_out, fan_in, 3, device=device, dtype=dtype, generator=generator)
    imag_unit = imag_unit / imag_unit.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(imag_unit.dtype).eps)

    weight = torch.empty(fan_out, fan_in, 4, device=device, dtype=dtype)
    weight[..., 0] = phi * torch.cos(theta)
    weight[..., 1:] = phi.unsqueeze(-1) * imag_unit * torch.sin(theta).unsqueeze(-1)
    return weight


@torch.no_grad()
def quaternion_adadelta_step_(
    param: torch.Tensor,
    direction: torch.Tensor,
    acc: torch.Tensor,
    *,
    lr: float,
    rho: float = 0.9,
    eps: float = 1e-9,
) -> None:
    if param.shape != direction.shape:
        raise ValueError(f"Expected param and direction to match, got {tuple(param.shape)} and {tuple(direction.shape)}.")
    if acc.shape != param.shape[:-1]:
        raise ValueError(f"Expected accumulator shape {tuple(param.shape[:-1])}, got {tuple(acc.shape)}.")

    acc.mul_(rho).add_(q_norm_sq(direction), alpha=1.0 - rho)
    param.add_(direction * (lr / torch.sqrt(acc + eps)).unsqueeze(-1))


@torch.no_grad()
def quaternion_sgd_step_(
    param: torch.Tensor,
    direction: torch.Tensor,
    *,
    lr: float,
) -> None:
    if param.shape != direction.shape:
        raise ValueError(f"Expected param and direction to match, got {tuple(param.shape)} and {tuple(direction.shape)}.")
    param.add_(direction, alpha=lr)


@torch.no_grad()
def quaternion_adagrad_step_(
    param: torch.Tensor,
    direction: torch.Tensor,
    acc: torch.Tensor,
    *,
    lr: float,
    eps: float = 1e-9,
) -> None:
    if param.shape != direction.shape:
        raise ValueError(f"Expected param and direction to match, got {tuple(param.shape)} and {tuple(direction.shape)}.")
    if acc.shape != param.shape[:-1]:
        raise ValueError(f"Expected accumulator shape {tuple(param.shape[:-1])}, got {tuple(acc.shape)}.")

    acc.copy_(q_norm_sq(direction))
    param.add_(direction * (lr / torch.sqrt(acc + eps)).unsqueeze(-1))


def pq_alpha_should_stop(
    train_losses: list[float],
    validation_loss: float,
    best_validation_loss: float,
    *,
    window: int = 5,
    threshold: float = 0.5,
) -> bool:
    if not train_losses or not math.isfinite(best_validation_loss) or best_validation_loss <= 0.0:
        return False

    k = min(len(train_losses), window)
    recent_train = train_losses[-k:]
    min_recent_train = min(recent_train)
    if min_recent_train <= 0.0:
        return False

    gl = 100.0 * ((validation_loss / best_validation_loss) - 1.0)
    pk = 1000.0 * ((sum(recent_train) / (k * min_recent_train)) - 1.0)
    if pk <= 0.0:
        return False
    return (gl / pk) > threshold


def clone_training_state(weights, biases, state_getter) -> TrainingState:
    return {
        "weights": [weight.detach().clone() for weight in weights],
        "biases": [bias.detach().clone() for bias in biases],
        "weight_accs": [state_getter(f"_weight_acc_{idx}").detach().clone() for idx in range(len(weights))],
        "bias_accs": [state_getter(f"_bias_acc_{idx}").detach().clone() for idx in range(len(biases))],
    }


@torch.no_grad()
def restore_training_state(weights, biases, state_getter, state: TrainingState) -> None:
    for weight, value in zip(weights, state["weights"]):
        weight.copy_(value)
    for bias, value in zip(biases, state["biases"]):
        bias.copy_(value)
    for idx, value in enumerate(state["weight_accs"]):
        state_getter(f"_weight_acc_{idx}").copy_(value)
    for idx, value in enumerate(state["bias_accs"]):
        state_getter(f"_bias_acc_{idx}").copy_(value)
