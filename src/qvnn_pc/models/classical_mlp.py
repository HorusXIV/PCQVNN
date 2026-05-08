"""Classical real-valued MLP baseline for the regression benchmarks."""

from __future__ import annotations

import math
import time
from typing import Literal, cast

import torch
from torch import nn

import tqdm

from qvnn_pc.models.shared import pq_alpha_should_stop as _pq_alpha_should_stop

ActivationName = Literal["relu", "leakyrelu", "sigmoid", "swish", "tanh"]
OptimizerName = Literal["SGD", "Adagrad", "Adadelta", "Adam"]


def _activation(name: ActivationName):
    if name == "relu":
        return torch.relu
    if name == "leakyrelu":
        negative_slope = 0.2

        def _leaky_relu(x: torch.Tensor) -> torch.Tensor:
            return torch.where(x >= 0.0, x, negative_slope * x)

        return _leaky_relu
    if name == "sigmoid":
        return torch.sigmoid
    if name == "swish":
        return lambda x: x * torch.sigmoid(x)
    if name == "tanh":
        return torch.tanh
    raise ValueError(
        f"Unsupported activation {name!r}. Expected one of: 'relu', 'leakyrelu', 'sigmoid', 'swish', 'tanh'."
    )


def regression_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, *, eps: float = 1e-9) -> dict[str, float]:
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Expected matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")
    error = y_true - y_pred
    mse = error.square().mean()
    mae = error.abs().mean()
    mape = (error.abs() / y_true.abs().clamp_min(eps)).mean()
    residual_ss = error.square().sum()
    centered_ss = (y_true - y_true.mean()).square().sum()
    if centered_ss <= eps:
        r2 = torch.zeros((), dtype=y_true.dtype, device=y_true.device)
    else:
        r2 = 1.0 - residual_ss / centered_ss
    return {
        "mse": float(mse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "mape": float(mape.detach().cpu()),
        "r2": float(r2.detach().cpu()),
    }


class ClassicalMLP(nn.Module):
    """Simple real-valued MLP baseline with a PCN-matched 4-20-20-20-20-1 shape."""

    def __init__(
        self,
        layer_sizes: tuple[int, ...] = (4, 20, 20, 20, 20, 1),
        *,
        activation: ActivationName = "tanh",
        optimizer: OptimizerName = "Adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("Expected at least input and output layer sizes.")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError(f"Layer sizes must be positive, got {layer_sizes}.")

        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.activation_name = activation
        self.optimizer_name = optimizer
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.activation = _activation(activation)

        layers: list[nn.Linear] = []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            layers.append(nn.Linear(n_in, n_out, device=device, dtype=dtype))
        self.layers = nn.ModuleList(layers)

        self._reset_parameters()
        self.optimizer = self._build_optimizer()

    def _reset_parameters(self) -> None:
        for layer in self.layers:
            linear = cast(nn.Linear, layer)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.parameters()
        if self.optimizer_name == "SGD":
            return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == "Adagrad":
            return torch.optim.Adagrad(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == "Adadelta":
            return torch.optim.Adadelta(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == "Adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(
            f"Unsupported optimizer {self.optimizer_name!r}. Expected one of: 'SGD', 'Adagrad', 'Adadelta', 'Adam'."
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        first_layer = cast(nn.Linear, self.layers[0])
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (B, {self.layer_sizes[0]}), got {tuple(x.shape)}.")
        if x.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Expected input dimension {self.layer_sizes[0]}, got {x.shape[1]}.")
        return x.to(device=first_layer.weight.device, dtype=first_layer.weight.dtype)

    def _prepare_target(self, y: torch.Tensor) -> torch.Tensor:
        first_layer = cast(nn.Linear, self.layers[0])
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.ndim != 2:
            raise ValueError(f"Expected target shape (B, {self.layer_sizes[-1]}), got {tuple(y.shape)}.")
        if y.shape[1] != self.layer_sizes[-1]:
            raise ValueError(f"Expected target dimension {self.layer_sizes[-1]}, got {y.shape[1]}.")
        return y.to(device=first_layer.weight.device, dtype=first_layer.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = self._prepare_input(x)
        for layer_idx, layer in enumerate(self.layers):
            current = layer(current)
            if layer_idx < len(self.layers) - 1:
                current = self.activation(current)
        return current

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        y_real = self._prepare_target(y)
        y_pred = self.forward(x)
        return regression_metrics(y_pred, y_real)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        x_real = self._prepare_input(x)
        y_real = self._prepare_target(y)
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        y_pred = self.forward(x_real)
        metrics = regression_metrics(y_pred, y_real)
        loss = torch.nn.functional.mse_loss(y_pred, y_real)
        loss.backward()
        self.optimizer.step()
        return metrics

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        epochs: int = 1,
        batch_size: int | None = None,
        shuffle_generator: torch.Generator | None = None,
        early_stopping: bool = False,
        early_stopping_window: int = 5,
        early_stopping_threshold: float = 0.5,
    ) -> dict[str, list[float] | float | int | bool]:
        x_train = self._prepare_input(x_train)
        y_train = self._prepare_target(y_train)
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Expected matching batch sizes, got {x_train.shape[0]} and {y_train.shape[0]}.")
        if epochs <= 0:
            raise ValueError(f"Expected epochs > 0, got {epochs}.")
        if (x_val is None) != (y_val is None):
            raise ValueError("Provide both x_val and y_val, or neither.")
        if batch_size is None:
            batch_size = x_train.shape[0]
        if batch_size <= 0:
            raise ValueError(f"Expected batch_size > 0, got {batch_size}.")

        x_val_real = None if x_val is None else self._prepare_input(x_val)
        y_val_real = None if y_val is None else self._prepare_target(y_val)

        history: dict[str, list[float] | float | int | bool] = {
            "mse": [],
            "mae": [],
            "mape": [],
            "r2": [],
            "validation_mse": [],
            "validation_mae": [],
            "validation_mape": [],
            "validation_r2": [],
            "runtime": 0.0,
            "best_validation_loss": math.inf,
            "epochs_ran": 0,
            "interrupted": False,
        }

        best_state = {name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()}
        best_validation_loss = math.inf
        start_time = time.perf_counter()

        for epoch_idx in tqdm.tqdm(range(epochs), desc="Training epochs", unit="epoch"):
            permutation = torch.randperm(x_train.shape[0], generator=shuffle_generator, device="cpu")
            for batch_start in range(0, x_train.shape[0], batch_size):
                batch_indices = permutation[batch_start:batch_start + batch_size]
                self.train_step(x_train[batch_indices], y_train[batch_indices])

            train_metrics = self.evaluate(x_train, y_train)
            history["mse"].append(train_metrics["mse"])  # type: ignore[index]
            history["mae"].append(train_metrics["mae"])  # type: ignore[index]
            history["mape"].append(train_metrics["mape"])  # type: ignore[index]
            history["r2"].append(train_metrics["r2"])  # type: ignore[index]

            validation_loss = train_metrics["mae"]
            if x_val_real is not None and y_val_real is not None:
                val_metrics = self.evaluate(x_val_real, y_val_real)
                history["validation_mse"].append(val_metrics["mse"])  # type: ignore[index]
                history["validation_mae"].append(val_metrics["mae"])  # type: ignore[index]
                history["validation_mape"].append(val_metrics["mape"])  # type: ignore[index]
                history["validation_r2"].append(val_metrics["r2"])  # type: ignore[index]
                validation_loss = val_metrics["mae"]

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()}

            history["epochs_ran"] = epoch_idx + 1

            if early_stopping and _pq_alpha_should_stop(
                history["mae"],  # type: ignore[arg-type]
                validation_loss,
                best_validation_loss,
                window=early_stopping_window,
                threshold=early_stopping_threshold,
            ):
                break

            if not torch.isfinite(torch.tensor(train_metrics["mse"])):
                history["interrupted"] = True
                break

        self.load_state_dict(best_state)
        history["runtime"] = time.perf_counter() - start_time
        history["best_validation_loss"] = best_validation_loss
        return history