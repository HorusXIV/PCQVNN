"""Bill et al. 2023 CMLP baseline for quaternion-valued regression."""

from __future__ import annotations

import math
import time
from typing import Literal

import torch
from torch import nn

from qvnn_pc.activations import (
    split_sigmoid,
    split_sigmoid_derivative,
    split_swish,
    split_swish_derivative,
)
from qvnn_pc.models.shared import (
    OptimizerName,
    clone_training_state,
    pq_alpha_should_stop as _pq_alpha_should_stop,
    quaternion_adagrad_step_,
    quaternion_adadelta_step_,
    quaternion_mae_loss,
    quaternion_sgd_step_,
    restore_training_state,
    scalar_regression_metrics,
    source_style_weight_init as _source_style_weight_init,
    validate_quaternion_batch as _validate_quaternion_batch,
)
from qvnn_pc.quaternion import q_conj, q_hadamard, q_matmul, q_mul

ActivationName = Literal["sigmoid", "swish"]


def _activation_pair(name: ActivationName):
    if name == "sigmoid":
        return split_sigmoid, split_sigmoid_derivative
    if name == "swish":
        return split_swish, split_swish_derivative
    raise ValueError(f"Unsupported activation {name!r}. Expected 'sigmoid' or 'swish'.")


class CMLPQuaternionMLP(nn.Module):
    """Quaternion CMLP with manual Bill et al.-style backward and Adadelta update."""

    def __init__(
        self,
        layer_sizes: tuple[int, ...] = (1, 5, 5, 5, 1),
        *,
        activation: ActivationName = "sigmoid",
        optimizer: OptimizerName = "Adadelta",
        lr: float = 0.00473,
        rho: float = 0.9,
        eps: float = 1e-9,
        init_scale: float = 0.1,
        source_style_init: bool = True,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("Expected at least input and output layer sizes.")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError(f"Layer sizes must be positive, got {layer_sizes}.")

        self.layer_sizes = tuple(layer_sizes)
        self.activation_name = activation
        self.optimizer_name = optimizer
        self.activation, self.activation_derivative = _activation_pair(activation)
        self.lr = lr
        self.rho = rho
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}
        weights = []
        biases = []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if source_style_init:
                weight = _source_style_weight_init(
                    n_out,
                    n_in,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            else:
                weight = torch.empty(n_out, n_in, 4, **factory_kwargs)
                weight.normal_(mean=0.0, std=init_scale, generator=generator)
            weights.append(nn.Parameter(weight))
            biases.append(nn.Parameter(torch.zeros(n_out, 4, **factory_kwargs)))

        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

        for idx, weight in enumerate(self.weights):
            self.register_buffer(f"_weight_acc_{idx}", torch.zeros_like(weight[..., 0]))
        for idx, bias in enumerate(self.biases):
            self.register_buffer(f"_bias_acc_{idx}", torch.zeros_like(bias[..., 0]))

    def _clone_training_state(self) -> dict[str, list[torch.Tensor]]:
        return clone_training_state(self.weights, self.biases, lambda name: getattr(self, name))

    @torch.no_grad()
    def _restore_training_state(self, state: dict[str, list[torch.Tensor]]) -> None:
        restore_training_state(self.weights, self.biases, lambda name: getattr(self, name), state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations, _ = self.forward_with_cache(x)
        return activations[-1]

    def forward_with_cache(
        self,
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        _validate_quaternion_batch("x", x)
        if x.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Expected {self.layer_sizes[0]} input quaternions, got {x.shape[1]}.")

        activations = [x]
        pre_activations = []
        current = x
        last_layer_idx = len(self.weights) - 1

        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            s = q_matmul(weight, current) + bias
            pre_activations.append(s)
            current = s if idx == last_layer_idx else self.activation(s)
            activations.append(current)

        return activations, pre_activations

    def compute_deltas(
        self,
        activations: list[torch.Tensor],
        preacts: list[torch.Tensor],
        y: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Compute Bill et al. CMLP deltas from cached forward tensors."""
        _validate_quaternion_batch("y", y)
        if len(activations) != len(self.weights) + 1:
            raise ValueError(
                f"Expected {len(self.weights) + 1} activation tensors, got {len(activations)}."
            )
        if len(preacts) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} pre-activation tensors, got {len(preacts)}.")

        y_pred = activations[-1]
        if y.shape != y_pred.shape:
            raise ValueError(f"Expected y to have shape {tuple(y_pred.shape)}, got {tuple(y.shape)}.")

        deltas: list[torch.Tensor] = [torch.empty(0, device=y.device, dtype=y.dtype)] * len(self.weights)
        deltas[-1] = y - y_pred

        for idx in range(len(self.weights) - 2, -1, -1):
            next_weight = self.weights[idx + 1]
            next_delta = deltas[idx + 1]
            back_signal = q_mul(
                q_conj(next_weight).unsqueeze(0),
                next_delta.unsqueeze(2),
            ).sum(dim=1)
            deltas[idx] = q_hadamard(self.activation_derivative(preacts[idx]), back_signal)

        return deltas

    def backward(
        self,
        activations: list[torch.Tensor],
        preacts: list[torch.Tensor],
        y: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Alias for manual CMLP delta computation."""
        return self.compute_deltas(activations, preacts, y)

    def compute_update_directions(
        self,
        activations: list[torch.Tensor],
        deltas: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compute mean CMLP weight and bias directions from activations and deltas."""
        if len(activations) != len(self.weights) + 1:
            raise ValueError(
                f"Expected {len(self.weights) + 1} activation tensors, got {len(activations)}."
            )
        if len(deltas) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} delta tensors, got {len(deltas)}.")

        weight_directions = []
        bias_directions = []
        for delta, prev_activation in zip(deltas, activations[:-1]):
            d_weight = q_mul(delta.unsqueeze(2), q_conj(prev_activation).unsqueeze(1)).mean(dim=0)
            d_bias = delta.mean(dim=0)
            weight_directions.append(d_weight)
            bias_directions.append(d_bias)

        return weight_directions, bias_directions

    @torch.no_grad()
    def print_preactivation_diagnostics(self, x: torch.Tensor) -> list[dict[str, int | float]]:
        """Print preactivation magnitude diagnostics for each layer."""
        _, preacts = self.forward_with_cache(x)
        diagnostics = []
        for idx, preact in enumerate(preacts, start=1):
            summary = {
                "layer": idx,
                "mean_abs": float(preact.abs().mean().detach().cpu()),
                "max_abs": float(preact.abs().max().detach().cpu()),
                "std": float(preact.std(unbiased=False).detach().cpu()),
            }
            diagnostics.append(summary)
            print(
                f"layer {idx}: mean_abs={summary['mean_abs']:.6f}, "
                f"max_abs={summary['max_abs']:.6f}, std={summary['std']:.6f}"
            )
        return diagnostics

    def backward_directions(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Convenience helper that runs forward cache, deltas, and directions."""
        activations, preacts = self.forward_with_cache(x)
        deltas = self.compute_deltas(activations, preacts, y)
        weight_directions, bias_directions = self.compute_update_directions(activations, deltas)
        return deltas, weight_directions, bias_directions

    @torch.no_grad()
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Apply one full-batch manual CMLP update and return pre-update metrics."""
        activations, preacts = self.forward_with_cache(x)
        y_pred = activations[-1]
        metrics = scalar_regression_metrics(y_pred, y)
        deltas = self.compute_deltas(activations, preacts, y)
        weight_directions, bias_directions = self.compute_update_directions(activations, deltas)

        for idx, (param, direction) in enumerate(zip(self.weights, weight_directions)):
            self._apply_update(
                idx,
                param=param,
                direction=direction,
                accumulator=getattr(self, f"_weight_acc_{idx}"),
            )

        for idx, (param, direction) in enumerate(zip(self.biases, bias_directions)):
            self._apply_update(
                idx,
                param=param,
                direction=direction,
                accumulator=getattr(self, f"_bias_acc_{idx}"),
            )

        return metrics

    @torch.no_grad()
    def _apply_update(
        self,
        idx: int,
        *,
        param: torch.Tensor,
        direction: torch.Tensor,
        accumulator: torch.Tensor,
    ) -> None:
        if self.optimizer_name == "SGD":
            quaternion_sgd_step_(param, direction, lr=self.lr)
            return
        if self.optimizer_name == "Adagrad":
            quaternion_adagrad_step_(param, direction, accumulator, lr=self.lr, eps=self.eps)
            return
        if self.optimizer_name == "Adadelta":
            quaternion_adadelta_step_(
                param,
                direction,
                accumulator,
                lr=self.lr,
                rho=self.rho,
                eps=self.eps,
            )
            return
        raise ValueError(
            f"Unsupported optimizer {self.optimizer_name!r} for CMLPQuaternionMLP at parameter index {idx}."
        )

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Evaluate MSE, MAPE, and R2 using y_pred[..., 0]."""
        return scalar_regression_metrics(self.forward(x), y)

    @torch.no_grad()
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
    ) -> dict[str, list[float] | float | int]:
        """Train with source-style shuffling and a selectable batch size."""
        _validate_quaternion_batch("x_train", x_train)
        _validate_quaternion_batch("y_train", y_train)
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Expected matching batch sizes, got {x_train.shape[0]} and {y_train.shape[0]}.")
        if x_train.shape[1:] != (self.layer_sizes[0], 4):
            raise ValueError(
                f"Expected x_train to have shape (B, {self.layer_sizes[0]}, 4), got {tuple(x_train.shape)}."
            )
        if y_train.shape[1:] != (self.layer_sizes[-1], 4):
            raise ValueError(
                f"Expected y_train to have shape (B, {self.layer_sizes[-1]}, 4), got {tuple(y_train.shape)}."
            )
        if epochs <= 0:
            raise ValueError(f"Expected epochs > 0, got {epochs}.")
        if (x_val is None) != (y_val is None):
            raise ValueError("Provide both x_val and y_val, or neither.")
        if batch_size is None:
            batch_size = x_train.shape[0]
        if batch_size <= 0:
            raise ValueError(f"Expected batch_size > 0, got {batch_size}.")
        if x_val is not None and y_val is not None:
            _validate_quaternion_batch("x_val", x_val)
            _validate_quaternion_batch("y_val", y_val)
            if x_val.shape[1:] != (self.layer_sizes[0], 4):
                raise ValueError(
                    f"Expected x_val to have shape (B, {self.layer_sizes[0]}, 4), got {tuple(x_val.shape)}."
                )
            if y_val.shape[1:] != (self.layer_sizes[-1], 4):
                raise ValueError(
                    f"Expected y_val to have shape (B, {self.layer_sizes[-1]}, 4), got {tuple(y_val.shape)}."
                )

        history: dict[str, list[float] | float | int] = {
            "quaternion_mae": [],
            "validation_quaternion_mae": [],
            "mape": [],
            "r2": [],
        }
        best_state = self._clone_training_state()
        best_checkpoint_loss = math.inf
        start = time.perf_counter()
        for _ in range(epochs):
            permutation = torch.randperm(
                x_train.shape[0],
                generator=shuffle_generator,
                device="cpu",
            )
            for batch_start in range(0, x_train.shape[0], batch_size):
                batch_indices = permutation[batch_start:batch_start + batch_size]
                self.train_step(x_train[batch_indices], y_train[batch_indices])

            train_pred = self.forward(x_train)
            train_qmae = float(quaternion_mae_loss(train_pred, y_train).detach().cpu())
            train_metrics = scalar_regression_metrics(train_pred, y_train)
            history["quaternion_mae"].append(train_qmae)  # type: ignore[index]
            history["mape"].append(train_metrics["mape"])  # type: ignore[index]
            history["r2"].append(train_metrics["r2"])  # type: ignore[index]
            if x_val is not None and y_val is not None:
                val_pred = self.forward(x_val)
                checkpoint_loss = float(quaternion_mae_loss(val_pred, y_val).detach().cpu())
                history["validation_quaternion_mae"].append(checkpoint_loss)  # type: ignore[index]
            else:
                checkpoint_loss = train_qmae

            if checkpoint_loss < best_checkpoint_loss:
                best_checkpoint_loss = checkpoint_loss
                best_state = self._clone_training_state()

            if (
                early_stopping
                and x_val is not None
                and y_val is not None
                and _pq_alpha_should_stop(
                    history["quaternion_mae"],  # type: ignore[arg-type]
                    checkpoint_loss,
                    best_checkpoint_loss,
                    window=early_stopping_window,
                    threshold=early_stopping_threshold,
                )
            ):
                break

            if not math.isfinite(train_qmae):
                break

        self._restore_training_state(best_state)
        history["runtime"] = time.perf_counter() - start
        history["best_validation_loss"] = best_checkpoint_loss
        history["epochs_ran"] = len(history["quaternion_mae"])  # type: ignore[arg-type]
        return history

    @torch.no_grad()
    def fit_online_source_style(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        epochs: int = 1,
        shuffle_generator: torch.Generator | None = None,
    ) -> dict[str, list[float] | float | int]:
        """Compatibility wrapper for the original sample-wise training behavior."""
        return self.fit(
            x_train,
            y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=1,
            shuffle_generator=shuffle_generator,
        )
