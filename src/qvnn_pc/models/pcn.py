"""Real-valued predictive coding network following the Stenlund (2025) setup."""

from __future__ import annotations

from contextlib import nullcontext
import time
from typing import Callable, Literal, cast

import torch
import torch.nn.functional as F
from torch import autocast, nn

import tqdm

ActivationName = Literal["leakyrelu", "sigmoid", "swish", "tanh"]
LatentInitName = Literal["feedforward", "random", "zeros"]


def _activation_pair(name: ActivationName) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    if name == "tanh":
        return torch.tanh, lambda x: 1.0 - torch.tanh(x).square()
    if name == "sigmoid":
        def _sigmoid(x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)

        def _sigmoid_deriv(x: torch.Tensor) -> torch.Tensor:
            sig = torch.sigmoid(x)
            return sig * (1.0 - sig)

        return _sigmoid, _sigmoid_deriv
    if name == "swish":
        def _swish(x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)

        def _swish_deriv(x: torch.Tensor) -> torch.Tensor:
            sig = torch.sigmoid(x)
            return sig + x * sig * (1.0 - sig)

        return _swish, _swish_deriv
    if name == "leakyrelu":
        negative_slope = 0.2

        def _leaky_relu(x: torch.Tensor) -> torch.Tensor:
            return F.leaky_relu(x, negative_slope=negative_slope)

        def _leaky_relu_deriv(x: torch.Tensor) -> torch.Tensor:
            return torch.where(x >= 0.0, torch.ones_like(x), torch.full_like(x, negative_slope))

        return _leaky_relu, _leaky_relu_deriv
    raise ValueError(
        f"Unsupported activation {name!r}. Expected one of: 'leakyrelu', 'sigmoid', 'swish', 'tanh'."
    )


def _autocast_context(device: torch.device) -> nullcontext | autocast:
    if device.type != "cuda":
        return nullcontext()
    return autocast(device_type="cuda", dtype=torch.float16)


def scalar_regression_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    eps: float = 1e-9,
) -> dict[str, float]:
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


class PCNLayer(nn.Module):
    activation_fn: Callable[[torch.Tensor], torch.Tensor]
    activation_deriv: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
        activation_deriv: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim, device=device, dtype=dtype))
        nn.init.xavier_uniform_(self.W)
        self.activation_fn = activation_fn
        self.activation_deriv = activation_deriv

    def forward(self, x_above: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_activation = x_above @ self.W.T
        prediction = self.activation_fn(pre_activation)
        return prediction, pre_activation


class PredictiveCodingNetwork(nn.Module):
    def __init__(
        self,
        dims: list[int] | tuple[int, ...],
        output_dim: int = 1,
        *,
        activation: ActivationName = "tanh",
        prediction_inference_steps: int = 25,
        prediction_eta_infer: float = 0.05,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("Expected at least one hidden layer.")
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"All dims must be positive, got {dims}.")
        if output_dim <= 0:
            raise ValueError(f"Expected output_dim > 0, got {output_dim}.")

        self.dims = tuple(int(dim) for dim in dims)
        self.output_dim = int(output_dim)
        self.L = len(self.dims) - 1
        self.activation_name = activation
        self.prediction_inference_steps = int(prediction_inference_steps)
        self.prediction_eta_infer = float(prediction_eta_infer)
        activation_fn, activation_deriv = _activation_pair(activation)

        self.layers = nn.ModuleList([
            PCNLayer(
                in_dim=self.dims[layer_idx + 1],
                out_dim=self.dims[layer_idx],
                activation_fn=activation_fn,
                activation_deriv=activation_deriv,
                device=device,
                dtype=dtype,
            )
            for layer_idx in range(self.L)
        ])
        self.readout = nn.Linear(self.dims[-1], self.output_dim, bias=False, device=device, dtype=dtype)
        nn.init.xavier_uniform_(self.readout.weight)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected real-valued input shape (B, {self.dims[0]}), got {tuple(x.shape)}.")
        if x.shape[1] != self.dims[0]:
            raise ValueError(f"Expected input feature dimension {self.dims[0]}, got {x.shape[1]}.")
        return x.to(device=self.readout.weight.device, dtype=self.readout.weight.dtype)

    def _prepare_target(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.ndim != 2:
            raise ValueError(f"Expected target shape (B, {self.output_dim}), got {tuple(y.shape)}.")
        if y.shape[1] != self.output_dim:
            raise ValueError(f"Expected target dimension {self.output_dim}, got {y.shape[1]}.")
        return y.to(device=self.readout.weight.device, dtype=self.readout.weight.dtype)

    def feedforward_latents(self, x: torch.Tensor) -> list[torch.Tensor]:
        current = self._prepare_input(x)
        latents: list[torch.Tensor] = []
        for layer in self.layers:
            current = layer.activation_fn(current @ layer.W)
            latents.append(current)
        return latents

    def init_latents(
        self,
        batch_size: int,
        device: torch.device,
        *,
        dtype: torch.dtype,
        mode: LatentInitName = "feedforward",
        x: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if mode == "feedforward":
            if x is None:
                raise ValueError("feedforward latent initialization requires an input tensor.")
            return [latent.detach().clone() for latent in self.feedforward_latents(x)]
        if mode == "zeros":
            return [torch.zeros(batch_size, dim, device=device, dtype=dtype) for dim in self.dims[1:]]
        if mode == "random":
            return [0.05 * torch.randn(batch_size, dim, device=device, dtype=dtype) for dim in self.dims[1:]]
        raise ValueError(f"Unsupported latent init mode {mode!r}.")

    def compute_errors(
        self,
        states: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        errors: list[torch.Tensor] = []
        gain_modulated_errors: list[torch.Tensor] = []
        pre_activations: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            prediction, pre_activation = layer(states[layer_idx + 1])
            error = states[layer_idx] - prediction
            gain_modulated_error = error * cast(Callable[[torch.Tensor], torch.Tensor], layer.activation_deriv)(pre_activation)
            errors.append(error)
            gain_modulated_errors.append(gain_modulated_error)
            pre_activations.append(pre_activation)
        return errors, gain_modulated_errors, pre_activations

    def _compute_energy(
        self,
        states: list[torch.Tensor],
        *,
        target: torch.Tensor | None = None,
    ) -> tuple[dict[str, float], torch.Tensor]:
        errors, _, _ = self.compute_errors(states)
        y_hat = self.readout(states[-1])
        reconstruction_mse = torch.stack([error.square().mean() for error in errors]).mean()
        supervised_mse = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
        if target is not None:
            supervised_mse = F.mse_loss(y_hat, target)
        free_energy = 0.5 * (reconstruction_mse + supervised_mse)
        return {
            "free_energy": float(free_energy.detach().cpu()),
            "reconstruction_mse": float(reconstruction_mse.detach().cpu()),
            "supervised_mse": float(supervised_mse.detach().cpu()),
        }, y_hat

    def _latent_gradients(
        self,
        states: list[torch.Tensor],
        *,
        target: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        errors, gain_modulated_errors, _ = self.compute_errors(states)
        y_hat = self.readout(states[-1])
        top_error = None if target is None else (y_hat - target) @ self.readout.weight

        gradients: list[torch.Tensor] = []
        for latent_idx in range(1, self.L + 1):
            gradient = -(gain_modulated_errors[latent_idx - 1] @ self.layers[latent_idx - 1].W)
            if latent_idx < self.L:
                gradient = gradient + errors[latent_idx]
            elif top_error is not None:
                gradient = gradient + top_error
            gradients.append(gradient)
        return gradients, y_hat

    @torch.no_grad()
    def infer_latents(
        self,
        x: torch.Tensor,
        *,
        y: torch.Tensor | None = None,
        steps: int | None = None,
        eta_infer: float | None = None,
        latent_init: LatentInitName = "feedforward",
        return_trace: bool = False,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor | dict[str, list[float]]]:
        x_real = self._prepare_input(x)
        y_real = None if y is None else self._prepare_target(y)
        infer_steps = self.prediction_inference_steps if steps is None else int(steps)
        infer_eta = self.prediction_eta_infer if eta_infer is None else float(eta_infer)
        if infer_steps < 0:
            raise ValueError(f"Expected steps >= 0, got {infer_steps}.")
        if infer_eta <= 0.0:
            raise ValueError(f"Expected eta_infer > 0, got {infer_eta}.")

        states = [x_real] + self.init_latents(
            x_real.shape[0],
            x_real.device,
            dtype=x_real.dtype,
            mode=latent_init,
            x=x_real,
        )
        trace: dict[str, list[float]] = {
            "free_energy": [],
            "reconstruction_mse": [],
            "supervised_mse": [],
            "mean_latent_grad_norm": [],
        }

        def _record_trace(gradients: list[torch.Tensor]) -> None:
            if not return_trace:
                return
            energy_metrics, _ = self._compute_energy(states, target=y_real)
            grad_norm = torch.stack([gradient.norm(dim=1).mean() for gradient in gradients]).mean()
            trace["free_energy"].append(energy_metrics["free_energy"])
            trace["reconstruction_mse"].append(energy_metrics["reconstruction_mse"])
            trace["supervised_mse"].append(energy_metrics["supervised_mse"])
            trace["mean_latent_grad_norm"].append(float(grad_norm.detach().cpu()))

        with _autocast_context(x_real.device):
            if infer_steps == 0:
                gradients = [torch.zeros_like(latent) for latent in states[1:]]
                prediction = self.readout(states[-1])
                _record_trace(gradients)
            else:
                for _ in range(infer_steps):
                    gradients, prediction = self._latent_gradients(states, target=y_real)
                    _record_trace(gradients)
                    for latent_idx, gradient in enumerate(gradients, start=1):
                        states[latent_idx].sub_(gradient, alpha=infer_eta)
                gradients, prediction = self._latent_gradients(states, target=y_real)
                _record_trace(gradients)

        return {
            "states": states,
            "prediction": prediction,
            "trace": trace,
        }

    @torch.no_grad()
    def fast_predict(self, x: torch.Tensor) -> torch.Tensor:
        latents = self.feedforward_latents(x)
        return self.readout(latents[-1])

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        *,
        infer_steps: int | None = None,
        eta_infer: float | None = None,
        latent_init: LatentInitName = "feedforward",
        fast: bool = False,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, list[float]]]:
        if fast:
            prediction = self.fast_predict(x)
            if return_trace:
                return prediction, {
                    "free_energy": [],
                    "reconstruction_mse": [],
                    "supervised_mse": [],
                    "mean_latent_grad_norm": [],
                }
            return prediction

        result = self.infer_latents(
            x,
            y=None,
            steps=infer_steps,
            eta_infer=eta_infer,
            latent_init=latent_init,
            return_trace=return_trace,
        )
        prediction = cast(torch.Tensor, result["prediction"])
        if return_trace:
            return prediction, cast(dict[str, list[float]], result["trace"])
        return prediction

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        *,
        infer_steps: int | None = None,
        eta_infer: float | None = None,
        latent_init: LatentInitName = "feedforward",
        fast: bool = False,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.predict(
                x,
                infer_steps=infer_steps,
                eta_infer=eta_infer,
                latent_init=latent_init,
                fast=fast,
            ),
        )

    @torch.no_grad()
    def train_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        eta_infer: float,
        eta_learn: float,
        T_infer: int,
        T_learn: int,
        latent_init: LatentInitName = "feedforward",
        record_trace: bool = False,
    ) -> dict[str, float | dict[str, list[float]]]:
        if T_infer < 0:
            raise ValueError(f"Expected T_infer >= 0, got {T_infer}.")
        if T_learn <= 0:
            raise ValueError(f"Expected T_learn > 0, got {T_learn}.")

        x_real = self._prepare_input(x)
        y_real = self._prepare_target(y)
        inference_result = self.infer_latents(
            x_real,
            y=y_real,
            steps=T_infer,
            eta_infer=eta_infer,
            latent_init=latent_init,
            return_trace=record_trace,
        )
        states = cast(list[torch.Tensor], inference_result["states"])
        batch_size = x_real.shape[0]

        with _autocast_context(x_real.device):
            for _ in range(T_learn):
                _, gain_modulated_errors, _ = self.compute_errors(states)
                for layer_idx, layer in enumerate(self.layers):
                    gradient = -(gain_modulated_errors[layer_idx].T @ states[layer_idx + 1]) / batch_size
                    layer.W.sub_(gradient, alpha=eta_learn)

                output_error = self.readout(states[-1]) - y_real
                readout_gradient = output_error.T @ states[-1] / batch_size
                self.readout.weight.sub_(readout_gradient, alpha=eta_learn)

            energy_metrics, y_hat = self._compute_energy(states, target=y_real)

        metrics: dict[str, float | dict[str, list[float]]] = {
            **energy_metrics,
            **scalar_regression_metrics(y_hat, y_real),
        }
        if record_trace:
            metrics["inference_trace"] = cast(dict[str, list[float]], inference_result["trace"])
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        infer_steps: int | None = None,
        eta_infer: float | None = None,
        latent_init: LatentInitName = "feedforward",
        fast: bool = False,
        return_trace: bool = False,
    ) -> dict[str, float | dict[str, list[float]]]:
        y_real = self._prepare_target(y)

        if fast:
            x_real = self._prepare_input(x)
            y_hat = self.fast_predict(x_real)
            states = [x_real, *self.feedforward_latents(x_real)]
            trace: dict[str, list[float]] = {
                "free_energy": [],
                "reconstruction_mse": [],
                "supervised_mse": [],
                "mean_latent_grad_norm": [],
            }
        else:
            result = self.infer_latents(
                x,
                y=None,
                steps=infer_steps,
                eta_infer=eta_infer,
                latent_init=latent_init,
                return_trace=return_trace,
            )
            states = cast(list[torch.Tensor], result["states"])
            y_hat = cast(torch.Tensor, result["prediction"])
            trace = cast(dict[str, list[float]], result["trace"])

        energy_metrics, _ = self._compute_energy(states, target=y_real)
        metrics: dict[str, float | dict[str, list[float]]] = {
            **energy_metrics,
            **scalar_regression_metrics(y_hat, y_real),
        }
        if return_trace:
            metrics["inference_trace"] = trace
        return metrics

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
        eta_infer: float = 0.05,
        eta_learn: float = 0.001,
        T_infer: int = 25,
        T_learn: int = 1,
        eval_infer_steps: int | None = None,
        latent_init: LatentInitName = "feedforward",
    ) -> dict[str, list[float] | float | int]:
        if epochs <= 0:
            raise ValueError(f"Expected epochs > 0, got {epochs}.")
        if batch_size is None:
            batch_size = x_train.shape[0]
        if batch_size <= 0:
            raise ValueError(f"Expected batch_size > 0, got {batch_size}.")
        if (x_val is None) != (y_val is None):
            raise ValueError("Provide both x_val and y_val, or neither.")

        history: dict[str, list[float] | float | int] = {
            "mse": [],
            "mae": [],
            "mape": [],
            "r2": [],
            "free_energy": [],
            "reconstruction_mse": [],
            "supervised_mse": [],
            "validation_mse": [],
            "validation_mae": [],
            "validation_mape": [],
            "validation_r2": [],
        }
        eval_steps = self.prediction_inference_steps if eval_infer_steps is None else int(eval_infer_steps)
        start = time.perf_counter()

        for _ in tqdm.tqdm(range(epochs), desc="Training PCN", unit="epoch"):
            permutation = torch.randperm(x_train.shape[0], generator=shuffle_generator, device="cpu")
            for batch_start in range(0, x_train.shape[0], batch_size):
                batch_indices = permutation[batch_start:batch_start + batch_size]
                self.train_batch(
                    x_train[batch_indices],
                    y_train[batch_indices],
                    eta_infer=eta_infer,
                    eta_learn=eta_learn,
                    T_infer=T_infer,
                    T_learn=T_learn,
                    latent_init=latent_init,
                )

            train_metrics = self.evaluate(
                x_train,
                y_train,
                infer_steps=eval_steps,
                eta_infer=eta_infer,
                latent_init=latent_init,
            )
            for key in ("mse", "mae", "mape", "r2", "free_energy", "reconstruction_mse", "supervised_mse"):
                history[key].append(float(train_metrics[key]))  # type: ignore[index]

            if x_val is not None and y_val is not None:
                val_metrics = self.evaluate(
                    x_val,
                    y_val,
                    infer_steps=eval_steps,
                    eta_infer=eta_infer,
                    latent_init=latent_init,
                )
                for key in ("mse", "mae", "mape", "r2"):
                    history[f"validation_{key}"].append(float(val_metrics[key]))  # type: ignore[index]

        history["runtime"] = time.perf_counter() - start
        history["epochs_ran"] = len(history["mse"])  # type: ignore[arg-type]
        return history


def train_pcn(
    model: PredictiveCodingNetwork,
    data_loader,
    num_epochs: int,
    eta_infer: float,
    eta_learn: float,
    T_infer: int,
    T_learn: int,
    device: str | torch.device = "cuda",
    *,
    latent_init: LatentInitName = "feedforward",
) -> dict[str, list[float] | float | int]:
    model.to(device).train()
    history: dict[str, list[float] | float | int] = {
        "free_energy": [],
        "reconstruction_mse": [],
        "supervised_mse": [],
        "mse": [],
        "mae": [],
        "mape": [],
        "r2": [],
    }
    start = time.perf_counter()

    for _ in tqdm.tqdm(range(num_epochs), desc="Training PCN", unit="epoch"):
        batch_metrics: list[dict[str, float | dict[str, list[float]]]] = []
        for x_batch, y_batch in data_loader:
            batch_metrics.append(
                model.train_batch(
                    x_batch.to(device),
                    y_batch.to(device),
                    eta_infer=eta_infer,
                    eta_learn=eta_learn,
                    T_infer=T_infer,
                    T_learn=T_learn,
                    latent_init=latent_init,
                )
            )

        for key in ("free_energy", "reconstruction_mse", "supervised_mse", "mse", "mae", "mape", "r2"):
            values = [float(metrics[key]) for metrics in batch_metrics]
            history[key].append(sum(values) / max(len(values), 1))  # type: ignore[index]

    history["runtime"] = time.perf_counter() - start
    history["epochs_ran"] = len(history["mse"])  # type: ignore[arg-type]
    return history
