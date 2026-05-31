"""Split-quaternion predictive coding network.

This is the practical PC/QVNN hybrid used as the first scalable research target:
states, weights, errors, and predictions are quaternion-valued, while nonlinear
gains use componentwise split activations instead of the full GHR derivative
machinery.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, cast

import torch
from torch import nn

from qvnn_pc.activations import (
    split_leaky_relu,
    split_leaky_relu_derivative,
    split_sigmoid,
    split_sigmoid_derivative,
    split_swish,
    split_swish_derivative,
    split_tanh,
    split_tanh_derivative,
)
from qvnn_pc.models.shared import (
    OptimizerName,
    quaternion_mae_loss,
    scalar_regression_metrics,
    source_style_weight_init,
    validate_quaternion_batch,
)
from qvnn_pc.quaternion import q_conj, q_linear, q_matmul, q_mul

ActivationName = Literal["leakyrelu", "sigmoid", "swish", "tanh"]
LatentInitName = Literal["feedforward", "random", "zeros"]


def _activation_pair(name: ActivationName) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    if name == "tanh":
        return split_tanh, split_tanh_derivative
    if name == "sigmoid":
        return split_sigmoid, split_sigmoid_derivative
    if name == "swish":
        return split_swish, split_swish_derivative
    if name == "leakyrelu":
        return split_leaky_relu, split_leaky_relu_derivative
    raise ValueError(
        f"Unsupported activation {name!r}. Expected one of: 'leakyrelu', 'sigmoid', 'swish', 'tanh'."
    )


def _all_finite(tensors: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> bool:
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def _adjoint_weight(weight: torch.Tensor) -> torch.Tensor:
    return q_conj(weight.transpose(0, 1))


def _coerce_generator_device(
    generator: torch.Generator | None,
    device: torch.device | str | None,
) -> torch.Generator | None:
    if generator is None or device is None:
        return generator
    target_device = torch.device(device)
    generator_device = torch.device(generator.device)
    if target_device.type == generator_device.type:
        return generator
    return None


class QuaternionPCNLayer(nn.Module):
    activation_fn: Callable[[torch.Tensor], torch.Tensor]
    activation_deriv: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
        activation_deriv: Callable[[torch.Tensor], torch.Tensor],
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        generator = _coerce_generator_device(generator, device)
        self.W = nn.Parameter(
            source_style_weight_init(out_dim, in_dim, generator=generator, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.zeros(out_dim, 4, device=device, dtype=dtype))
        self.activation_fn = activation_fn
        self.activation_deriv = activation_deriv

    def forward(self, x_above: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_activation = q_linear(x_above, self.W, self.bias)
        prediction = self.activation_fn(pre_activation)
        return prediction, pre_activation


class SplitQuaternionPredictiveCodingNetwork(nn.Module):
    def __init__(
        self,
        layer_sizes: tuple[int, ...] = (1, 5, 5, 5, 1),
        *,
        activation: ActivationName = "tanh",
        optimizer: OptimizerName = "SGD",
        prediction_inference_steps: int = 16,
        prediction_eta_infer: float = 0.01,
        optimizer_eps: float = 1e-9,
        adadelta_rho: float = 0.9,
        adaptive_inference: bool = True,
        inference_tolerance: float = 1e-3,
        min_inference_steps: int = 4,
        check_finite: bool = False,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("Expected at least one hidden layer.")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError(f"Layer sizes must be positive, got {layer_sizes}.")

        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.L = len(self.layer_sizes) - 1
        self.activation_name = activation
        self.optimizer_name = optimizer
        self.prediction_inference_steps = int(prediction_inference_steps)
        self.prediction_eta_infer = float(prediction_eta_infer)
        self.optimizer_eps = float(optimizer_eps)
        self.adadelta_rho = float(adadelta_rho)
        self.adaptive_inference = bool(adaptive_inference)
        self.inference_tolerance = float(inference_tolerance)
        self.min_inference_steps = int(min_inference_steps)
        self.check_finite = bool(check_finite)

        activation_fn, activation_deriv = _activation_pair(activation)
        generator = _coerce_generator_device(generator, device)
        self.layers = nn.ModuleList([
            QuaternionPCNLayer(
                in_dim=self.layer_sizes[layer_idx + 1],
                out_dim=self.layer_sizes[layer_idx],
                activation_fn=activation_fn,
                activation_deriv=activation_deriv,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            for layer_idx in range(self.L)
        ])
        self.readout_weight = nn.Parameter(
            source_style_weight_init(self.layer_sizes[-1], self.layer_sizes[-1], generator=generator, device=device, dtype=dtype)
        )
        self.readout_bias = nn.Parameter(torch.zeros(self.layer_sizes[-1], 4, device=device, dtype=dtype))

        for layer_idx, layer in enumerate(self.layers):
            self.register_buffer(f"_weight_acc_{layer_idx}", torch.zeros_like(layer.W))
            self.register_buffer(f"_weight_delta_acc_{layer_idx}", torch.zeros_like(layer.W))
            self.register_buffer(f"_bias_acc_{layer_idx}", torch.zeros_like(layer.bias))
            self.register_buffer(f"_bias_delta_acc_{layer_idx}", torch.zeros_like(layer.bias))
        self.register_buffer("_readout_acc", torch.zeros_like(self.readout_weight))
        self.register_buffer("_readout_delta_acc", torch.zeros_like(self.readout_weight))
        self.register_buffer("_readout_bias_acc", torch.zeros_like(self.readout_bias))
        self.register_buffer("_readout_bias_delta_acc", torch.zeros_like(self.readout_bias))

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        validate_quaternion_batch("x", x)
        if x.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Expected {self.layer_sizes[0]} input quaternions, got {x.shape[1]}.")
        return x.to(device=self.readout_weight.device, dtype=self.readout_weight.dtype)

    def _prepare_target(self, y: torch.Tensor) -> torch.Tensor:
        validate_quaternion_batch("y", y)
        if y.shape[1] != self.layer_sizes[-1]:
            raise ValueError(f"Expected {self.layer_sizes[-1]} target quaternions, got {y.shape[1]}.")
        return y.to(device=self.readout_weight.device, dtype=self.readout_weight.dtype)

    def _readout(self, latent: torch.Tensor) -> torch.Tensor:
        return q_linear(latent, self.readout_weight, self.readout_bias)

    @torch.no_grad()
    def _apply_update(
        self,
        *,
        param: torch.Tensor,
        gradient: torch.Tensor,
        accumulator: torch.Tensor,
        delta_accumulator: torch.Tensor,
        lr: float,
    ) -> None:
        if self.optimizer_name == "SGD":
            param.sub_(gradient, alpha=lr)
            return
        if self.optimizer_name == "Adagrad":
            accumulator.addcmul_(gradient, gradient, value=1.0)
            param.addcdiv_(gradient, accumulator.sqrt().add_(self.optimizer_eps), value=-lr)
            return
        if self.optimizer_name == "Adadelta":
            accumulator.mul_(self.adadelta_rho).addcmul_(gradient, gradient, value=1.0 - self.adadelta_rho)
            update = -gradient * torch.sqrt(delta_accumulator + self.optimizer_eps) / torch.sqrt(
                accumulator + self.optimizer_eps
            )
            if lr != 1.0:
                update.mul_(lr)
            param.add_(update)
            delta_accumulator.mul_(self.adadelta_rho).addcmul_(update, update, value=1.0 - self.adadelta_rho)
            return
        raise ValueError(f"Unsupported optimizer {self.optimizer_name!r}.")

    def feedforward_latents(self, x: torch.Tensor) -> list[torch.Tensor]:
        current = self._prepare_input(x)
        latents: list[torch.Tensor] = []
        for layer in self.layers:
            current = layer.activation_fn(q_matmul(_adjoint_weight(layer.W), current - layer.bias))
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
            return [torch.zeros(batch_size, dim, 4, device=device, dtype=dtype) for dim in self.layer_sizes[1:]]
        if mode == "random":
            return [0.05 * torch.randn(batch_size, dim, 4, device=device, dtype=dtype) for dim in self.layer_sizes[1:]]
        raise ValueError(f"Unsupported latent init mode {mode!r}.")

    def compute_errors(
        self,
        states: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        errors: list[torch.Tensor] = []
        gain_modulated_errors: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            prediction, pre_activation = layer(states[layer_idx + 1])
            error = states[layer_idx] - prediction
            gain = error * cast(Callable[[torch.Tensor], torch.Tensor], layer.activation_deriv)(pre_activation)
            errors.append(error)
            gain_modulated_errors.append(gain)
        return errors, gain_modulated_errors

    def _latent_gradients(
        self,
        states: list[torch.Tensor],
        *,
        target: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        errors, gain_modulated_errors = self.compute_errors(states)
        y_hat = self._readout(states[-1])
        top_error = None
        if target is not None:
            readout_feedback_scale = self.readout_weight.square().sum().clamp_min(1.0)
            top_error = q_matmul(_adjoint_weight(self.readout_weight), y_hat - target) / readout_feedback_scale

        gradients: list[torch.Tensor] = []
        for latent_idx in range(1, self.L + 1):
            gradient = -q_matmul(_adjoint_weight(self.layers[latent_idx - 1].W), gain_modulated_errors[latent_idx - 1])
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
        adaptive: bool | None = None,
        tolerance: float | None = None,
        min_steps: int | None = None,
        check_finite: bool | None = None,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        x_quat = self._prepare_input(x)
        y_quat = None if y is None else self._prepare_target(y)
        infer_steps = self.prediction_inference_steps if steps is None else int(steps)
        infer_eta = self.prediction_eta_infer if eta_infer is None else float(eta_infer)
        use_adaptive = self.adaptive_inference if adaptive is None else bool(adaptive)
        infer_tolerance = self.inference_tolerance if tolerance is None else float(tolerance)
        infer_min_steps = self.min_inference_steps if min_steps is None else int(min_steps)
        do_finite_checks = self.check_finite if check_finite is None else bool(check_finite)

        states = [x_quat] + self.init_latents(
            x_quat.shape[0],
            x_quat.device,
            dtype=x_quat.dtype,
            mode=latent_init,
            x=x_quat,
        )
        prediction = self._readout(states[-1])
        for step_idx in range(infer_steps):
            gradients, prediction = self._latent_gradients(states, target=y_quat)
            if do_finite_checks and not _all_finite(tuple(gradients) + (prediction,)):
                raise FloatingPointError("Quaternion PCN latent inference became non-finite.")
            if use_adaptive:
                grad_norm = torch.stack([gradient.norm(dim=-1).mean() for gradient in gradients]).mean()
                if step_idx + 1 >= infer_min_steps and float(grad_norm.detach().cpu()) <= infer_tolerance:
                    break
            for latent_idx, gradient in enumerate(gradients, start=1):
                states[latent_idx].sub_(gradient, alpha=infer_eta)
        gradients, prediction = self._latent_gradients(states, target=y_quat)
        if do_finite_checks and not _all_finite(tuple(gradients) + (prediction,)):
            raise FloatingPointError("Quaternion PCN final latent state became non-finite.")
        return {"states": states, "prediction": prediction}

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        *,
        infer_steps: int | None = None,
        eta_infer: float | None = None,
        latent_init: LatentInitName = "feedforward",
        fast: bool = False,
    ) -> torch.Tensor:
        if fast:
            return self._readout(self.feedforward_latents(x)[-1])
        result = self.infer_latents(x, steps=infer_steps, eta_infer=eta_infer, latent_init=latent_init)
        return cast(torch.Tensor, result["prediction"])

    forward = predict

    @torch.no_grad()
    def train_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        eta_infer: float,
        eta_learn: float,
        T_infer: int,
        T_learn: int = 1,
        latent_init: LatentInitName = "feedforward",
    ) -> dict[str, float]:
        x_quat = self._prepare_input(x)
        y_quat = self._prepare_target(y)
        inference_result = self.infer_latents(
            x_quat,
            y=y_quat,
            steps=T_infer,
            eta_infer=eta_infer,
            latent_init=latent_init,
        )
        states = cast(list[torch.Tensor], inference_result["states"])
        batch_size = x_quat.shape[0]

        for _ in range(T_learn):
            _, gain_modulated_errors = self.compute_errors(states)
            for layer_idx, layer in enumerate(self.layers):
                gradient = -q_mul(
                    gain_modulated_errors[layer_idx].unsqueeze(2),
                    q_conj(states[layer_idx + 1]).unsqueeze(1),
                ).mean(dim=0)
                bias_gradient = -gain_modulated_errors[layer_idx].mean(dim=0)
                self._apply_update(
                    param=layer.W,
                    gradient=gradient,
                    accumulator=cast(torch.Tensor, getattr(self, f"_weight_acc_{layer_idx}")),
                    delta_accumulator=cast(torch.Tensor, getattr(self, f"_weight_delta_acc_{layer_idx}")),
                    lr=eta_learn,
                )
                self._apply_update(
                    param=layer.bias,
                    gradient=bias_gradient,
                    accumulator=cast(torch.Tensor, getattr(self, f"_bias_acc_{layer_idx}")),
                    delta_accumulator=cast(torch.Tensor, getattr(self, f"_bias_delta_acc_{layer_idx}")),
                    lr=eta_learn,
                )

            # top_latent = self.feedforward_latents(x_quat)[-1]
            # output_error = self._readout(top_latent) - y_quat
            # readout_scale = (top_latent.square().sum(dim=(-1, -2)) + 1.0).mean().clamp_min(1.0)
            # readout_gradient = q_mul(output_error.unsqueeze(2), q_conj(top_latent).unsqueeze(1)).mean(dim=0)
            # readout_gradient = readout_gradient / readout_scale
            # readout_bias_gradient = output_error.mean(dim=0) / readout_scale

            top_latent = states[-1]
            output_error = self._readout(top_latent) - y_quat
            readout_gradient = q_mul(
                output_error.unsqueeze(2),
                q_conj(top_latent).unsqueeze(1),
            ).mean(dim=0)
            readout_bias_gradient = output_error.mean(dim=0)
            self._apply_update(
                param=self.readout_weight,
                gradient=readout_gradient,
                accumulator=cast(torch.Tensor, self._readout_acc),
                delta_accumulator=cast(torch.Tensor, self._readout_delta_acc),
                lr=eta_learn,
            )
            self._apply_update(
                param=self.readout_bias,
                gradient=readout_bias_gradient,
                accumulator=cast(torch.Tensor, self._readout_bias_acc),
                delta_accumulator=cast(torch.Tensor, self._readout_bias_delta_acc),
                lr=eta_learn,
            )

        y_hat = self.predict(x_quat, infer_steps=0, fast=True)
        metrics = scalar_regression_metrics(y_hat, y_quat)
        metrics["quaternion_mae"] = float(quaternion_mae_loss(y_hat, y_quat).detach().cpu())
        metrics["free_energy"] = self.energy(x_quat, y_quat)
        return metrics

    @torch.no_grad()
    def energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> float:
        x_quat = self._prepare_input(x)
        y_quat = None if y is None else self._prepare_target(y)
        states = [x_quat, *self.feedforward_latents(x_quat)]
        errors, _ = self.compute_errors(states)
        reconstruction = torch.stack([error.square().mean() for error in errors]).mean()
        supervised = torch.zeros((), device=x_quat.device, dtype=x_quat.dtype)
        if y_quat is not None:
            supervised = (self._readout(states[-1]) - y_quat).square().mean()
        free_energy = 0.5 * (reconstruction + supervised)
        if not math.isfinite(float(free_energy.detach().cpu())):
            raise FloatingPointError("Quaternion PCN energy became non-finite.")
        return float(free_energy.detach().cpu())

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
    ) -> dict[str, float]:
        y_quat = self._prepare_target(y)
        y_hat = self.predict(x, infer_steps=infer_steps, eta_infer=eta_infer, latent_init=latent_init, fast=fast)
        metrics = scalar_regression_metrics(y_hat, y_quat)
        metrics["quaternion_mae"] = float(quaternion_mae_loss(y_hat, y_quat).detach().cpu())
        metrics["free_energy"] = self.energy(x, y_quat)
        return metrics
