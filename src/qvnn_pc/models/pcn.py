"""Implementation of Stenlund 2025 Predictive Coding Network"""

from __future__ import annotations

from typing import Callable, Literal, cast

import torch
import torch.nn.functional as F
from torch import autocast, nn

import tqdm

ActivationName = Literal["leakyrelu", "sigmoid", "swish", "tanh"]



def _default_activation_deriv(a: torch.Tensor) -> torch.Tensor:
    return (a > 0).float()


class PCNLayer(nn.Module):
    activation_fn: Callable[[torch.Tensor], torch.Tensor]
    activation_deriv: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        in_dim,
        out_dim,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
        activation_deriv: Callable[[torch.Tensor], torch.Tensor] = _default_activation_deriv,
    ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.activation_fn: Callable[[torch.Tensor], torch.Tensor] = activation_fn
        self.activation_deriv: Callable[[torch.Tensor], torch.Tensor] = activation_deriv

    def forward(self, x_above):
        with autocast(device_type="cuda"):
            a = x_above @ self.W.T
            x_hat = self.activation_fn(a)
            return x_hat, a


class PredictiveCodingNetwork(nn.Module):
    def __init__(self, dims: list[int], output_dim: int) -> None:
        super().__init__()
        self.dims = dims
        self.L = len(dims) - 1
        self.layers = nn.ModuleList([
            PCNLayer(in_dim=dims[layer_idx + 1], out_dim=dims[layer_idx])
            for layer_idx in range(self.L)
        ])
        self.readout = nn.Linear(dims[-1], output_dim, bias=False)

    def init_latents(self, batch_size, device):
        return [
            torch.randn(batch_size, d, device=device, requires_grad=False)
            for d in self.dims[1:]
        ]

    def compute_errors(self, inputs_latents):
        errors, gain_modulated_errors = [], []
        for layer_idx, layer in enumerate(self.layers):
            x_hat, a = layer(inputs_latents[layer_idx + 1])
            err = inputs_latents[layer_idx] - x_hat
            activation_deriv = cast(Callable[[torch.Tensor], torch.Tensor], layer.activation_deriv)
            gm_err = err * activation_deriv(a)
            errors.append(err)
            gain_modulated_errors.append(gm_err)

        return errors, gain_modulated_errors


def train_pcn(model, data_loader, num_epochs, eta_infer, eta_learn,
    T_infer,
    T_learn,
    device="cuda",
):
    model.to(device).train()

    for epoch in tqdm.tqdm(range(num_epochs)):
        for x_batch, y_batch in data_loader:
            batch_size = x_batch.size(0)
            input_dim = model.dims[0]

            x_batch = x_batch.view(batch_size, input_dim).to(device)
            y_batch = F.one_hot(y_batch, num_classes=model.readout.out_features).float().to(device)

            inputs_latents = [x_batch] + model.init_latents(batch_size, device)
            weights = [layer.W for layer in model.layers] + [model.readout.weight]

            # INFERENCE - T_infer steps
            with torch.no_grad(), autocast(device_type="cuda"):
                for t in range(1, T_infer + 1):
                    errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                    y_hat = model.readout(inputs_latents[-1])
                    eps_sup = y_hat - y_batch
                    eps_L = eps_sup @ weights[-1]
                    errors_extended = errors + [eps_L]

                    # Latent gradients and updates
                    for latent_idx in range(1, model.L + 1):
                        grad_Xl = errors_extended[latent_idx] - gain_modulated_errors[latent_idx - 1] @ weights[latent_idx - 1]
                        inputs_latents[latent_idx] -= eta_infer * grad_Xl

            # LEARNING - T_learn steps
            with torch.no_grad():
                for t in range(T_infer + 1, T_learn + T_infer + 1):
                    errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                    y_hat = model.readout(inputs_latents[-1])
                    eps_sup = y_hat - y_batch

                    # Weight gradients and updates
                    for weight_idx in range(model.L):
                        grad_Wl = -(gain_modulated_errors[weight_idx].T @ inputs_latents[weight_idx + 1]) / batch_size
                        weights[weight_idx] -= eta_learn * grad_Wl

                    grad_Wout = eps_sup.T @ inputs_latents[-1] / batch_size
                    weights[-1] -= eta_learn * grad_Wout