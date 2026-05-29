"""Bill et al.-style benchmark dataset preparation for quaternion regression. Similar to code in '../capsule-2567237/code/Utils/ackley-data.jl'"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.stats import qmc

from qvnn_pc.functions import ackley_4d, griewank_4d, rastrigin_4d, schwefel_4d

BENCHMARK_FUNCTIONS = {
    "ackley": ackley_4d,
    "griewank": griewank_4d,
    "rastrigin": rastrigin_4d,
    "schwefel": schwefel_4d,
}

BENCHMARK_BOUNDS = {
    "ackley": (-30.0, 30.0),
    "griewank": (-600.0, 600.0),
    "rastrigin": (-5.12, 5.12),
    "schwefel": (-500.0, 500.0),
}


@dataclass(frozen=True)
class QuaternionRegressionDataSplit:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    raw_x_train: torch.Tensor
    raw_x_val: torch.Tensor
    raw_x_test: torch.Tensor
    y_train_scalar: torch.Tensor
    y_val_scalar: torch.Tensor
    y_test_scalar: torch.Tensor
    y_train_scalar_normalized: torch.Tensor
    y_val_scalar_normalized: torch.Tensor
    y_test_scalar_normalized: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


def _resolve_benchmark(name: str) -> tuple[str, tuple[float, float]]:
    key = name.lower()
    if key not in BENCHMARK_FUNCTIONS:
        valid = ", ".join(sorted(BENCHMARK_FUNCTIONS))
        raise ValueError(f"Unknown benchmark {name!r}. Expected one of: {valid}.")
    return key, BENCHMARK_BOUNDS[key]


def _quaternionize_inputs(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(1)


def _quaternionize_targets(y: torch.Tensor) -> torch.Tensor:
    y_quat = torch.zeros(y.shape[0], 1, 4, dtype=y.dtype, device=y.device)
    y_quat[:, 0, 0] = y
    return y_quat


def _lhs_scaled_inputs(
    function_name: str,
    *,
    n: int,
    sample_seed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    _, (low, high) = _resolve_benchmark(function_name)
    sampler = qmc.LatinHypercube(d=4, optimization="random-cd", seed=sample_seed)
    sample_unit = sampler.random(n)
    sample_scaled = qmc.scale(sample_unit, [low] * 4, [high] * 4)
    return torch.tensor(sample_scaled, dtype=dtype)


def make_benchmark_regression_split(
    function_name: str,
    *,
    n: int = 30_000,
    sample_seed: int = 0,
    split_seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
) -> QuaternionRegressionDataSplit:
    """Prepare one shuffled, split, train-standardized benchmark dataset."""
    if n <= 0:
        raise ValueError(f"Expected n > 0, got {n}.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"Expected train_ratio in (0, 1), got {train_ratio}.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"Expected val_ratio in (0, 1), got {val_ratio}.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("Expected train_ratio + val_ratio < 1.0.")

    key, _ = _resolve_benchmark(function_name)
    raw_x = _lhs_scaled_inputs(key, n=n, sample_seed=sample_seed, dtype=dtype)
    y_scalar = BENCHMARK_FUNCTIONS[key](raw_x)

    shuffle_generator = torch.Generator(device="cpu").manual_seed(split_seed)
    permutation = torch.randperm(n, generator=shuffle_generator)
    raw_x = raw_x[permutation]
    y_scalar = y_scalar[permutation]

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Split sizes must all be positive, got train={n_train}, val={n_val}, test={n_test}."
        )

    raw_x_train = raw_x[:n_train]
    raw_x_val = raw_x[n_train:n_train + n_val]
    raw_x_test = raw_x[n_train + n_val:]
    y_train_scalar = y_scalar[:n_train]
    y_val_scalar = y_scalar[n_train:n_train + n_val]
    y_test_scalar = y_scalar[n_train + n_val:]

    x_mean = raw_x_train.mean(dim=0)
    x_std = raw_x_train.std(dim=0, unbiased=False).clamp_min(eps)
    y_mean = y_train_scalar.mean()
    y_std = y_train_scalar.std(unbiased=False).clamp_min(eps)
    y_train_scalar_normalized = (y_train_scalar - y_mean) / y_std
    y_val_scalar_normalized = (y_val_scalar - y_mean) / y_std
    y_test_scalar_normalized = (y_test_scalar - y_mean) / y_std

    x_train = _quaternionize_inputs((raw_x_train - x_mean) / x_std)
    x_val = _quaternionize_inputs((raw_x_val - x_mean) / x_std)
    x_test = _quaternionize_inputs((raw_x_test - x_mean) / x_std)
    y_train = _quaternionize_targets(y_train_scalar)
    y_val = _quaternionize_targets(y_val_scalar)
    y_test = _quaternionize_targets(y_test_scalar)

    target_device = torch.device(device) if device is not None else None

    def _move(tensor: torch.Tensor) -> torch.Tensor:
        if target_device is None:
            return tensor
        return tensor.to(target_device)

    return QuaternionRegressionDataSplit(
        x_train=_move(x_train),
        y_train=_move(y_train),
        x_val=_move(x_val),
        y_val=_move(y_val),
        x_test=_move(x_test),
        y_test=_move(y_test),
        raw_x_train=_move(raw_x_train),
        raw_x_val=_move(raw_x_val),
        raw_x_test=_move(raw_x_test),
        y_train_scalar=_move(y_train_scalar),
        y_val_scalar=_move(y_val_scalar),
        y_test_scalar=_move(y_test_scalar),
        y_train_scalar_normalized=_move(y_train_scalar_normalized),
        y_val_scalar_normalized=_move(y_val_scalar_normalized),
        y_test_scalar_normalized=_move(y_test_scalar_normalized),
        x_mean=_move(x_mean),
        x_std=_move(x_std),
        y_mean=_move(y_mean),
        y_std=_move(y_std),
    )
