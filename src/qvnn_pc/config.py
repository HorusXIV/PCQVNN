"""Configuration loading for experiment hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def default_hyperparameters_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "hyperparameters.yml"


@dataclass(frozen=True)
class ModelHyperparameters:
    lr: float
    optimizer: str
    activation: str
    batch_size: int = 1


def load_hyperparameters(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path is not None else default_hyperparameters_path()
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter config not found at {path}.")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a top-level mapping.")
    return data


def get_model_hyperparameters(
    architecture: str,
    benchmark: str,
    *,
    config_path: str | Path | None = None,
) -> ModelHyperparameters:
    raw = load_hyperparameters(config_path)
    try:
        architecture_map = raw["architectures"][architecture.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(raw.get("architectures", {})))
        raise ValueError(
            f"Unknown architecture {architecture!r} in hyperparameter config. Available: {available}."
        ) from exc

    try:
        benchmark_map = architecture_map[benchmark.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(architecture_map))
        raise ValueError(
            f"Unknown benchmark {benchmark!r} for architecture {architecture!r} in hyperparameter config. "
            f"Available: {available}."
        ) from exc

    try:
        return ModelHyperparameters(
            lr=float(benchmark_map["lr"]),
            optimizer=str(benchmark_map["optimizer"]),
            activation=str(benchmark_map["activation"]),
            batch_size=int(benchmark_map.get("batch_size", 1)),
        )
    except KeyError as exc:
        raise ValueError(
            f"Missing required hyperparameter key {exc.args[0]!r} for architecture {architecture!r}, "
            f"benchmark {benchmark!r}."
        ) from exc

