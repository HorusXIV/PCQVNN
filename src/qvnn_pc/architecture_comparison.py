"""Benchmark runner for comparing quaternion MLP architectures in this package."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Literal

import pandas as pd
import torch

from qvnn_pc.config import default_hyperparameters_path, get_model_hyperparameters
from qvnn_pc.data import BENCHMARK_FUNCTIONS, make_benchmark_regression_split
from qvnn_pc.models import (
    CMLPQuaternionMLP,
    GHRQuaternionMLP,
    quaternion_mae_loss,
    scalar_regression_metrics,
)

ArchitectureName = Literal["cmlp", "ghr"]
DEFAULT_ARCHITECTURES: tuple[ArchitectureName, ...] = ("cmlp", "ghr")
DEFAULT_BENCHMARKS: tuple[str, ...] = tuple(BENCHMARK_FUNCTIONS)
ARCHITECTURE_LABELS = {
    "cmlp": "CMLP",
    "ghr": "GHR",
}


@dataclass(frozen=True)
class ArchitectureComparisonConfig:
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS
    architectures: tuple[ArchitectureName, ...] = DEFAULT_ARCHITECTURES
    layer_sizes: tuple[int, ...] = (1, 5, 5, 5, 1)
    epochs: int = 75
    repetitions: int = 2
    dataset_size: int = 30_000
    sample_seed: int = 0
    split_seed: int = 0
    early_stopping: bool = True
    batch_size: int | None = None
    hyperparameters_path: Path = default_hyperparameters_path()
    results_root: Path = Path("results")
    run_name: str | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype = torch.float32


def _canonical_benchmark_name(name: str) -> str:
    key = name.lower()
    if key not in BENCHMARK_FUNCTIONS:
        valid = ", ".join(sorted(BENCHMARK_FUNCTIONS))
        raise ValueError(f"Unknown benchmark {name!r}. Expected one of: {valid}.")
    return key


def _canonical_architecture_name(name: str) -> ArchitectureName:
    key = name.lower()
    if key not in ARCHITECTURE_LABELS:
        valid = ", ".join(sorted(ARCHITECTURE_LABELS))
        raise ValueError(f"Unknown architecture {name!r}. Expected one of: {valid}.")
    return key  # type: ignore[return-value]


def _build_model(
    architecture: ArchitectureName,
    *,
    activation: str,
    optimizer: str,
    lr: float,
    layer_sizes: tuple[int, ...],
    init_seed: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
):
    generator = torch.Generator(device="cpu").manual_seed(init_seed)
    if architecture == "cmlp":
        return CMLPQuaternionMLP(
            activation=activation,
            optimizer=optimizer,
            lr=lr,
            layer_sizes=layer_sizes,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    if architecture == "ghr":
        return GHRQuaternionMLP(
            activation=activation,
            optimizer=optimizer,
            lr=lr,
            layer_sizes=layer_sizes,
            generator=generator,
            device=device,
            dtype=dtype,
    )
    raise ValueError(f"Unsupported architecture {architecture!r}.")


def _timestamped_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _prepare_run_directory(config: ArchitectureComparisonConfig, output_path: Path | None) -> tuple[Path, Path]:
    if output_path is not None:
        run_dir = output_path.parent
        summary_path = output_path
    else:
        run_dir = config.results_root / (config.run_name or _timestamped_run_name())
        summary_path = run_dir / "architecture_comparison.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "history").mkdir(exist_ok=True)
    return run_dir, summary_path


def _artifact_stem(benchmark: str, architecture: ArchitectureName, repetition: int) -> str:
    return f"{benchmark}_{architecture}_rep{repetition:02d}"


def _save_run_metadata(
    run_dir: Path,
    config: ArchitectureComparisonConfig,
    *,
    summary_path: Path,
) -> None:
    metadata = {
        "benchmarks": list(config.benchmarks),
        "architectures": list(config.architectures),
        "layer_sizes": list(config.layer_sizes),
        "epochs": config.epochs,
        "repetitions": config.repetitions,
        "dataset_size": config.dataset_size,
        "sample_seed": config.sample_seed,
        "split_seed": config.split_seed,
        "early_stopping": config.early_stopping,
        "batch_size_override": config.batch_size,
        "hyperparameters_path": str(config.hyperparameters_path),
        "results_root": str(config.results_root),
        "run_name": config.run_name,
        "summary_path": str(summary_path),
        "device": None if config.device is None else str(config.device),
        "dtype": str(config.dtype),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if config.hyperparameters_path.exists():
        shutil.copy2(config.hyperparameters_path, run_dir / "hyperparameters.yml")


def _save_training_artifacts(
    run_dir: Path,
    *,
    benchmark: str,
    architecture: ArchitectureName,
    repetition: int,
    model,
    history: dict[str, list[float] | float | int],
    hyperparameters,
    batch_size: int,
) -> None:
    stem = _artifact_stem(benchmark, architecture, repetition)
    history_path = run_dir / "history" / f"{stem}.json"
    model_path = run_dir / "models" / f"{stem}.pt"

    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "benchmark": benchmark,
            "architecture": architecture,
            "repetition": repetition,
            "layer_sizes": list(model.layer_sizes),
            "activation": hyperparameters.activation,
            "optimizer": hyperparameters.optimizer,
            "learning_rate": hyperparameters.lr,
            "batch_size": batch_size,
            "history_path": str(history_path),
            "checkpoint_note": "Model parameters restored to the best tracked validation checkpoint.",
        },
        model_path,
    )


def run_architecture_comparison(
    config: ArchitectureComparisonConfig = ArchitectureComparisonConfig(),
    *,
    run_dir: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for benchmark_idx, benchmark_name in enumerate(config.benchmarks):
        benchmark = _canonical_benchmark_name(benchmark_name)
        split = make_benchmark_regression_split(
            benchmark,
            n=config.dataset_size,
            sample_seed=config.sample_seed + benchmark_idx,
            split_seed=config.split_seed + benchmark_idx,
            device=config.device,
            dtype=config.dtype,
        )

        for architecture_name in config.architectures:
            architecture = _canonical_architecture_name(architecture_name)
            hyperparameters = get_model_hyperparameters(
                architecture,
                benchmark,
                config_path=config.hyperparameters_path,
            )
            batch_size = config.batch_size if config.batch_size is not None else hyperparameters.batch_size

            for repetition in range(1, config.repetitions + 1):
                model = _build_model(
                    architecture,
                    activation=hyperparameters.activation,
                    optimizer=hyperparameters.optimizer,
                    lr=hyperparameters.lr,
                    layer_sizes=config.layer_sizes,
                    init_seed=int(repetition * 1e3),
                    device=config.device,
                    dtype=config.dtype,
                )
                shuffle_generator = torch.Generator(device="cpu").manual_seed(int(repetition * 1e6))
                history = model.fit(
                    split.x_train,
                    split.y_train,
                    x_val=split.x_val,
                    y_val=split.y_val,
                    epochs=config.epochs,
                    batch_size=batch_size,
                    shuffle_generator=shuffle_generator,
                    early_stopping=config.early_stopping and architecture == "ghr",
                )

                test_pred = model.forward(split.x_test)
                test_metrics = scalar_regression_metrics(test_pred, split.y_test)
                test_qmae = float(quaternion_mae_loss(test_pred, split.y_test).detach().cpu())
                if run_dir is not None:
                    _save_training_artifacts(
                        run_dir,
                        benchmark=benchmark,
                        architecture=architecture,
                        repetition=repetition,
                        model=model,
                        history=history,
                        hyperparameters=hyperparameters,
                        batch_size=batch_size,
                    )
                rows.append(
                    {
                        "repetition": repetition,
                        "algorithm": ARCHITECTURE_LABELS[architecture],
                        "test_function": benchmark,
                        "learning_rate": hyperparameters.lr,
                        "optimizer": hyperparameters.optimizer,
                        "activation": hyperparameters.activation,
                        "batch_size": batch_size,
                        "test_quaternion_mae": test_qmae,
                        "test_mse": test_metrics["mse"],
                        "test_mae": test_metrics["mae"],
                        "test_mape": test_metrics["mape"],
                        "test_r2": test_metrics["r2"],
                        "epochs": int(history.get("epochs_ran", config.epochs)),
                        "runtime_seconds": float(history["runtime"]),
                    }
                )

    return pd.DataFrame(rows)


def write_architecture_comparison(results: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".xlsx":
        results.to_excel(output, index=False)
    else:
        results.to_csv(output, index=False)
    return output


def _dtype_from_name(name: str) -> torch.dtype:
    lookup = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return lookup[name.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(lookup))
        raise ValueError(f"Unsupported dtype {name!r}. Expected one of: {valid}.") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="*", default=list(DEFAULT_BENCHMARKS))
    parser.add_argument("--architectures", nargs="*", default=list(DEFAULT_ARCHITECTURES))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--dataset-size", type=int, default=30_000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", type=Path, default=default_hyperparameters_path())
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    config = ArchitectureComparisonConfig(
        benchmarks=tuple(args.benchmarks),
        architectures=tuple(_canonical_architecture_name(name) for name in args.architectures),
        epochs=args.epochs,
        repetitions=args.repetitions,
        dataset_size=args.dataset_size,
        sample_seed=args.sample_seed,
        split_seed=args.split_seed,
        early_stopping=args.early_stopping,
        batch_size=args.batch_size,
        hyperparameters_path=args.config,
        results_root=args.results_root,
        run_name=args.run_name,
        dtype=_dtype_from_name(args.dtype),
        device=args.device,
    )
    run_dir, summary_path = _prepare_run_directory(config, args.output)
    _save_run_metadata(run_dir, config, summary_path=summary_path)
    results = run_architecture_comparison(config, run_dir=run_dir)
    path = write_architecture_comparison(results, summary_path)
    print(f"Wrote {len(results)} rows to {path}.")
    print(f"Saved run artifacts to {run_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
