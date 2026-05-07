"""Simple benchmark runner for the saved architecture-comparison experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import signal
import shutil
import time
from typing import Literal

import pandas as pd
from pandas.errors import EmptyDataError
import torch

from qvnn_pc.config import default_hyperparameters_path, get_model_hyperparameters
from qvnn_pc.data import BENCHMARK_FUNCTIONS, make_benchmark_regression_split
from qvnn_pc.models import (
    CMLPQuaternionMLP,
    GHRQuaternionMLP,
    quaternion_mae_loss,
    scalar_regression_metrics,
)
from qvnn_pc.models.shared import pq_alpha_should_stop as _pq_alpha_should_stop

ArchitectureName = Literal["cmlp", "ghr"]
DEFAULT_ARCHITECTURES: tuple[ArchitectureName, ...] = ("cmlp", "ghr")
DEFAULT_BENCHMARKS: tuple[str, ...] = tuple(BENCHMARK_FUNCTIONS)
DEFAULT_LAYER_SIZES: tuple[int, ...] = (1, 5, 5, 5, 1)
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_HYPERPARAMETERS_PATH = default_hyperparameters_path()

ARCHITECTURE_LABELS = {
    "cmlp": "CMLP",
    "ghr": "GHR",
}
_STOP_REQUESTED = False


@dataclass(frozen=True)
class ArchitectureComparisonConfig:
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS
    architectures: tuple[ArchitectureName, ...] = DEFAULT_ARCHITECTURES
    layer_sizes: tuple[int, ...] = DEFAULT_LAYER_SIZES
    epochs: int = 5
    repetitions: int = 2
    dataset_size: int = 30_000
    sample_seed: int = 0
    split_seed: int = 0
    early_stopping: bool = True
    device: torch.device | str | None = None
    dtype: torch.dtype = torch.float32


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    signal_name = signal.Signals(signum).name
    print(f"Received {signal_name}. Saving progress after the current batch.")


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


def _prepare_run_directory(output_path: Path | None) -> tuple[Path, Path]:
    if output_path is not None:
        run_dir = output_path.parent
        summary_path = output_path
    else:
        run_dir = DEFAULT_RESULTS_ROOT / _timestamped_run_name()
        summary_path = run_dir / "architecture_comparison.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "history").mkdir(exist_ok=True)
    return run_dir, summary_path


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
        "summary_path": str(summary_path),
        "device": None if config.device is None else str(config.device),
        "dtype": str(config.dtype),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if DEFAULT_HYPERPARAMETERS_PATH.exists():
        shutil.copy2(DEFAULT_HYPERPARAMETERS_PATH, run_dir / "hyperparameters.yml")


def _save_training_artifacts(
    run_dir: Path,
    *,
    benchmark: str,
    architecture: ArchitectureName,
    repetition: int,
    model,
    history: dict[str, list[float] | float | int],
    activation: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
) -> None:
    stem = f"{benchmark}_{architecture}_rep{repetition:02d}"
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
            "activation": activation,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "history_path": str(history_path),
            "checkpoint_note": "Model parameters restored to the best tracked validation checkpoint.",
        },
        model_path,
    )


def _checkpoint_path(run_dir: Path, benchmark: str, architecture: ArchitectureName, repetition: int) -> Path:
    return run_dir / "checkpoints" / f"{benchmark}_{architecture}_rep{repetition:02d}.pt"


def _clone_state_dict(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _save_resume_checkpoint(
    checkpoint_path: Path,
    *,
    model,
    best_state_dict: dict[str, torch.Tensor],
    history: dict[str, list[float] | float | int],
    epoch_index: int,
    next_batch_start: int,
    permutation: torch.Tensor | None,
    best_validation_loss: float,
    shuffle_generator: torch.Generator,
    runtime_seconds: float,
    interrupted: bool,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_state_dict": best_state_dict,
            "history": history,
            "epoch_index": epoch_index,
            "next_batch_start": next_batch_start,
            "permutation": None if permutation is None else permutation.detach().cpu().clone(),
            "best_validation_loss": best_validation_loss,
            "shuffle_generator_state": shuffle_generator.get_state(),
            "runtime_seconds": runtime_seconds,
            "interrupted": interrupted,
        },
        checkpoint_path,
    )


def _fit_with_resume(
    model,
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    batch_size: int,
    shuffle_generator: torch.Generator,
    early_stopping: bool,
    checkpoint_path: Path | None,
) -> dict[str, list[float] | float | int]:
    global _STOP_REQUESTED
    history: dict[str, list[float] | float | int] = {
        "quaternion_mae": [],
        "validation_quaternion_mae": [],
        "mape": [],
        "r2": [],
    }
    best_state_dict = _clone_state_dict(model)
    best_validation_loss = float("inf")
    start_epoch = 0
    next_batch_start = 0
    pending_permutation: torch.Tensor | None = None
    runtime_seconds = 0.0

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        best_state_dict = checkpoint["best_state_dict"]
        history = checkpoint["history"]
        start_epoch = int(checkpoint.get("epoch_index", len(history["quaternion_mae"])))  # type: ignore[index]
        next_batch_start = int(checkpoint.get("next_batch_start", 0))
        pending_permutation = checkpoint.get("permutation")
        best_validation_loss = float(checkpoint["best_validation_loss"])
        shuffle_generator.set_state(checkpoint["shuffle_generator_state"])
        runtime_seconds = float(checkpoint.get("runtime_seconds", 0.0))
        if next_batch_start > 0:
            print(f"Resuming {checkpoint_path.name} from epoch {start_epoch + 1}, batch offset {next_batch_start}.")
        else:
            print(f"Resuming {checkpoint_path.name} from epoch {start_epoch + 1}.")

    start_time = time.perf_counter()

    for epoch_idx in range(start_epoch, epochs):
        if epoch_idx == start_epoch and pending_permutation is not None:
            permutation = pending_permutation
        else:
            permutation = torch.randperm(
                x_train.shape[0],
                generator=shuffle_generator,
                device="cpu",
            )

        batch_starts = range(next_batch_start, x_train.shape[0], batch_size)
        for batch_start in batch_starts:
            batch_indices = permutation[batch_start:batch_start + batch_size]
            model.train_step(x_train[batch_indices], y_train[batch_indices])
            if _STOP_REQUESTED:
                elapsed = runtime_seconds + (time.perf_counter() - start_time)
                if checkpoint_path is not None:
                    _save_resume_checkpoint(
                        checkpoint_path,
                        model=model,
                        best_state_dict=best_state_dict,
                        history=history,
                        epoch_index=epoch_idx,
                        next_batch_start=batch_start + batch_size,
                        permutation=permutation,
                        best_validation_loss=best_validation_loss,
                        shuffle_generator=shuffle_generator,
                        runtime_seconds=elapsed,
                        interrupted=True,
                    )
                history["runtime"] = elapsed
                history["best_validation_loss"] = best_validation_loss
                history["epochs_ran"] = len(history["quaternion_mae"])  # type: ignore[arg-type]
                history["interrupted"] = True
                return history

        train_pred = model.forward(x_train)
        train_qmae = float(quaternion_mae_loss(train_pred, y_train).detach().cpu())
        train_metrics = scalar_regression_metrics(train_pred, y_train)

        val_pred = model.forward(x_val)
        validation_loss = float(quaternion_mae_loss(val_pred, y_val).detach().cpu())

        history["quaternion_mae"].append(train_qmae)  # type: ignore[index]
        history["validation_quaternion_mae"].append(validation_loss)  # type: ignore[index]
        history["mape"].append(train_metrics["mape"])  # type: ignore[index]
        history["r2"].append(train_metrics["r2"])  # type: ignore[index]

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state_dict = _clone_state_dict(model)

        epochs_completed = epoch_idx + 1
        elapsed = runtime_seconds + (time.perf_counter() - start_time)
        if checkpoint_path is not None:
            _save_resume_checkpoint(
                checkpoint_path,
                model=model,
                best_state_dict=best_state_dict,
                history=history,
                epoch_index=epochs_completed,
                next_batch_start=0,
                permutation=None,
                best_validation_loss=best_validation_loss,
                shuffle_generator=shuffle_generator,
                runtime_seconds=elapsed,
                interrupted=False,
            )

        if early_stopping and _pq_alpha_should_stop(
            history["quaternion_mae"],  # type: ignore[arg-type]
            validation_loss,
            best_validation_loss,
            window=5,
            threshold=0.5,
        ):
            break

        if not torch.isfinite(torch.tensor(train_qmae)):
            break

        next_batch_start = 0
        pending_permutation = None

    model.load_state_dict(best_state_dict)
    history["runtime"] = runtime_seconds + (time.perf_counter() - start_time)
    history["best_validation_loss"] = best_validation_loss
    history["epochs_ran"] = len(history["quaternion_mae"])  # type: ignore[arg-type]
    history["interrupted"] = False
    return history


def run_architecture_comparison(
    config: ArchitectureComparisonConfig = ArchitectureComparisonConfig(),
    *,
    run_dir: Path | None = None,
    existing_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    global _STOP_REQUESTED
    rows: list[dict[str, float | int | str]] = []
    completed: set[tuple[str, str, int]] = set()

    if existing_results is not None and not existing_results.empty:
        rows.extend(existing_results.to_dict("records"))
        completed = {
            (str(row.test_function), str(row.algorithm), int(row.repetition))
            for row in existing_results.itertuples(index=False)
        }

    if run_dir is not None:
        (run_dir / "checkpoints").mkdir(exist_ok=True)

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
                config_path=DEFAULT_HYPERPARAMETERS_PATH,
            )

            for repetition in range(1, config.repetitions + 1):
                result_key = (benchmark, ARCHITECTURE_LABELS[architecture], repetition)
                if result_key in completed:
                    print(f"Skipping completed run: {benchmark} {ARCHITECTURE_LABELS[architecture]} rep {repetition}.")
                    continue

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
                checkpoint_path = None if run_dir is None else _checkpoint_path(run_dir, benchmark, architecture, repetition)
                history = _fit_with_resume(
                    model,
                    x_train=split.x_train,
                    y_train=split.y_train,
                    x_val=split.x_val,
                    y_val=split.y_val,
                    epochs=config.epochs,
                    batch_size=hyperparameters.batch_size,
                    shuffle_generator=shuffle_generator,
                    early_stopping=config.early_stopping,
                    checkpoint_path=checkpoint_path,
                )
                if history.get("interrupted"):
                    print("Training paused after checkpoint save. Returning partial results for resume.")
                    results = pd.DataFrame(rows)
                    if results.empty:
                        return results
                    return results.sort_values(["test_function", "algorithm", "repetition"]).reset_index(drop=True)

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
                        activation=hyperparameters.activation,
                        optimizer=hyperparameters.optimizer,
                        learning_rate=hyperparameters.lr,
                        batch_size=hyperparameters.batch_size,
                    )
                if checkpoint_path is not None and checkpoint_path.exists():
                    checkpoint_path.unlink()

                rows.append(
                    {
                        "repetition": repetition,
                        "algorithm": ARCHITECTURE_LABELS[architecture],
                        "test_function": benchmark,
                        "learning_rate": hyperparameters.lr,
                        "optimizer": hyperparameters.optimizer,
                        "activation": hyperparameters.activation,
                        "batch_size": hyperparameters.batch_size,
                        "test_quaternion_mae": test_qmae,
                        "test_mse": test_metrics["mse"],
                        "test_mae": test_metrics["mae"],
                        "test_mape": test_metrics["mape"],
                        "test_r2": test_metrics["r2"],
                        "epochs": int(history.get("epochs_ran", config.epochs)),
                        "runtime_seconds": float(history["runtime"]),
                    }
                )
                completed.add(result_key)

    results = pd.DataFrame(rows)
    if results.empty:
        return results
    return results.sort_values(["test_function", "algorithm", "repetition"]).reset_index(drop=True)


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
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

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
    parser.add_argument("--no-early-stopping", action="store_true")
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
        early_stopping=not args.no_early_stopping,
        dtype=_dtype_from_name(args.dtype),
        device=args.device,
    )
    run_dir, summary_path = _prepare_run_directory(args.output)
    _save_run_metadata(run_dir, config, summary_path=summary_path)
    if summary_path.exists():
        try:
            existing_results = pd.read_csv(summary_path)
        except EmptyDataError:
            existing_results = None
    else:
        existing_results = None
    results = run_architecture_comparison(config, run_dir=run_dir, existing_results=existing_results)
    path = write_architecture_comparison(results, summary_path)
    print(f"Wrote {len(results)} rows to {path}.")
    print(f"Saved run artifacts to {run_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
