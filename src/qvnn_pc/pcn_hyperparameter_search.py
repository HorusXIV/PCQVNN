"""Batch-stratified Latin-hypercube search for the real-valued predictive coding network."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc
import torch

from qvnn_pc.data import BENCHMARK_FUNCTIONS, make_benchmark_regression_split
from qvnn_pc.models import PredictiveCodingNetwork
from qvnn_pc.models.pcn import ActivationName, LatentInitName
from qvnn_pc.models.shared import OptimizerName
from qvnn_pc.pcn_benchmark import (
    DEFAULT_RESULTS_ROOT,
    PredictiveCodingBenchmarkConfig,
    RealRegressionSplit,
    _canonical_benchmark_name,
    _plot_history,
    _raw_scale_metrics,
    _standardized_real_split,
    _validate_requested_device,
)


@dataclass(frozen=True)
class PredictiveCodingSearchConfig:
    benchmarks: tuple[str, ...] = tuple(BENCHMARK_FUNCTIONS)
    dims: tuple[int, ...] = (4, 20, 20, 20, 20)
    output_dim: int = 1
    epochs: int = 80
    trials: int = 48
    dataset_size: int = 30_000
    sample_seed: int = 0
    split_seed: int = 0
    search_seed: int = 17
    batch_sizes: tuple[int, ...] = (1, 32, 512)
    latent_init: LatentInitName = "feedforward"
    probe_size: int = 128
    normalize_targets: bool = True
    adaptive_inference: bool = True
    inference_tolerance: float = 1e-3
    min_inference_steps: int = 4
    check_finite: bool = False
    device: torch.device | str | None = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    eta_log10_min: float = -5.0
    eta_log10_max: float = -1.5
    T_infer_train_min: int = 8
    T_infer_train_max: int = 48
    T_infer_eval_min: int = 8
    T_infer_eval_max: int = 32
    T_learn_min: int = 1
    T_learn_max: int = 8
    activations: tuple[ActivationName, ...] = ("tanh", "sigmoid", "swish", "leakyrelu")
    optimizers: tuple[OptimizerName, ...] = ("SGD", "Adagrad", "Adadelta")
    search_eval_every: int = 5
    search_train_eval_size: int = 2048
    search_val_eval_size: int = 2048
    early_stopping: bool = True
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4
    max_trial_runtime_seconds: float | None = 1800.0


SCRIPT_CONFIG = PredictiveCodingSearchConfig()


def _timestamped_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _prepare_run_directory(output_path: Path | None) -> tuple[Path, Path]:
    if output_path is not None:
        run_dir = output_path.parent
        summary_path = output_path
    else:
        run_dir = DEFAULT_RESULTS_ROOT / _timestamped_run_name()
        summary_path = run_dir / "pcn_hyperparameter_search.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("trial_history", "models", "evaluations", "plots"):
        (run_dir / child).mkdir(exist_ok=True)
    return run_dir, summary_path


def _save_run_metadata(run_dir: Path, config: PredictiveCodingSearchConfig, *, summary_path: Path) -> None:
    metadata = {
        "benchmarks": list(config.benchmarks),
        "dims": list(config.dims),
        "output_dim": config.output_dim,
        "epochs": config.epochs,
        "trials": config.trials,
        "dataset_size": config.dataset_size,
        "sample_seed": config.sample_seed,
        "split_seed": config.split_seed,
        "search_seed": config.search_seed,
        "batch_sizes": list(config.batch_sizes),
        "latent_init": config.latent_init,
        "probe_size": config.probe_size,
        "normalize_targets": config.normalize_targets,
        "adaptive_inference": config.adaptive_inference,
        "inference_tolerance": config.inference_tolerance,
        "min_inference_steps": config.min_inference_steps,
        "check_finite": config.check_finite,
        "device": None if config.device is None else str(config.device),
        "dtype": str(config.dtype),
        "eta_log10_range": [config.eta_log10_min, config.eta_log10_max],
        "T_infer_train_range": [config.T_infer_train_min, config.T_infer_train_max],
        "T_infer_eval_range": [config.T_infer_eval_min, config.T_infer_eval_max],
        "T_learn_range": [config.T_learn_min, config.T_learn_max],
        "activations": list(config.activations),
        "optimizers": list(config.optimizers),
        "search_eval_every": config.search_eval_every,
        "search_train_eval_size": config.search_train_eval_size,
        "search_val_eval_size": config.search_val_eval_size,
        "early_stopping": config.early_stopping,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "max_trial_runtime_seconds": config.max_trial_runtime_seconds,
        "sampler": "batch-stratified scipy.stats.qmc.LatinHypercube(optimization='random-cd')",
        "trial_order": "round-robin across batch sizes",
        "summary_path": str(summary_path),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _scaled_int(sample_value: float, lower: int, upper: int) -> int:
    span = upper - lower + 1
    idx = min(int(sample_value * span), span - 1)
    return lower + idx


def _validate_batch_sizes(batch_sizes: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(dict.fromkeys(int(batch_size) for batch_size in batch_sizes))
    if not normalized:
        raise ValueError("Expected at least one candidate batch size.")
    if any(batch_size <= 0 for batch_size in normalized):
        raise ValueError(f"Expected all batch sizes > 0, got {normalized}.")
    return normalized


def _sample_trial_grid(
    config: PredictiveCodingSearchConfig,
    *,
    benchmark_idx: int,
) -> list[dict[str, float | int | str]]:
    batch_sizes = _validate_batch_sizes(config.batch_sizes)
    if config.trials <= 0:
        raise ValueError(f"Expected trials > 0, got {config.trials}.")

    base_trials = config.trials // len(batch_sizes)
    remainder = config.trials % len(batch_sizes)
    trials_by_batch: list[list[dict[str, float | int | str]]] = []
    for batch_idx, batch_size in enumerate(batch_sizes):
        batch_trials = base_trials + (1 if batch_idx < remainder else 0)
        if batch_trials <= 0:
            continue

        sampler = qmc.LatinHypercube(
            d=7,
            optimization="random-cd",
            seed=config.search_seed + benchmark_idx * 10_000 + batch_idx,
        )
        unit = sampler.random(batch_trials)

        batch_specs: list[dict[str, float | int | str]] = []
        for row in unit:
            eta_infer = 10.0 ** (config.eta_log10_min + row[0] * (config.eta_log10_max - config.eta_log10_min))
            eta_learn = 10.0 ** (config.eta_log10_min + row[1] * (config.eta_log10_max - config.eta_log10_min))
            optimizer = config.optimizers[min(int(row[5] * len(config.optimizers)), len(config.optimizers) - 1)]
            activation = config.activations[min(int(row[6] * len(config.activations)), len(config.activations) - 1)]
            batch_specs.append(
                {
                    "batch_size": int(batch_size),
                    "eta_infer": float(eta_infer),
                    "eta_learn": float(eta_learn),
                    "T_infer_train": _scaled_int(float(row[2]), config.T_infer_train_min, config.T_infer_train_max),
                    "T_infer_eval": _scaled_int(float(row[3]), config.T_infer_eval_min, config.T_infer_eval_max),
                    "T_learn": _scaled_int(float(row[4]), config.T_learn_min, config.T_learn_max),
                    "optimizer": optimizer,
                    "activation": activation,
                }
            )
        trials_by_batch.append(batch_specs)

    trials: list[dict[str, float | int | str]] = []
    for round_idx in range(max(len(batch_specs) for batch_specs in trials_by_batch)):
        for batch_specs in trials_by_batch:
            if round_idx < len(batch_specs):
                trials.append(batch_specs[round_idx])

    for trial_idx, trial in enumerate(trials, start=1):
        trial["trial"] = trial_idx
    return trials


def _plot_search_results(run_dir: Path, benchmark: str, results: pd.DataFrame) -> None:
    completed = results[results["status"].isin(["completed", "early_stopped"])].copy()
    if completed.empty:
        return

    sorted_results = completed.sort_values("val_mae", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(sorted_results.index + 1, sorted_results["val_mae"], marker="o", linewidth=1.2)
    axes[0].set_title("Validation MAE by Trial Rank")
    axes[0].set_xlabel("Trial Rank")
    axes[0].set_ylabel("Validation MAE")

    scatter = axes[1].scatter(
        completed["eta_learn"],
        completed["val_mae"],
        c=completed["eta_infer"],
        cmap="viridis",
        s=32,
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Search Landscape")
    axes[1].set_xlabel("eta_learn")
    axes[1].set_ylabel("Validation MAE")
    fig.colorbar(scatter, ax=axes[1], label="eta_infer")

    fig.tight_layout()
    fig.savefig(run_dir / "plots" / f"{benchmark}_pcn_search.png", dpi=160)
    plt.close(fig)


def _save_trial_history(run_dir: Path, benchmark: str, trial: int, history: dict) -> Path:
    path = run_dir / "trial_history" / f"{benchmark}_trial{trial:03d}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return path


def _save_progress_summary(run_dir: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).sort_values(["benchmark", "trial"]).to_csv(
        run_dir / "pcn_hyperparameter_search_progress.csv",
        index=False,
    )


def _move_real_split_to_device(
    real_split: RealRegressionSplit,
    device: torch.device | str | None,
) -> RealRegressionSplit:
    return real_split.to(device)


def _screening_subset(x: torch.Tensor, y: torch.Tensor, max_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if max_size <= 0 or x.shape[0] <= max_size:
        return x, y
    return x[:max_size], y[:max_size]


def _search_history_template() -> dict[str, list[float] | list[int] | list[list[float]] | float | int | str]:
    return {
        "eval_epoch": [],
        "train_mse": [],
        "train_mae": [],
        "train_mape": [],
        "train_r2": [],
        "train_free_energy": [],
        "train_reconstruction_mse": [],
        "train_supervised_mse": [],
        "validation_mse": [],
        "validation_mae": [],
        "validation_mape": [],
        "validation_r2": [],
        "validation_free_energy": [],
        "validation_reconstruction_mse": [],
        "validation_supervised_mse": [],
        "probe_train_free_energy_trace": [],
        "probe_train_supervised_mse_trace": [],
        "probe_train_latent_grad_trace": [],
        "probe_val_free_energy_trace": [],
        "probe_val_supervised_mse_trace": [],
        "probe_val_latent_grad_trace": [],
    }


def _metrics_are_finite(metrics: dict[str, float | dict[str, list[float]]]) -> bool:
    return all(math.isfinite(value) for value in metrics.values() if isinstance(value, float))


def _fit_model_for_search(
    model: PredictiveCodingNetwork,
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    benchmark_config: PredictiveCodingBenchmarkConfig,
    search_config: PredictiveCodingSearchConfig,
    shuffle_generator: torch.Generator,
) -> tuple[
    dict[str, list[float] | list[int] | list[list[float]] | float | int | str],
    dict[str, float | dict[str, list[float]]],
    dict[str, float | dict[str, list[float]]],
    dict[str, torch.Tensor],
]:
    history = _search_history_template()
    history["status"] = "completed"
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_train_metrics: dict[str, float | dict[str, list[float]]] = {
        "mse": float("nan"),
        "mae": float("nan"),
        "mape": float("nan"),
        "r2": float("nan"),
    }
    best_val_metrics: dict[str, float | dict[str, list[float]]] = {
        "mse": float("nan"),
        "mae": float("inf"),
        "mape": float("nan"),
        "r2": float("nan"),
    }
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    epochs_ran = 0
    eval_every = max(1, int(search_config.search_eval_every))
    train_eval_x, train_eval_y = _screening_subset(x_train, y_train, search_config.search_train_eval_size)
    val_eval_x, val_eval_y = _screening_subset(x_val, y_val, search_config.search_val_eval_size)

    start = time.perf_counter()
    for epoch_idx in range(benchmark_config.epochs):
        epochs_ran = epoch_idx + 1
        permutation = torch.randperm(x_train.shape[0], generator=shuffle_generator, device="cpu")
        try:
            for batch_start in range(0, x_train.shape[0], benchmark_config.batch_size):
                batch_indices = permutation[batch_start:batch_start + benchmark_config.batch_size]
                batch_indices = batch_indices.to(x_train.device)
                model.train_batch(
                    x_train[batch_indices],
                    y_train[batch_indices],
                    eta_infer=benchmark_config.eta_infer,
                    eta_learn=benchmark_config.eta_learn,
                    T_infer=benchmark_config.T_infer_train,
                    T_learn=benchmark_config.T_learn,
                    latent_init=benchmark_config.latent_init,
                    adaptive=benchmark_config.adaptive_inference,
                    tolerance=benchmark_config.inference_tolerance,
                    min_steps=benchmark_config.min_inference_steps,
                    check_finite=benchmark_config.check_finite,
                )
        except FloatingPointError:
            history["status"] = "non_finite"
            break

        should_evaluate = (
            epoch_idx == 0
            or epochs_ran == benchmark_config.epochs
            or epochs_ran % eval_every == 0
        )
        if not should_evaluate:
            if (
                search_config.max_trial_runtime_seconds is not None
                and time.perf_counter() - start >= search_config.max_trial_runtime_seconds
            ):
                history["status"] = "time_capped"
                break
            continue

        try:
            train_metrics = model.evaluate(
                train_eval_x,
                train_eval_y,
                infer_steps=benchmark_config.T_infer_eval,
                eta_infer=benchmark_config.eta_infer,
                latent_init=benchmark_config.latent_init,
                adaptive=benchmark_config.adaptive_inference,
                tolerance=benchmark_config.inference_tolerance,
                min_steps=benchmark_config.min_inference_steps,
            )
            val_metrics = model.evaluate(
                val_eval_x,
                val_eval_y,
                infer_steps=benchmark_config.T_infer_eval,
                eta_infer=benchmark_config.eta_infer,
                latent_init=benchmark_config.latent_init,
                adaptive=benchmark_config.adaptive_inference,
                tolerance=benchmark_config.inference_tolerance,
                min_steps=benchmark_config.min_inference_steps,
            )
        except FloatingPointError:
            history["status"] = "non_finite"
            break

        if not _metrics_are_finite(train_metrics) or not _metrics_are_finite(val_metrics):
            history["status"] = "non_finite"
            break

        history["eval_epoch"].append(epochs_ran)  # type: ignore[index]
        for key in ("mse", "mae", "mape", "r2", "free_energy", "reconstruction_mse", "supervised_mse"):
            history[f"train_{key}"].append(float(train_metrics[key]))  # type: ignore[index]
            history[f"validation_{key}"].append(float(val_metrics[key]))  # type: ignore[index]

        validation_loss = float(val_metrics["mae"])
        if validation_loss < best_validation_loss - search_config.early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if search_config.early_stopping and epochs_without_improvement >= search_config.early_stopping_patience:
            history["status"] = "early_stopped"
            break
        if (
            search_config.max_trial_runtime_seconds is not None
            and time.perf_counter() - start >= search_config.max_trial_runtime_seconds
        ):
            history["status"] = "time_capped"
            break

    model.load_state_dict(best_state)
    history["runtime"] = time.perf_counter() - start
    history["best_validation_loss"] = best_validation_loss
    history["epochs_ran"] = epochs_ran
    history["evals_ran"] = len(history["validation_mae"])  # type: ignore[arg-type]
    return history, best_train_metrics, best_val_metrics, best_state


def _save_best_artifacts(
    run_dir: Path,
    *,
    benchmark: str,
    best_row: dict[str, float | int | str],
    best_history: dict,
    best_evaluation: dict[str, dict[str, float | dict[str, list[float]]]],
    best_model: PredictiveCodingNetwork,
    base_config: PredictiveCodingSearchConfig,
) -> None:
    with (run_dir / f"{benchmark}_best_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(best_row, handle, indent=2)
    with (run_dir / "evaluations" / f"{benchmark}_best_search_evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(best_evaluation, handle, indent=2)
    with (run_dir / "trial_history" / f"{benchmark}_best_search_history.json").open("w", encoding="utf-8") as handle:
        json.dump(best_history, handle, indent=2)

    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "benchmark": benchmark,
            "algorithm": "PCN-search",
            "dims": list(base_config.dims),
            "output_dim": base_config.output_dim,
            "best_hyperparameters": best_row,
            "history_path": str(run_dir / "trial_history" / f"{benchmark}_best_search_history.json"),
            "evaluation_path": str(run_dir / "evaluations" / f"{benchmark}_best_search_evaluation.json"),
        },
        run_dir / "models" / f"{benchmark}_best_search_model.pt",
    )

    _plot_history(run_dir, f"{benchmark}_best_search", best_history)


def run_pcn_hyperparameter_search(
    config: PredictiveCodingSearchConfig = PredictiveCodingSearchConfig(),
    *,
    run_dir: Path | None = None,
) -> pd.DataFrame:
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        for child in ("trial_history", "models", "evaluations", "plots"):
            (run_dir / child).mkdir(exist_ok=True)

    rows: list[dict[str, float | int | str]] = []

    for benchmark_idx, benchmark_name in enumerate(config.benchmarks):
        benchmark = _canonical_benchmark_name(benchmark_name)
        split = make_benchmark_regression_split(
            benchmark,
            n=config.dataset_size,
            sample_seed=config.sample_seed + benchmark_idx,
            split_seed=config.split_seed + benchmark_idx,
            device=None,
            dtype=config.dtype,
        )
        real_split = _move_real_split_to_device(
            _standardized_real_split(split, normalize_targets=config.normalize_targets),
            config.device,
        )
        x_train, y_train = real_split.x_train, real_split.y_train
        x_val, y_val = real_split.x_val, real_split.y_val
        x_test, y_test = real_split.x_test, real_split.y_test
        trial_specs = _sample_trial_grid(config, benchmark_idx=benchmark_idx)

        best_val_mae = float("inf")
        best_row: dict[str, float | int | str] | None = None
        best_history: dict | None = None
        best_evaluation: dict[str, dict[str, float | dict[str, list[float]]]] | None = None
        best_state: dict[str, torch.Tensor] | None = None

        for trial_spec in trial_specs:
            trial = int(trial_spec["trial"])
            benchmark_config = PredictiveCodingBenchmarkConfig(
                benchmarks=(benchmark,),
                dims=config.dims,
                output_dim=config.output_dim,
                activation=trial_spec["activation"],  # type: ignore[arg-type]
                optimizer=trial_spec["optimizer"],  # type: ignore[arg-type]
                epochs=config.epochs,
                repetitions=1,
                dataset_size=config.dataset_size,
                sample_seed=config.sample_seed + benchmark_idx,
                split_seed=config.split_seed + benchmark_idx,
                batch_size=int(trial_spec["batch_size"]),
                eta_infer=float(trial_spec["eta_infer"]),
                eta_learn=float(trial_spec["eta_learn"]),
                T_infer_train=int(trial_spec["T_infer_train"]),
                T_infer_eval=int(trial_spec["T_infer_eval"]),
                T_learn=int(trial_spec["T_learn"]),
                latent_init=config.latent_init,
                probe_size=config.probe_size,
                normalize_targets=config.normalize_targets,
                adaptive_inference=config.adaptive_inference,
                inference_tolerance=config.inference_tolerance,
                min_inference_steps=config.min_inference_steps,
                check_finite=config.check_finite,
                early_stopping=config.early_stopping,
                device=config.device,
                dtype=config.dtype,
            )

            model = PredictiveCodingNetwork(
                list(config.dims),
                output_dim=config.output_dim,
                activation=benchmark_config.activation,
                optimizer=benchmark_config.optimizer,
                prediction_inference_steps=benchmark_config.T_infer_eval,
                prediction_eta_infer=benchmark_config.eta_infer,
                adaptive_inference=benchmark_config.adaptive_inference,
                inference_tolerance=benchmark_config.inference_tolerance,
                min_inference_steps=benchmark_config.min_inference_steps,
                check_finite=benchmark_config.check_finite,
                device=config.device,
                dtype=config.dtype,
            )
            shuffle_generator = torch.Generator(device="cpu").manual_seed(config.search_seed * 10_000 + trial)
            trial_start = time.perf_counter()
            status = "completed"

            try:
                history, train_metrics, val_metrics, trial_best_state = _fit_model_for_search(
                    model,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    benchmark_config=benchmark_config,
                    search_config=config,
                    shuffle_generator=shuffle_generator,
                )
                status = str(history["status"])
                test_metrics = {"mse": float("nan"), "mae": float("nan"), "mape": float("nan"), "r2": float("nan")}
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                history = {
                    "status": "failed",
                    "runtime": time.perf_counter() - trial_start,
                    "epochs_ran": 0,
                    "error": str(exc),
                }
                status = "failed"
                train_metrics = {"mse": float("nan"), "mae": float("nan"), "mape": float("nan"), "r2": float("nan")}
                val_metrics = {"mse": float("nan"), "mae": float("inf"), "mape": float("nan"), "r2": float("nan")}
                test_metrics = {"mse": float("nan"), "mae": float("nan"), "mape": float("nan"), "r2": float("nan")}

            if run_dir is not None:
                _save_trial_history(run_dir, benchmark, trial, history)

            row = {
                "benchmark": benchmark,
                "trial": trial,
                "activation": str(trial_spec["activation"]),
                "optimizer": str(trial_spec["optimizer"]),
                "batch_size": int(trial_spec["batch_size"]),
                "eta_infer": float(trial_spec["eta_infer"]),
                "eta_learn": float(trial_spec["eta_learn"]),
                "T_infer_train": int(trial_spec["T_infer_train"]),
                "T_infer_eval": int(trial_spec["T_infer_eval"]),
                "T_learn": int(trial_spec["T_learn"]),
                "status": status,
                "epochs": int(history.get("epochs_ran", 0)),
                "runtime_seconds": float(history.get("runtime", time.perf_counter() - trial_start)),
                "train_mae": float(train_metrics["mae"]),
                "val_mae": float(val_metrics["mae"]),
                "test_mae": float(test_metrics["mae"]),
                "train_mse": float(train_metrics["mse"]),
                "val_mse": float(val_metrics["mse"]),
                "test_mse": float(test_metrics["mse"]),
                "val_r2": float(val_metrics["r2"]),
                "test_r2": float(test_metrics["r2"]),
                "screening_train_size": min(config.search_train_eval_size, x_train.shape[0]),
                "screening_val_size": min(config.search_val_eval_size, x_val.shape[0]),
                "evals": int(history.get("evals_ran", 0)),
            }
            rows.append(row)
            if run_dir is not None:
                _save_progress_summary(run_dir, rows)

            if status in {"completed", "early_stopped", "time_capped"} and row["val_mae"] < best_val_mae:
                best_val_mae = float(row["val_mae"])
                best_row = row
                best_history = history
                best_evaluation = None
                best_state = trial_best_state

        if best_row is None or best_history is None or best_state is None:
            raise RuntimeError(f"No successful PCN hyperparameter-search trial completed for benchmark {benchmark!r}.")

        best_model = PredictiveCodingNetwork(
            list(config.dims),
            output_dim=config.output_dim,
            activation=best_row["activation"],  # type: ignore[arg-type]
            optimizer=best_row["optimizer"],  # type: ignore[arg-type]
            prediction_inference_steps=int(best_row["T_infer_eval"]),
            prediction_eta_infer=float(best_row["eta_infer"]),
            adaptive_inference=config.adaptive_inference,
            inference_tolerance=config.inference_tolerance,
            min_inference_steps=config.min_inference_steps,
            check_finite=config.check_finite,
            device=config.device,
            dtype=config.dtype,
        )
        best_model.load_state_dict(best_state)
        full_train_metrics = best_model.evaluate(
            x_train,
            y_train,
            infer_steps=int(best_row["T_infer_eval"]),
            eta_infer=float(best_row["eta_infer"]),
            latent_init=config.latent_init,
            adaptive=config.adaptive_inference,
            tolerance=config.inference_tolerance,
            min_steps=config.min_inference_steps,
        )
        full_val_metrics = best_model.evaluate(
            x_val,
            y_val,
            infer_steps=int(best_row["T_infer_eval"]),
            eta_infer=float(best_row["eta_infer"]),
            latent_init=config.latent_init,
            adaptive=config.adaptive_inference,
            tolerance=config.inference_tolerance,
            min_steps=config.min_inference_steps,
        )
        full_test_metrics = best_model.evaluate(
            x_test,
            y_test,
            infer_steps=int(best_row["T_infer_eval"]),
            eta_infer=float(best_row["eta_infer"]),
            latent_init=config.latent_init,
            adaptive=config.adaptive_inference,
            tolerance=config.inference_tolerance,
            min_steps=config.min_inference_steps,
        )
        best_eval_config = replace(
            PredictiveCodingBenchmarkConfig(),
            T_infer_eval=int(best_row["T_infer_eval"]),
            eta_infer=float(best_row["eta_infer"]),
            latent_init=config.latent_init,
            normalize_targets=config.normalize_targets,
            adaptive_inference=config.adaptive_inference,
            inference_tolerance=config.inference_tolerance,
            min_inference_steps=config.min_inference_steps,
            check_finite=config.check_finite,
        )
        full_train_raw_metrics = _raw_scale_metrics(
            best_model,
            real_split.x_train,
            real_split.y_train_raw,
            split=real_split,
            config=best_eval_config,
        )
        full_val_raw_metrics = _raw_scale_metrics(
            best_model,
            real_split.x_val,
            real_split.y_val_raw,
            split=real_split,
            config=best_eval_config,
        )
        full_test_raw_metrics = _raw_scale_metrics(
            best_model,
            real_split.x_test,
            real_split.y_test_raw,
            split=real_split,
            config=best_eval_config,
        )
        best_evaluation = {
            "train": full_train_metrics,
            "val": full_val_metrics,
            "test": full_test_metrics,
            "train_raw_scale": full_train_raw_metrics,
            "val_raw_scale": full_val_raw_metrics,
            "test_raw_scale": full_test_raw_metrics,
        }
        best_row.update(
            {
                "train_mae": float(full_train_raw_metrics["mae"]),
                "val_mae": float(full_val_raw_metrics["mae"]),
                "test_mae": float(full_test_raw_metrics["mae"]),
                "train_mse": float(full_train_raw_metrics["mse"]),
                "val_mse": float(full_val_raw_metrics["mse"]),
                "test_mse": float(full_test_raw_metrics["mse"]),
                "val_r2": float(full_val_raw_metrics["r2"]),
                "test_r2": float(full_test_raw_metrics["r2"]),
                "train_normalized_mae": float(full_train_metrics["mae"]),
                "val_normalized_mae": float(full_val_metrics["mae"]),
                "test_normalized_mae": float(full_test_metrics["mae"]),
                "full_evaluation": 1,
            }
        )

        benchmark_results = pd.DataFrame([row for row in rows if row["benchmark"] == benchmark])
        if run_dir is not None:
            _plot_search_results(run_dir, benchmark, benchmark_results)

        if run_dir is not None:
            _save_best_artifacts(
                run_dir,
                benchmark=benchmark,
                best_row=best_row,
                best_history=best_history,
                best_evaluation=best_evaluation,
                best_model=best_model,
                base_config=config,
            )

    return pd.DataFrame(rows).sort_values(["benchmark", "trial"]).reset_index(drop=True)


def write_pcn_hyperparameter_search(results: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".xlsx":
        results.to_excel(output, index=False)
    else:
        results.to_csv(output, index=False)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a batch-stratified Latin-hypercube hyperparameter search for the real-valued PCN."
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-trial-runtime-seconds", type=float, default=None)
    parser.add_argument("--no-trial-runtime-cap", action="store_true")
    args = parser.parse_args(argv)

    config = SCRIPT_CONFIG
    if args.benchmark is not None:
        config = replace(config, benchmarks=(_canonical_benchmark_name(args.benchmark),))
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.no_trial_runtime_cap:
        config = replace(config, max_trial_runtime_seconds=None)
    elif args.max_trial_runtime_seconds is not None:
        config = replace(config, max_trial_runtime_seconds=args.max_trial_runtime_seconds)

    _validate_requested_device(config.device)

    run_dir, summary_path = _prepare_run_directory(args.output)
    _save_run_metadata(run_dir, config, summary_path=summary_path)
    results = run_pcn_hyperparameter_search(config, run_dir=run_dir)
    path = write_pcn_hyperparameter_search(results, summary_path)
    print(f"Wrote {len(results)} rows to {path}.")
    print(f"Saved search artifacts to {run_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
