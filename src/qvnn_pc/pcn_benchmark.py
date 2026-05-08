"""Benchmark runner for the real-valued predictive coding network."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import time
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from qvnn_pc.config import default_hyperparameters_path
from qvnn_pc.data import BENCHMARK_FUNCTIONS, make_benchmark_regression_split
from qvnn_pc.models import PredictiveCodingNetwork
from qvnn_pc.models.pcn import ActivationName, LatentInitName
from qvnn_pc.models.shared import OptimizerName
from qvnn_pc.models.shared import pq_alpha_should_stop as _pq_alpha_should_stop

DEFAULT_BENCHMARKS: tuple[str, ...] = tuple(BENCHMARK_FUNCTIONS)
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_HYPERPARAMETERS_PATH = default_hyperparameters_path()
ALGORITHM_LABEL = "PCN"


@dataclass(frozen=True)
class PredictiveCodingBenchmarkConfig:
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS
    dims: tuple[int, ...] = (4, 20, 20, 20, 20)
    output_dim: int = 1
    activation: ActivationName = "tanh"
    optimizer: OptimizerName = "SGD"
    epochs: int = 250
    repetitions: int = 1
    dataset_size: int = 30_000
    sample_seed: int = 0
    split_seed: int = 0
    batch_size: int = 512
    eta_infer: float = 0.01
    eta_learn: float = 1e-4
    T_infer_train: int = 24
    T_infer_eval: int = 32
    T_learn: int = 1
    latent_init: LatentInitName = "feedforward"
    probe_size: int = 128
    early_stopping: bool = False
    device: torch.device | str | None = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


SCRIPT_CONFIG = PredictiveCodingBenchmarkConfig()


def _canonical_benchmark_name(name: str) -> str:
    key = name.lower()
    if key not in BENCHMARK_FUNCTIONS:
        valid = ", ".join(sorted(BENCHMARK_FUNCTIONS))
        raise ValueError(f"Unknown benchmark {name!r}. Expected one of: {valid}.")
    return key


def _validate_requested_device(device: torch.device | str | None) -> None:
    if device is None:
        return
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "CUDA was explicitly requested for the PCN benchmark, but PyTorch could not initialize CUDA. "
        "This usually means the container's PyTorch/CUDA build is incompatible with the cluster driver. "
        "Rebuild the container with a compatible pinned torch wheel and rerun the job."
    )


def _timestamped_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _prepare_run_directory(output_path: Path | None) -> tuple[Path, Path]:
    if output_path is not None:
        run_dir = output_path.parent
        summary_path = output_path
    else:
        run_dir = DEFAULT_RESULTS_ROOT / _timestamped_run_name()
        summary_path = run_dir / "pcn_benchmark.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("models", "history", "evaluations", "plots"):
        (run_dir / child).mkdir(exist_ok=True)
    return run_dir, summary_path


def _save_run_metadata(
    run_dir: Path,
    config: PredictiveCodingBenchmarkConfig,
    *,
    summary_path: Path,
) -> None:
    metadata = {
        "benchmarks": list(config.benchmarks),
        "dims": list(config.dims),
        "output_dim": config.output_dim,
        "activation": config.activation,
        "optimizer": config.optimizer,
        "epochs": config.epochs,
        "repetitions": config.repetitions,
        "dataset_size": config.dataset_size,
        "sample_seed": config.sample_seed,
        "split_seed": config.split_seed,
        "batch_size": config.batch_size,
        "eta_infer": config.eta_infer,
        "eta_learn": config.eta_learn,
        "T_infer_train": config.T_infer_train,
        "T_infer_eval": config.T_infer_eval,
        "T_learn": config.T_learn,
        "latent_init": config.latent_init,
        "probe_size": config.probe_size,
        "early_stopping": config.early_stopping,
        "summary_path": str(summary_path),
        "device": None if config.device is None else str(config.device),
        "dtype": str(config.dtype),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if DEFAULT_HYPERPARAMETERS_PATH.exists():
        shutil.copy2(DEFAULT_HYPERPARAMETERS_PATH, run_dir / "hyperparameters.yml")


def _standardized_real_split(split):
    return (
        split.x_train[:, 0, :],
        split.y_train_scalar.unsqueeze(-1),
        split.x_val[:, 0, :],
        split.y_val_scalar.unsqueeze(-1),
        split.x_test[:, 0, :],
        split.y_test_scalar.unsqueeze(-1),
    )


def _history_template() -> dict[str, list[float] | list[list[float]] | float | int]:
    return {
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


def _metric_dict_is_finite(metrics: dict[str, float]) -> bool:
    return all(math.isfinite(value) for value in metrics.values())


def _record_probe_trace(
    history: dict[str, list[float] | list[list[float]] | float | int],
    *,
    prefix: str,
    metrics: dict[str, float | dict[str, list[float]]],
) -> None:
    trace = metrics.get("inference_trace")
    if not isinstance(trace, dict):
        return
    history[f"{prefix}_free_energy_trace"].append(list(trace["free_energy"]))  # type: ignore[index]
    history[f"{prefix}_supervised_mse_trace"].append(list(trace["supervised_mse"]))  # type: ignore[index]
    history[f"{prefix}_latent_grad_trace"].append(list(trace["mean_latent_grad_norm"]))  # type: ignore[index]


def _plot_history(run_dir: Path, stem: str, history: dict[str, list[float] | list[list[float]] | float | int]) -> None:
    epochs = range(1, len(history["train_mae"]) + 1)  # type: ignore[arg-type]
    if not list(epochs):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_mae"], label="train MAE")  # type: ignore[arg-type]
    axes[0].plot(epochs, history["validation_mae"], label="val MAE")  # type: ignore[arg-type]
    axes[0].set_title("Prediction Error")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    axes[0].legend()

    axes[1].plot(epochs, history["train_free_energy"], label="train free energy")  # type: ignore[arg-type]
    axes[1].plot(epochs, history["validation_free_energy"], label="val free energy")  # type: ignore[arg-type]
    axes[1].set_title("PC Objective")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Energy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(run_dir / "plots" / f"{stem}_training.png", dpi=160)
    plt.close(fig)

    probe_val = history["probe_val_free_energy_trace"]  # type: ignore[assignment]
    if probe_val:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        last_free = probe_val[-1]
        last_sup = history["probe_val_supervised_mse_trace"][-1]  # type: ignore[index]
        axes[0].plot(range(len(last_free)), last_free)
        axes[0].set_title("Final Val Probe Free-Energy Trace")
        axes[0].set_xlabel("Inference Step")
        axes[0].set_ylabel("Energy")
        axes[1].plot(range(len(last_sup)), last_sup)
        axes[1].set_title("Final Val Probe Supervised MSE Trace")
        axes[1].set_xlabel("Inference Step")
        axes[1].set_ylabel("MSE")
        fig.tight_layout()
        fig.savefig(run_dir / "plots" / f"{stem}_inference.png", dpi=160)
        plt.close(fig)


def _save_training_artifacts(
    run_dir: Path,
    *,
    benchmark: str,
    repetition: int,
    model: PredictiveCodingNetwork,
    history: dict[str, list[float] | list[list[float]] | float | int],
    evaluation: dict[str, dict[str, float | dict[str, list[float]]]],
    config: PredictiveCodingBenchmarkConfig,
) -> None:
    stem = f"{benchmark}_pcn_rep{repetition:02d}"
    history_path = run_dir / "history" / f"{stem}.json"
    model_path = run_dir / "models" / f"{stem}.pt"
    evaluation_path = run_dir / "evaluations" / f"{stem}.json"

    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    with evaluation_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "benchmark": benchmark,
            "algorithm": ALGORITHM_LABEL,
            "repetition": repetition,
            "dims": list(config.dims),
            "output_dim": config.output_dim,
            "activation": config.activation,
            "optimizer": config.optimizer,
            "eta_infer": config.eta_infer,
            "eta_learn": config.eta_learn,
            "T_infer_train": config.T_infer_train,
            "T_infer_eval": config.T_infer_eval,
            "T_learn": config.T_learn,
            "batch_size": config.batch_size,
            "latent_init": config.latent_init,
            "history_path": str(history_path),
            "evaluation_path": str(evaluation_path),
        },
        model_path,
    )

    _plot_history(run_dir, stem, history)


def _fit_model(
    model: PredictiveCodingNetwork,
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    config: PredictiveCodingBenchmarkConfig,
    shuffle_generator: torch.Generator,
) -> dict[str, list[float] | list[list[float]] | float | int | str]:
    history = _history_template()
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_validation_loss = float("inf")
    probe_train_x = x_train[:config.probe_size]
    probe_train_y = y_train[:config.probe_size]
    probe_val_x = x_val[:config.probe_size]
    probe_val_y = y_val[:config.probe_size]

    start = time.perf_counter()
    history["status"] = "completed"
    for _ in range(config.epochs):
        permutation = torch.randperm(x_train.shape[0], generator=shuffle_generator, device="cpu")
        try:
            for batch_start in range(0, x_train.shape[0], config.batch_size):
                batch_indices = permutation[batch_start:batch_start + config.batch_size]
                model.train_batch(
                    x_train[batch_indices],
                    y_train[batch_indices],
                    eta_infer=config.eta_infer,
                    eta_learn=config.eta_learn,
                    T_infer=config.T_infer_train,
                    T_learn=config.T_learn,
                    latent_init=config.latent_init,
                )

            train_metrics = model.evaluate(
                x_train,
                y_train,
                infer_steps=config.T_infer_eval,
                eta_infer=config.eta_infer,
                latent_init=config.latent_init,
            )
            val_metrics = model.evaluate(
                x_val,
                y_val,
                infer_steps=config.T_infer_eval,
                eta_infer=config.eta_infer,
                latent_init=config.latent_init,
            )
        except FloatingPointError:
            history["status"] = "non_finite"
            break
        if not _metric_dict_is_finite(cast(dict[str, float], train_metrics)) or not _metric_dict_is_finite(
            cast(dict[str, float], val_metrics)
        ):
            history["status"] = "non_finite"
            break

        for key in ("mse", "mae", "mape", "r2", "free_energy", "reconstruction_mse", "supervised_mse"):
            history[f"train_{key}"].append(float(train_metrics[key]))  # type: ignore[index]
            history[f"validation_{key}"].append(float(val_metrics[key]))  # type: ignore[index]

        probe_train_trace = cast(
            dict[str, list[float]],
            model.infer_latents(
                probe_train_x,
                y=probe_train_y,
                steps=config.T_infer_eval,
                eta_infer=config.eta_infer,
                latent_init=config.latent_init,
                return_trace=True,
            )["trace"],
        )
        probe_val_trace = cast(
            dict[str, list[float]],
            model.infer_latents(
                probe_val_x,
                y=probe_val_y,
                steps=config.T_infer_eval,
                eta_infer=config.eta_infer,
                latent_init=config.latent_init,
                return_trace=True,
            )["trace"],
        )
        _record_probe_trace(history, prefix="probe_train", metrics={"inference_trace": probe_train_trace})
        _record_probe_trace(history, prefix="probe_val", metrics={"inference_trace": probe_val_trace})

        validation_loss = float(val_metrics["mae"])
        best_validation_so_far = best_validation_loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        if config.early_stopping and _pq_alpha_should_stop(
            cast(list[float], history["train_mae"]),
            validation_loss,
            best_validation_so_far,
            window=5,
            threshold=0.5,
        ):
            history["status"] = "early_stopped"
            break

    model.load_state_dict(best_state)
    history["runtime"] = time.perf_counter() - start
    history["best_validation_loss"] = best_validation_loss
    history["epochs_ran"] = len(history["train_mae"])  # type: ignore[arg-type]
    return history


def run_pcn_benchmark(
    config: PredictiveCodingBenchmarkConfig = PredictiveCodingBenchmarkConfig(),
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
            device=None,
            dtype=config.dtype,
        )
        x_train, y_train, x_val, y_val, x_test, y_test = _standardized_real_split(split)

        for repetition in range(1, config.repetitions + 1):
            model = PredictiveCodingNetwork(
                list(config.dims),
                output_dim=config.output_dim,
                activation=config.activation,
                optimizer=config.optimizer,
                prediction_inference_steps=config.T_infer_eval,
                prediction_eta_infer=config.eta_infer,
                device=config.device,
                dtype=config.dtype,
            )
            shuffle_generator = torch.Generator(device="cpu").manual_seed(int(repetition * 1e6))
            history = _fit_model(
                model,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                config=config,
                shuffle_generator=shuffle_generator,
            )

            train_metrics = model.evaluate(
                x_train,
                y_train,
                infer_steps=config.T_infer_eval,
                eta_infer=config.eta_infer,
                latent_init=config.latent_init,
            )
            val_metrics = model.evaluate(
                x_val,
                y_val,
                infer_steps=config.T_infer_eval,
                eta_infer=config.eta_infer,
                latent_init=config.latent_init,
            )
            test_metrics = model.evaluate(
                x_test,
                y_test,
                infer_steps=config.T_infer_eval,
                eta_infer=config.eta_infer,
                latent_init=config.latent_init,
                return_trace=True,
            )

            evaluation = {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            }
            if run_dir is not None:
                _save_training_artifacts(
                    run_dir,
                    benchmark=benchmark,
                    repetition=repetition,
                    model=model,
                    history=history,
                    evaluation=evaluation,
                    config=config,
                )

            rows.append(
                {
                    "repetition": repetition,
                    "algorithm": ALGORITHM_LABEL,
                    "test_function": benchmark,
                    "activation": config.activation,
                    "optimizer": config.optimizer,
                    "layer_dims": "-".join(str(dim) for dim in config.dims),
                    "batch_size": config.batch_size,
                    "eta_infer": config.eta_infer,
                    "eta_learn": config.eta_learn,
                    "T_infer_train": config.T_infer_train,
                    "T_infer_eval": config.T_infer_eval,
                    "T_learn": config.T_learn,
                    "latent_init": config.latent_init,
                    "test_mse": float(test_metrics["mse"]),
                    "test_mae": float(test_metrics["mae"]),
                    "test_mape": float(test_metrics["mape"]),
                    "test_r2": float(test_metrics["r2"]),
                    "test_free_energy": float(test_metrics["free_energy"]),
                    "test_reconstruction_mse": float(test_metrics["reconstruction_mse"]),
                    "test_supervised_mse": float(test_metrics["supervised_mse"]),
                    "status": str(history["status"]),
                    "epochs": int(history["epochs_ran"]),
                    "runtime_seconds": float(history["runtime"]),
                }
            )

    return pd.DataFrame(rows).sort_values(["test_function", "repetition"]).reset_index(drop=True)


def write_pcn_benchmark(results: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".xlsx":
        results.to_excel(output, index=False)
    else:
        results.to_csv(output, index=False)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real-valued predictive coding regression benchmarks.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    config = SCRIPT_CONFIG
    if args.benchmark is not None:
        config = replace(config, benchmarks=(_canonical_benchmark_name(args.benchmark),))
    if args.device is not None:
        config = replace(config, device=args.device)

    _validate_requested_device(config.device)

    run_dir, summary_path = _prepare_run_directory(args.output)
    _save_run_metadata(run_dir, config, summary_path=summary_path)
    results = run_pcn_benchmark(config, run_dir=run_dir)
    path = write_pcn_benchmark(results, summary_path)
    print(f"Wrote {len(results)} rows to {path}.")
    print(f"Saved run artifacts to {run_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
