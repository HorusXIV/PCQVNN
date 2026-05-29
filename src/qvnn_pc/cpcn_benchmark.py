"""Benchmark runner for the split-quaternion predictive coding network."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from typing import cast

import pandas as pd
import torch

from qvnn_pc.config import default_hyperparameters_path
from qvnn_pc.data import BENCHMARK_FUNCTIONS, make_benchmark_regression_split
from qvnn_pc.models import SplitQuaternionPredictiveCodingNetwork
from qvnn_pc.models.cpcn import ActivationName, LatentInitName
from qvnn_pc.models.pcn import scalar_regression_metrics as real_scalar_regression_metrics
from qvnn_pc.models.shared import OptimizerName, quaternion_mae_loss

DEFAULT_BENCHMARKS: tuple[str, ...] = tuple(BENCHMARK_FUNCTIONS)
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_HYPERPARAMETERS_PATH = default_hyperparameters_path()
ALGORITHM_LABEL = "CPCN"


@dataclass(frozen=True)
class SplitQuaternionPCNBenchmarkConfig:
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS
    layer_sizes: tuple[int, ...] = (1, 5, 5, 5, 1)
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
    T_infer_train: int = 16
    T_infer_eval: int = 24
    T_learn: int = 1
    latent_init: LatentInitName = "feedforward"
    normalize_targets: bool = True
    adaptive_inference: bool = True
    inference_tolerance: float = 1e-3
    min_inference_steps: int = 4
    check_finite: bool = False
    evaluation_interval: int = 5
    device: torch.device | str | None = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


SCRIPT_CONFIG = SplitQuaternionPCNBenchmarkConfig()


@dataclass(frozen=True)
class QuaternionPCNRegressionSplit:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    y_train_raw_scalar: torch.Tensor
    y_val_raw_scalar: torch.Tensor
    y_test_raw_scalar: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    def to(self, device: torch.device | str | None) -> "QuaternionPCNRegressionSplit":
        if device is None:
            return self
        target_device = torch.device(device)
        return QuaternionPCNRegressionSplit(
            x_train=self.x_train.to(target_device),
            y_train=self.y_train.to(target_device),
            x_val=self.x_val.to(target_device),
            y_val=self.y_val.to(target_device),
            x_test=self.x_test.to(target_device),
            y_test=self.y_test.to(target_device),
            y_train_raw_scalar=self.y_train_raw_scalar.to(target_device),
            y_val_raw_scalar=self.y_val_raw_scalar.to(target_device),
            y_test_raw_scalar=self.y_test_raw_scalar.to(target_device),
            y_mean=self.y_mean.to(target_device),
            y_std=self.y_std.to(target_device),
        )


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
    raise RuntimeError("CUDA was requested, but PyTorch could not initialize CUDA.")


def _timestamped_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _prepare_run_directory(output_path: Path | None) -> tuple[Path, Path]:
    if output_path is not None:
        run_dir = output_path.parent
        summary_path = output_path
    else:
        run_dir = DEFAULT_RESULTS_ROOT / _timestamped_run_name()
        summary_path = run_dir / "cpcn_benchmark.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("models", "history", "evaluations"):
        (run_dir / child).mkdir(exist_ok=True)
    return run_dir, summary_path


def _save_run_metadata(
    run_dir: Path,
    config: SplitQuaternionPCNBenchmarkConfig,
    *,
    summary_path: Path,
) -> None:
    metadata = {
        "benchmarks": list(config.benchmarks),
        "layer_sizes": list(config.layer_sizes),
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
        "normalize_targets": config.normalize_targets,
        "adaptive_inference": config.adaptive_inference,
        "inference_tolerance": config.inference_tolerance,
        "min_inference_steps": config.min_inference_steps,
        "check_finite": config.check_finite,
        "evaluation_interval": config.evaluation_interval,
        "summary_path": str(summary_path),
        "device": None if config.device is None else str(config.device),
        "dtype": str(config.dtype),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if DEFAULT_HYPERPARAMETERS_PATH.exists():
        shutil.copy2(DEFAULT_HYPERPARAMETERS_PATH, run_dir / "hyperparameters.yml")


def _normalized_quaternion_target(y_scalar: torch.Tensor) -> torch.Tensor:
    y = torch.zeros(y_scalar.shape[0], 1, 4, dtype=y_scalar.dtype, device=y_scalar.device)
    y[:, 0, 0] = y_scalar
    return y


def _quaternion_split(split, *, normalize_targets: bool = True) -> QuaternionPCNRegressionSplit:
    if normalize_targets:
        y_train = _normalized_quaternion_target(split.y_train_scalar_normalized)
        y_val = _normalized_quaternion_target(split.y_val_scalar_normalized)
        y_test = _normalized_quaternion_target(split.y_test_scalar_normalized)
    else:
        y_train = split.y_train
        y_val = split.y_val
        y_test = split.y_test

    return QuaternionPCNRegressionSplit(
        x_train=split.x_train,
        y_train=y_train,
        x_val=split.x_val,
        y_val=y_val,
        x_test=split.x_test,
        y_test=y_test,
        y_train_raw_scalar=split.y_train_scalar.unsqueeze(-1),
        y_val_raw_scalar=split.y_val_scalar.unsqueeze(-1),
        y_test_raw_scalar=split.y_test_scalar.unsqueeze(-1),
        y_mean=split.y_mean.reshape(1, 1),
        y_std=split.y_std.reshape(1, 1),
    )


def _raw_scalar_metrics(
    model: SplitQuaternionPredictiveCodingNetwork,
    x: torch.Tensor,
    y_raw_scalar: torch.Tensor,
    *,
    split: QuaternionPCNRegressionSplit,
    config: SplitQuaternionPCNBenchmarkConfig,
) -> dict[str, float]:
    y_pred = model.predict(
        x,
        infer_steps=config.T_infer_eval,
        eta_infer=config.eta_infer,
        latent_init=config.latent_init,
    )
    pred_scalar = y_pred[..., 0]
    if config.normalize_targets:
        pred_scalar = pred_scalar * split.y_std + split.y_mean
    return real_scalar_regression_metrics(pred_scalar, y_raw_scalar)


def _evaluate_normalized(
    model: SplitQuaternionPredictiveCodingNetwork,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    config: SplitQuaternionPCNBenchmarkConfig,
) -> dict[str, float]:
    return model.evaluate(
        x,
        y,
        infer_steps=config.T_infer_eval,
        eta_infer=config.eta_infer,
        latent_init=config.latent_init,
    )


def _history_template() -> dict[str, list[float] | float | int | str]:
    return {
        "train_mse": [],
        "train_mae": [],
        "train_r2": [],
        "validation_mse": [],
        "validation_mae": [],
        "validation_r2": [],
        "train_raw_mae": [],
        "validation_raw_mae": [],
        "train_free_energy": [],
        "validation_free_energy": [],
        "status": "completed",
    }


def _fit_model(
    model: SplitQuaternionPredictiveCodingNetwork,
    *,
    split: QuaternionPCNRegressionSplit,
    config: SplitQuaternionPCNBenchmarkConfig,
    shuffle_generator: torch.Generator,
) -> dict[str, list[float] | float | int | str]:
    history = _history_template()
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_validation_loss = float("inf")
    eval_every = max(1, int(config.evaluation_interval))
    epochs_ran = 0
    start = time.perf_counter()

    for epoch_idx in range(config.epochs):
        epochs_ran = epoch_idx + 1
        permutation = torch.randperm(split.x_train.shape[0], generator=shuffle_generator, device="cpu")
        try:
            for batch_start in range(0, split.x_train.shape[0], config.batch_size):
                batch_indices = permutation[batch_start:batch_start + config.batch_size].to(split.x_train.device)
                model.train_batch(
                    split.x_train[batch_indices],
                    split.y_train[batch_indices],
                    eta_infer=config.eta_infer,
                    eta_learn=config.eta_learn,
                    T_infer=config.T_infer_train,
                    T_learn=config.T_learn,
                    latent_init=config.latent_init,
                )

            should_evaluate = epoch_idx == 0 or epochs_ran == config.epochs or epochs_ran % eval_every == 0
            if not should_evaluate:
                continue

            train_metrics = _evaluate_normalized(model, split.x_train, split.y_train, config=config)
            val_metrics = _evaluate_normalized(model, split.x_val, split.y_val, config=config)
            train_raw_metrics = _raw_scalar_metrics(
                model,
                split.x_train,
                split.y_train_raw_scalar,
                split=split,
                config=config,
            )
            val_raw_metrics = _raw_scalar_metrics(
                model,
                split.x_val,
                split.y_val_raw_scalar,
                split=split,
                config=config,
            )
        except FloatingPointError:
            history["status"] = "non_finite"
            break

        for prefix, metrics in (("train", train_metrics), ("validation", val_metrics)):
            history[f"{prefix}_mse"].append(float(metrics["mse"]))  # type: ignore[index]
            history[f"{prefix}_mae"].append(float(metrics["mae"]))  # type: ignore[index]
            history[f"{prefix}_r2"].append(float(metrics["r2"]))  # type: ignore[index]
            history[f"{prefix}_free_energy"].append(float(metrics["free_energy"]))  # type: ignore[index]
        history["train_raw_mae"].append(float(train_raw_metrics["mae"]))  # type: ignore[index]
        history["validation_raw_mae"].append(float(val_raw_metrics["mae"]))  # type: ignore[index]

        validation_loss = float(val_raw_metrics["mae"])
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    model.load_state_dict(best_state)
    history["runtime"] = time.perf_counter() - start
    history["best_validation_loss"] = best_validation_loss
    history["epochs_ran"] = epochs_ran
    history["evals_ran"] = len(cast(list[float], history["validation_raw_mae"]))
    return history


def _save_training_artifacts(
    run_dir: Path,
    *,
    benchmark: str,
    repetition: int,
    model: SplitQuaternionPredictiveCodingNetwork,
    history: dict[str, list[float] | float | int | str],
    evaluation: dict[str, dict[str, float]],
    config: SplitQuaternionPCNBenchmarkConfig,
) -> None:
    stem = f"{benchmark}_cpcn_rep{repetition:02d}"
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
            "layer_sizes": list(config.layer_sizes),
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


def run_cpcn_benchmark(
    config: SplitQuaternionPCNBenchmarkConfig = SplitQuaternionPCNBenchmarkConfig(),
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
        quat_split = _quaternion_split(split, normalize_targets=config.normalize_targets).to(config.device)

        for repetition in range(1, config.repetitions + 1):
            generator = torch.Generator(device="cpu").manual_seed(int(repetition * 1e6))
            model = SplitQuaternionPredictiveCodingNetwork(
                layer_sizes=config.layer_sizes,
                activation=config.activation,
                optimizer=config.optimizer,
                prediction_inference_steps=config.T_infer_eval,
                prediction_eta_infer=config.eta_infer,
                adaptive_inference=config.adaptive_inference,
                inference_tolerance=config.inference_tolerance,
                min_inference_steps=config.min_inference_steps,
                check_finite=config.check_finite,
                generator=generator,
                device=config.device,
                dtype=config.dtype,
            )
            shuffle_generator = torch.Generator(device="cpu").manual_seed(int(repetition * 1e6) + 17)
            history = _fit_model(model, split=quat_split, config=config, shuffle_generator=shuffle_generator)

            train_metrics = _evaluate_normalized(model, quat_split.x_train, quat_split.y_train, config=config)
            val_metrics = _evaluate_normalized(model, quat_split.x_val, quat_split.y_val, config=config)
            test_metrics = _evaluate_normalized(model, quat_split.x_test, quat_split.y_test, config=config)
            train_raw_metrics = _raw_scalar_metrics(
                model,
                quat_split.x_train,
                quat_split.y_train_raw_scalar,
                split=quat_split,
                config=config,
            )
            val_raw_metrics = _raw_scalar_metrics(
                model,
                quat_split.x_val,
                quat_split.y_val_raw_scalar,
                split=quat_split,
                config=config,
            )
            test_raw_metrics = _raw_scalar_metrics(
                model,
                quat_split.x_test,
                quat_split.y_test_raw_scalar,
                split=quat_split,
                config=config,
            )

            evaluation = {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "train_raw_scale": train_raw_metrics,
                "val_raw_scale": val_raw_metrics,
                "test_raw_scale": test_raw_metrics,
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

            test_pred = model.predict(
                quat_split.x_test,
                infer_steps=config.T_infer_eval,
                eta_infer=config.eta_infer,
                latent_init=config.latent_init,
            )
            rows.append(
                {
                    "repetition": repetition,
                    "algorithm": ALGORITHM_LABEL,
                    "test_function": benchmark,
                    "activation": config.activation,
                    "optimizer": config.optimizer,
                    "layer_sizes": "-".join(str(size) for size in config.layer_sizes),
                    "batch_size": config.batch_size,
                    "eta_infer": config.eta_infer,
                    "eta_learn": config.eta_learn,
                    "T_infer_train": config.T_infer_train,
                    "T_infer_eval": config.T_infer_eval,
                    "T_learn": config.T_learn,
                    "latent_init": config.latent_init,
                    "normalize_targets": int(config.normalize_targets),
                    "adaptive_inference": int(config.adaptive_inference),
                    "test_mse": float(test_raw_metrics["mse"]),
                    "test_mae": float(test_raw_metrics["mae"]),
                    "test_mape": float(test_raw_metrics["mape"]),
                    "test_r2": float(test_raw_metrics["r2"]),
                    "test_normalized_mse": float(test_metrics["mse"]),
                    "test_normalized_mae": float(test_metrics["mae"]),
                    "test_quaternion_mae": float(quaternion_mae_loss(test_pred, quat_split.y_test).detach().cpu()),
                    "test_free_energy": float(test_metrics["free_energy"]),
                    "status": str(history["status"]),
                    "epochs": int(history["epochs_ran"]),
                    "runtime_seconds": float(history["runtime"]),
                    "evals": int(history["evals_ran"]),
                }
            )

    return pd.DataFrame(rows).sort_values(["test_function", "repetition"]).reset_index(drop=True)


def write_cpcn_benchmark(results: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".xlsx":
        results.to_excel(output, index=False)
    else:
        results.to_csv(output, index=False)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run split-quaternion predictive coding regression benchmarks.")
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
    results = run_cpcn_benchmark(config, run_dir=run_dir)
    path = write_cpcn_benchmark(results, summary_path)
    print(f"Wrote {len(results)} rows to {path}.")
    print(f"Saved run artifacts to {run_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
