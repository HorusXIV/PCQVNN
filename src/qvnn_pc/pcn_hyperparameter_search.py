"""Latin-hypercube hyperparameter search for the real-valued predictive coding network."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
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
    _canonical_benchmark_name,
    _fit_model,
    _plot_history,
    _standardized_real_split,
    _validate_requested_device,
)


@dataclass(frozen=True)
class PredictiveCodingSearchConfig:
    benchmarks: tuple[str, ...] = tuple(BENCHMARK_FUNCTIONS)
    dims: tuple[int, ...] = (4, 20, 20, 20, 20)
    output_dim: int = 1
    epochs: int = 30
    trials: int = 24
    dataset_size: int = 30_000
    sample_seed: int = 0
    split_seed: int = 0
    search_seed: int = 17
    batch_size: int = 512
    latent_init: LatentInitName = "feedforward"
    probe_size: int = 128
    device: torch.device | str | None = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    eta_log10_min: float = -7.0
    eta_log10_max: float = -2.0
    T_infer_train_min: int = 4
    T_infer_train_max: int = 64
    T_infer_eval_min: int = 4
    T_infer_eval_max: int = 64
    T_learn_min: int = 1
    T_learn_max: int = 12
    activations: tuple[ActivationName, ...] = ("tanh", "sigmoid", "swish", "leakyrelu")
    optimizers: tuple[OptimizerName, ...] = ("SGD", "Adagrad", "Adadelta")


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
        "batch_size": config.batch_size,
        "latent_init": config.latent_init,
        "probe_size": config.probe_size,
        "device": None if config.device is None else str(config.device),
        "dtype": str(config.dtype),
        "eta_log10_range": [config.eta_log10_min, config.eta_log10_max],
        "T_infer_train_range": [config.T_infer_train_min, config.T_infer_train_max],
        "T_infer_eval_range": [config.T_infer_eval_min, config.T_infer_eval_max],
        "T_learn_range": [config.T_learn_min, config.T_learn_max],
        "activations": list(config.activations),
        "optimizers": list(config.optimizers),
        "sampler": "scipy.stats.qmc.LatinHypercube(optimization='random-cd')",
        "summary_path": str(summary_path),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _scaled_int(sample_value: float, lower: int, upper: int) -> int:
    span = upper - lower + 1
    idx = min(int(sample_value * span), span - 1)
    return lower + idx


def _sample_trial_grid(
    config: PredictiveCodingSearchConfig,
    *,
    benchmark_idx: int,
) -> list[dict[str, float | int | str]]:
    sampler = qmc.LatinHypercube(d=7, optimization="random-cd", seed=config.search_seed + benchmark_idx)
    unit = sampler.random(config.trials)

    trials: list[dict[str, float | int | str]] = []
    for trial_idx, row in enumerate(unit, start=1):
        eta_infer = 10.0 ** (config.eta_log10_min + row[0] * (config.eta_log10_max - config.eta_log10_min))
        eta_learn = 10.0 ** (config.eta_log10_min + row[1] * (config.eta_log10_max - config.eta_log10_min))
        optimizer = config.optimizers[min(int(row[5] * len(config.optimizers)), len(config.optimizers) - 1)]
        activation = config.activations[min(int(row[6] * len(config.activations)), len(config.activations) - 1)]
        trials.append(
            {
                "trial": trial_idx,
                "eta_infer": float(eta_infer),
                "eta_learn": float(eta_learn),
                "T_infer_train": _scaled_int(float(row[2]), config.T_infer_train_min, config.T_infer_train_max),
                "T_infer_eval": _scaled_int(float(row[3]), config.T_infer_eval_min, config.T_infer_eval_max),
                "T_learn": _scaled_int(float(row[4]), config.T_learn_min, config.T_learn_max),
                "optimizer": optimizer,
                "activation": activation,
            }
        )
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
                batch_size=config.batch_size,
                eta_infer=float(trial_spec["eta_infer"]),
                eta_learn=float(trial_spec["eta_learn"]),
                T_infer_train=int(trial_spec["T_infer_train"]),
                T_infer_eval=int(trial_spec["T_infer_eval"]),
                T_learn=int(trial_spec["T_learn"]),
                latent_init=config.latent_init,
                probe_size=config.probe_size,
                early_stopping=False,
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
                device=config.device,
                dtype=config.dtype,
            )
            shuffle_generator = torch.Generator(device="cpu").manual_seed(config.search_seed * 10_000 + trial)
            trial_start = time.perf_counter()
            status = "completed"

            try:
                history = _fit_model(
                    model,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    config=benchmark_config,
                    shuffle_generator=shuffle_generator,
                )
                status = str(history["status"])
                train_metrics = model.evaluate(
                    x_train,
                    y_train,
                    infer_steps=benchmark_config.T_infer_eval,
                    eta_infer=benchmark_config.eta_infer,
                    latent_init=benchmark_config.latent_init,
                )
                val_metrics = model.evaluate(
                    x_val,
                    y_val,
                    infer_steps=benchmark_config.T_infer_eval,
                    eta_infer=benchmark_config.eta_infer,
                    latent_init=benchmark_config.latent_init,
                )
                test_metrics = model.evaluate(
                    x_test,
                    y_test,
                    infer_steps=benchmark_config.T_infer_eval,
                    eta_infer=benchmark_config.eta_infer,
                    latent_init=benchmark_config.latent_init,
                )
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
            }
            rows.append(row)

            if status in {"completed", "early_stopped"} and row["val_mae"] < best_val_mae:
                best_val_mae = float(row["val_mae"])
                best_row = row
                best_history = history
                best_evaluation = {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics,
                }
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        benchmark_results = pd.DataFrame([row for row in rows if row["benchmark"] == benchmark])
        if run_dir is not None:
            _plot_search_results(run_dir, benchmark, benchmark_results)

        if best_row is None or best_history is None or best_evaluation is None or best_state is None:
            raise RuntimeError(f"No successful PCN hyperparameter-search trial completed for benchmark {benchmark!r}.")

        best_model = PredictiveCodingNetwork(
            list(config.dims),
            output_dim=config.output_dim,
            activation=best_row["activation"],  # type: ignore[arg-type]
            optimizer=best_row["optimizer"],  # type: ignore[arg-type]
            prediction_inference_steps=int(best_row["T_infer_eval"]),
            prediction_eta_infer=float(best_row["eta_infer"]),
            device=config.device,
            dtype=config.dtype,
        )
        best_model.load_state_dict(best_state)
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
    parser = argparse.ArgumentParser(description="Run a Latin-hypercube hyperparameter search for the real-valued PCN.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args(argv)

    config = SCRIPT_CONFIG
    if args.benchmark is not None:
        config = replace(config, benchmarks=(_canonical_benchmark_name(args.benchmark),))
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.trials is not None:
        config = replace(config, trials=args.trials)

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
