import pandas as pd
import torch

from qvnn_pc.architecture_comparison import (
    ArchitectureComparisonConfig,
    _load_existing_results,
    run_architecture_comparison,
    write_architecture_comparison,
)


def test_architecture_comparison_smoke_runs_requested_models():
    results = run_architecture_comparison(
        ArchitectureComparisonConfig(
            benchmarks=("ackley",),
            architectures=("cmlp", "ghr"),
            layer_sizes=(1, 2, 1),
            epochs=1,
            repetitions=1,
            dataset_size=32,
            sample_seed=7,
            split_seed=11,
            dtype=torch.float64,
        )
    )

    assert len(results) == 2
    assert set(results["algorithm"]) == {"CMLP", "GHR"}
    assert set(results["test_function"]) == {"ackley"}
    assert {"test_quaternion_mae", "test_mape", "test_r2", "runtime_seconds", "epochs"} <= set(results.columns)
    assert (results["epochs"] >= 1).all()


def test_architecture_comparison_skips_completed_rows_from_existing_results():
    existing = pd.DataFrame(
        [
            {
                "repetition": 1,
                "algorithm": "CMLP",
                "test_function": "ackley",
                "learning_rate": 0.00473375604375846,
                "optimizer": "Adadelta",
                "activation": "sigmoid",
                "batch_size": 1,
                "test_quaternion_mae": 0.1,
                "test_mse": 0.2,
                "test_mae": 0.3,
                "test_mape": 0.4,
                "test_r2": 0.5,
                "epochs": 1,
                "runtime_seconds": 0.6,
            }
        ]
    )

    results = run_architecture_comparison(
        ArchitectureComparisonConfig(
            benchmarks=("ackley",),
            architectures=("cmlp", "ghr"),
            layer_sizes=(1, 2, 1),
            epochs=1,
            repetitions=1,
            dataset_size=32,
            sample_seed=7,
            split_seed=11,
            dtype=torch.float64,
        ),
        existing_results=existing,
    )

    assert len(results) == 2
    assert set(results["algorithm"]) == {"CMLP", "GHR"}
    cmlp_row = results.loc[results["algorithm"] == "CMLP"].iloc[0]
    assert cmlp_row["runtime_seconds"] == 0.6


def test_architecture_comparison_reconstructs_missing_summary_from_saved_artifacts(tmp_path):
    config = ArchitectureComparisonConfig(
        benchmarks=("ackley",),
        architectures=("cmlp",),
        layer_sizes=(1, 2, 1),
        epochs=1,
        repetitions=1,
        dataset_size=32,
        sample_seed=7,
        split_seed=11,
        dtype=torch.float64,
    )
    summary_path = tmp_path / "architecture_comparison_ackley.csv"

    initial = run_architecture_comparison(config, run_dir=tmp_path)
    assert len(initial) == 1
    assert not summary_path.exists()

    recovered = _load_existing_results(summary_path, tmp_path, config)

    assert recovered is not None
    assert len(recovered) == 1
    assert recovered.loc[0, "test_function"] == "ackley"
    assert recovered.loc[0, "algorithm"] == "CMLP"
    assert recovered.loc[0, "epochs"] >= 1
    assert recovered.loc[0, "runtime_seconds"] >= 0.0
    assert summary_path.exists()
    assert (tmp_path / "architecture_comparison.csv").exists()


def test_write_architecture_comparison_merges_benchmark_fragments(tmp_path):
    ackley = pd.DataFrame(
        [
            {
                "repetition": 1,
                "algorithm": "CMLP",
                "test_function": "ackley",
                "learning_rate": 0.1,
                "optimizer": "SGD",
                "activation": "relu",
                "batch_size": 4,
                "test_quaternion_mae": 0.2,
                "test_mse": 0.3,
                "test_mae": 0.4,
                "test_mape": 0.5,
                "test_r2": 0.6,
                "epochs": 3,
                "runtime_seconds": 1.2,
            }
        ]
    )
    griewank = pd.DataFrame(
        [
            {
                "repetition": 1,
                "algorithm": "GHR",
                "test_function": "griewank",
                "learning_rate": 0.2,
                "optimizer": "Adam",
                "activation": "tanh",
                "batch_size": 8,
                "test_quaternion_mae": 0.7,
                "test_mse": 0.8,
                "test_mae": 0.9,
                "test_mape": 1.0,
                "test_r2": 0.1,
                "epochs": 4,
                "runtime_seconds": 2.3,
            }
        ]
    )

    write_architecture_comparison(ackley, tmp_path / "architecture_comparison_ackley.csv")
    write_architecture_comparison(griewank, tmp_path / "architecture_comparison_griewank.csv")

    merged = pd.read_csv(tmp_path / "architecture_comparison.csv")
    assert len(merged) == 2
    assert set(merged["test_function"]) == {"ackley", "griewank"}
