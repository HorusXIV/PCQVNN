import pandas as pd
import torch

from qvnn_pc.architecture_comparison import ArchitectureComparisonConfig, run_architecture_comparison


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
