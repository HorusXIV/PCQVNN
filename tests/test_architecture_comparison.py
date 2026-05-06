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

