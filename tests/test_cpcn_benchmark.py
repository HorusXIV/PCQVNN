import torch

from qvnn_pc.cpcn_benchmark import SplitQuaternionPCNBenchmarkConfig, run_cpcn_benchmark


def test_cpcn_benchmark_smoke_runs_requested_function():
    results = run_cpcn_benchmark(
        SplitQuaternionPCNBenchmarkConfig(
            benchmarks=("ackley",),
            layer_sizes=(1, 3, 1),
            epochs=1,
            repetitions=1,
            dataset_size=32,
            batch_size=8,
            eta_infer=0.01,
            eta_learn=0.001,
            T_infer_train=1,
            T_infer_eval=1,
            T_learn=1,
            evaluation_interval=1,
            device="cpu",
            dtype=torch.float64,
        )
    )

    assert len(results) == 1
    row = results.iloc[0]
    assert row["algorithm"] == "CPCN"
    assert row["test_function"] == "ackley"
    assert row["epochs"] >= 1
    assert torch.isfinite(torch.tensor(row["test_mae"]))
