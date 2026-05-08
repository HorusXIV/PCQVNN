import torch

from qvnn_pc.pcn_benchmark import PredictiveCodingBenchmarkConfig, run_pcn_benchmark


def test_pcn_benchmark_smoke_runs_requested_function():
    results = run_pcn_benchmark(
        PredictiveCodingBenchmarkConfig(
            benchmarks=("ackley",),
            dims=(4, 4, 4),
            epochs=1,
            repetitions=1,
            dataset_size=32,
            batch_size=8,
            eta_infer=0.02,
            eta_learn=0.001,
            T_infer_train=1,
            T_infer_eval=1,
            T_learn=1,
            probe_size=8,
            early_stopping=False,
            device="cpu",
            dtype=torch.float64,
        )
    )

    assert len(results) == 1
    row = results.iloc[0]
    assert row["algorithm"] == "PCN"
    assert row["test_function"] == "ackley"
    assert row["epochs"] >= 1
