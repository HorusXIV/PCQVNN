import torch

from qvnn_pc.pcn_hyperparameter_search import PredictiveCodingSearchConfig, run_pcn_hyperparameter_search


def test_pcn_hyperparameter_search_smoke_runs_requested_function():
    results = run_pcn_hyperparameter_search(
        PredictiveCodingSearchConfig(
            benchmarks=("ackley",),
            dims=(4, 4, 4),
            epochs=1,
            trials=4,
            dataset_size=32,
            batch_sizes=(1, 8),
            probe_size=8,
            device="cpu",
            dtype=torch.float64,
            T_infer_train_min=1,
            T_infer_train_max=2,
            T_infer_eval_min=1,
            T_infer_eval_max=2,
            T_learn_min=1,
            T_learn_max=2,
        )
    )

    assert len(results) == 4
    assert set(results["benchmark"]) == {"ackley"}
    assert set(results["batch_size"]) == {1, 8}
    assert {"activation", "optimizer", "batch_size", "val_mae", "status"} <= set(results.columns)
