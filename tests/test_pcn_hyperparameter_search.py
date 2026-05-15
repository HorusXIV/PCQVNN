from pathlib import Path

import torch

from qvnn_pc.pcn_hyperparameter_search import (
    PredictiveCodingSearchConfig,
    _sample_trial_grid,
    run_pcn_hyperparameter_search,
)


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
            max_trial_runtime_seconds=None,
        )
    )

    assert len(results) == 4
    assert set(results["benchmark"]) == {"ackley"}
    assert set(results["batch_size"]) == {1, 8}
    assert {"activation", "optimizer", "batch_size", "val_mae", "status"} <= set(results.columns)


def test_pcn_hyperparameter_search_interleaves_batch_sizes():
    specs = _sample_trial_grid(
        PredictiveCodingSearchConfig(
            trials=6,
            batch_sizes=(1, 32, 512),
        ),
        benchmark_idx=0,
    )

    assert [spec["batch_size"] for spec in specs] == [1, 32, 512, 1, 32, 512]
    assert [spec["trial"] for spec in specs] == [1, 2, 3, 4, 5, 6]


def test_pcn_hyperparameter_search_writes_progress_summary(tmp_path: Path):
    run_dir = tmp_path / "pcn-search"

    run_pcn_hyperparameter_search(
        PredictiveCodingSearchConfig(
            benchmarks=("ackley",),
            dims=(4, 4, 4),
            epochs=1,
            trials=2,
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
            max_trial_runtime_seconds=None,
        ),
        run_dir=run_dir,
    )

    assert (run_dir / "pcn_hyperparameter_search_progress.csv").exists()
