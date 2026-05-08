import torch

from qvnn_pc.models import PredictiveCodingNetwork


def test_pcn_forward_returns_real_regression_shape():
    model = PredictiveCodingNetwork([4, 6, 6], output_dim=1, prediction_inference_steps=1)
    x = torch.randn(5, 4)

    y = model.forward(x, infer_steps=1)

    assert y.shape == (5, 1)


def test_pcn_fast_and_iterative_predictions_have_scalar_shape():
    model = PredictiveCodingNetwork([4, 6, 6], output_dim=1, prediction_inference_steps=2)
    x = torch.randn(7, 4)

    fast = model.fast_predict(x)
    slow = model.predict(x, infer_steps=2)

    assert fast.shape == (7, 1)
    assert slow.shape == (7, 1)


def test_pcn_train_batch_returns_scalar_metrics():
    model = PredictiveCodingNetwork([4, 5, 5], output_dim=1, optimizer="Adagrad", prediction_inference_steps=1)
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)

    metrics = model.train_batch(
        x,
        y,
        eta_infer=0.05,
        eta_learn=0.001,
        T_infer=1,
        T_learn=1,
    )

    assert {"free_energy", "reconstruction_mse", "supervised_mse", "mse", "mae", "r2"} <= set(metrics)
