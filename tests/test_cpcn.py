import torch

from qvnn_pc.models import SplitQuaternionPredictiveCodingNetwork


def test_split_quaternion_pcn_predicts_quaternion_shape():
    model = SplitQuaternionPredictiveCodingNetwork((1, 3, 1), prediction_inference_steps=1)
    x = torch.randn(4, 1, 4)

    y = model.predict(x, infer_steps=1)

    assert y.shape == (4, 1, 4)


def test_split_quaternion_pcn_train_batch_returns_metrics():
    model = SplitQuaternionPredictiveCodingNetwork((1, 3, 1), prediction_inference_steps=1, check_finite=True)
    x = torch.randn(5, 1, 4)
    y = torch.zeros(5, 1, 4)
    y[:, 0, 0] = 1.0

    metrics = model.train_batch(x, y, eta_infer=0.01, eta_learn=0.001, T_infer=1)

    assert {"mse", "mae", "r2", "quaternion_mae", "free_energy"} <= set(metrics)
    assert all(torch.isfinite(param).all() for param in model.parameters())
