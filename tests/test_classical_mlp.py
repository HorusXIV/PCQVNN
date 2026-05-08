import torch

from qvnn_pc.models import ClassicalMLP


def test_classical_mlp_forward_returns_scalar_shape():
    model = ClassicalMLP(layer_sizes=(4, 20, 20, 20, 20, 1), activation="tanh")
    x = torch.randn(5, 4)

    y = model.forward(x)

    assert y.shape == (5, 1)


def test_classical_mlp_train_step_and_fit_smoke_run():
    model = ClassicalMLP(layer_sizes=(4, 20, 20, 20, 20, 1), activation="tanh", optimizer="Adam", lr=1e-3)
    x = torch.randn(16, 4)
    y = torch.randn(16, 1)

    step_metrics = model.train_step(x, y)
    history = model.fit(x, y, epochs=2, batch_size=8, early_stopping=False)

    assert {"mse", "mae", "mape", "r2"} <= set(step_metrics)
    assert history["epochs_ran"] == 2
    assert history["runtime"] >= 0.0