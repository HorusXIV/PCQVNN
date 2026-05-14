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


def test_pcn_readout_bias_learns_constant_target():
    model = PredictiveCodingNetwork([4, 5], output_dim=1, prediction_inference_steps=1)
    x = torch.zeros(8, 4)
    y = torch.full((8, 1), 3.0)

    initial_bias = model.readout_bias.detach().clone()
    model.train_batch(
        x,
        y,
        eta_infer=0.01,
        eta_learn=0.1,
        T_infer=1,
        T_learn=1,
    )

    assert torch.all(model.readout_bias > initial_bias)


def test_pcn_evaluate_does_not_clamp_target_during_prediction():
    model = PredictiveCodingNetwork([1, 1], output_dim=1, prediction_inference_steps=1)
    with torch.no_grad():
        model.layers[0].W.zero_()
        model.layers[0].bias.zero_()
        model.readout_weight.fill_(1.0)
        model.readout_bias.zero_()

    metrics = model.evaluate(
        torch.zeros(1, 1),
        torch.full((1, 1), 10.0),
        infer_steps=1,
        eta_infer=0.1,
    )

    assert metrics["mse"] == 100.0


def test_pcn_target_clamped_inference_stays_finite_with_large_readout():
    model = PredictiveCodingNetwork([1, 2], output_dim=1, prediction_inference_steps=30)
    with torch.no_grad():
        model.readout_weight.fill_(25.0)

    result = model.infer_latents(
        torch.zeros(1, 1),
        y=torch.ones(1, 1),
        steps=30,
        eta_infer=0.05,
    )

    assert torch.isfinite(result["prediction"]).all()


def test_pcn_repeated_single_sample_readout_updates_stay_finite():
    model = PredictiveCodingNetwork([4, 20, 20], output_dim=1, activation="leakyrelu", prediction_inference_steps=5)
    x = torch.randn(1, 4)
    y = torch.ones(1, 1)

    for _ in range(20):
        model.train_batch(
            x,
            y,
            eta_infer=0.05,
            eta_learn=0.005,
            T_infer=5,
            T_learn=32,
            latent_init="random",
        )

    assert all(torch.isfinite(param).all() for param in model.parameters())
