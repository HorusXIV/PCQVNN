import math

import pytest
import torch

from qvnn_pc.models import CMLPQuaternionMLP, quaternion_mae_loss, scalar_regression_metrics
from qvnn_pc.quaternion import q_conj, q_mul, q_norm_sq


DTYPE = torch.float64
ATOL = 1e-9
RTOL = 1e-7


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def q(r, i=0.0, j=0.0, k=0.0):
    return torch.tensor([r, i, j, k], dtype=DTYPE)


def assert_qclose(actual, expected, atol=ATOL, rtol=RTOL):
    actual = actual.detach().to(dtype=DTYPE)
    expected = expected.detach().to(dtype=DTYPE)
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"\nactual:   {actual}\nexpected: {expected}\ndiff:     {actual - expected}"
    )


def manual_q_mul(a, b):
    """
    Manual Hamilton product.

    a ⊗ b =
    (ar br - ai bi - aj bj - ak bk)
    + (ar bi + ai br + aj bk - ak bj)i
    + (ar bj - ai bk + aj br + ak bi)j
    + (ar bk + ai bj - aj bi + ak br)k
    """
    ar, ai, aj, ak = a.unbind(-1)
    br, bi, bj, bk = b.unbind(-1)

    return torch.stack(
        [
            ar * br - ai * bi - aj * bj - ak * bk,
            ar * bi + ai * br + aj * bk - ak * bj,
            ar * bj - ai * bk + aj * br + ak * bi,
            ar * bk + ai * bj - aj * bi + ak * br,
        ],
        dim=-1,
    )


def manual_q_conj(a):
    out = a.clone()
    out[..., 1:] *= -1
    return out


def manual_q_linear(x, W, b):
    """
    x: (B, n_in, 4)
    W: (n_out, n_in, 4)
    b: (n_out, 4)

    Returns:
        y: (B, n_out, 4)

    y[b, n] = sum_m W[n, m] ⊗ x[b, m] + b[n]
    """
    B = x.shape[0]
    n_out, n_in, _ = W.shape
    y = torch.zeros((B, n_out, 4), dtype=x.dtype)

    for batch in range(B):
        for n in range(n_out):
            acc = b[n].clone()
            for m in range(n_in):
                acc = acc + manual_q_mul(W[n, m], x[batch, m])
            y[batch, n] = acc

    return y


def split_sigmoid(x):
    return torch.sigmoid(x)


def split_sigmoid_deriv(preact):
    s = torch.sigmoid(preact)
    return s * (1.0 - s)


def split_swish(x):
    return x * torch.sigmoid(x)


def split_swish_deriv(preact):
    sig = torch.sigmoid(preact)
    return sig + preact * sig * (1.0 - sig)


def _make_model(layer_sizes, activation="sigmoid"):
    """
    Adapter for possible constructor variants.
    Adjust this helper if your model constructor is different.
    """
    candidates = [
        lambda: CMLPQuaternionMLP(layer_sizes=layer_sizes, activation=activation),
        lambda: CMLPQuaternionMLP(layer_sizes=layer_sizes, hidden_activation=activation),
        lambda: CMLPQuaternionMLP(sizes=layer_sizes, activation=activation),
        lambda: CMLPQuaternionMLP(architecture=layer_sizes, activation=activation),
        lambda: CMLPQuaternionMLP(layer_sizes, activation=activation),
        lambda: CMLPQuaternionMLP(layer_sizes, hidden_activation=activation),
        lambda: CMLPQuaternionMLP(layer_sizes),
    ]

    last_error = None
    for make in candidates:
        try:
            model = make()
            if hasattr(model, "to"):
                model = model.to(dtype=DTYPE)
            return model
        except TypeError as e:
            last_error = e

    raise TypeError(
        "Could not construct CMLPQuaternionMLP. "
        "Adjust _make_model() to match your constructor."
    ) from last_error


def _get_weights_biases(model):
    assert hasattr(model, "weights"), "Model must expose model.weights"
    assert hasattr(model, "biases"), "Model must expose model.biases"
    return model.weights, model.biases


def _set_params(model, weights, biases):
    model_weights, model_biases = _get_weights_biases(model)

    assert len(model_weights) == len(weights)
    assert len(model_biases) == len(biases)

    with torch.no_grad():
        for target, source in zip(model_weights, weights):
            assert target.shape == source.shape, f"{target.shape=} != {source.shape=}"
            target.copy_(source.to(dtype=target.dtype, device=target.device))

        for target, source in zip(model_biases, biases):
            assert target.shape == source.shape, f"{target.shape=} != {source.shape=}"
            target.copy_(source.to(dtype=target.dtype, device=target.device))


def _forward(model, x):
    """
    Supports either:
      - forward_with_cache returns (activations, preacts)
      - forward returns (activations, preacts)
      - forward returns prediction only
    """
    if hasattr(model, "forward_with_cache"):
        activations, preacts = model.forward_with_cache(x)
        prediction = activations[-1]
        return prediction, activations, preacts

    out = model(x)

    if isinstance(out, tuple) and len(out) == 2:
        activations, preacts = out
        prediction = activations[-1]
        return prediction, activations, preacts

    prediction = out
    return prediction, None, None


def _loss_real(y_pred, y_true):
    return torch.mean((y_true[..., 0] - y_pred[..., 0]) ** 2)


# ---------------------------------------------------------------------
# Quaternion operation tests
# ---------------------------------------------------------------------

def test_manual_hamilton_product_matches_q_mul():
    a = q(1.0, 2.0, -3.0, 0.5)
    b = q(-0.25, 4.0, 1.5, -2.0)

    expected = manual_q_mul(a, b)
    actual = q_mul(a, b)

    assert_qclose(actual, expected)


def test_basis_multiplication_rules():
    one = q(1.0, 0.0, 0.0, 0.0)
    i = q(0.0, 1.0, 0.0, 0.0)
    j = q(0.0, 0.0, 1.0, 0.0)
    k = q(0.0, 0.0, 0.0, 1.0)
    minus_one = q(-1.0, 0.0, 0.0, 0.0)

    assert_qclose(q_mul(i, i), minus_one)
    assert_qclose(q_mul(j, j), minus_one)
    assert_qclose(q_mul(k, k), minus_one)

    assert_qclose(q_mul(i, j), k)
    assert_qclose(q_mul(j, k), i)
    assert_qclose(q_mul(k, i), j)

    assert_qclose(q_mul(j, i), -k)
    assert_qclose(q_mul(k, j), -i)
    assert_qclose(q_mul(i, k), -j)

    assert_qclose(q_mul(one, i), i)
    assert_qclose(q_mul(i, one), i)


def test_conjugate_norm_identity():
    a = q(1.5, -2.0, 3.0, 0.25)

    prod = q_mul(a, q_conj(a))
    norm_sq = q_norm_sq(a)

    expected = torch.tensor([norm_sq.item(), 0.0, 0.0, 0.0], dtype=DTYPE)
    assert_qclose(prod, expected)


# ---------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------

def test_single_layer_forward_matches_manual_hamilton_linear():
    """
    Catches:
    - wrong Hamilton product order
    - incorrect bias broadcasting
    - accidental activation on output layer
    """
    model = _make_model([1, 1], activation="sigmoid")

    W0 = torch.tensor([[[0.5, -1.0, 2.0, 0.25]]], dtype=DTYPE)
    b0 = torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=DTYPE)
    _set_params(model, [W0], [b0])

    x = torch.tensor([[[1.0, 2.0, -1.0, 0.5]]], dtype=DTYPE)

    expected = manual_q_linear(x, W0, b0)
    actual, _, _ = _forward(model, x)

    assert actual.shape == (1, 1, 4)
    assert_qclose(actual, expected)


def test_two_layer_forward_matches_manual_computation_with_hidden_sigmoid():
    """
    Architecture: 1 -> 1 -> 1

    Hidden layer must use split sigmoid.
    Output layer must be identity.
    """
    model = _make_model([1, 1, 1], activation="sigmoid")

    W0 = torch.tensor([[[0.5, -0.25, 0.75, 0.1]]], dtype=DTYPE)
    b0 = torch.tensor([[0.05, -0.1, 0.2, 0.0]], dtype=DTYPE)

    W1 = torch.tensor([[[-0.3, 0.6, 0.2, -0.4]]], dtype=DTYPE)
    b1 = torch.tensor([[0.01, 0.02, -0.03, 0.04]], dtype=DTYPE)

    _set_params(model, [W0, W1], [b0, b1])

    x = torch.tensor([[[1.0, -2.0, 0.5, 0.25]]], dtype=DTYPE)

    s1 = manual_q_linear(x, W0, b0)
    a1 = split_sigmoid(s1)
    expected = manual_q_linear(a1, W1, b1)

    actual, activations, preacts = _forward(model, x)

    assert_qclose(actual, expected)

    if activations is not None and preacts is not None:
        assert_qclose(preacts[0], s1)
        assert_qclose(activations[1], a1)


def test_output_layer_is_identity_not_sigmoid():
    """
    This catches a very common reason for constant Ackley predictions:
    applying sigmoid/swish to the output layer in a scalar regression task.
    """
    model = _make_model([1, 1], activation="sigmoid")

    W0 = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]], dtype=DTYPE)
    b0 = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    _set_params(model, [W0], [b0])

    x = torch.tensor([[[3.0, 0.0, 0.0, 0.0]]], dtype=DTYPE)

    y_pred, _, _ = _forward(model, x)

    # Identity output should be exactly 6 in the real component.
    # Sigmoid output would be ~0.9975.
    assert_qclose(y_pred, torch.tensor([[[6.0, 0.0, 0.0, 0.0]]], dtype=DTYPE))


# ---------------------------------------------------------------------
# Manual CMLP backward tests
# ---------------------------------------------------------------------

def test_cmlp_backward_deltas_match_manual_computation_sigmoid():
    """
    Architecture: 1 -> 1 -> 1

    Checks the exact CMLP hidden delta:

        delta_hidden =
            sigmoid'(s_hidden) ⊙ (conj(W_out) ⊗ delta_out)

    This catches:
    - missing conjugate
    - wrong product order
    - wrong derivative
    - wrong output delta sign
    """
    model = _make_model([1, 1, 1], activation="sigmoid")

    W0 = torch.tensor([[[0.2, -0.4, 0.3, 0.1]]], dtype=DTYPE)
    b0 = torch.tensor([[0.01, -0.02, 0.03, -0.04]], dtype=DTYPE)

    W1 = torch.tensor([[[0.7, -0.2, 0.5, -0.1]]], dtype=DTYPE)
    b1 = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=DTYPE)

    _set_params(model, [W0, W1], [b0, b1])

    x = torch.tensor([[[1.0, 0.5, -1.0, 2.0]]], dtype=DTYPE)
    y = torch.tensor([[[1.5, 0.0, 0.0, 0.0]]], dtype=DTYPE)

    y_pred, activations, preacts = _forward(model, x)

    assert activations is not None and preacts is not None, (
        "For this test, model.forward must return (activations, preacts)."
    )
    assert hasattr(model, "backward"), "Model must implement manual backward()."

    deltas = model.backward(activations, preacts, y)

    delta_out_expected = y - y_pred

    hidden_back_signal = manual_q_mul(
        manual_q_conj(W1[0, 0]),
        delta_out_expected[0, 0],
    ).reshape(1, 1, 4)

    hidden_delta_expected = split_sigmoid_deriv(preacts[0]) * hidden_back_signal

    assert_qclose(deltas[-1], delta_out_expected)
    assert_qclose(deltas[0], hidden_delta_expected)


def test_cmlp_update_directions_match_manual_computation():
    """
    Checks:

        dW[l][n,m] = mean_b delta[l][b,n] ⊗ conj(x[l-1][b,m])
        db[l][n]   = mean_b delta[l][b,n]

    This catches:
    - wrong outer product order
    - missing conjugate of previous activation
    - summing instead of averaging over batch
    """
    model = _make_model([1, 1, 1], activation="sigmoid")

    W0 = torch.tensor([[[0.2, -0.4, 0.3, 0.1]]], dtype=DTYPE)
    b0 = torch.tensor([[0.01, -0.02, 0.03, -0.04]], dtype=DTYPE)

    W1 = torch.tensor([[[0.7, -0.2, 0.5, -0.1]]], dtype=DTYPE)
    b1 = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=DTYPE)

    _set_params(model, [W0, W1], [b0, b1])

    x = torch.tensor([[[1.0, 0.5, -1.0, 2.0]]], dtype=DTYPE)
    y = torch.tensor([[[1.5, 0.0, 0.0, 0.0]]], dtype=DTYPE)

    y_pred, activations, preacts = _forward(model, x)

    assert hasattr(model, "backward"), "Model must implement manual backward()."
    assert hasattr(model, "compute_update_directions"), (
        "Model must implement compute_update_directions()."
    )

    deltas = model.backward(activations, preacts, y)
    dWs, dbs = model.compute_update_directions(activations, deltas)

    # Layer 0 expected update
    dW0_expected = manual_q_mul(
        deltas[0][0, 0],
        manual_q_conj(activations[0][0, 0]),
    ).reshape(1, 1, 4)
    db0_expected = deltas[0].mean(dim=0)

    # Layer 1 expected update
    dW1_expected = manual_q_mul(
        deltas[1][0, 0],
        manual_q_conj(activations[1][0, 0]),
    ).reshape(1, 1, 4)
    db1_expected = deltas[1].mean(dim=0)

    assert_qclose(dWs[0], dW0_expected)
    assert_qclose(dbs[0], db0_expected)

    assert_qclose(dWs[1], dW1_expected)
    assert_qclose(dbs[1], db1_expected)


def test_cmlp_update_direction_reduces_simple_one_layer_loss():
    """
    This is a sign test.

    With:
        x = 1
        y = 1
        W = 0
        b = 0

    Prediction is 0. The CMLP direction should increase W and b,
    reducing MSE after a small manual update.

    If this fails, your update sign is probably wrong.
    """
    model = _make_model([1, 1], activation="sigmoid")

    W0 = torch.zeros((1, 1, 4), dtype=DTYPE)
    b0 = torch.zeros((1, 4), dtype=DTYPE)
    _set_params(model, [W0], [b0])

    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=DTYPE)
    y = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=DTYPE)

    pred_before, activations, preacts = _forward(model, x)
    loss_before = _loss_real(pred_before, y)

    deltas = model.backward(activations, preacts, y)
    dWs, dbs = model.compute_update_directions(activations, deltas)

    model_weights, model_biases = _get_weights_biases(model)

    with torch.no_grad():
        model_weights[0].add_(0.1 * dWs[0])
        model_biases[0].add_(0.1 * dbs[0])

    pred_after, _, _ = _forward(model, x)
    loss_after = _loss_real(pred_after, y)

    assert loss_after < loss_before, (
        f"Expected loss to decrease after one small CMLP update. "
        f"Before={loss_before.item()}, after={loss_after.item()}"
    )


# ---------------------------------------------------------------------
# Batch averaging tests
# ---------------------------------------------------------------------

def test_update_direction_averages_over_batch_not_sum():
    """
    Uses a 1-layer model and two samples.

    Expected:
        dW = mean_b delta_b ⊗ conj(x_b)

    If implementation sums over the batch instead of averaging,
    this test fails by roughly a factor of B.
    """
    model = _make_model([1, 1], activation="sigmoid")

    W0 = torch.zeros((1, 1, 4), dtype=DTYPE)
    b0 = torch.zeros((1, 4), dtype=DTYPE)
    _set_params(model, [W0], [b0])

    x = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0, 0.0]],
        ],
        dtype=DTYPE,
    )
    y = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0, 0.0]],
        ],
        dtype=DTYPE,
    )

    y_pred, activations, preacts = _forward(model, x)
    deltas = model.backward(activations, preacts, y)
    dWs, dbs = model.compute_update_directions(activations, deltas)

    expected_terms = []
    for batch in range(2):
        expected_terms.append(
            manual_q_mul(deltas[0][batch, 0], manual_q_conj(x[batch, 0]))
        )
    dW_expected = torch.stack(expected_terms, dim=0).mean(dim=0).reshape(1, 1, 4)
    db_expected = deltas[0].mean(dim=0)

    assert_qclose(dWs[0], dW_expected)
    assert_qclose(dbs[0], db_expected)


# ---------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------

def test_scalar_regression_metrics_match_manual_values():
    y_true = torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0, 0.0]]], dtype=DTYPE)
    y_pred = torch.tensor([[[1.0, 9.0, 8.0, 7.0]], [[1.0, 6.0, 5.0, 4.0]], [[5.0, 3.0, 2.0, 1.0]]], dtype=DTYPE)

    metrics = scalar_regression_metrics(y_pred, y_true)

    y_true_real = y_true[..., 0]
    y_pred_real = y_pred[..., 0]

    mse_expected = torch.mean((y_true_real - y_pred_real) ** 2).item()
    mae_expected = torch.mean(torch.abs(y_true_real - y_pred_real)).item()
    mape_expected = torch.mean(torch.abs(y_true_real - y_pred_real) / torch.abs(y_true_real)).item()
    r2_expected = (
        1.0
        - torch.sum((y_true_real - y_pred_real) ** 2)
        / torch.sum((y_true_real - torch.mean(y_true_real)) ** 2)
    ).item()

    assert math.isclose(metrics["mse"], mse_expected, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(metrics["mae"], mae_expected, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(metrics["mape"], mape_expected, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(metrics["r2"], r2_expected, rel_tol=1e-9, abs_tol=1e-9)


def test_metrics_use_real_scalar_output_only_for_function_approximation():
    """
    Function targets are encoded as f(x)+0i+0j+0k.

    This test ensures the benchmark metric is not accidentally computed
    over all quaternion components.
    """
    y_true_quat = torch.tensor(
        [
            [[10.0, 0.0, 0.0, 0.0]],
            [[20.0, 0.0, 0.0, 0.0]],
        ],
        dtype=DTYPE,
    )

    # Imaginary components are wrong on purpose.
    # Scalar regression metrics should only use real component.
    y_pred_quat = torch.tensor(
        [
            [[11.0, 999.0, -999.0, 500.0]],
            [[18.0, -500.0, 700.0, -300.0]],
        ],
        dtype=DTYPE,
    )

    y_true = y_true_quat[..., 0].reshape(-1)
    y_pred = y_pred_quat[..., 0].reshape(-1)

    metrics = scalar_regression_metrics(y_pred_quat, y_true_quat)

    mse_expected = torch.mean((y_true - y_pred) ** 2).item()
    assert math.isclose(metrics["mse"], mse_expected, rel_tol=1e-9, abs_tol=1e-9)


def test_quaternion_mae_loss_matches_mean_quaternion_norm():
    y_true = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, -2.0, 2.0]],
        ],
        dtype=DTYPE,
    )
    y_pred = torch.tensor(
        [
            [[4.0, 0.0, 0.0, 0.0]],
            [[0.0, -2.0, 2.0, -1.0]],
        ],
        dtype=DTYPE,
    )

    expected = torch.linalg.vector_norm(y_pred - y_true, dim=-1).mean()

    torch.testing.assert_close(quaternion_mae_loss(y_pred, y_true), expected)


def test_adadelta_accumulators_persist_across_multiple_train_steps():
    model = _make_model([1, 1], activation="sigmoid")

    W0 = torch.zeros((1, 1, 4), dtype=DTYPE)
    b0 = torch.zeros((1, 4), dtype=DTYPE)
    _set_params(model, [W0], [b0])

    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=DTYPE)
    y = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=DTYPE)

    weight_acc_before = model._weight_acc_0
    bias_acc_before = model._bias_acc_0

    model.train_step(x, y)
    weight_acc_after_first = model._weight_acc_0.clone()
    bias_acc_after_first = model._bias_acc_0.clone()

    model.train_step(x, y)

    assert model._weight_acc_0 is weight_acc_before
    assert model._bias_acc_0 is bias_acc_before
    assert torch.all(model._weight_acc_0 >= 0.9 * weight_acc_after_first)
    assert torch.all(model._bias_acc_0 >= 0.9 * bias_acc_after_first)


def test_preactivation_diagnostic_reports_expected_statistics(capsys):
    model = _make_model([1, 1, 1], activation="sigmoid")
    x = torch.tensor([[[0.25, -0.5, 0.75, -1.0]]], dtype=DTYPE)

    diagnostics = model.print_preactivation_diagnostics(x)
    captured = capsys.readouterr()

    assert len(diagnostics) == 2
    assert "mean_abs=" in captured.out
    assert "max_abs=" in captured.out
    assert "std=" in captured.out


def test_source_style_initialization_uses_bounded_quaternion_radius():
    model = CMLPQuaternionMLP(layer_sizes=(1, 5), generator=torch.Generator().manual_seed(0))
    sigma = 1.0 / math.sqrt(2.0 * (1 + 5))
    radii = torch.linalg.vector_norm(model.weights[0].detach(), dim=-1)

    assert radii.shape == (5, 1)
    assert torch.all(radii <= sigma + 1e-12)
    torch.testing.assert_close(model.biases[0], torch.zeros_like(model.biases[0]))


def test_fit_online_source_style_updates_one_sample_at_a_time_and_tracks_checkpoint_metric():
    model = CMLPQuaternionMLP(layer_sizes=(1, 1), activation="sigmoid", source_style_init=False, init_scale=0.0)
    with torch.no_grad():
        model.weights[0].zero_()
        model.biases[0].zero_()

    x_train = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0, 0.0]],
        ],
        dtype=DTYPE,
    )
    y_train = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0, 0.0]],
        ],
        dtype=DTYPE,
    )
    x_val = x_train[:1]
    y_val = y_train[:1]

    history = model.fit_online_source_style(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=2,
        shuffle_generator=torch.Generator().manual_seed(0),
    )

    assert len(history["quaternion_mae"]) == 2
    assert len(history["validation_quaternion_mae"]) == 2
    assert len(history["mape"]) == 2
    assert len(history["r2"]) == 2
    assert history["runtime"] >= 0.0
    assert torch.any(model._weight_acc_0 > 0.0)
    assert torch.any(model._bias_acc_0 > 0.0)
