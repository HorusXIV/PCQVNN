import math

import torch

from qvnn_pc.activations import split_sigmoid, split_sigmoid_derivative
from qvnn_pc.models import CMLPQuaternionMLP, GHRQuaternionMLP
from qvnn_pc.quaternion import q_conj, q_mul


DTYPE = torch.float64
ATOL = 1e-9
RTOL = 1e-7


def assert_qclose(actual, expected, atol=ATOL, rtol=RTOL):
    actual = actual.detach().to(dtype=DTYPE)
    expected = expected.detach().to(dtype=DTYPE)
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"\nactual:   {actual}\nexpected: {expected}\ndiff:     {actual - expected}"
    )


def manual_q_mul(a, b):
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


def q_involution(q, axis):
    if axis == "i":
        signs = q.new_tensor((1.0, 1.0, -1.0, -1.0))
    elif axis == "j":
        signs = q.new_tensor((1.0, -1.0, 1.0, -1.0))
    elif axis == "k":
        signs = q.new_tensor((1.0, -1.0, -1.0, 1.0))
    else:
        raise ValueError(axis)
    return q * signs


def set_params(model, weights, biases):
    with torch.no_grad():
        for target, source in zip(model.weights, weights):
            target.copy_(source.to(dtype=target.dtype, device=target.device))
        for target, source in zip(model.biases, biases):
            target.copy_(source.to(dtype=target.dtype, device=target.device))


def test_ghr_single_layer_forward_matches_manual_hamilton_linear():
    model = GHRQuaternionMLP(layer_sizes=(1, 1), activation="sigmoid", source_style_init=False, init_scale=0.0).to(
        dtype=DTYPE
    )
    weight = torch.tensor([[[0.5, -1.0, 2.0, 0.25]]], dtype=DTYPE)
    bias = torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=DTYPE)
    set_params(model, [weight], [bias])

    x = torch.tensor([[[1.0, 2.0, -1.0, 0.5]]], dtype=DTYPE)
    expected = manual_q_mul(weight[0, 0], x[0, 0]).reshape(1, 1, 4) + bias.reshape(1, 1, 4)
    actual = model(x)

    assert_qclose(actual, expected)


def test_ghr_hidden_delta_matches_manual_source_style_rule():
    model = GHRQuaternionMLP(layer_sizes=(1, 1, 1), activation="sigmoid", optimizer="SGD", source_style_init=False, init_scale=0.0).to(
        dtype=DTYPE
    )

    w0 = torch.tensor([[[0.2, -0.4, 0.3, 0.1]]], dtype=DTYPE)
    b0 = torch.tensor([[0.01, -0.02, 0.03, -0.04]], dtype=DTYPE)
    w1 = torch.tensor([[[0.7, -0.2, 0.5, -0.1]]], dtype=DTYPE)
    b1 = torch.zeros((1, 4), dtype=DTYPE)
    set_params(model, [w0, w1], [b0, b1])

    x = torch.tensor([[[1.0, 0.5, -1.0, 2.0]]], dtype=DTYPE)
    y = torch.tensor([[[1.5, 0.0, 0.0, 0.0]]], dtype=DTYPE)

    activations, preacts = model.forward_with_cache(x)
    deltas = model.backward(activations, preacts, y)

    delta_out_expected = y - activations[-1]
    hidden_activation = activations[1]
    derivative_components = split_sigmoid_derivative(hidden_activation)[0, 0]
    d_r, d_i, d_j, d_k = derivative_components.unbind(dim=-1)

    coeff_0 = 0.25 * (d_r + d_i + d_j + d_k)
    coeff_i = 0.25 * (d_r + d_i - d_j - d_k)
    coeff_j = 0.25 * (d_r - d_i + d_j - d_k)
    coeff_k = 0.25 * (d_r - d_i - d_j + d_k)

    back_signal = manual_q_mul(q_conj(w1[0, 0]), delta_out_expected[0, 0])
    hidden_delta_expected = (
        coeff_0 * back_signal
        + q_involution(coeff_i * back_signal, "i")
        + q_involution(coeff_j * back_signal, "j")
        + q_involution(coeff_k * back_signal, "k")
    ).reshape(1, 1, 4)

    assert_qclose(deltas[-1], delta_out_expected)
    assert_qclose(deltas[0], hidden_delta_expected)


def test_ghr_backward_differs_from_cmlp_on_mixed_quaternion_hidden_state():
    ghr = GHRQuaternionMLP(layer_sizes=(1, 1, 1), activation="sigmoid", optimizer="SGD", source_style_init=False, init_scale=0.0).to(
        dtype=DTYPE
    )
    cmlp = CMLPQuaternionMLP(layer_sizes=(1, 1, 1), activation="sigmoid", source_style_init=False, init_scale=0.0).to(
        dtype=DTYPE
    )

    weights = [
        torch.tensor([[[0.8, -0.2, 0.6, 0.4]]], dtype=DTYPE),
        torch.tensor([[[0.3, 0.7, -0.5, 0.1]]], dtype=DTYPE),
    ]
    biases = [
        torch.tensor([[0.1, -0.3, 0.2, -0.4]], dtype=DTYPE),
        torch.zeros((1, 4), dtype=DTYPE),
    ]
    set_params(ghr, weights, biases)
    set_params(cmlp, weights, biases)

    x = torch.tensor([[[0.9, -1.1, 0.75, 0.5]]], dtype=DTYPE)
    y = torch.tensor([[[0.25, 0.0, 0.0, 0.0]]], dtype=DTYPE)

    ghr_activations, ghr_preacts = ghr.forward_with_cache(x)
    cmlp_activations, cmlp_preacts = cmlp.forward_with_cache(x)
    ghr_hidden_delta = ghr.backward(ghr_activations, ghr_preacts, y)[0]
    cmlp_hidden_delta = cmlp.backward(cmlp_activations, cmlp_preacts, y)[0]

    assert not torch.allclose(ghr_hidden_delta, cmlp_hidden_delta, atol=ATOL, rtol=RTOL)


def test_ghr_train_step_reduces_simple_one_layer_loss():
    model = GHRQuaternionMLP(
        layer_sizes=(1, 1),
        activation="sigmoid",
        optimizer="SGD",
        lr=0.1,
        source_style_init=False,
        init_scale=0.0,
    ).to(dtype=DTYPE)
    set_params(model, [torch.zeros((1, 1, 4), dtype=DTYPE)], [torch.zeros((1, 4), dtype=DTYPE)])

    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=DTYPE)
    y = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=DTYPE)

    pred_before = model(x)
    loss_before = torch.mean((y[..., 0] - pred_before[..., 0]) ** 2)

    metrics = model.train_step(x, y)
    pred_after = model(x)
    loss_after = torch.mean((y[..., 0] - pred_after[..., 0]) ** 2)

    assert math.isfinite(metrics["mse"])
    assert loss_after < loss_before

