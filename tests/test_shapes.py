import torch

from qvnn_pc.activations import (
    split_leaky_relu,
    split_leaky_relu_derivative,
    split_sigmoid,
    split_sigmoid_derivative,
    split_swish,
    split_swish_derivative,
    split_tanh,
    split_tanh_derivative,
)
from qvnn_pc.quaternion import q_matmul


def test_q_matmul_supports_batched_inputs():
    W = torch.randn(3, 2, 4)
    x = torch.randn(5, 2, 4)
    y = q_matmul(W, x)
    assert y.shape == (5, 3, 4)


def test_split_activations_preserve_shape():
    q = torch.randn(7, 4)
    assert split_tanh(q).shape == q.shape
    assert split_sigmoid(q).shape == q.shape
    assert split_swish(q).shape == q.shape
    assert split_leaky_relu(q).shape == q.shape


def test_split_activation_derivatives_match_autograd():
    q = torch.tensor(
        [[-1.25, -0.5, 0.75, 1.5], [0.2, -2.0, 1.1, -0.7]],
        requires_grad=True,
    )

    split_tanh(q).sum().backward(retain_graph=True)
    tanh_grad = q.grad.detach().clone()
    q.grad.zero_()

    split_sigmoid(q).sum().backward(retain_graph=True)
    sigmoid_grad = q.grad.detach().clone()
    q.grad.zero_()

    split_swish(q).sum().backward(retain_graph=True)
    swish_grad = q.grad.detach().clone()
    q.grad.zero_()

    split_leaky_relu(q, negative_slope=0.2).sum().backward()
    leaky_relu_grad = q.grad.detach().clone()

    q_detached = q.detach()
    torch.testing.assert_close(tanh_grad, split_tanh_derivative(q_detached))
    torch.testing.assert_close(sigmoid_grad, split_sigmoid_derivative(q_detached))
    torch.testing.assert_close(swish_grad, split_swish_derivative(q_detached))
    torch.testing.assert_close(
        leaky_relu_grad,
        split_leaky_relu_derivative(q_detached, negative_slope=0.2),
    )
