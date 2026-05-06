import torch

from qvnn_pc.quaternion import (
    hamilton_product,
    q_conj,
    q_linear,
    q_matmul,
    q_mul,
    q_norm,
    quaternion_abs,
    quaternion_mse,
)


def test_hamilton_product_matches_known_basis_rule():
    i = torch.tensor([0.0, 1.0, 0.0, 0.0])
    j = torch.tensor([0.0, 0.0, 1.0, 0.0])
    expected_k = torch.tensor([0.0, 0.0, 0.0, 1.0])
    torch.testing.assert_close(hamilton_product(i, j), expected_k)


def test_hamilton_product_is_non_commutative():
    i = torch.tensor([0.0, 1.0, 0.0, 0.0])
    j = torch.tensor([0.0, 0.0, 1.0, 0.0])
    k = q_mul(i, j)
    minus_k = q_mul(j, i)
    torch.testing.assert_close(k, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(minus_k, torch.tensor([0.0, 0.0, 0.0, -1.0]))


def test_conjugate_and_norm_behave_consistently():
    q = torch.tensor([1.0, 2.0, 2.0, 0.0])
    torch.testing.assert_close(q_conj(q), torch.tensor([1.0, -2.0, -2.0, 0.0]))
    torch.testing.assert_close(q_norm(q), torch.tensor(3.0))
    torch.testing.assert_close(quaternion_abs(q), torch.tensor(3.0))


def test_quaternion_matmul_uses_left_multiplication_convention():
    i = torch.tensor([0.0, 1.0, 0.0, 0.0])
    j = torch.tensor([0.0, 0.0, 1.0, 0.0])
    k = torch.tensor([0.0, 0.0, 0.0, 1.0])

    W = torch.stack((i, j), dim=0).unsqueeze(0)
    x = torch.stack((j, k), dim=0)

    y = q_matmul(W, x)
    torch.testing.assert_close(y.squeeze(0), torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_q_linear_adds_bias_after_quaternion_matmul():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    W = identity.view(1, 1, 4)
    x = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
    b = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    torch.testing.assert_close(q_linear(x, W, b), x + b)


def test_quaternion_mse_is_zero_for_identical_inputs():
    y = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.5, 2.0, -3.0],
        ]
    )
    torch.testing.assert_close(quaternion_mse(y, y), torch.tensor(0.0))
