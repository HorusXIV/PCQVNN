"""Core quaternion algebra implemented with PyTorch tensors."""

from __future__ import annotations

import torch


def _validate_quaternion_shape(q: torch.Tensor) -> None:
    if q.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to have size 4, got {q.shape}.")


def hamilton_product(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Compute the Hamilton product q ⊗ p."""
    _validate_quaternion_shape(q)
    _validate_quaternion_shape(p)

    qr, qi, qj, qk = q.unbind(dim=-1)
    pr, pi, pj, pk = p.unbind(dim=-1)

    r = qr * pr - qi * pi - qj * pj - qk * pk
    i = qr * pi + qi * pr + qj * pk - qk * pj
    j = qr * pj - qi * pk + qj * pr + qk * pi
    k = qr * pk + qi * pj - qj * pi + qk * pr

    return torch.stack((r, i, j, k), dim=-1)


def q_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add two quaternion tensors componentwise."""
    _validate_quaternion_shape(a)
    _validate_quaternion_shape(b)
    return a + b


def q_conj(q: torch.Tensor) -> torch.Tensor:
    """Return the quaternion conjugate."""
    _validate_quaternion_shape(q)
    signs = q.new_tensor((1.0, -1.0, -1.0, -1.0))
    return q * signs


def q_hadamard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Componentwise quaternion product."""
    _validate_quaternion_shape(a)
    _validate_quaternion_shape(b)
    return a * b


def q_norm_sq(q: torch.Tensor) -> torch.Tensor:
    """Return the squared Euclidean norm of each quaternion."""
    _validate_quaternion_shape(q)
    return q.square().sum(dim=-1)


def q_norm(q: torch.Tensor) -> torch.Tensor:
    """Return the Euclidean norm of each quaternion."""
    _validate_quaternion_shape(q)
    return torch.linalg.vector_norm(q, dim=-1)


def quaternion_abs(q: torch.Tensor) -> torch.Tensor:
    """Alias for the quaternion magnitude."""
    return q_norm(q)


def q_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product alias."""
    return hamilton_product(a, b)


def q_matmul(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute sum_j W_ij ⊗ x_j using the fixed left-multiplication convention."""
    _validate_quaternion_shape(W)
    _validate_quaternion_shape(x)

    if W.ndim < 3:
        raise ValueError("Expected W to have shape (..., out_features, in_features, 4).")
    if x.ndim < 2:
        raise ValueError("Expected x to have shape (..., in_features, 4).")
    if W.shape[-2] != x.shape[-2]:
        raise ValueError(
            f"Incompatible shapes for quaternion matmul: W has {W.shape[-2]} inputs "
            f"but x has {x.shape[-2]} entries."
        )

    if W.ndim == 3 and x.ndim in {2, 3}:
        wr, wi, wj, wk = W.unbind(dim=-1)
        xr, xi, xj, xk = x.unbind(dim=-1)
        if x.ndim == 2:
            r = wr @ xr - wi @ xi - wj @ xj - wk @ xk
            i = wr @ xi + wi @ xr + wj @ xk - wk @ xj
            j = wr @ xj - wi @ xk + wj @ xr + wk @ xi
            k = wr @ xk + wi @ xj - wj @ xi + wk @ xr
        else:
            r = xr @ wr.T - xi @ wi.T - xj @ wj.T - xk @ wk.T
            i = xi @ wr.T + xr @ wi.T + xk @ wj.T - xj @ wk.T
            j = xj @ wr.T - xk @ wi.T + xr @ wj.T + xi @ wk.T
            k = xk @ wr.T + xj @ wi.T - xi @ wj.T + xr @ wk.T
        return torch.stack((r, i, j, k), dim=-1)

    products = hamilton_product(W, x.unsqueeze(-3))
    return products.sum(dim=-2)


def quaternion_matmul(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Alias for quaternion matrix multiplication."""
    return q_matmul(W, x)


def q_linear(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """Quaternion affine map using output = W ⊗ x + b."""
    y = q_matmul(W, x)
    if b is not None:
        _validate_quaternion_shape(b)
        y = q_add(y, b)
    return y


def quaternion_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean squared Euclidean distance between quaternion-valued targets."""
    _validate_quaternion_shape(y_pred)
    _validate_quaternion_shape(y_true)
    diff = y_pred - y_true
    return diff.square().sum(dim=-1).mean()
