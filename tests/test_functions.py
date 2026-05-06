import torch

from qvnn_pc.functions import ackley_4d, griewank_4d, rastrigin_4d, schwefel_4d


def test_ackley_has_zero_value_at_origin():
    x = torch.zeros(4)
    torch.testing.assert_close(ackley_4d(x), torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_griewank_has_zero_value_at_origin():
    x = torch.zeros(4)
    torch.testing.assert_close(griewank_4d(x), torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_rastrigin_has_zero_value_at_origin():
    x = torch.zeros(4)
    torch.testing.assert_close(rastrigin_4d(x), torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_schwefel_is_near_zero_at_standard_optimum():
    optimum = torch.full((4,), 420.968746)
    value = schwefel_4d(optimum)
    assert value.abs() < 1e-2
