import torch

from qvnn_pc.data import make_benchmark_regression_split


def test_bill_style_split_sizes_match_30000_protocol():
    split = make_benchmark_regression_split("ackley", n=30_000, sample_seed=7, split_seed=11)

    assert split.x_train.shape == (21_000, 1, 4)
    assert split.y_train.shape == (21_000, 1, 4)
    assert split.x_val.shape == (4_500, 1, 4)
    assert split.y_val.shape == (4_500, 1, 4)
    assert split.x_test.shape == (4_500, 1, 4)
    assert split.y_test.shape == (4_500, 1, 4)


def test_training_inputs_are_standardized_with_train_only_statistics():
    split = make_benchmark_regression_split("griewank", n=30_000, sample_seed=3, split_seed=5)
    x_train = split.x_train[:, 0, :]

    torch.testing.assert_close(x_train.mean(dim=0), torch.zeros(4), atol=2e-6, rtol=0.0)
    torch.testing.assert_close(
        x_train.std(dim=0, unbiased=False),
        torch.ones(4),
        atol=2e-5,
        rtol=0.0,
    )


def test_targets_are_quaternion_encoded_with_zero_imaginary_components():
    split = make_benchmark_regression_split("schwefel", n=30_000, sample_seed=13, split_seed=17)

    for target in (split.y_train, split.y_val, split.y_test):
        assert target.ndim == 3
        assert target.shape[1:] == (1, 4)
        torch.testing.assert_close(target[..., 1:], torch.zeros_like(target[..., 1:]))


def test_scalar_targets_expose_train_only_normalized_view():
    split = make_benchmark_regression_split("ackley", n=30_000, sample_seed=13, split_seed=17)

    torch.testing.assert_close(split.y_train_scalar_normalized.mean(), torch.tensor(0.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        split.y_train_scalar_normalized.std(unbiased=False),
        torch.tensor(1.0),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        split.y_train_scalar_normalized * split.y_std + split.y_mean,
        split.y_train_scalar,
        atol=1e-5,
        rtol=1e-5,
    )


def test_ackley_source_style_bounds_use_minus_30_to_30():
    split = make_benchmark_regression_split("ackley", n=30_000, sample_seed=29, split_seed=31)

    assert float(split.raw_x_train.min()) >= -30.0
    assert float(split.raw_x_train.max()) <= 30.0
    assert float(split.raw_x_val.min()) >= -30.0
    assert float(split.raw_x_val.max()) <= 30.0
    assert float(split.raw_x_test.min()) >= -30.0
    assert float(split.raw_x_test.max()) <= 30.0


def test_validation_and_test_use_training_standardization_not_separate_fit():
    split = make_benchmark_regression_split("rastrigin", n=30_000, sample_seed=19, split_seed=23)

    expected_val = (split.raw_x_val - split.x_mean) / split.x_std
    expected_test = (split.raw_x_test - split.x_mean) / split.x_std
    torch.testing.assert_close(split.x_val[:, 0, :], expected_val)
    torch.testing.assert_close(split.x_test[:, 0, :], expected_test)

    val_fit = (split.raw_x_val - split.raw_x_val.mean(dim=0)) / split.raw_x_val.std(dim=0, unbiased=False)
    test_fit = (split.raw_x_test - split.raw_x_test.mean(dim=0)) / split.raw_x_test.std(dim=0, unbiased=False)

    assert not torch.allclose(split.x_val[:, 0, :], val_fit, atol=1e-3, rtol=1e-3)
    assert not torch.allclose(split.x_test[:, 0, :], test_fit, atol=1e-3, rtol=1e-3)
